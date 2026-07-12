import copy
import importlib.util
from dataclasses import replace
import json
import pathlib

import pytest
import torch


from audiolens.fitting import (
    LIBRISPEECH_REVISION,
    audit_manifest_rows,
    canonical_json_bytes,
    config_digest,
    lens_from_fit_checkpoint,
    paired_resume_prefix,
    validate_lens,
    validate_runtime_lens_file,
)
from audiolens.models import DEFAULT_MODEL_PROFILE
from audiolens.models.base import AudioFitContractError
from audiolens.models.gemma4 import (
    GemmaPreparedAudioLensModel,
    crop_attention_mapping,
    expand_batch,
    validate_audio_layout,
)


def _tiny_profile():
    return replace(
        DEFAULT_MODEL_PROFILE,
        key="fake-audio",
        slug="fake-audio",
        model_id="test/fake-audio",
        model_revision="fake-revision",
        adapter_source="tests:FakeAudioRuntime",
        d_model=2,
        source_layers=(1, 3),
        target_layer=4,
        max_sequence_length=80,
        skip_first=1,
        dimension_batch_size=2,
        read_layer=3,
        read_layers=(1, 3),
    )

_FIT_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "modal_fit_lens.py"
_FIT_SPEC = importlib.util.spec_from_file_location("modal_fit_lens", _FIT_SCRIPT)
assert _FIT_SPEC is not None and _FIT_SPEC.loader is not None
canonical_fit = importlib.util.module_from_spec(_FIT_SPEC)
_FIT_SPEC.loader.exec_module(canonical_fit)


def test_fit_manifest_loader_imports_without_modal_project_tree(
    tmp_path,
    monkeypatch,
):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    bundled_script = script_dir / "modal_fit_lens.py"
    bundled_script.write_bytes(_FIT_SCRIPT.read_bytes())
    monkeypatch.setenv("AUDIOLENS_GIT_REVISION", "a" * 40)
    monkeypatch.setenv("AUDIOLENS_SOURCE_DIGEST", "b" * 64)
    monkeypatch.setenv("AUDIOLENS_LOCK_SHA256", "c" * 64)
    spec = importlib.util.spec_from_file_location(
        "bundled_modal_fit_lens",
        bundled_script,
    )
    assert spec is not None and spec.loader is not None
    bundled = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundled)
    assert bundled.image is bundled.app is bundled.vol is None
    assert callable(bundled.load_completed_fit_manifest)

    class MountedVolume:
        commits = 0

        def commit(self):
            self.commits += 1

    mounted = MountedVolume()

    class VolumeAPI:
        @staticmethod
        def from_name(name, *, create_if_missing):
            assert name == "audiolens-vol"
            assert create_if_missing is True
            return mounted

    monkeypatch.setattr(bundled.modal, "Volume", VolumeAPI)
    bundled._commit_volume()
    assert bundled.vol is mounted
    assert mounted.commits == 1


@pytest.mark.parametrize(
    "relative",
    [
        "src/audiolens/__init__.py",
        "src/audiolens/models/__init__.py",
        "src/audiolens/models/base.py",
        "src/audiolens/models/gemma4.py",
    ],
)
def test_fit_source_digest_tracks_model_profile_sources(
    tmp_path,
    monkeypatch,
    relative,
):
    for source in canonical_fit.FIT_SOURCE_RELATIVES:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(canonical_fit, "REPO_ROOT", tmp_path)
    before = canonical_fit._source_digest()
    (tmp_path / relative).write_text(f"changed:{relative}", encoding="utf-8")
    assert canonical_fit._source_digest() != before


def test_canonical_config_digest_is_stable_but_order_sensitive():
    a = {"model": "gemma", "prompts": ["a", "b"]}
    assert canonical_json_bytes(a) == canonical_json_bytes(
        {"prompts": ["a", "b"], "model": "gemma"}
    )
    assert config_digest(a) == config_digest(copy.deepcopy(a))
    changed = copy.deepcopy(a)
    changed["prompts"].reverse()
    assert config_digest(a) != config_digest(changed)


def test_output_metadata_is_outside_immutable_config_digest():
    run = {"config": {"manifest_sha256": "a" * 64}, "outputs": {}}
    before = config_digest(run["config"])
    run["outputs"]["mixed"] = {"sha256": "b" * 64}
    assert config_digest(run["config"]) == before


def test_audio_layout_selects_only_audio_positions():
    # Five framing tokens, 50 contiguous audio tokens, then three closers.
    ids = torch.tensor([[2, 105, 2364, 107, 256000] + [258881] * 50 + [258883, 106, 107]])
    layout = validate_audio_layout(ids, 258881)
    assert layout.audio_start == 5
    assert layout.n_audio_tokens == 50
    assert layout.stop == 55
    assert layout.n_valid_positions == 38
    assert torch.equal(ids[0, : layout.stop][layout.valid_mask], torch.full((38,), 258881))


def test_audio_layout_rejects_gaps_and_over_budget():
    with pytest.raises(AudioFitContractError, match="not contiguous"):
        validate_audio_layout(torch.tensor([[2] + [9] * 20 + [8, 9]]), 9)
    with pytest.raises(AudioFitContractError, match="above max_length"):
        validate_audio_layout(torch.tensor([[2] + [9] * 130]), 9)


def test_crop_attention_mapping_and_expand_batch():
    full = torch.arange(36).view(1, 1, 6, 6)
    sliding = torch.ones(1, 1, 6, 6)
    cropped = crop_attention_mapping(
        {"full_attention": full, "sliding_attention": sliding}, 4
    )
    assert cropped["full_attention"].shape == (1, 1, 4, 4)
    assert torch.equal(cropped["full_attention"], full[..., :4, :4])
    expanded = expand_batch(cropped["sliding_attention"], 8)
    assert expanded.shape == (8, 1, 4, 4)
    assert expanded.untyped_storage().data_ptr() == cropped["sliding_attention"].untyped_storage().data_ptr()
    with pytest.raises(AudioFitContractError, match="unexpected attention"):
        crop_attention_mapping({"full_attention": full}, 4)
    with pytest.raises(AudioFitContractError, match="must be None or a Tensor"):
        crop_attention_mapping(
            {"full_attention": lambda: None, "sliding_attention": sliding}, 4
        )


def test_adapter_rejects_forward_before_encode():
    adapter = object.__new__(GemmaPreparedAudioLensModel)
    adapter._prepared = None
    with pytest.raises(AudioFitContractError, match="before encode"):
        adapter.forward(torch.ones(1, 2, dtype=torch.long))


def _manifest_rows():
    rows = []
    for config, split, offset in (("clean", "train.100", 0), ("other", "train.500", 64)):
        for i in range(64):
            ident = offset + i
            rows.append(
                {
                    "dataset": "openslr/librispeech_asr",
                    "revision": LIBRISPEECH_REVISION,
                    "config": config,
                    "split": split,
                    "selection_index": ident,
                    "id": f"sample-{ident}",
                    "speaker_id": ident,
                    "chapter_id": ident,
                    "transcript": "a nonempty transcript",
                    "duration_seconds": 3.0,
                    "sampling_rate": 16_000,
                    "audio_sha256": f"{ident:064x}",
                    "audio_start": 5,
                    "n_audio_tokens": 75,
                    "sliced_seq_len": 80,
                    "n_valid_positions": 63,
                    "volume_path": f"/vol/audio/{ident}.flac",
                }
            )
    return rows


def test_manifest_audit_accepts_exact_fixed_design():
    audit_manifest_rows(_manifest_rows())


@pytest.mark.parametrize(
    "mutation", ["duplicate", "stratum", "duration", "hash", "reordered"]
)
def test_manifest_audit_rejects_drift(mutation):
    rows = _manifest_rows()
    if mutation == "duplicate":
        rows[1]["speaker_id"] = rows[0]["speaker_id"]
    elif mutation == "stratum":
        rows[1]["config"] = "other"
        rows[1]["split"] = "train.500"
    elif mutation == "duration":
        rows[1]["duration_seconds"] = 5.0
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[1]["audio_sha256"] = "not-a-hash"
    with pytest.raises(AudioFitContractError):
        audit_manifest_rows(rows)


def test_manifest_audit_uses_profile_sequence_geometry():
    rows = _manifest_rows()
    profile = replace(_tiny_profile(), max_sequence_length=79)
    with pytest.raises(AudioFitContractError, match="max sequence length 79"):
        audit_manifest_rows(rows, profile=profile)


def test_fit_checkpoint_reconstructs_exact_fp32_mean(tmp_path):
    profile = _tiny_profile()
    checkpoint = tmp_path / "fit.pt"
    jacobian_sum = {layer: torch.eye(2) * 4 for layer in profile.source_layers}
    torch.save(
        {
            "jacobian_sum": jacobian_sum,
            "n_done": 4,
            "next_idx": 4,
            "source_layers": list(profile.source_layers),
            "target_layer": profile.target_layer,
            "skip_first": profile.skip_first,
        },
        checkpoint,
    )
    lens = lens_from_fit_checkpoint(checkpoint, 4, profile=profile)
    validate_lens(lens, 4, profile=profile)
    assert torch.equal(lens.jacobians[3], torch.eye(2))


def test_fit_checkpoint_rejects_wrong_resume_count(tmp_path):
    profile = _tiny_profile()
    checkpoint = tmp_path / "fit.pt"
    torch.save(
        {
            "jacobian_sum": {
                layer: torch.eye(profile.d_model) for layer in profile.source_layers
            },
            "n_done": 3,
            "next_idx": 4,
            "source_layers": list(profile.source_layers),
            "target_layer": profile.target_layer,
            "skip_first": profile.skip_first,
        },
        checkpoint,
    )
    with pytest.raises(AudioFitContractError, match="counts"):
        lens_from_fit_checkpoint(checkpoint, 4, profile=profile)


@pytest.mark.parametrize("mutation", ["target", "width", "layers"])
def test_fit_checkpoint_rejects_profile_geometry_mismatch(tmp_path, mutation):
    profile = _tiny_profile()
    state = {
        "jacobian_sum": {
            layer: torch.eye(profile.d_model) for layer in profile.source_layers
        },
        "n_done": 4,
        "next_idx": 4,
        "source_layers": list(profile.source_layers),
        "target_layer": profile.target_layer,
        "skip_first": profile.skip_first,
    }
    if mutation == "target":
        state["target_layer"] = profile.target_layer + 1
    elif mutation == "width":
        state["jacobian_sum"][profile.source_layers[0]] = torch.eye(
            profile.d_model + 1
        )
    else:
        state["source_layers"] = [0, *profile.source_layers[1:]]
    checkpoint = tmp_path / "mismatched-fit.pt"
    torch.save(state, checkpoint)
    with pytest.raises(AudioFitContractError):
        lens_from_fit_checkpoint(checkpoint, 4, profile=profile)


def test_runtime_lens_file_requires_fp16(tmp_path):
    import jlens

    profile = _tiny_profile()
    lens = jlens.JacobianLens(
        jacobians={layer: torch.eye(2) for layer in profile.source_layers},
        n_prompts=4,
        d_model=profile.d_model,
    )
    path = tmp_path / "runtime.pt"
    lens.save(path)
    validate_runtime_lens_file(path, 4, profile=profile)
    state = torch.load(path, weights_only=True)
    state["J"] = {layer: value.float() for layer, value in state["J"].items()}
    torch.save(state, path)
    with pytest.raises(AudioFitContractError, match="not serialized as fp16"):
        validate_runtime_lens_file(path, 4, profile=profile)


def test_validate_lens_rejects_alternate_profile_layers_and_width():
    import jlens

    profile = _tiny_profile()
    wrong_layers = jlens.JacobianLens(
        jacobians={0: torch.eye(2), 3: torch.eye(2)},
        n_prompts=4,
        d_model=2,
    )
    with pytest.raises(AudioFitContractError, match="layers/d_model"):
        validate_lens(wrong_layers, 4, profile=profile)
    wrong_width = jlens.JacobianLens(
        jacobians={layer: torch.eye(3) for layer in profile.source_layers},
        n_prompts=4,
        d_model=3,
    )
    with pytest.raises(AudioFitContractError, match="layers/d_model"):
        validate_lens(wrong_width, 4, profile=profile)


def test_synthetic_prompt_weighted_merge_is_exact():
    import jlens

    text = jlens.JacobianLens(
        jacobians={layer: torch.ones(2, 2) for layer in range(34)},
        n_prompts=400,
        d_model=2,
    )
    audio = jlens.JacobianLens(
        jacobians={layer: torch.full((2, 2), 3.0) for layer in range(34)},
        n_prompts=128,
        d_model=2,
    )
    mixed = jlens.JacobianLens.merge([text, audio])
    assert mixed.n_prompts == 528
    assert torch.equal(mixed.jacobians[29], torch.full((2, 2), (400 + 384) / 528))


def _paired_config():
    return {
        "lenses": {"text400": {}, "mixed528": {}},
        "read_layers": [29],
        "anchors": {"clusters": ["joy"]},
        "topk": 1,
    }


def _paired_row(clip: str):
    actor = clip.removesuffix(".wav").split("-")[-1]
    layer = {"anchor_mass": {"joy": 0.1}, "topk_ids": [1], "topk_toks": ["x"]}
    return {
        "clip": clip,
        "meta": {
            "emotion": "happy",
            "intensity": "normal",
            "statement": "Kids are talking by the door",
            "rep": "01",
            "actor": actor,
        },
        "n_audio_tokens": 50,
        "seq_len": 58,
        "readouts": {
            "text400": {"layers": {"29": layer}},
            "mixed528": {"layers": {"29": copy.deepcopy(layer)}},
        },
    }


def test_paired_resume_accepts_only_sorted_complete_prefix_and_repairs_torn_tail(tmp_path):
    clips = ["03-01-03-01-01-01-01.wav", "03-01-03-01-01-01-02.wav"]
    path = tmp_path / "paired.jsonl"
    first = json.dumps(_paired_row(clips[0]), sort_keys=True) + "\n"
    path.write_text(first + '{"clip":')
    assert paired_resume_prefix(path, _paired_config(), clips) == {clips[0]}
    assert path.read_text() == first

    path.write_text(json.dumps(_paired_row(clips[1])) + "\n")
    with pytest.raises(AudioFitContractError, match="expected sorted clip"):
        paired_resume_prefix(path, _paired_config(), clips)

    broken = _paired_row(clips[0])
    broken["readouts"]["mixed528"]["layers"].clear()
    path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(AudioFitContractError, match="layer mismatch"):
        paired_resume_prefix(path, _paired_config(), clips)


def test_neuronpedia_wikitext_rechunking_exact_golden():
    rows = [
        {"text": "   "},
        {"text": " = Section = "},
        {"text": " abc "},
        {"text": "defghij"},
        {"text": " klmn "},
    ]
    assert canonical_fit._wikitext_prompts(
        2, rows=rows, max_chars=10, min_chars=3
    ) == ["abc defgh", "ij klmn"]


def test_neuronpedia_chunk_boundary_is_strictly_greater_than():
    consumed = []

    def rows():
        consumed.append("exact")
        yield {"text": "123456789"}  # Leading separator makes the buffer exactly 10.
        consumed.append("overflow")
        yield {"text": "A"}

    assert canonical_fit._wikitext_prompts(
        1, rows=rows(), max_chars=10, min_chars=1
    ) == ["123456789"]
    assert consumed == ["exact", "overflow"]


def test_neuronpedia_rechunking_rejects_short_tail_and_insufficient_count():
    with pytest.raises(RuntimeError, match=r"0/1 prompts"):
        canonical_fit._wikitext_prompts(
            1,
            rows=[{"text": "short"}],
            max_chars=20,
            min_chars=6,
        )
    with pytest.raises(RuntimeError, match=r"1/2 prompts"):
        canonical_fit._wikitext_prompts(
            2,
            rows=[{"text": "123456789 A"}],
            max_chars=10,
            min_chars=2,
        )


def _canonical_text_config(prompts=None):
    if prompts is None:
        prompts = ["first corpus chunk", "second corpus chunk"]
    return canonical_fit._build_fit_config(
        prompts,
        requested_count=len(prompts),
        runtime_identity={
            "packages": {"torch": "pinned"},
            "python": "3.12",
            "cuda": "pinned",
            "device": "H100",
            "torch_cuda_alloc_conf": "expandable_segments:True",
            "modal_image_id": "im-pinned",
        },
        source_identity={"git_revision": "a" * 40, "digest": "b" * 64},
    )


def test_modal_timeout_is_maximum_24_hours_and_content_addressed():
    config = _canonical_text_config()
    assert canonical_fit.MODAL_FUNCTION_TIMEOUT_SECONDS == 86_400
    assert config["runtime"]["modal_function_timeout_seconds"] == 86_400
    changed = copy.deepcopy(config)
    changed["runtime"]["modal_function_timeout_seconds"] = 4 * 60 * 60
    assert config_digest(changed) != config_digest(config)
    with pytest.raises(AudioFitContractError, match="source/runtime identity"):
        canonical_fit._validate_pinned_fit_config(changed)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("model revision", lambda c: c["model"].update(revision="changed")),
        ("tokenizer revision", lambda c: c["tokenizer"].update(revision="changed")),
        ("data revision", lambda c: c["dataset"].update(revision="changed")),
        ("JLens revision", lambda c: c["jlens"].update(revision="changed")),
        (
            "chunk width",
            lambda c: c["dataset"]["chunking"].update(max_chars=1_999),
        ),
        (
            "raw/chat policy",
            lambda c: c["prompt_policy"].update(kind="chat", chat_template=True),
        ),
        ("BOS policy", lambda c: c["prompt_policy"].update(force_bos=False)),
        ("geometry", lambda c: c["fit"].update(skip_first=15)),
        ("model dtype", lambda c: c["fit"].update(model_dtype="float16")),
        ("backend", lambda c: c["fit"].update(attention_backend="sdpa")),
        ("lock identity", lambda c: c["lock"].update(uv_lock_sha256="c" * 64)),
        ("source identity", lambda c: c["source"].update(digest="d" * 64)),
        ("runtime identity", lambda c: c["runtime"].update(device="other")),
    ],
)
def test_every_frozen_fit_input_changes_content_identity(name, mutate):
    del name
    base = _canonical_text_config()
    changed = copy.deepcopy(base)
    mutate(changed)
    assert config_digest(changed) != config_digest(base)


def test_prompt_selection_and_order_are_part_of_fit_identity():
    rechunked = _canonical_text_config(["joined row one", "joined row two"])
    legacy_long_rows = _canonical_text_config(["joined row one"[:10], "joined row two"[:10]])
    reordered = _canonical_text_config(["joined row two", "joined row one"])
    assert config_digest(rechunked) != config_digest(legacy_long_rows)
    assert config_digest(rechunked) != config_digest(reordered)


def test_fit_manifest_is_idempotent_and_rejects_stale_identity(tmp_path):
    config = _canonical_text_config()
    manifest_path, paths, first = canonical_fit._initialize_fit_manifest(tmp_path, config)
    _, same_paths, second = canonical_fit._initialize_fit_manifest(tmp_path, copy.deepcopy(config))
    assert same_paths == paths
    assert second == first

    changed = copy.deepcopy(config)
    changed["fit"]["max_seq_len"] = 127
    with pytest.raises(AudioFitContractError, match="identity mismatch"):
        canonical_fit._ensure_fit_manifest_at(
            manifest_path,
            changed,
            canonical_fit._run_paths(changed),
        )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"\xff\xfe", id="invalid-utf8"),
        pytest.param(b"[]", id="non-mapping-root"),
    ],
)
def test_fit_manifest_readers_wrap_invalid_text_and_roots(tmp_path, payload):
    config = _canonical_text_config()
    paths = canonical_fit._run_paths(config)
    manifest_path = tmp_path / paths["manifest"]
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(payload)
    with pytest.raises(AudioFitContractError, match="invalid fit manifest"):
        canonical_fit._ensure_fit_manifest_at(
            manifest_path,
            config,
            paths,
        )
    with pytest.raises(AudioFitContractError, match="invalid fit manifest"):
        canonical_fit.load_completed_fit_manifest(
            manifest_path,
            volume_root=tmp_path,
        )


def test_checkpoint_reader_wraps_nonmapping_and_corrupt_torch_payloads(tmp_path):
    config = _canonical_text_config()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(["not", "a", "mapping"], checkpoint_path)
    with pytest.raises(AudioFitContractError, match="invalid fit checkpoint root"):
        canonical_fit._read_checkpoint_state(
            checkpoint_path,
            config,
            maximum_count=2,
        )

    checkpoint_path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(AudioFitContractError, match="invalid fit checkpoint"):
        canonical_fit._read_checkpoint_state(
            checkpoint_path,
            config,
            maximum_count=2,
        )


def test_prefix_snapshot_survives_resume_then_is_deleted_after_manifest_commit(
    tmp_path,
    monkeypatch,
):
    config = _canonical_text_config(["prompt"] * 1_000)
    paths = canonical_fit._run_paths(config)
    assert set(paths) == {"manifest", "checkpoint", "lens"}
    prefix_path = tmp_path / canonical_fit._diagnostic_prefix_checkpoint(config)
    prefix_path.parent.mkdir(parents=True)
    prefix_path.write_bytes(b"fp32-prefix")

    pending = {
        "status": "pending",
        "canonical": False,
        "config": config,
        "stability": None,
    }
    assert canonical_fit._cleanup_completed_prefix_snapshot(pending, tmp_path) is False
    assert prefix_path.is_file()

    complete = {
        "status": "complete",
        "canonical": True,
        "config": config,
        "paths": paths,
        "stability": {"kind": "fp32_prefix_500_disjoint_halves"},
    }
    manifest_path = tmp_path / paths["manifest"]
    manifest_path.parent.mkdir(parents=True)
    commits = []

    def observe_commit():
        persisted = json.loads(manifest_path.read_text())
        commits.append(
            (
                persisted["status"],
                persisted["stability"]["kind"],
                prefix_path.exists(),
            )
        )

    monkeypatch.setattr(canonical_fit, "_commit_volume", observe_commit)
    canonical_fit._persist_completed_manifest(
        manifest_path,
        complete,
        tmp_path,
    )
    assert commits == [
        ("complete", "fp32_prefix_500_disjoint_halves", True),
        ("complete", "fp32_prefix_500_disjoint_halves", False),
    ]
    assert not prefix_path.exists()
    assert set(json.loads(manifest_path.read_text())["paths"]) == {
        "manifest",
        "checkpoint",
        "lens",
    }


def test_stale_manifest_rejects_before_model_loading(tmp_path):
    rows = [{"text": "x" * 2_001}]
    prompts = canonical_fit._wikitext_prompts(1, rows=rows)
    runtime = _canonical_text_config(["x"])["runtime"]
    source = {"git_revision": "a" * 40, "digest": "b" * 64}
    config = canonical_fit._build_fit_config(
        prompts,
        requested_count=1,
        runtime_identity=runtime,
        source_identity=source,
    )
    manifest_path, _, record = canonical_fit._initialize_fit_manifest(tmp_path, config)
    record["config"]["fit"]["target_layer"] = 33
    manifest_path.write_text(json.dumps(record))
    model_loaded = False

    def forbidden_model_loader():
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model loader must not run")

    with pytest.raises(AudioFitContractError, match="identity mismatch"):
        canonical_fit._fit_lens_impl(
            1,
            volume_root=tmp_path,
            rows=rows,
            runtime_identity=runtime,
            source_identity=source,
            model_loader=forbidden_model_loader,
        )
    assert not model_loaded


def test_completed_manifest_binds_relative_fp16_lens_bytes(tmp_path, monkeypatch):
    import jlens

    monkeypatch.setattr(canonical_fit, "D_MODEL", 2)
    config = _canonical_text_config(["one smoke prompt"])
    manifest_path, paths, record = canonical_fit._initialize_fit_manifest(
        tmp_path,
        config,
    )
    lens_path = tmp_path / paths["lens"]
    lens_path.parent.mkdir(parents=True)
    lens = jlens.JacobianLens(
        jacobians={layer: torch.eye(2) for layer in range(34)},
        n_prompts=1,
        d_model=2,
    )
    lens.save(lens_path)
    record.update(
        {
            "status": "complete",
            "canonical": False,
            "lens": canonical_fit._lens_manifest_metadata(
                lens_path,
                paths["lens"],
                config,
            ),
            "checkpoint": {
                "relative_path": paths["checkpoint"],
                "sha256": "c" * 64,
                "bytes": 123,
                "dtype": "float32",
                "kind": "running_sum",
                "n_done": 1,
                "next_idx": 1,
                "d_model": 2,
                "source_layers": list(range(34)),
                "target_layer": 34,
                "skip_first": 16,
            },
        }
    )
    manifest_path.write_text(json.dumps(record, sort_keys=True))
    loaded = canonical_fit.load_completed_fit_manifest(manifest_path)
    assert loaded["lens"]["relative_path"] == paths["lens"]
    assert loaded["lens"]["dtype"] == "float16"
    assert loaded["canonical"] is False

    state = torch.load(lens_path, weights_only=True)
    state["J"][0] = state["J"][0].float()
    torch.save(state, lens_path)
    with pytest.raises(AudioFitContractError, match="lens metadata"):
        canonical_fit.load_completed_fit_manifest(manifest_path)


class _CausalLinear(torch.nn.Module):
    def __init__(self, local, past):
        super().__init__()
        self.register_buffer("local", torch.tensor(local, dtype=torch.float64))
        self.register_buffer("past", torch.tensor(past, dtype=torch.float64))

    def forward(self, hidden):
        prior = torch.cat(
            [torch.zeros_like(hidden[:, :1]), hidden.cumsum(dim=1)[:, :-1]],
            dim=1,
        )
        return hidden @ self.local.T + prior @ self.past.T


class _AsymmetricCausalModel:
    n_layers = 3
    d_model = 2

    def __init__(self):
        self.layers = torch.nn.ModuleList(
            [
                _CausalLinear(
                    [[1.1, -0.2], [0.3, 0.8]],
                    [[0.2, 0.4], [-0.1, 0.3]],
                ),
                _CausalLinear(
                    [[0.7, 0.5], [-0.4, 1.2]],
                    [[0.6, -0.2], [0.1, 0.35]],
                ),
                _CausalLinear(
                    [[1.3, -0.7], [0.2, 0.9]],
                    [[-0.3, 0.45], [0.55, 0.15]],
                ),
            ]
        )
        self._ids = {
            "short": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "long": torch.tensor([[2, 1, 3, 5, 4]], dtype=torch.long),
        }
        self.norm_weight = torch.tensor([1.4, 0.6], dtype=torch.float64)
        self.unembedding = torch.tensor(
            [[0.8, -0.3], [0.2, 1.1], [-0.5, 0.7]],
            dtype=torch.float64,
        )

    def encode(self, text, *, max_length=512):
        return self._ids[text][:, :max_length]

    @staticmethod
    def _embed(input_ids):
        values = input_ids.to(torch.float64)
        return torch.stack((values / 5, (values.square() + 1) / 7), dim=-1)

    def forward(self, input_ids):
        hidden = self._embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def unembed(self, residual):
        residual = residual.to(torch.float64)
        normalized = residual * torch.rsqrt(
            residual.square().mean(dim=-1, keepdim=True) + 0.25
        )
        normalized = normalized * self.norm_weight
        return normalized @ self.unembedding.T


def _independent_prompt_oracle(
    model,
    prompt,
    *,
    source_layers=(0, 1),
    target_layer=2,
    skip_first=1,
    same_position_only=False,
):
    input_ids = model.encode(prompt, max_length=8)
    hidden = model._embed(input_ids)
    activations = []
    for layer in model.layers:
        hidden = layer(hidden)
        activations.append(hidden)
    valid = list(range(skip_first, input_ids.shape[1] - 1))
    matrices = {}
    for source_layer in source_layers:
        source = activations[source_layer].detach().clone().requires_grad_(True)
        target = source
        for layer in model.layers[source_layer + 1 : target_layer + 1]:
            target = layer(target)
        rows = []
        for output_dim in range(model.d_model):
            if same_position_only:
                current_rows = []
                for position in valid:
                    gradient = torch.autograd.grad(
                        target[0, position, output_dim],
                        source,
                        retain_graph=True,
                    )[0]
                    current_rows.append(gradient[0, position])
                rows.append(torch.stack(current_rows).mean(dim=0))
            else:
                gradient = torch.autograd.grad(
                    target[0, valid, output_dim].sum(),
                    source,
                    retain_graph=True,
                )[0]
                rows.append(gradient[0, valid].mean(dim=0))
        matrices[source_layer] = torch.stack(rows)
    return matrices, len(valid)


def _equal_prompt_oracle(model, prompts, **kwargs):
    per_prompt = [
        _independent_prompt_oracle(model, prompt, **kwargs)[0]
        for prompt in prompts
    ]
    return {
        layer: sum(result[layer] for result in per_prompt) / len(per_prompt)
        for layer in per_prompt[0]
    }


def test_same_config_resume_preserves_exact_fp32_running_sum(tmp_path):
    import jlens

    prompts = ["short", "long", "short"]
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    resumed_path = tmp_path / "resumed.pt"
    uninterrupted = jlens.fit(
        _AsymmetricCausalModel(),
        prompts=prompts,
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=2,
        max_seq_len=8,
        skip_first=1,
        checkpoint_path=str(uninterrupted_path),
        checkpoint_every=1,
        resume=False,
    )
    jlens.fit(
        _AsymmetricCausalModel(),
        prompts=prompts[:1],
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=2,
        max_seq_len=8,
        skip_first=1,
        checkpoint_path=str(resumed_path),
        checkpoint_every=1,
        resume=False,
    )
    resumed = jlens.fit(
        _AsymmetricCausalModel(),
        prompts=prompts,
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=2,
        max_seq_len=8,
        skip_first=1,
        checkpoint_path=str(resumed_path),
        checkpoint_every=1,
        resume=True,
    )
    uninterrupted_state = torch.load(uninterrupted_path, weights_only=True)
    resumed_state = torch.load(resumed_path, weights_only=True)
    assert resumed_state["n_done"] == resumed_state["next_idx"] == len(prompts)
    for layer in (0, 1):
        assert torch.equal(
            resumed_state["jacobian_sum"][layer],
            uninterrupted_state["jacobian_sum"][layer],
        )
        assert torch.equal(resumed.jacobians[layer], uninterrupted.jacobians[layer])


def test_released_causal_estimator_matches_independent_asymmetric_oracle():
    import jlens

    model = _AsymmetricCausalModel()
    prompts = ["short", "long", "short"]
    lens = jlens.fit(
        model,
        prompts=prompts,
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=2,
        max_seq_len=8,
        skip_first=1,
        checkpoint_path=None,
        checkpoint_every=None,
        resume=False,
    )
    expected = _equal_prompt_oracle(model, prompts)
    for layer in (0, 1):
        torch.testing.assert_close(
            lens.jacobians[layer].double(),
            expected[layer],
            rtol=1e-6,
            atol=1e-6,
        )

    same_position = _equal_prompt_oracle(
        model,
        prompts,
        same_position_only=True,
    )
    wrong_target = _equal_prompt_oracle(
        model,
        prompts,
        source_layers=(0,),
        target_layer=1,
    )
    prompt_results = [
        _independent_prompt_oracle(model, prompt)
        for prompt in prompts
    ]
    causal_pair_weights = [
        n_valid * (n_valid + 1) // 2
        for _, n_valid in prompt_results
    ]
    causal_source_target_pair_weighted = {
        layer: sum(
            result[layer] * pair_weight
            for (result, _), pair_weight in zip(
                prompt_results,
                causal_pair_weights,
                strict=True,
            )
        )
        / sum(causal_pair_weights)
        for layer in (0, 1)
    }
    assert not torch.allclose(expected[0], expected[0].T)
    assert not torch.allclose(expected[0], same_position[0])
    assert not torch.allclose(expected[0], wrong_target[0])
    assert not torch.allclose(
        expected[0],
        causal_source_target_pair_weighted[0],
    )

    residual = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    transported = lens.transport(residual, 0)
    assert torch.equal(transported, residual @ lens.jacobians[0].T)
    assert not torch.allclose(transported, residual @ lens.jacobians[0])

    logits = model.unembed(transported)
    transported64 = (residual @ lens.jacobians[0].T).double()
    normalized = transported64 * torch.rsqrt(
        transported64.square().mean(dim=-1, keepdim=True) + 0.25
    )
    expected_logits = (
        normalized * torch.tensor([1.4, 0.6], dtype=torch.float64)
    ) @ torch.tensor(
        [[0.8, -0.3], [0.2, 1.1], [-0.5, 0.7]],
        dtype=torch.float64,
    ).T
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
