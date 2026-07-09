import copy
from dataclasses import replace
import json

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
from audiolens.models import (
    DEFAULT_MODEL_PROFILE,
    AudioFitContractError,
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
