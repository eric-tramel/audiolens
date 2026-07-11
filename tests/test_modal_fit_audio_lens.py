import builtins
import importlib.util
import os
import pathlib
import sys

import pytest

from audiolens.models.base import AudioFitContractError


_SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "modal_fit_audio_lens.py"


def _load_script(module_name):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULES_BEFORE_IMPORT = set(sys.modules)
_PREVIOUS_DISABLE_MODAL = os.environ.get("AUDIOLENS_DISABLE_MODAL")
os.environ["AUDIOLENS_DISABLE_MODAL"] = "1"
try:
    modal_fit_audio_lens = _load_script("modal_fit_audio_lens")
finally:
    if _PREVIOUS_DISABLE_MODAL is None:
        os.environ.pop("AUDIOLENS_DISABLE_MODAL", None)
    else:
        os.environ["AUDIOLENS_DISABLE_MODAL"] = _PREVIOUS_DISABLE_MODAL
_MODULES_LOADED_BY_SCRIPT = set(sys.modules) - _MODULES_BEFORE_IMPORT
_FORBIDDEN_IMPORT_ROOTS = {"torch", "transformers", "datasets", "jlens"}


def test_disabled_script_import_skips_deployment_and_is_import_light(monkeypatch):
    imported = []
    original_import = builtins.__import__

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("AUDIOLENS_DISABLE_MODAL", "1")
    monkeypatch.setattr(builtins, "__import__", recording_import)
    reloaded = _load_script("disabled_modal_fit_audio_lens")

    attempted_roots = {name.partition(".")[0] for name in imported}
    loaded_roots = {name.partition(".")[0] for name in _MODULES_LOADED_BY_SCRIPT}
    assert attempted_roots.isdisjoint(_FORBIDDEN_IMPORT_ROOTS)
    assert loaded_roots.isdisjoint(_FORBIDDEN_IMPORT_ROOTS)
    assert reloaded.image is reloaded.app is reloaded.vol is None


def test_source_metadata_and_ranks_are_stable_across_enumeration_order():
    from audiolens.audio_fitting import (
        AUDIO_SELECTION_SEED,
        DATASET_ID,
        LIBRISPEECH_REVISION,
        metadata_rank,
        source_pool_ranks,
    )
    from audiolens.models import AudioFitContractError

    inventory = {
        ("clean", "train.360"): [
            {
                "id": "7-2-b",
                "speaker_id": "7",
                "chapter_id": "2",
                "text": "  exact transcript B  ",
            },
            {
                "id": "3-1-a",
                "speaker_id": 3,
                "chapter_id": 1,
                "text": "exact transcript A",
            },
            {
                "id": "7-2-a",
                "speaker_id": 7,
                "chapter_id": 2,
                "text": "exact transcript C",
            },
        ],
        ("other", "train.500"): [
            {
                "id": "9-4-a",
                "speaker_id": 9,
                "chapter_id": 4,
                "text": "exact transcript D",
            }
        ],
    }
    loader_calls = []

    def source_loader(reverse):
        def load(config, split, *, metadata_only):
            loader_calls.append((config, split, metadata_only))
            rows = list(inventory[(config, split)])
            return list(reversed(rows)) if reverse else rows

        return load

    forward = modal_fit_audio_lens._collect_source_metadata(source_loader(False))
    reversed_enumeration = modal_fit_audio_lens._collect_source_metadata(source_loader(True))
    assert forward == reversed_enumeration
    assert loader_calls == [
        ("clean", "train.360", True),
        ("other", "train.500", True),
        ("clean", "train.360", True),
        ("other", "train.500", True),
    ]
    assert forward[2] == {
        "dataset": DATASET_ID,
        "revision": LIBRISPEECH_REVISION,
        "config": "clean",
        "split": "train.360",
        "source_id": "7-2-b",
        "speaker_id": 7,
        "chapter_id": 2,
        "transcript": "  exact transcript B  ",
    }

    forward_ranks = source_pool_ranks(forward)
    reversed_ranks = source_pool_ranks(reversed_enumeration)
    assert forward_ranks == reversed_ranks
    ranks_by_source = {rank["source_id"]: rank for rank in forward_ranks}
    clean_b = ranks_by_source["7-2-b"]
    assert clean_b["speaker_rank_sha256"] == metadata_rank(
        AUDIO_SELECTION_SEED,
        DATASET_ID,
        LIBRISPEECH_REVISION,
        "clean",
        "train.360",
        7,
    )
    assert clean_b["utterance_rank_sha256"] == metadata_rank(
        AUDIO_SELECTION_SEED,
        DATASET_ID,
        LIBRISPEECH_REVISION,
        "clean",
        "train.360",
        7,
        "7-2-b",
    )
    assert ranks_by_source["7-2-a"]["speaker_rank_sha256"] == clean_b["speaker_rank_sha256"]
    assert ranks_by_source["7-2-a"]["utterance_rank_sha256"] != clean_b["utterance_rank_sha256"]

    with pytest.raises(AudioFitContractError, match="invalid speaker/chapter identity"):
        modal_fit_audio_lens._source_metadata(
            {
                "id": "bad-speaker",
                "speaker_id": True,
                "chapter_id": 1,
                "text": "transcript",
            },
            "clean",
            "train.360",
        )
    with pytest.raises(AudioFitContractError, match="no exact transcript"):
        modal_fit_audio_lens._source_metadata(
            {
                "id": "missing-transcript",
                "speaker_id": 1,
                "chapter_id": 1,
                "text": "",
            },
            "clean",
            "train.360",
        )


def test_small_ranked_selection_obeys_attempt_order_quota_and_interleaving(
    tmp_path,
    monkeypatch,
):
    import audiolens.audio_fitting as audio_fitting
    from audiolens.models import AudioFitContractError

    clean_split = "train.360"
    other_split = "train.500"

    def candidate(config, split, speaker_id, source_id, speaker_rank, utterance_rank):
        return (
            {
                "config": config,
                "split": split,
                "speaker_id": speaker_id,
                "source_id": source_id,
            },
            {
                "config": config,
                "split": split,
                "speaker_id": speaker_id,
                "source_id": source_id,
                "speaker_rank_sha256": speaker_rank,
                "utterance_rank_sha256": utterance_rank,
            },
        )

    fixtures = [
        candidate("clean", clean_split, 101, "clean-reject", "1" * 64, "1" * 64),
        candidate("clean", clean_split, 101, "clean-accept", "1" * 64, "2" * 64),
        candidate("clean", clean_split, 102, "clean-second", "2" * 64, "1" * 64),
        candidate("other", other_split, 201, "other-first", "1" * 64, "1" * 64),
        candidate("other", other_split, 202, "other-second", "2" * 64, "1" * 64),
    ]
    pool = {
        "rows": [metadata for metadata, _rank in reversed(fixtures)],
    }
    source_calls = []
    source_rows = {
        ("clean", clean_split): [
            {"id": "clean-second"},
            {"id": "clean-accept"},
            {"id": "clean-reject"},
        ],
        ("other", other_split): [
            {"id": "other-second"},
            {"id": "other-first"},
        ],
    }

    def source_loader(config, split, *, metadata_only):
        source_calls.append((config, split, metadata_only))
        return source_rows[(config, split)]

    prepared_attempts = []
    processor = object()

    def prepare_candidate(
        metadata,
        source,
        observed_processor,
        *,
        selection_index,
        stratum_index,
        volume_root,
    ):
        assert observed_processor is processor
        assert source["id"] == metadata["source_id"]
        assert volume_root == tmp_path
        prepared_attempts.append(source["id"])
        if source["id"] == "clean-reject":
            raise modal_fit_audio_lens._IneligibleAudio("fixture_rejection")
        return {
            "config": metadata["config"],
            "source_id": metadata["source_id"],
            "speaker_id": metadata["speaker_id"],
            "selection_index": selection_index,
            "stratum_index": stratum_index,
            "pair_id": f"pair-{source['id']}",
        }

    monkeypatch.setattr(audio_fitting, "STRATUM_SIZE", 2)
    monkeypatch.setattr(
        audio_fitting,
        "source_pool_ranks",
        lambda _rows: [rank for _metadata, rank in reversed(fixtures)],
    )
    monkeypatch.setattr(modal_fit_audio_lens, "_prepare_candidate", prepare_candidate)
    rows, ledger = modal_fit_audio_lens._select_corpus(
        pool,
        {"selection": {"max_attempts_per_stratum": 10}},
        processor,
        source_loader,
        volume_root=tmp_path,
    )

    assert source_calls == [
        ("clean", clean_split, False),
        ("other", other_split, False),
    ]
    assert prepared_attempts == [
        "clean-reject",
        "clean-accept",
        "clean-second",
        "other-first",
        "other-second",
    ]
    assert [row["source_id"] for row in rows] == [
        "clean-accept",
        "other-first",
        "clean-second",
        "other-second",
    ]
    assert [row["selection_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["stratum_index"] for row in rows] == [0, 0, 1, 1]
    assert [
        (
            attempt["attempt_index"],
            attempt["stratum_attempt_index"],
            attempt["source_id"],
            attempt["outcome"],
            attempt["reason"],
        )
        for attempt in ledger
    ] == [
        (0, 0, "clean-reject", "rejected", "fixture_rejection"),
        (1, 1, "clean-accept", "selected", None),
        (2, 2, "clean-second", "selected", None),
        (3, 0, "other-first", "selected", None),
        (4, 1, "other-second", "selected", None),
    ]

    with pytest.raises(AudioFitContractError, match="reached bounded attempt limit 1"):
        modal_fit_audio_lens._select_corpus(
            pool,
            {"selection": {"max_attempts_per_stratum": 1}},
            processor,
            source_loader,
            volume_root=tmp_path,
        )


def test_representative_replay_uses_first_clean_and_other_rows_only(
    tmp_path,
    monkeypatch,
):
    rows = [
        {
            "config": "clean",
            "pair_id": "pair-clean",
            "volume_path": "audio/clean.flac",
        },
        {
            "config": "other",
            "pair_id": "pair-other",
            "volume_path": "audio/other.flac",
        },
        {
            "config": "clean",
            "pair_id": "pair-unused",
            "volume_path": "audio/unused.flac",
        },
    ]
    runtime = object()
    replayed_paths = []

    def validate_replay(observed_runtime, path):
        assert observed_runtime is runtime
        replayed_paths.append(path)
        return {"validated_path": path}

    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_validate_one_prepared_replay",
        validate_replay,
    )
    evidence = modal_fit_audio_lens._representative_replay_parity(runtime, rows, tmp_path)

    clean_path = str(tmp_path / "audio/clean.flac")
    other_path = str(tmp_path / "audio/other.flac")
    assert replayed_paths == [clean_path, other_path]
    assert evidence == {
        "clean": {
            "pair_id": "pair-clean",
            "validated_path": clean_path,
        },
        "other": {
            "pair_id": "pair-other",
            "validated_path": other_path,
        },
    }

    with pytest.raises(RuntimeError, match="first clean/other"):
        modal_fit_audio_lens._representative_replay_parity(runtime, rows[1:], tmp_path)
    assert replayed_paths == [clean_path, other_path]


def test_immutable_write_is_idempotent_but_refuses_different_content(tmp_path):
    path = tmp_path / "sealed" / "artifact.json"
    modal_fit_audio_lens._write_immutable_bytes(path, b"sealed\n")
    modal_fit_audio_lens._write_immutable_bytes(path, b"sealed\n")

    with pytest.raises(RuntimeError, match="immutable content-addressed artifact"):
        modal_fit_audio_lens._write_immutable_bytes(path, b"changed\n")
    assert path.read_bytes() == b"sealed\n"

    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="immutable content-addressed artifact"):
        modal_fit_audio_lens._write_immutable_bytes(directory, b"content")


def test_jsonl_loader_rejects_first_record_beyond_bound_before_parsing(tmp_path):
    path = tmp_path / "oversized.jsonl"
    path.write_text('{"index": 0}\n{"index": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(AudioFitContractError, match="exceeds its bounded maximum"):
        modal_fit_audio_lens._load_jsonl(
            path,
            "bounded rows",
            max_rows=2,
        )

    valid_path = tmp_path / "bounded.jsonl"
    valid_path.write_text('{"index": 0}\n{"index": 1}\n', encoding="utf-8")
    assert modal_fit_audio_lens._load_jsonl(
        valid_path,
        "bounded rows",
        max_rows=2,
    ) == [{"index": 0}, {"index": 1}]

    overlong_path = tmp_path / "overlong.jsonl"
    overlong_path.write_bytes(
        b'{"value":"' + b"x" * modal_fit_audio_lens.MAX_JSONL_LINE_BYTES + b'"}\n'
    )
    with pytest.raises(AudioFitContractError, match="exceeds 1048576 bytes"):
        modal_fit_audio_lens._load_jsonl(
            overlong_path,
            "bounded rows",
            max_rows=2,
        )


_ACTION_ENDPOINTS = (
    "rank_audio_source",
    "stage_audio_corpus",
    "replay_audio_selection",
    "restore_audio_sources",
    "preflight_audio_fit",
    "validate_audio_replay",
    "smoke_audio_fit",
    "fit_audio_lens",
)


@pytest.mark.parametrize(
    ("action", "endpoint", "expected_kwargs"),
    [
        ("rank_source_only", "rank_audio_source", {}),
        (
            "stage_corpus_only",
            "stage_audio_corpus",
            {"source_pool_sha256": "a" * 64},
        ),
        (
            "selection_replay_only",
            "replay_audio_selection",
            {
                "source_pool_sha256": "a" * 64,
                "ordered_corpus_sha256": "b" * 64,
            },
        ),
        (
            "restore_source_only",
            "restore_audio_sources",
            {"ordered_corpus_sha256": "b" * 64},
        ),
        (
            "preflight_only",
            "preflight_audio_fit",
            {"ordered_corpus_sha256": "b" * 64},
        ),
        (
            "replay_parity_only",
            "validate_audio_replay",
            {"ordered_corpus_sha256": "b" * 64},
        ),
        (
            "smoke_only",
            "smoke_audio_fit",
            {"ordered_corpus_sha256": "b" * 64},
        ),
        (
            "fit",
            "fit_audio_lens",
            {"ordered_corpus_sha256": "b" * 64},
        ),
    ],
)
def test_main_dispatches_exactly_the_selected_staged_action(
    monkeypatch,
    capsys,
    action,
    endpoint,
    expected_kwargs,
):
    class EndpointStub:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def remote(self, **kwargs):
            self.calls.append(kwargs)
            return f"response:{self.name}"

    endpoints = {name: EndpointStub(name) for name in _ACTION_ENDPOINTS}
    for name, stub in endpoints.items():
        monkeypatch.setattr(modal_fit_audio_lens, name, stub)

    modal_fit_audio_lens.main(
        **{action: True},
        source_pool_sha256="a" * 64,
        ordered_corpus_sha256="b" * 64,
    )

    assert endpoints[endpoint].calls == [expected_kwargs]
    assert all(not stub.calls for name, stub in endpoints.items() if name != endpoint)
    assert capsys.readouterr().out == f"response:{endpoint}\n"


def test_main_rejects_missing_or_ambiguous_actions_before_dispatch(monkeypatch):
    calls = []

    class EndpointStub:
        def remote(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("invalid action selection reached dispatch")

    for name in _ACTION_ENDPOINTS:
        monkeypatch.setattr(modal_fit_audio_lens, name, EndpointStub())

    with pytest.raises(SystemExit, match="select exactly one staged action"):
        modal_fit_audio_lens.main()
    with pytest.raises(SystemExit, match="select exactly one staged action"):
        modal_fit_audio_lens.main(rank_source_only=True, fit=True)
    assert calls == []


def test_processor_replay_checks_every_persisted_layout_field(
    tmp_path,
    monkeypatch,
):
    import types

    import torch

    import audiolens.models as models
    from audiolens.audio_fitting import input_ids_sha256

    processor_ids = torch.tensor([[2, 105, 107, 256000, 9, 1, 2, 3]])
    prepared = types.SimpleNamespace(
        input_ids=processor_ids,
        layout=types.SimpleNamespace(
            audio_start=3,
            n_audio_tokens=2,
            stop=7,
            n_valid_positions=3,
        ),
    )
    row = {
        "volume_path": "audio/fixture.flac",
        "processor_input_ids_sha256": input_ids_sha256(processor_ids),
        "fit_input_ids_sha256": input_ids_sha256(processor_ids[:, :7]),
        "audio_start": 3,
        "n_audio_tokens": 2,
        "processor_seq_len": 8,
        "sliced_seq_len": 7,
        "n_valid_positions": 3,
    }
    observed_paths = []

    def prepare_audio(_processor, path, *, profile):
        observed_paths.append(path)
        assert profile.key == "gemma-4-e2b-it"
        return prepared

    monkeypatch.setattr(models, "prepare_audio", prepare_audio)
    context = {"root": tmp_path, "rows": [dict(row), dict(row)]}
    assert modal_fit_audio_lens._processor_replay_impl(
        context,
        processor_loader=object,
    ) == {"row_count": 2, "processor_replayed": True}
    assert observed_paths == [tmp_path / "audio/fixture.flac"] * 2

    changed = dict(row)
    changed["n_valid_positions"] = 4
    with pytest.raises(AudioFitContractError, match="processor replay changed"):
        modal_fit_audio_lens._processor_replay_impl(
            {"root": tmp_path, "rows": [changed]},
            processor_loader=object,
        )


def test_fit_rejects_internal_snapshot_as_requested_prefix():
    with pytest.raises(ValueError, match=r"\[10, 20, 1000\]"):
        modal_fit_audio_lens._fit_audio_lens_impl(requested_prefix=500)


def test_smoke_gate_requires_separately_persisted_ten_to_twenty_resume(
    tmp_path,
    monkeypatch,
):
    import json

    import audiolens.audio_fitting as audio_fitting

    context = {
        "root": tmp_path,
        "config": {"schema_version": audio_fitting.SCHEMA_VERSION},
    }
    counts = iter((0, 10, 20))
    preflight_calls = []

    def preflight(**kwargs):
        preflight_calls.append(kwargs)
        return {
            "status": "pending",
            "current_count": next(counts),
            "context": context,
        }

    remote_calls = []

    def call_deployed(name, **kwargs):
        remote_calls.append((name, kwargs))
        return json.dumps({"current_count": kwargs["requested_prefix"]})

    gate_requirements = []
    reloads = []
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_preflight_audio_fit_impl",
        preflight,
    )
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_require_gates",
        lambda _context, gates: gate_requirements.append(tuple(gates)) or {},
    )
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_write_gate",
        lambda _context, gate: {"gate": gate},
    )
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_call_deployed_function",
        call_deployed,
    )
    monkeypatch.setattr(modal_fit_audio_lens, "_commit_volume", lambda: None)
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_reload_volume",
        lambda: reloads.append(True),
    )
    monkeypatch.setattr(
        audio_fitting,
        "gate_path",
        lambda _config, _gate: "missing-smoke-gate.json",
    )

    payload = json.loads(
        modal_fit_audio_lens.smoke_audio_fit(
            ordered_corpus_sha256="a" * 64,
        )
    )
    assert payload["current_count"] == 20
    assert payload["first_prefix"] == 10
    assert payload["resumed_prefix"] == 20
    assert remote_calls == [
        (
            "_fit_audio_lens_gpu",
            {
                "requested_prefix": 10,
                "ordered_corpus_sha256": "a" * 64,
            },
        ),
        (
            "_fit_audio_lens_gpu",
            {
                "requested_prefix": 20,
                "ordered_corpus_sha256": "a" * 64,
            },
        ),
    ]
    assert len(preflight_calls) == 3
    assert reloads == [True, True]
    assert gate_requirements == [audio_fitting.REQUIRED_GATES[:4]] * 3


def test_periodic_stock_checkpoint_writes_keep_identity_during_interruption(
    monkeypatch,
):
    import types

    import jlens.fitting as jlens_fitting
    import torch

    import audiolens.audio_fitting as audio_fitting

    writes = []

    def write_checkpoint(state, path):
        writes.append((state, path))
        raise RuntimeError("interrupted")

    identity = {
        "fit_config_sha256": "a" * 64,
        "ordered_corpus_sha256": "b" * 64,
    }
    monkeypatch.setattr(jlens_fitting, "_atomic_save", write_checkpoint)
    monkeypatch.setattr(
        jlens_fitting,
        "jacobian_for_prompt",
        lambda *_args, **_kwargs: ({0: torch.ones(2, 2)}, 2, 1),
    )
    monkeypatch.setattr(
        audio_fitting,
        "validate_checkpoint_identity",
        lambda value: value,
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        with modal_fit_audio_lens._stamped_jlens_checkpoint_writer(identity):
            jlens_fitting.fit(
                types.SimpleNamespace(n_layers=2, d_model=2),
                prompts=["fixture"] * 5,
                source_layers=[0],
                target_layer=1,
                checkpoint_path="checkpoint.pt",
                checkpoint_every=5,
                resume=False,
            )

    assert len(writes) == 1
    state, path = writes[0]
    assert path == "checkpoint.pt"
    assert state["n_done"] == state["next_idx"] == 5
    assert state["fit_config_sha256"] == "a" * 64
    assert state["ordered_corpus_sha256"] == "b" * 64
    assert jlens_fitting._atomic_save is write_checkpoint


def test_corpus_staging_rejects_regular_file_root_before_selection(
    tmp_path,
    monkeypatch,
):
    import audiolens.audio_fitting as audio_fitting

    corpus_parent = tmp_path / "audio-corpora" / "config-digest"
    corpus_parent.parent.mkdir(parents=True)
    corpus_parent.write_bytes(b"not-a-directory")
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_corpus_config",
        lambda: {"schema_version": audio_fitting.SCHEMA_VERSION},
    )
    monkeypatch.setattr(
        modal_fit_audio_lens,
        "_load_source_pool",
        lambda *_args: ({}, "pool.json", "a" * 64),
    )
    monkeypatch.setattr(
        audio_fitting,
        "corpus_config_digest",
        lambda _config: "config-digest",
    )

    with pytest.raises(RuntimeError, match="corpus root exists but is not a directory"):
        modal_fit_audio_lens._stage_audio_corpus_impl(volume_root=tmp_path)


@pytest.mark.parametrize("metadata_only", [True, False])
def test_default_source_loader_pins_revision_and_disables_authentication(
    monkeypatch,
    metadata_only,
):
    import types

    from audiolens.audio_fitting import DATASET_ID, LIBRISPEECH_REVISION

    calls = []

    class Audio:
        def __init__(self, *, decode):
            self.decode = decode

    class Dataset:
        def select_columns(self, columns):
            calls.append(("select_columns", columns))
            return "metadata-dataset"

        def cast_column(self, name, feature):
            calls.append(("cast_column", name, feature.decode))
            return "audio-dataset"

    def load_dataset(*args, **kwargs):
        calls.append(("load_dataset", args, kwargs))
        return Dataset()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(Audio=Audio, load_dataset=load_dataset),
    )

    result = modal_fit_audio_lens._default_source_loader(
        "clean",
        "train.360",
        metadata_only=metadata_only,
    )
    assert calls[0] == (
        "load_dataset",
        (DATASET_ID, "clean"),
        {
            "split": "train.360",
            "revision": LIBRISPEECH_REVISION,
            "streaming": metadata_only,
            "trust_remote_code": False,
            "token": False,
        },
    )
    if metadata_only:
        assert result == "metadata-dataset"
        assert calls[1] == (
            "select_columns",
            ["id", "speaker_id", "chapter_id", "text"],
        )
    else:
        assert result == "audio-dataset"
        assert calls[1] == ("cast_column", "audio", False)


def test_source_census_rejects_regular_file_root_before_enumeration(
    tmp_path,
    monkeypatch,
):
    import audiolens.audio_fitting as audio_fitting

    monkeypatch.setattr(
        audio_fitting,
        "corpus_config_digest",
        lambda _config: "config-digest",
    )
    census_root = tmp_path / "audio-source-census" / "config-digest"
    census_root.parent.mkdir()
    census_root.write_text("not a directory")

    def source_loader(*_args, **_kwargs):
        raise AssertionError("source enumeration must not start")

    with pytest.raises(AudioFitContractError, match="source census root"):
        modal_fit_audio_lens._collect_source_metadata_durable(
            source_loader,
            root=tmp_path,
            corpus_config={},
            commit=lambda: None,
        )


def test_source_census_resumes_from_immutable_shards_after_preemption(
    tmp_path,
    monkeypatch,
):
    import audiolens.audio_fitting as audio_fitting

    inventories = {
        ("clean", "train.360"): [
            {"id": f"clean-{index}", "speaker_id": index, "chapter_id": 1, "text": "clean"}
            for index in range(3)
        ],
        ("other", "train.500"): [
            {
                "id": f"other-{index}",
                "speaker_id": index + 10,
                "chapter_id": 2,
                "text": "other",
            }
            for index in range(2)
        ],
    }
    loader_calls = []
    skips = []

    class Source(list):
        def skip(self, count):
            skips.append(count)
            return Source(self[count:])

    def source_loader(config, split, *, metadata_only):
        loader_calls.append((config, split, metadata_only))
        return Source(inventories[(config, split)])

    monkeypatch.setattr(
        audio_fitting,
        "SOURCE_POOL_COUNTS",
        {"clean": 3, "other": 2},
    )
    monkeypatch.setattr(
        audio_fitting,
        "corpus_config_digest",
        lambda _config: "config-digest",
    )
    monkeypatch.setattr(modal_fit_audio_lens, "SOURCE_CENSUS_CHUNK_SIZE", 2)

    commits = 0

    def preempt_after_first_shard():
        nonlocal commits
        commits += 1
        if commits == 1:
            raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        modal_fit_audio_lens._collect_source_metadata_durable(
            source_loader,
            root=tmp_path,
            corpus_config={},
            commit=preempt_after_first_shard,
        )

    rows = modal_fit_audio_lens._collect_source_metadata_durable(
        source_loader,
        root=tmp_path,
        corpus_config={},
        commit=lambda: None,
    )
    assert len(rows) == 5
    assert skips == [2]
    assert loader_calls == [
        ("clean", "train.360", True),
        ("clean", "train.360", True),
        ("other", "train.500", True),
    ]

    loader_calls.clear()
    restored = modal_fit_audio_lens._collect_source_metadata_durable(
        source_loader,
        root=tmp_path,
        corpus_config={},
        commit=lambda: None,
    )
    assert restored == rows
    assert loader_calls == []


def test_source_census_requires_eof_proof_after_final_shard_preemption(
    tmp_path,
    monkeypatch,
):
    import audiolens.audio_fitting as audio_fitting

    inventories = {
        ("clean", "train.360"): [
            {
                "id": f"clean-{index}",
                "speaker_id": index,
                "chapter_id": 1,
                "text": "clean",
            }
            for index in range(4)
        ],
        ("other", "train.500"): [
            {
                "id": f"other-{index}",
                "speaker_id": index + 10,
                "chapter_id": 2,
                "text": "other",
            }
            for index in range(2)
        ],
    }
    skips = []

    class Source(list):
        def skip(self, count):
            skips.append(count)
            return Source(self[count:])

    def source_loader(config, split, *, metadata_only):
        return Source(inventories[(config, split)])

    monkeypatch.setattr(
        audio_fitting,
        "SOURCE_POOL_COUNTS",
        {"clean": 3, "other": 2},
    )
    monkeypatch.setattr(
        audio_fitting,
        "corpus_config_digest",
        lambda _config: "config-digest",
    )
    monkeypatch.setattr(modal_fit_audio_lens, "SOURCE_CENSUS_CHUNK_SIZE", 3)

    with pytest.raises(RuntimeError, match="preempted"):
        modal_fit_audio_lens._collect_source_metadata_durable(
            source_loader,
            root=tmp_path,
            corpus_config={},
            commit=lambda: (_ for _ in ()).throw(RuntimeError("preempted")),
        )

    completion = (
        tmp_path / "audio-source-census" / "config-digest" / "clean-train-360" / "complete.json"
    )
    assert not completion.exists()
    with pytest.raises(AudioFitContractError, match="exceeds pinned count"):
        modal_fit_audio_lens._collect_source_metadata_durable(
            source_loader,
            root=tmp_path,
            corpus_config={},
            commit=lambda: None,
        )
    assert skips == [3]


def test_complete_source_pool_cache_rebinds_only_source_implementation(
    tmp_path,
    monkeypatch,
):
    import json

    import audiolens.audio_fitting as audio_fitting

    path = tmp_path / "audio-source-pools" / "old-config" / "pool" / "pool.json"
    path.parent.mkdir(parents=True)
    record = {
        "config": {
            "schema_version": 2,
            "source_sha256": "a" * 64,
            "selection": {"seed": 7},
        },
        "source_pool_sha256": "b" * 64,
        "rows": [{"source_id": "fixture"}],
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        audio_fitting,
        "validate_source_pool_record",
        lambda value, _config: value,
    )

    current = {
        "schema_version": 2,
        "source_sha256": "c" * 64,
        "selection": {"seed": 7},
    }
    assert modal_fit_audio_lens._reusable_source_metadata(
        tmp_path,
        current,
    ) == [{"source_id": "fixture"}]
    changed_selection = {
        **current,
        "selection": {"seed": 8},
    }
    assert (
        modal_fit_audio_lens._reusable_source_metadata(
            tmp_path,
            changed_selection,
        )
        is None
    )


def test_bf16_batch_bounds_are_geometrically_consistent():
    modal_fit_audio_lens._validate_batched_replay_bounds(
        {
            "layer": {
                "cosine": 0.998860239982605,
                "relative_l2": 0.04773113504052162,
            }
        }
    )

    with pytest.raises(RuntimeError, match="exceeds bf16 parity bounds"):
        modal_fit_audio_lens._validate_batched_replay_bounds(
            {
                "layer": {
                    "cosine": modal_fit_audio_lens.BF16_BATCH_COSINE_MIN - 1e-6,
                    "relative_l2": 0.01,
                }
            }
        )
    with pytest.raises(RuntimeError, match="exceeds bf16 parity bounds"):
        modal_fit_audio_lens._validate_batched_replay_bounds(
            {
                "layer": {
                    "cosine": 1.0,
                    "relative_l2": (modal_fit_audio_lens.BF16_BATCH_RELATIVE_L2_MAX + 1e-6),
                }
            }
        )
