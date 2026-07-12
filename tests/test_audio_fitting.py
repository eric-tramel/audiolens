from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import audiolens.models as model_contracts
from audiolens import audio_fitting as af
from audiolens.fitting import canonical_json_bytes
from audiolens.models.base import AudioFitContractError
from audiolens.models.gemma4 import GEMMA4_PROFILE

_SOURCE_SHA256 = "a" * 64
_LOCK_SHA256 = "b" * 64
_RUNTIME = {
    "python_version": "3.12.9",
    "torch_version": "2.7.1",
    "transformers_version": "4.53.2",
    "datasets_version": "4.0.0",
    "jlens_revision": af.JLENS_REVISION,
    "cuda_version": "12.8",
}
_SOURCE_FIELDS = (
    "dataset",
    "revision",
    "config",
    "split",
    "source_id",
    "speaker_id",
    "chapter_id",
    "transcript",
)
_PAIR_FIELDS = (
    *_SOURCE_FIELDS,
    "transcript_sha256",
    "audio_sha256",
    "decoded_pcm_sha256",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _different_sha256(digest: str) -> str:
    replacement = "0" if digest[0] != "0" else "1"
    return replacement + digest[1:]


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _SOURCE_FIELDS}


def _ranked_speakers(config: str, split: str, first_speaker: int) -> list[int]:
    ranked = [
        (
            af.metadata_rank(
                af.AUDIO_SELECTION_SEED,
                af.DATASET_ID,
                af.LIBRISPEECH_REVISION,
                config,
                split,
                speaker_id,
            ),
            speaker_id,
        )
        for speaker_id in range(first_speaker, first_speaker + af.STRATUM_SIZE)
    ]
    return [speaker_id for _, speaker_id in sorted(ranked)]


def build_synthetic_corpus_rows(
    *,
    audio_sha256: str | None = None,
    pcm_sha256: str | None = None,
    num_samples: int = 32_000,
) -> list[dict[str, Any]]:
    """Build the exact alternating, globally unique 500/500 frozen corpus."""

    shared_audio = audio_sha256 or hashlib.sha256(b"synthetic shared FLAC").hexdigest()
    shared_pcm = pcm_sha256 or af.decoded_pcm_sha256(np.array([-3, -1, 0, 1, 3], dtype=np.int16))
    processor_ids = [[2, 105, 2364, 107, 256000] + [258881] * 20 + [258883, 106, 107]]
    fit_ids = [processor_ids[0][:25]]
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for stratum_position, (config, split) in enumerate(af.STRATA):
        first_speaker = stratum_position * 1_000
        rows: list[dict[str, Any]] = []
        for stratum_index, speaker_id in enumerate(_ranked_speakers(config, split, first_speaker)):
            source_id = f"{config}-{speaker_id:04d}"
            transcript = f"Exact transcript for {source_id}."
            row = {
                "schema_version": af.SCHEMA_VERSION,
                "kind": af.CORPUS_ROW_KIND,
                "selection_index": 2 * stratum_index + stratum_position,
                "stratum_index": stratum_index,
                "dataset": af.DATASET_ID,
                "revision": af.LIBRISPEECH_REVISION,
                "config": config,
                "split": split,
                "source_id": source_id,
                "speaker_id": speaker_id,
                "chapter_id": 10_000 + speaker_id,
                "transcript": transcript,
                "transcript_sha256": af.transcript_sha256(transcript),
                "audio_sha256": shared_audio,
                "decoded_pcm_sha256": shared_pcm,
                "sampling_rate": af.SAMPLE_RATE,
                "num_samples": num_samples,
                "duration_seconds": round(num_samples / af.SAMPLE_RATE, 6),
                "volume_path": af.audio_blob_path(shared_audio),
                "audio_start": af.AUDIO_START,
                "n_audio_tokens": 20,
                "processor_seq_len": 28,
                "sliced_seq_len": 25,
                "n_valid_positions": 8,
                "processor_input_ids_sha256": af.input_ids_sha256(processor_ids),
                "fit_input_ids_sha256": af.input_ids_sha256(fit_ids),
            }
            row["pair_id"] = af.make_pair_id(row)
            rows.append(row)
        by_stratum[config] = rows

    alternating: list[dict[str, Any]] = []
    for stratum_index in range(af.STRATUM_SIZE):
        for config, _ in af.STRATA:
            alternating.append(by_stratum[config][stratum_index])
    return alternating


def build_synthetic_attempt_ledger(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the exact all-selected trace in canonical rank and stratum order."""

    ledger: list[dict[str, Any]] = []
    for config, split in af.STRATA:
        selected = sorted(
            (row for row in rows if row["config"] == config),
            key=lambda row: row["stratum_index"],
        )
        for stratum_attempt_index, row in enumerate(selected):
            ledger.append(
                {
                    "schema_version": af.SCHEMA_VERSION,
                    "kind": af.ATTEMPT_ROW_KIND,
                    "attempt_index": len(ledger),
                    "stratum": config,
                    "stratum_attempt_index": stratum_attempt_index,
                    "config": config,
                    "split": split,
                    "speaker_id": row["speaker_id"],
                    "source_id": row["source_id"],
                    "speaker_rank_sha256": af.metadata_rank(
                        af.AUDIO_SELECTION_SEED,
                        af.DATASET_ID,
                        af.LIBRISPEECH_REVISION,
                        config,
                        split,
                        row["speaker_id"],
                    ),
                    "utterance_rank_sha256": af.metadata_rank(
                        af.AUDIO_SELECTION_SEED,
                        af.DATASET_ID,
                        af.LIBRISPEECH_REVISION,
                        config,
                        split,
                        row["speaker_id"],
                        row["source_id"],
                    ),
                    "outcome": "selected",
                    "reason": None,
                    "pair_id": row["pair_id"],
                }
            )
    return ledger


def build_synthetic_source_pool(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = sorted(
        (_metadata(row) for row in rows),
        key=lambda row: (
            af.STRATA.index((row["config"], row["split"])),
            row["speaker_id"],
            row["source_id"],
        ),
    )
    return {
        "schema_version": af.SCHEMA_VERSION,
        "kind": af.SOURCE_POOL_KIND,
        "config": dict(config),
        "corpus_config_sha256": af.corpus_config_digest(config),
        "source_pool_sha256": af.source_pool_digest(metadata),
        "row_count": len(metadata),
        "stratum_counts": {
            name: sum(row["config"] == name and row["split"] == split for row in metadata)
            for name, split in af.STRATA
        },
        "rows": metadata,
    }


def build_synthetic_protocol(
    *,
    profile: Any = GEMMA4_PROFILE,
    audio_sha256: str | None = None,
    pcm_sha256: str | None = None,
) -> dict[str, Any]:
    """Build config, rows, bounded ledger, envelope, and sealed corpus artifact."""

    config = af.build_corpus_config(
        profile,
        source_digest=_SOURCE_SHA256,
        lock_sha256=_LOCK_SHA256,
    )
    rows = build_synthetic_corpus_rows(
        audio_sha256=audio_sha256,
        pcm_sha256=pcm_sha256,
    )
    ledger = build_synthetic_attempt_ledger(rows)
    source_pool = build_synthetic_source_pool(config, rows)
    source_pool_sha256 = source_pool["source_pool_sha256"]
    ordered_sha256 = af.ordered_corpus_digest(rows)
    paths = af.corpus_paths(config, ordered_sha256)
    source_pool_path = af.source_pool_path(config, source_pool_sha256)
    source_pool_file_sha256 = hashlib.sha256(_canonical_bytes(source_pool) + b"\n").hexdigest()
    envelope = {
        "schema_version": af.SCHEMA_VERSION,
        "kind": af.CORPUS_ENVELOPE_KIND,
        "config": config,
        "corpus_config_sha256": af.corpus_config_digest(config),
        "source_pool_sha256": source_pool_sha256,
        "source_pool_path": source_pool_path,
        "source_pool_file_sha256": source_pool_file_sha256,
        "row_count": af.CORPUS_SIZE,
        "stratum_counts": {config: af.STRATUM_SIZE for config, _ in af.STRATA},
        "ordered_corpus_sha256": ordered_sha256,
        "rows_path": paths["rows"],
        "rows_sha256": af.canonical_jsonl_sha256(rows),
        "attempt_ledger_path": paths["attempt_ledger"],
        "attempt_ledger_sha256": af.canonical_jsonl_sha256(ledger),
        "attempt_count": len(ledger),
        "max_attempts_per_stratum": config["selection"]["max_attempts_per_stratum"],
        "audio_root": "audio-blobs",
    }
    artifact = af.corpus_artifact(envelope, rows, ledger)
    return {
        "config": config,
        "rows": rows,
        "source_pool": source_pool,
        "ledger": ledger,
        "envelope": envelope,
        "artifact": artifact,
    }


def build_synthetic_fit_bundle(
    protocol: Mapping[str, Any], *, profile: Any = GEMMA4_PROFILE
) -> dict[str, Any]:
    config = af.build_fit_config(
        protocol["artifact"],
        profile,
        source_digest=_SOURCE_SHA256,
        lock_sha256=_LOCK_SHA256,
        runtime=_RUNTIME,
    )
    paths = af.run_paths(config)
    return {
        "config": config,
        "paths": paths,
        "identity": af.checkpoint_identity(config, paths),
    }


def _refresh_pair_identity(row: dict[str, Any]) -> None:
    row["transcript_sha256"] = af.transcript_sha256(row["transcript"])
    row["volume_path"] = af.audio_blob_path(row["audio_sha256"])
    row["pair_id"] = af.make_pair_id(row)


def _replace_nested(value: Mapping[str, Any], path: str, replacement: Any) -> dict[str, Any]:
    changed = copy.deepcopy(value)
    cursor = changed
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = replacement
    return changed


def _tiny_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    tiny = copy.deepcopy(identity)
    tiny["source_layers"] = [1]
    tiny["target_layer"] = 2
    tiny["d_model"] = 2
    return af.validate_checkpoint_identity(tiny)


def _checkpoint_state(
    identity: Mapping[str, Any], count: int, means: Mapping[int, Any]
) -> dict[str, Any]:
    return {
        "source_layers": identity["source_layers"],
        "target_layer": identity["target_layer"],
        "skip_first": identity["skip_first"],
        "n_done": count,
        "next_idx": count,
        "fit_config_sha256": identity["fit_config_sha256"],
        "ordered_corpus_sha256": identity["ordered_corpus_sha256"],
        "jacobian_sum": {layer: mean * count for layer, mean in means.items()},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _seal_checkpoint(
    root: Path,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    expected_count: int,
) -> tuple[dict[str, Any], Path]:
    import torch

    if expected_count == af.CORPUS_SIZE:
        path = root / identity["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), path)
        artifact = af.checkpoint_artifact(path, identity, expected_count)
    else:
        candidate = root / "prefix-candidate.pt"
        torch.save(dict(state), candidate)
        artifact = af.checkpoint_artifact(candidate, identity, expected_count)
        path = root / artifact["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(path)
    assert af.validate_checkpoint_artifact(artifact, path, identity, expected_count) == artifact
    return artifact, path


@pytest.fixture(scope="module")
def synthetic_protocol() -> dict[str, Any]:
    return build_synthetic_protocol()


@pytest.fixture(scope="module")
def synthetic_fit_bundle(synthetic_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return build_synthetic_fit_bundle(synthetic_protocol)


def test_source_pool_rejects_truncated_pinned_splits(
    synthetic_protocol: Mapping[str, Any],
):
    with pytest.raises(AudioFitContractError, match="complete pinned LibriSpeech"):
        af.validate_source_pool_record(
            synthetic_protocol["source_pool"],
            synthetic_protocol["config"],
        )


def test_source_pool_digest_and_metadata_ranks_are_exact_and_order_independent(
    synthetic_protocol: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        af,
        "SOURCE_POOL_COUNTS",
        {name: af.STRATUM_SIZE for name, _ in af.STRATA},
    )
    metadata = [_metadata(row) for row in synthetic_protocol["rows"]]
    row_hashes = sorted(hashlib.sha256(_canonical_bytes(row)).digest() for row in metadata)
    oracle = hashlib.sha256()
    oracle.update(b"source_bound_audio_source_pool_v1\0")
    oracle.update(len(row_hashes).to_bytes(8, "big"))
    for row_hash in row_hashes:
        oracle.update(row_hash)

    expected_pool = oracle.hexdigest()
    assert af.source_pool_digest(metadata) == expected_pool
    assert af.source_pool_digest(reversed(metadata)) == expected_pool
    assert (
        af.validate_source_pool_record(
            synthetic_protocol["source_pool"],
            synthetic_protocol["config"],
        )
        == synthetic_protocol["source_pool"]
    )
    assert (
        af.validate_attempt_ledger_against_source_pool(
            synthetic_protocol["ledger"],
            synthetic_protocol["source_pool"],
            corpus_config=synthetic_protocol["config"],
        )
        == synthetic_protocol["ledger"]
    )

    sample = metadata[137]
    speaker_payload = {
        "seed": af.AUDIO_SELECTION_SEED,
        "dataset": af.DATASET_ID,
        "revision": af.LIBRISPEECH_REVISION,
        "config": sample["config"],
        "split": sample["split"],
        "speaker_id": sample["speaker_id"],
    }
    utterance_payload = {**speaker_payload, "source_id": sample["source_id"]}
    assert (
        af.metadata_rank(
            **speaker_payload,
        )
        == hashlib.sha256(_canonical_bytes(speaker_payload)).hexdigest()
    )
    assert (
        af.metadata_rank(
            **utterance_payload,
        )
        == hashlib.sha256(_canonical_bytes(utterance_payload)).hexdigest()
    )

    forward = sorted(
        metadata,
        key=lambda row: (
            af.metadata_rank(
                af.AUDIO_SELECTION_SEED,
                row["dataset"],
                row["revision"],
                row["config"],
                row["split"],
                row["speaker_id"],
                row["source_id"],
            ),
            row["source_id"],
        ),
    )
    reverse = sorted(
        reversed(metadata),
        key=lambda row: (
            af.metadata_rank(
                af.AUDIO_SELECTION_SEED,
                row["dataset"],
                row["revision"],
                row["config"],
                row["split"],
                row["speaker_id"],
                row["source_id"],
            ),
            row["source_id"],
        ),
    )
    assert [row["source_id"] for row in forward] == [row["source_id"] for row in reverse]

    with pytest.raises(AudioFitContractError, match="duplicate source metadata"):
        af.source_pool_digest([*metadata, metadata[0]])


def test_synthetic_builders_follow_frozen_schemas_and_content_addressed_paths(
    synthetic_protocol: Mapping[str, Any],
    synthetic_fit_bundle: Mapping[str, Any],
):
    config = synthetic_protocol["config"]
    rows = synthetic_protocol["rows"]
    ledger = synthetic_protocol["ledger"]
    envelope = synthetic_protocol["envelope"]
    artifact = synthetic_protocol["artifact"]

    assert set(config) == {
        "schema_version",
        "kind",
        "source_sha256",
        "lock_sha256",
        "dataset",
        "selection",
        "audio",
        "processor",
        "transcript",
    }
    assert set(rows[0]) == {
        "schema_version",
        "kind",
        "selection_index",
        "stratum_index",
        *_SOURCE_FIELDS,
        "pair_id",
        "transcript_sha256",
        "audio_sha256",
        "decoded_pcm_sha256",
        "sampling_rate",
        "num_samples",
        "duration_seconds",
        "volume_path",
        "audio_start",
        "n_audio_tokens",
        "processor_seq_len",
        "sliced_seq_len",
        "n_valid_positions",
        "processor_input_ids_sha256",
        "fit_input_ids_sha256",
    }
    assert set(ledger[0]) == {
        "schema_version",
        "kind",
        "attempt_index",
        "stratum",
        "stratum_attempt_index",
        "config",
        "split",
        "speaker_id",
        "source_id",
        "speaker_rank_sha256",
        "utterance_rank_sha256",
        "outcome",
        "reason",
        "pair_id",
    }
    assert set(envelope) == {
        "schema_version",
        "kind",
        "config",
        "corpus_config_sha256",
        "source_pool_sha256",
        "source_pool_path",
        "source_pool_file_sha256",
        "row_count",
        "stratum_counts",
        "ordered_corpus_sha256",
        "rows_path",
        "rows_sha256",
        "attempt_ledger_path",
        "attempt_ledger_sha256",
        "attempt_count",
        "max_attempts_per_stratum",
        "audio_root",
    }
    assert set(artifact) == {
        "schema_version",
        "kind",
        "config",
        "source_sha256",
        "lock_sha256",
        "corpus_config_sha256",
        "source_pool_sha256",
        "source_pool_path",
        "source_pool_file_sha256",
        "row_count",
        "stratum_counts",
        "ordered_corpus_sha256",
        "envelope_path",
        "envelope_sha256",
        "rows_path",
        "rows_sha256",
        "attempt_ledger_path",
        "attempt_ledger_sha256",
        "attempt_count",
        "audio_root",
    }

    corpus_root = (
        f"audio-corpora/{af.corpus_config_digest(config)}/{envelope['ordered_corpus_sha256']}"
    )
    assert af.source_pool_path(config, envelope["source_pool_sha256"]) == (
        f"audio-source-pools/{af.corpus_config_digest(config)}/"
        f"{envelope['source_pool_sha256']}/pool.json"
    )
    assert envelope["rows_path"] == f"{corpus_root}/rows.jsonl"
    assert envelope["attempt_ledger_path"] == f"{corpus_root}/attempt-ledger.jsonl"
    assert artifact["envelope_path"] == f"{corpus_root}/envelope.json"

    fit_config = synthetic_fit_bundle["config"]
    fit_digest = af.fit_config_digest(fit_config)
    assert set(fit_config) == {
        "schema_version",
        "kind",
        "source_sha256",
        "lock_sha256",
        "corpus",
        "model",
        "runtime",
        "fit",
    }
    assert synthetic_fit_bundle["paths"] == {
        "manifest": f"audio-fit-runs/{fit_digest}/run.json",
        "checkpoint": f"audio-fit-runs/{fit_digest}/working-checkpoint.pt",
        "snapshot_dir": f"audio-fit-runs/{fit_digest}/prefix-500",
        "stability_dir": f"audio-fit-runs/{fit_digest}/stability",
        "lens_dir": f"audio-fit-runs/{fit_digest}/lens",
        "gate_dir": f"audio-fit-runs/{fit_digest}/gates",
    }


def test_corpus_config_rejects_alternate_attempt_bound(
    synthetic_protocol: Mapping[str, Any],
):
    changed = copy.deepcopy(synthetic_protocol["config"])
    changed["selection"]["max_attempts_per_stratum"] -= 1
    with pytest.raises(AudioFitContractError, match="must equal"):
        af.validate_corpus_config(changed)


def test_exact_alternating_unique_speaker_corpus_is_accepted(
    synthetic_protocol: Mapping[str, Any],
):
    rows = synthetic_protocol["rows"]
    assert af.validate_corpus_rows(rows) == rows
    assert len(rows) == 1_000
    assert Counter(row["config"] for row in rows) == {"clean": 500, "other": 500}
    assert len({row["speaker_id"] for row in rows}) == 1_000
    assert all(
        (row["config"], row["split"]) == af.STRATA[index % 2]
        and row["selection_index"] == index
        and row["stratum_index"] == index // 2
        for index, row in enumerate(rows)
    )
    assert Counter(row["config"] for row in rows[:500]) == {
        "clean": 250,
        "other": 250,
    }
    assert Counter(row["config"] for row in rows[500:]) == {
        "clean": 250,
        "other": 250,
    }

    valid_reordering = copy.deepcopy(rows)
    first, second = valid_reordering[0], valid_reordering[2]
    valid_reordering[0], valid_reordering[2] = second, first
    valid_reordering[0]["selection_index"] = 0
    valid_reordering[0]["stratum_index"] = 0
    valid_reordering[2]["selection_index"] = 2
    valid_reordering[2]["stratum_index"] = 1
    assert af.validate_corpus_rows(valid_reordering) == valid_reordering
    assert af.ordered_corpus_digest(valid_reordering) != af.ordered_corpus_digest(rows)


@pytest.mark.parametrize("mutation", ["reordered", "pair", "source", "speaker", "stratum"])
def test_corpus_rejects_reordering_duplicates_and_wrong_strata(
    synthetic_protocol: Mapping[str, Any], mutation: str
):
    rows = copy.deepcopy(synthetic_protocol["rows"])
    if mutation == "reordered":
        rows[0], rows[2] = rows[2], rows[0]
    elif mutation == "pair":
        for field in _PAIR_FIELDS:
            rows[2][field] = rows[0][field]
        _refresh_pair_identity(rows[2])
    elif mutation == "source":
        rows[2]["source_id"] = rows[0]["source_id"]
        _refresh_pair_identity(rows[2])
    elif mutation == "speaker":
        rows[2]["speaker_id"] = rows[0]["speaker_id"]
        _refresh_pair_identity(rows[2])
    else:
        rows[0]["config"], rows[0]["split"] = af.STRATA[1]
        _refresh_pair_identity(rows[0])

    with pytest.raises(AudioFitContractError):
        af.validate_corpus_rows(rows)


def test_attempt_ledger_enforces_rank_order_selection_link_and_quota(
    synthetic_protocol: Mapping[str, Any],
):
    ledger = synthetic_protocol["ledger"]
    config = synthetic_protocol["config"]
    assert af.validate_attempt_ledger(ledger, corpus_config=config) == ledger

    bool_index = copy.deepcopy(ledger)
    bool_index[0]["attempt_index"] = False
    with pytest.raises(AudioFitContractError, match="must be an integer"):
        af.validate_attempt_ledger(bool_index, corpus_config=config)

    float_stratum_index = copy.deepcopy(ledger)
    float_stratum_index[0]["stratum_attempt_index"] = 0.0
    with pytest.raises(AudioFitContractError, match="must be an integer"):
        af.validate_attempt_ledger(float_stratum_index, corpus_config=config)

    wrong_rank = copy.deepcopy(ledger)
    wrong_rank[0]["speaker_rank_sha256"] = _different_sha256(wrong_rank[0]["speaker_rank_sha256"])
    with pytest.raises(AudioFitContractError, match="speaker rank changed"):
        af.validate_attempt_ledger(wrong_rank, corpus_config=config)

    wrong_utterance_rank = copy.deepcopy(ledger)
    wrong_utterance_rank[0]["utterance_rank_sha256"] = _different_sha256(
        wrong_utterance_rank[0]["utterance_rank_sha256"]
    )
    with pytest.raises(AudioFitContractError, match="utterance rank changed"):
        af.validate_attempt_ledger(wrong_utterance_rank, corpus_config=config)

    same_speaker_attempts = copy.deepcopy(ledger)
    selected = same_speaker_attempts[0]
    earlier_source = None
    earlier_rank = None
    for suffix in range(100_000):
        candidate_source = f"{selected['source_id']}-rejected-{suffix}"
        candidate_rank = af.metadata_rank(
            af.AUDIO_SELECTION_SEED,
            af.DATASET_ID,
            af.LIBRISPEECH_REVISION,
            selected["config"],
            selected["split"],
            selected["speaker_id"],
            candidate_source,
        )
        if candidate_rank < selected["utterance_rank_sha256"]:
            earlier_source, earlier_rank = candidate_source, candidate_rank
            break
    assert earlier_source is not None and earlier_rank is not None
    rejected = copy.deepcopy(selected)
    rejected.update(
        {
            "source_id": earlier_source,
            "utterance_rank_sha256": earlier_rank,
            "outcome": "rejected",
            "reason": "processor_contract",
            "pair_id": None,
        }
    )
    same_speaker_attempts.insert(0, rejected)
    attempt_counts = {name: 0 for name, _ in af.STRATA}
    for attempt_index, attempt in enumerate(same_speaker_attempts):
        attempt["attempt_index"] = attempt_index
        attempt["stratum_attempt_index"] = attempt_counts[attempt["config"]]
        attempt_counts[attempt["config"]] += 1
    assert (
        af.validate_attempt_ledger(same_speaker_attempts, corpus_config=config)
        == same_speaker_attempts
    )

    wrong_utterance_order = copy.deepcopy(same_speaker_attempts)
    wrong_utterance_order[0], wrong_utterance_order[1] = (
        wrong_utterance_order[1],
        wrong_utterance_order[0],
    )
    for index in (0, 1):
        wrong_utterance_order[index]["attempt_index"] = index
        wrong_utterance_order[index]["stratum_attempt_index"] = index
    with pytest.raises(AudioFitContractError, match="utterance-rank order"):
        af.validate_attempt_ledger(wrong_utterance_order, corpus_config=config)

    wrong_order = copy.deepcopy(ledger)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    for index in (0, 1):
        wrong_order[index]["attempt_index"] = index
        wrong_order[index]["stratum_attempt_index"] = index
    with pytest.raises(AudioFitContractError, match="speaker-rank order"):
        af.validate_attempt_ledger(wrong_order, corpus_config=config)

    wrong_selection = copy.deepcopy(ledger)
    wrong_selection[0]["pair_id"] = _different_sha256(wrong_selection[0]["pair_id"])
    assert af.validate_attempt_ledger(wrong_selection, corpus_config=config) == wrong_selection
    wrong_selection_envelope = copy.deepcopy(synthetic_protocol["envelope"])
    wrong_selection_envelope["attempt_ledger_sha256"] = af.canonical_jsonl_sha256(wrong_selection)
    with pytest.raises(AudioFitContractError, match="selected identities"):
        af.validate_corpus_envelope(
            wrong_selection_envelope,
            synthetic_protocol["rows"],
            wrong_selection,
        )

    short_quota = copy.deepcopy(ledger[:-1])
    with pytest.raises(AudioFitContractError, match="selected counts"):
        af.validate_attempt_ledger(short_quota, corpus_config=config)


def test_attempt_ledger_enforces_per_stratum_attempt_bound(
    synthetic_protocol: Mapping[str, Any],
    monkeypatch,
):
    monkeypatch.setattr(af, "MAX_ATTEMPTS_PER_STRATUM", af.STRATUM_SIZE)
    bounded_config = af.build_corpus_config(
        GEMMA4_PROFILE,
        source_digest=_SOURCE_SHA256,
        lock_sha256=_LOCK_SHA256,
    )
    ledger = copy.deepcopy(synthetic_protocol["ledger"])
    last_clean = ledger[af.STRATUM_SIZE - 1]
    extra_speaker = last_clean["speaker_id"]
    extra_source = None
    extra_utterance_rank = None
    for suffix in range(100_000):
        candidate_source = f"{last_clean['source_id']}-rejected-{suffix}"
        candidate_rank = af.metadata_rank(
            af.AUDIO_SELECTION_SEED,
            af.DATASET_ID,
            af.LIBRISPEECH_REVISION,
            "clean",
            "train.360",
            extra_speaker,
            candidate_source,
        )
        if candidate_rank < last_clean["utterance_rank_sha256"]:
            extra_source, extra_utterance_rank = candidate_source, candidate_rank
            break
    assert extra_source is not None and extra_utterance_rank is not None
    ledger.insert(
        af.STRATUM_SIZE - 1,
        {
            "schema_version": af.SCHEMA_VERSION,
            "kind": af.ATTEMPT_ROW_KIND,
            "attempt_index": -1,
            "stratum": "clean",
            "stratum_attempt_index": -1,
            "config": "clean",
            "split": "train.360",
            "speaker_id": extra_speaker,
            "source_id": extra_source,
            "speaker_rank_sha256": last_clean["speaker_rank_sha256"],
            "utterance_rank_sha256": extra_utterance_rank,
            "outcome": "rejected",
            "reason": "processor_contract",
            "pair_id": None,
        },
    )
    stratum_attempts = {config: 0 for config, _ in af.STRATA}
    for attempt_index, attempt in enumerate(ledger):
        attempt["attempt_index"] = attempt_index
        attempt["stratum_attempt_index"] = stratum_attempts[attempt["config"]]
        stratum_attempts[attempt["config"]] += 1

    with pytest.raises(AudioFitContractError, match="exceeds 500 bounded attempts"):
        af.validate_attempt_ledger(ledger, corpus_config=bounded_config)


def test_transcript_input_id_and_pcm_hashes_are_exact_and_portable():
    transcript = " Café\u0301 stays exact. "
    assert (
        af.transcript_sha256(transcript) == hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    )
    assert af.transcript_sha256(transcript) != af.transcript_sha256(transcript.strip())
    assert af.transcript_sha256("é") != af.transcript_sha256("e\u0301")

    ids = [[0, 2, 258881, 9]]
    payload = {
        "encoding": "canonical_json_int64_matrix_v1",
        "shape": [1, 4],
        "values": ids,
    }
    expected_ids = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    assert af.input_ids_sha256(ids) == expected_ids
    assert af.input_ids_sha256(np.asarray(ids, dtype=np.int64)) == expected_ids
    assert af.input_ids_sha256([[0, 2, 258881]]) != expected_ids
    with pytest.raises(AudioFitContractError, match="nonnegative integer"):
        af.input_ids_sha256([[0, True]])

    samples = [-32768, -1, 0, 1, 32767]
    little = np.asarray(samples, dtype="<i2")
    big = np.asarray(samples, dtype=">i2")
    expected_pcm = hashlib.sha256(little.tobytes(order="C")).hexdigest()
    assert af.decoded_pcm_sha256(little) == expected_pcm
    assert af.decoded_pcm_sha256(big) == expected_pcm
    with pytest.raises(AudioFitContractError, match="signed-int16"):
        af.decoded_pcm_sha256(np.asarray([0, 1, 32767], dtype=np.uint16))


@pytest.mark.parametrize(
    "field",
    [
        "transcript",
        "source_id",
        "audio_sha256",
        "decoded_pcm_sha256",
        "processor_input_ids_sha256",
        "fit_input_ids_sha256",
    ],
)
def test_restored_row_rejects_every_pair_or_processor_identity_change(
    synthetic_protocol: Mapping[str, Any], field: str
):
    expected = synthetic_protocol["rows"][0]
    assert af.validate_restored_row(expected, copy.deepcopy(expected)) == expected
    restored = copy.deepcopy(expected)
    if field == "transcript":
        restored[field] += " changed"
        _refresh_pair_identity(restored)
    elif field == "source_id":
        restored[field] += "-changed"
        _refresh_pair_identity(restored)
    elif field in {"audio_sha256", "decoded_pcm_sha256"}:
        restored[field] = _different_sha256(restored[field])
        _refresh_pair_identity(restored)
    else:
        restored[field] = _different_sha256(restored[field])

    assert af.validate_corpus_row(restored) == restored
    with pytest.raises(AudioFitContractError, match="restored corpus row"):
        af.validate_restored_row(expected, restored)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema_version", True),
        ("dataset.token", 0),
        ("audio.channels", True),
        ("audio.crop", 0),
        ("processor.require_contiguous_audio_span", 1),
        ("transcript.required", 1),
        ("transcript.fit", 0),
    ],
)
def test_corpus_config_rejects_bool_int_aliases_for_fixed_scalars(
    synthetic_protocol: Mapping[str, Any], path: str, value: Any
):
    changed = _replace_nested(synthetic_protocol["config"], path, value)
    with pytest.raises(AudioFitContractError):
        af.validate_corpus_config(changed)


def test_row_and_fit_config_reject_bool_int_aliases_for_fixed_scalars(
    synthetic_protocol: Mapping[str, Any], synthetic_fit_bundle: Mapping[str, Any]
):
    row = copy.deepcopy(synthetic_protocol["rows"][0])
    row["schema_version"] = True
    with pytest.raises(AudioFitContractError):
        af.validate_corpus_row(row)

    for path, value in (
        ("schema_version", True),
        ("fit.transcript_fit", 0),
        ("fit.resume", 1),
        ("fit.compile", 0),
    ):
        changed = _replace_nested(synthetic_fit_bundle["config"], path, value)
        with pytest.raises(AudioFitContractError):
            af.validate_fit_config(changed)


def test_fit_run_and_checkpoint_identity_bind_model_source_corpus_and_config(
    synthetic_protocol: Mapping[str, Any], synthetic_fit_bundle: Mapping[str, Any]
):
    config = synthetic_fit_bundle["config"]
    paths = synthetic_fit_bundle["paths"]
    identity = synthetic_fit_bundle["identity"]
    digest = af.fit_config_digest(config)

    assert af.validate_fit_config(config) == config
    assert identity["fit_config_sha256"] == digest
    assert (
        identity["ordered_corpus_sha256"] == synthetic_protocol["artifact"]["ordered_corpus_sha256"]
    )
    assert identity["expected_count"] == 1_000
    assert identity["path"] == paths["checkpoint"]
    assert af.validate_checkpoint_identity(identity) == identity

    with pytest.raises(AudioFitContractError, match="source/lock identity"):
        af.build_fit_config(
            synthetic_protocol["artifact"],
            GEMMA4_PROFILE,
            source_digest="f" * 64,
            lock_sha256=_LOCK_SHA256,
            runtime=_RUNTIME,
        )

    wrong_source = copy.deepcopy(config)
    wrong_source["source_sha256"] = "f" * 64
    with pytest.raises(AudioFitContractError, match="source/lock identity"):
        af.validate_fit_config(wrong_source)

    wrong_model = copy.deepcopy(config)
    wrong_model["model"]["model_revision"] = "wrong-revision"
    with pytest.raises(AudioFitContractError, match="model identity"):
        af.validate_fit_config(wrong_model)

    changed_config = copy.deepcopy(config)
    changed_config["runtime"]["python_version"] = "3.13.0"
    assert af.validate_fit_config(changed_config) == changed_config
    with pytest.raises(AudioFitContractError, match="checkpoint paths"):
        af.checkpoint_identity(changed_config, paths)

    windows_paths = copy.deepcopy(paths)
    windows_paths["checkpoint"] = "C:/audio-fit/working-checkpoint.pt"
    with pytest.raises(AudioFitContractError, match="checkpoint paths"):
        af.checkpoint_identity(config, windows_paths)


def test_checkpoint_state_rejects_wrong_stamps_geometry_counts_and_bool_counts(
    tmp_path: Path, synthetic_fit_bundle: Mapping[str, Any]
):
    import torch

    identity = _tiny_identity(synthetic_fit_bundle["identity"])
    means = {1: torch.eye(2, dtype=torch.float32)}
    state = _checkpoint_state(identity, af.PREFIX_COUNT, means)
    path = tmp_path / "working.pt"
    torch.save(state, path)
    assert (
        af.validate_checkpoint_state(
            path,
            identity,
            maximum_count=af.PREFIX_COUNT,
            exact_count=af.PREFIX_COUNT,
        )["n_done"]
        == af.PREFIX_COUNT
    )

    mutations = []
    wrong_fit = copy.deepcopy(state)
    wrong_fit["fit_config_sha256"] = _different_sha256(identity["fit_config_sha256"])
    mutations.append(wrong_fit)
    wrong_corpus = copy.deepcopy(state)
    wrong_corpus["ordered_corpus_sha256"] = _different_sha256(identity["ordered_corpus_sha256"])
    mutations.append(wrong_corpus)
    wrong_geometry = copy.deepcopy(state)
    wrong_geometry["target_layer"] += 1
    mutations.append(wrong_geometry)
    torn_count = copy.deepcopy(state)
    torn_count["next_idx"] -= 1
    mutations.append(torn_count)
    bool_count = copy.deepcopy(state)
    bool_count["n_done"] = True
    bool_count["next_idx"] = True
    mutations.append(bool_count)
    wrong_shape = copy.deepcopy(state)
    wrong_shape["jacobian_sum"] = {1: torch.ones(3, 3, dtype=torch.float32)}
    mutations.append(wrong_shape)

    for mutation in mutations:
        torch.save(mutation, path)
        with pytest.raises(AudioFitContractError):
            af.validate_checkpoint_state(
                path,
                identity,
                maximum_count=af.PREFIX_COUNT,
                exact_count=af.PREFIX_COUNT,
            )


def test_reconstruct_layer_means_is_exact_bounded_and_nonmutating():
    import torch

    first_mean = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    second_mean = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    prefix_sum = first_mean * af.PREFIX_COUNT
    final_sum = prefix_sum + second_mean * (af.CORPUS_SIZE - af.PREFIX_COUNT)
    prefix = {"n_done": af.PREFIX_COUNT, "jacobian_sum": {1: prefix_sum.clone()}}
    final = {"n_done": af.CORPUS_SIZE, "jacobian_sum": {1: final_sum.clone()}}

    first, full, second = af.reconstruct_layer_means(prefix, final, 1)
    assert torch.equal(first, first_mean)
    assert torch.equal(full, (first_mean + second_mean) / 2)
    assert torch.equal(second, second_mean)
    assert torch.equal(prefix["jacobian_sum"][1], prefix_sum)
    assert torch.equal(final["jacobian_sum"][1], final_sum)
    assert first.data_ptr() != prefix_sum.data_ptr()
    assert full.data_ptr() != final_sum.data_ptr()

    wrong_count = copy.deepcopy(prefix)
    wrong_count["n_done"] = af.PREFIX_COUNT - 1
    with pytest.raises(AudioFitContractError, match="exact 500/1,000"):
        af.reconstruct_layer_means(wrong_count, final, 1)

    nonfinite = copy.deepcopy(final)
    nonfinite["jacobian_sum"][1][0, 0] = float("inf")
    with pytest.raises(AudioFitContractError, match="nonfinite"):
        af.reconstruct_layer_means(prefix, nonfinite, 1)


def test_stability_metrics_and_checkpoint_artifacts_are_exact(
    tmp_path: Path, synthetic_fit_bundle: Mapping[str, Any]
):
    import torch

    identity = _tiny_identity(synthetic_fit_bundle["identity"])
    first_mean = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    second_mean = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    prefix_state = _checkpoint_state(identity, af.PREFIX_COUNT, {1: first_mean})
    final_state = _checkpoint_state(identity, af.CORPUS_SIZE, {1: (first_mean + second_mean) / 2})
    prefix_artifact, prefix_path = _seal_checkpoint(
        tmp_path, identity, prefix_state, af.PREFIX_COUNT
    )
    final_artifact, final_path = _seal_checkpoint(tmp_path, identity, final_state, af.CORPUS_SIZE)

    report = af.stability_from_checkpoints(prefix_path, final_path, identity)
    layer = report["layers"]["1"]
    assert layer["identity_centered_split_half_cosine"] == pytest.approx(0.0, abs=1e-7)
    assert layer["first_half_to_full_relative_l2"] == pytest.approx(math.sqrt(0.1), rel=1e-7)
    assert report["first_half_count"] == report["second_half_count"] == 500
    assert report["full_count"] == 1_000
    assert set(layer) == {
        "identity_centered_split_half_cosine",
        "first_half_to_full_relative_l2",
    }
    assert prefix_artifact["role"] == "prefix_500"
    assert final_artifact["role"] == "final_checkpoint"

    stability = af.stability_artifact(report, identity)
    report_path = tmp_path / stability["relative_path"]
    _write_json(report_path, report)
    assert af.validate_stability_artifact(stability, identity, path=report_path) == stability

    nonfinite = copy.deepcopy(report)
    nonfinite["layers"]["1"]["first_half_to_full_relative_l2"] = float("nan")
    with pytest.raises(AudioFitContractError, match="finite"):
        af.validate_stability_report(nonfinite, identity)


def _build_completed_run_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path]:
    import jlens
    import soundfile as sf
    import torch

    tiny_profile = replace(
        GEMMA4_PROFILE,
        d_model=2,
        source_layers=(1,),
        target_layer=2,
        dimension_batch_size=1,
    )
    monkeypatch.setattr(model_contracts, "DEFAULT_MODEL_PROFILE", tiny_profile)
    monkeypatch.setattr(
        af,
        "SOURCE_POOL_COUNTS",
        {name: af.STRATUM_SIZE for name, _ in af.STRATA},
    )

    pcm = np.zeros(32_000, dtype=np.int16)
    temporary_flac = root / "source.flac"
    sf.write(temporary_flac, pcm, af.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    audio_sha256 = hashlib.sha256(temporary_flac.read_bytes()).hexdigest()
    pcm_sha256 = af.decoded_pcm_sha256(pcm)
    blob_path = root / af.audio_blob_path(audio_sha256)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_flac.replace(blob_path)

    protocol = build_synthetic_protocol(
        profile=tiny_profile,
        audio_sha256=audio_sha256,
        pcm_sha256=pcm_sha256,
    )
    fit_bundle = build_synthetic_fit_bundle(protocol, profile=tiny_profile)
    artifact = protocol["artifact"]
    _write_json(root / artifact["source_pool_path"], protocol["source_pool"])
    envelope_path = root / artifact["envelope_path"]
    _write_json(envelope_path, protocol["envelope"])
    _write_jsonl(root / artifact["rows_path"], protocol["rows"])
    _write_jsonl(root / artifact["attempt_ledger_path"], protocol["ledger"])

    assert (
        af.validate_corpus_row(protocol["rows"][0], volume_root=root, require_file=True)
        == protocol["rows"][0]
    )
    flac_info = sf.info(blob_path)
    monkeypatch.setattr(sf, "info", lambda _path: flac_info)
    monkeypatch.setattr(
        sf,
        "read",
        lambda _path, **_kwargs: (pcm.copy(), af.SAMPLE_RATE),
    )

    identity = fit_bundle["identity"]
    first_mean = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    second_mean = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    prefix_state = _checkpoint_state(identity, af.PREFIX_COUNT, {1: first_mean})
    final_state = _checkpoint_state(identity, af.CORPUS_SIZE, {1: (first_mean + second_mean) / 2})
    prefix_artifact, prefix_path = _seal_checkpoint(root, identity, prefix_state, af.PREFIX_COUNT)
    final_artifact, final_path = _seal_checkpoint(root, identity, final_state, af.CORPUS_SIZE)

    report = af.stability_from_checkpoints(prefix_path, final_path, identity)
    stability = af.stability_artifact(report, identity)
    _write_json(root / stability["relative_path"], report)

    lens = jlens.JacobianLens(
        jacobians={1: (first_mean + second_mean) / 2},
        n_prompts=af.CORPUS_SIZE,
        d_model=2,
    )
    lens_candidate = root / "lens-candidate.pt"
    lens.save(str(lens_candidate), dtype=torch.float16)
    lens_artifact = af.lens_artifact(lens_candidate, identity, final_path)
    lens_path = root / lens_artifact["relative_path"]
    lens_path.parent.mkdir(parents=True, exist_ok=True)
    lens_candidate.replace(lens_path)
    assert (
        af.validate_lens_artifact(
            lens_artifact,
            lens_path,
            identity,
            final_path,
        )
        == lens_artifact
    )

    gates: dict[str, dict[str, Any]] = {}
    for gate in af.REQUIRED_GATES:
        gate_record = af.gate_record(fit_bundle["config"], gate)
        gate_file = root / af.gate_path(fit_bundle["config"], gate)
        _write_json(gate_file, gate_record)
        gates[gate] = af.gate_artifact(
            gate_file,
            fit_bundle["config"],
            gate,
        )

    record = {
        "schema_version": af.SCHEMA_VERSION,
        "kind": af.COMPLETED_RUN_KIND,
        "status": "complete",
        "fit_config_sha256": af.fit_config_digest(fit_bundle["config"]),
        "config": fit_bundle["config"],
        "paths": fit_bundle["paths"],
        "corpus": artifact,
        "checkpoint_identity": identity,
        "prefix_snapshot": prefix_artifact,
        "checkpoint": final_artifact,
        "stability": stability,
        "lens": lens_artifact,
        "gates": gates,
    }
    return record, envelope_path


def test_completed_run_validates_local_artifacts_and_rejects_mutation_and_drive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    record, envelope_path = _build_completed_run_fixture(tmp_path, monkeypatch)
    assert af.validate_completed_run(record, volume_root=tmp_path) == record

    float_checkpoint_count = copy.deepcopy(record)
    float_checkpoint_count["checkpoint"]["n_done"] = float(af.CORPUS_SIZE)
    with pytest.raises(AudioFitContractError, match="metadata changed"):
        af.validate_completed_run(float_checkpoint_count, volume_root=tmp_path)

    float_lens_count = copy.deepcopy(record)
    float_lens_count["lens"]["n_prompts"] = float(af.CORPUS_SIZE)
    with pytest.raises(AudioFitContractError, match="metadata changed"):
        af.validate_completed_run(float_lens_count, volume_root=tmp_path)

    windows_drive = copy.deepcopy(record)
    windows_drive["prefix_snapshot"]["relative_path"] = "C:/outside-prefix.pt"
    with pytest.raises(AudioFitContractError, match="portable relative path"):
        af.validate_completed_run(windows_drive, volume_root=tmp_path)

    missing_gate = copy.deepcopy(record)
    del missing_gate["gates"]["processor_replay"]
    with pytest.raises(AudioFitContractError, match="gate set is invalid"):
        af.validate_completed_run(missing_gate, volume_root=tmp_path)

    gate_path = tmp_path / record["gates"]["processor_replay"]["relative_path"]
    original_gate = gate_path.read_bytes()
    gate_path.write_bytes(original_gate + b" ")
    with pytest.raises(AudioFitContractError, match="gate artifact.*changed"):
        af.validate_completed_run(record, volume_root=tmp_path)
    gate_path.write_bytes(original_gate)

    original_envelope = envelope_path.read_bytes()
    envelope_path.write_bytes(original_envelope + b" ")
    with pytest.raises(AudioFitContractError, match="envelope bytes changed"):
        af.validate_completed_run(record, volume_root=tmp_path)


def test_checkpoint_loader_rejects_oversized_storage_before_torch_load(
    tmp_path: Path,
    synthetic_fit_bundle: Mapping[str, Any],
):
    identity = synthetic_fit_bundle["identity"]
    maximum_bytes = (
        len(identity["source_layers"]) * identity["d_model"] * identity["d_model"] * 4
        + 32 * 1024 * 1024
    )
    oversized = tmp_path / "oversized-checkpoint.pt"
    with oversized.open("wb") as handle:
        handle.seek(maximum_bytes)
        handle.write(b"\0")

    with pytest.raises(AudioFitContractError, match="above the bounded maximum"):
        af.validate_checkpoint_state(
            oversized,
            identity,
            maximum_count=af.CORPUS_SIZE,
        )
