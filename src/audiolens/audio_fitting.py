"""Pure, source-bound corpus and artifact contracts for audio J-lens fitting.

This module deliberately contains no Modal, dataset, processor, model, or eager
Torch imports.  Heavy dependencies are loaded only by the validators that need
to inspect a waveform or a serialized fit artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .models import AudioFitContractError

SCHEMA_VERSION = 2

DATASET_ID = "openslr/librispeech_asr"
LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
AUDIO_SELECTION_SEED = 20260710

STRATA = (("clean", "train.360"), ("other", "train.500"))
STRATUM_SIZE = 500
CORPUS_SIZE = 2 * STRATUM_SIZE
PREFIX_COUNT = CORPUS_SIZE // 2
SOURCE_POOL_COUNTS = {
    "clean": 104_014,
    "other": 148_688,
}
MAX_ATTEMPTS_PER_STRATUM = max(SOURCE_POOL_COUNTS.values())

SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 4.0
MAX_SEQUENCE_LENGTH = 128
SKIP_FIRST = 16
AUDIO_START = 5
CLOSING_TOKEN_COUNT = 3
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024


SOURCE_POOL_KIND = "source_bound_audio_source_pool"
CORPUS_CONFIG_KIND = "source_bound_audio_corpus_config"
CORPUS_ROW_KIND = "source_bound_audio_corpus_row"
ATTEMPT_ROW_KIND = "source_bound_audio_selection_attempt"
CORPUS_ENVELOPE_KIND = "source_bound_audio_corpus"
CORPUS_ARTIFACT_KIND = "source_bound_audio_corpus_artifact"
FIT_CONFIG_KIND = "source_bound_audio_jlens_fit_config"
CHECKPOINT_IDENTITY_KIND = "source_bound_audio_checkpoint_identity"
PREFIX_SNAPSHOT_KIND = "source_bound_audio_prefix_snapshot"
FINAL_CHECKPOINT_KIND = "source_bound_audio_final_checkpoint"
STABILITY_REPORT_KIND = "source_bound_audio_stability_report"
STABILITY_ARTIFACT_KIND = "source_bound_audio_stability_artifact"
LENS_ARTIFACT_KIND = "source_bound_audio_lens_artifact"
COMPLETED_RUN_KIND = "source_bound_audio_jlens_run"
GATE_RECORD_KIND = "source_bound_audio_gate"
GATE_ARTIFACT_KIND = "source_bound_audio_gate_artifact"
REQUIRED_GATES = (
    "selection_replay",
    "source_restore",
    "processor_replay",
    "decoder_replay",
    "smoke_resume",
)

REJECTION_REASONS = frozenset(
    {
        "speaker_already_selected",
        "missing_audio",
        "missing_audio_bytes",
        "decode_error",
        "not_native_mono_16khz",
        "nonfinite_waveform",
        "decoded_sample_count_mismatch",
        "duration_out_of_range",
        "processor_contract",
    }
)

_HASH_ALGORITHM = "sha256_canonical_json_v1"
_PCM_ENCODING = "signed_int16_little_endian_c_order"
_INPUT_IDS_ENCODING = "canonical_json_int64_matrix_v1"
_SOURCE_METADATA_FIELDS = frozenset(
    {
        "dataset",
        "revision",
        "config",
        "split",
        "source_id",
        "speaker_id",
        "chapter_id",
        "transcript",
    }
)
_SOURCE_POOL_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "config",
        "corpus_config_sha256",
        "source_pool_sha256",
        "row_count",
        "stratum_counts",
        "rows",
    }
)
_CORPUS_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "selection_index",
        "stratum_index",
        *_SOURCE_METADATA_FIELDS,
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
)
_ATTEMPT_ROW_FIELDS = frozenset(
    {
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
)
_CORPUS_ENVELOPE_FIELDS = frozenset(
    {
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
)
_CORPUS_ARTIFACT_FIELDS = frozenset(
    {
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
)
_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "role",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "path",
        "expected_count",
        "source_layers",
        "target_layer",
        "skip_first",
        "d_model",
        "dtype",
        "snapshot_dir",
        "stability_dir",
        "lens_dir",
    }
)
_CHECKPOINT_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "role",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "checkpoint_identity_sha256",
        "relative_path",
        "sha256",
        "bytes",
        "n_done",
        "next_idx",
        "dtype",
        "d_model",
        "source_layers",
        "target_layer",
        "skip_first",
    }
)
_STABILITY_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "prefix_checkpoint_sha256",
        "final_checkpoint_sha256",
        "first_half_count",
        "second_half_count",
        "full_count",
        "cosine_centering",
        "relative_l2_reference",
        "layers",
    }
)
_STABILITY_LAYER_FIELDS = frozenset(
    {
        "identity_centered_split_half_cosine",
        "first_half_to_full_relative_l2",
    }
)
_STABILITY_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "role",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "relative_path",
        "sha256",
        "bytes",
    }
)
_GATE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "gate",
        "status",
        "fit_config_sha256",
        "ordered_corpus_sha256",
    }
)
_GATE_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "gate",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "relative_path",
        "sha256",
        "bytes",
    }
)


_LENS_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "role",
        "fit_config_sha256",
        "ordered_corpus_sha256",
        "final_checkpoint_sha256",
        "relative_path",
        "sha256",
        "bytes",
        "dtype",
        "n_prompts",
        "d_model",
        "source_layers",
    }
)
_COMPLETED_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "fit_config_sha256",
        "config",
        "paths",
        "corpus",
        "checkpoint_identity",
        "prefix_snapshot",
        "checkpoint",
        "stability",
        "lens",
        "gates",
    }
)


def _fail(message: str) -> None:
    raise AudioFitContractError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        _fail(f"{label} fields are invalid; missing={missing}, extra={extra}")
    if "schema_version" in fields:
        _require_int(value["schema_version"], f"{label}.schema_version", minimum=0)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if not _is_int(value) or (minimum is not None and value < minimum):
        suffix = "an integer" if minimum is None else f"an integer >= {minimum}"
        _fail(f"{label} must be {suffix}, got {value!r}")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        _fail(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase 64-hex SHA-256 digest")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _validate_json_value(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if _is_int(value):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{label} contains a nonfinite number")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            _fail(f"{label} contains a non-string object key")
        for key, item in value.items():
            _validate_json_value(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    _fail(f"{label} is not portable JSON: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _same_json(left: Any, right: Any) -> bool:
    """Compare portable JSON without Python's bool/int equality coercion."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(config: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(config)))


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} must be a nonempty portable relative path")
    path = pathlib.PurePosixPath(value)
    windows_path = pathlib.PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"{label} must be a normalized portable relative path, got {value!r}")
    return path.as_posix()


def _physical_path(volume_root: str | pathlib.Path, relative: str, label: str) -> pathlib.Path:
    portable = _relative_path(relative, label)
    root = pathlib.Path(volume_root).resolve()
    candidate = (root / portable).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes volume root: {portable}")
    return candidate


def transcript_sha256(transcript: str) -> str:
    """Hash the exact, unnormalized UTF-8 source transcript."""

    _require_nonempty_string(transcript, "transcript")
    return _sha256_bytes(transcript.encode("utf-8"))


def decoded_pcm_sha256(pcm: Any) -> str:
    """Hash mono signed-int16 PCM as little-endian C-order sample bytes."""

    import numpy as np

    array = np.asarray(pcm)
    if array.ndim != 1 or array.dtype.kind != "i" or array.dtype.itemsize != 2:
        _fail(
            "decoded PCM must be a one-dimensional signed-int16 array, "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    little = np.ascontiguousarray(array.astype(np.dtype("<i2"), copy=False))
    return _sha256_bytes(little.tobytes(order="C"))


def input_ids_sha256(input_ids: Any) -> str:
    """Hash one complete ``[1, seq]`` ID matrix using portable canonical JSON."""

    values = input_ids
    if hasattr(values, "detach"):
        values = values.detach().to(device="cpu").tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], list)
        or not values[0]
    ):
        _fail("input IDs must have shape [1, seq] with a nonempty sequence")
    row: list[int] = []
    for index, value in enumerate(values[0]):
        if not _is_int(value) or value < 0:
            _fail(f"input ID at [0,{index}] must be a nonnegative integer")
        row.append(value)
    payload = {
        "encoding": _INPUT_IDS_ENCODING,
        "shape": [1, len(row)],
        "values": [row],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def source_rank_payload(
    seed: int,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    speaker_id: int,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Return the exact metadata-only speaker or utterance rank payload."""

    _require_int(seed, "rank seed", minimum=0)
    _require_nonempty_string(dataset, "rank dataset")
    _require_nonempty_string(revision, "rank revision")
    _require_nonempty_string(config, "rank config")
    _require_nonempty_string(split, "rank split")
    _require_int(speaker_id, "rank speaker_id", minimum=0)
    payload: dict[str, Any] = {
        "seed": seed,
        "dataset": dataset,
        "revision": revision,
        "config": config,
        "split": split,
        "speaker_id": speaker_id,
    }
    if source_id is not None:
        payload["source_id"] = _require_nonempty_string(source_id, "rank source_id")
    return payload


def metadata_rank(
    seed: int,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    speaker_id: int,
    source_id: str | None = None,
) -> str:
    """Cryptographically rank a speaker or utterance independently of enumeration."""

    return _sha256_bytes(
        _canonical_json_bytes(
            source_rank_payload(seed, dataset, revision, config, split, speaker_id, source_id)
        )
    )


def _validate_source_metadata(row: Any, label: str) -> dict[str, Any]:
    value = _require_mapping(row, label)
    _require_exact_fields(value, _SOURCE_METADATA_FIELDS, label)
    if value["dataset"] != DATASET_ID:
        _fail(f"{label}.dataset must be {DATASET_ID!r}")
    if value["revision"] != LIBRISPEECH_REVISION:
        _fail(f"{label}.revision must be the pinned LibriSpeech revision")
    coordinates = (value["config"], value["split"])
    if coordinates not in STRATA:
        _fail(f"{label} has unsupported stratum {coordinates!r}")
    _require_nonempty_string(value["source_id"], f"{label}.source_id")
    _require_int(value["speaker_id"], f"{label}.speaker_id", minimum=0)
    _require_int(value["chapter_id"], f"{label}.chapter_id", minimum=0)
    _require_nonempty_string(value["transcript"], f"{label}.transcript")
    return dict(value)


def source_pool_digest(metadata_rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash a metadata pool as an order-independent set of canonical rows."""

    row_hashes: list[bytes] = []
    source_ids: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(metadata_rows):
        row = _validate_source_metadata(raw, f"source metadata row {index}")
        source_key = (row["config"], row["split"], row["source_id"])
        if source_key in source_ids:
            _fail(f"duplicate source metadata identity {source_key!r}")
        source_ids.add(source_key)
        row_hashes.append(bytes.fromhex(_sha256_bytes(_canonical_json_bytes(row))))
    if not row_hashes:
        _fail("source metadata pool must not be empty")
    row_hashes.sort()
    digest = hashlib.sha256()
    digest.update(b"source_bound_audio_source_pool_v1\0")
    digest.update(len(row_hashes).to_bytes(8, "big"))
    for row_hash in row_hashes:
        digest.update(row_hash)
    return digest.hexdigest()


def _source_pool_ranks_validated(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "config": row["config"],
            "split": row["split"],
            "speaker_id": row["speaker_id"],
            "source_id": row["source_id"],
            "speaker_rank_sha256": metadata_rank(
                AUDIO_SELECTION_SEED,
                DATASET_ID,
                LIBRISPEECH_REVISION,
                row["config"],
                row["split"],
                row["speaker_id"],
            ),
            "utterance_rank_sha256": metadata_rank(
                AUDIO_SELECTION_SEED,
                DATASET_ID,
                LIBRISPEECH_REVISION,
                row["config"],
                row["split"],
                row["speaker_id"],
                row["source_id"],
            ),
        }
        for row in rows
    ]


def source_pool_ranks(
    metadata_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive deterministic traversal ranks without duplicating them on disk."""

    rows = [
        _validate_source_metadata(row, f"source metadata row {index}")
        for index, row in enumerate(metadata_rows)
    ]
    if not rows:
        _fail("source metadata pool must not be empty")
    return _source_pool_ranks_validated(rows)


def validate_source_pool_record(
    record: Mapping[str, Any],
    corpus_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete frozen metadata pool used to derive traversal ranks."""

    config = validate_corpus_config(corpus_config)
    value = _require_mapping(record, "source pool")
    _require_exact_fields(value, _SOURCE_POOL_RECORD_FIELDS, "source pool")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != SOURCE_POOL_KIND:
        _fail("source pool kind/schema version is invalid")
    if not _same_json(value["config"], config):
        _fail("source pool corpus config changed")
    if value["corpus_config_sha256"] != corpus_config_digest(config):
        _fail("source pool corpus-config digest changed")
    rows_value = value["rows"]
    if not isinstance(rows_value, list) or not rows_value:
        _fail("source pool rows must be a nonempty list")
    rows = [
        _validate_source_metadata(row, f"source pool row {index}")
        for index, row in enumerate(rows_value)
    ]
    expected_rows = sorted(
        rows,
        key=lambda row: (
            STRATA.index((row["config"], row["split"])),
            row["speaker_id"],
            row["source_id"],
        ),
    )
    if not _same_json(rows, expected_rows):
        _fail("source pool rows are outside canonical storage order")
    digest = source_pool_digest(rows)
    if value["source_pool_sha256"] != digest:
        _fail("source pool content digest changed")
    _require_int(value["row_count"], "source pool row_count", minimum=1)
    if value["row_count"] != len(rows):
        _fail("source pool row_count changed")
    expected_counts = {
        name: sum(row["config"] == name and row["split"] == split for row in rows)
        for name, split in STRATA
    }
    if (
        not _same_json(value["stratum_counts"], expected_counts)
        or not _same_json(expected_counts, SOURCE_POOL_COUNTS)
        or value["row_count"] != sum(SOURCE_POOL_COUNTS.values())
    ):
        _fail(
            "source pool does not contain the complete pinned LibriSpeech "
            f"splits: expected {SOURCE_POOL_COUNTS}, got {expected_counts}"
        )
    return dict(value)


def build_corpus_config(
    profile: Any,
    *,
    source_digest: str,
    lock_sha256: str,
) -> dict[str, Any]:
    """Build the one immutable production corpus-selection configuration."""

    source_digest = _require_sha256(source_digest, "audio-fit source digest")
    lock_sha256 = _require_sha256(lock_sha256, "lock digest")
    _require_int(
        MAX_ATTEMPTS_PER_STRATUM,
        "maximum attempts per stratum",
        minimum=STRATUM_SIZE,
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": CORPUS_CONFIG_KIND,
        "source_sha256": source_digest,
        "lock_sha256": lock_sha256,
        "dataset": {
            "id": DATASET_ID,
            "revision": LIBRISPEECH_REVISION,
            "token": False,
        },
        "selection": {
            "seed": AUDIO_SELECTION_SEED,
            "strata": [
                {"config": name, "split": split, "count": STRATUM_SIZE} for name, split in STRATA
            ],
            "unique_speakers": "global",
            "speaker_rank": _HASH_ALGORITHM,
            "utterance_rank": _HASH_ALGORITHM,
            "alternation": [f"{name}/{split}" for name, split in STRATA],
            "max_attempts_per_stratum": MAX_ATTEMPTS_PER_STRATUM,
        },
        "audio": {
            "channels": 1,
            "sampling_rate": SAMPLE_RATE,
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "crop": False,
            "pad": False,
            "resample": False,
            "decoded_pcm_encoding": _PCM_ENCODING,
        },
        "processor": {
            "profile_key": profile.key,
            "profile_version": profile.version,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "adapter_source": profile.adapter_source,
            "max_sequence_length": profile.max_sequence_length,
            "skip_first": profile.skip_first,
            "require_contiguous_audio_span": True,
            "require_nonempty_stock_positions": True,
        },
        "transcript": {
            "required": True,
            "normalization": "none_utf8_exact",
            "fit": False,
        },
    }
    return validate_corpus_config(config)


def validate_corpus_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(config, "corpus config")
    expected_top = frozenset(
        {
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
    )
    _require_exact_fields(value, expected_top, "corpus config")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != CORPUS_CONFIG_KIND:
        _fail("corpus config kind/schema version is invalid")
    _require_sha256(value["source_sha256"], "corpus config source_sha256")
    _require_sha256(value["lock_sha256"], "corpus config lock_sha256")

    dataset = _require_mapping(value["dataset"], "corpus config dataset")
    _require_exact_fields(dataset, frozenset({"id", "revision", "token"}), "corpus config dataset")
    if not _same_json(
        dataset,
        {"id": DATASET_ID, "revision": LIBRISPEECH_REVISION, "token": False},
    ):
        _fail("corpus config dataset identity is not the pinned anonymous LibriSpeech source")

    selection = _require_mapping(value["selection"], "corpus config selection")
    _require_exact_fields(
        selection,
        frozenset(
            {
                "seed",
                "strata",
                "unique_speakers",
                "speaker_rank",
                "utterance_rank",
                "alternation",
                "max_attempts_per_stratum",
            }
        ),
        "corpus config selection",
    )
    expected_strata = [
        {"config": name, "split": split, "count": STRATUM_SIZE} for name, split in STRATA
    ]
    if selection["seed"] != AUDIO_SELECTION_SEED:
        _fail(f"corpus selection seed must be {AUDIO_SELECTION_SEED}")
    if not _same_json(selection["strata"], expected_strata):
        _fail("corpus selection strata/counts are invalid")
    if selection["unique_speakers"] != "global":
        _fail("corpus selection must require globally unique speakers")
    if (
        selection["speaker_rank"] != _HASH_ALGORITHM
        or selection["utterance_rank"] != _HASH_ALGORITHM
    ):
        _fail("corpus selection metadata rank algorithm is invalid")
    if selection["alternation"] != [f"{name}/{split}" for name, split in STRATA]:
        _fail("corpus selection alternation is invalid")
    _require_int(
        selection["max_attempts_per_stratum"],
        "corpus selection max_attempts_per_stratum",
        minimum=STRATUM_SIZE,
    )
    if selection["max_attempts_per_stratum"] != MAX_ATTEMPTS_PER_STRATUM:
        _fail(f"corpus selection max_attempts_per_stratum must equal {MAX_ATTEMPTS_PER_STRATUM}")

    audio = _require_mapping(value["audio"], "corpus config audio")
    expected_audio = {
        "channels": 1,
        "sampling_rate": SAMPLE_RATE,
        "min_duration_seconds": MIN_DURATION_SECONDS,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "crop": False,
        "pad": False,
        "resample": False,
        "decoded_pcm_encoding": _PCM_ENCODING,
    }
    _require_exact_fields(audio, frozenset(expected_audio), "corpus config audio")
    if not _same_json(audio, expected_audio):
        _fail("corpus audio contract is invalid")

    processor = _require_mapping(value["processor"], "corpus config processor")
    _require_exact_fields(
        processor,
        frozenset(
            {
                "profile_key",
                "profile_version",
                "model_id",
                "model_revision",
                "adapter_source",
                "max_sequence_length",
                "skip_first",
                "require_contiguous_audio_span",
                "require_nonempty_stock_positions",
            }
        ),
        "corpus config processor",
    )
    for key in ("profile_key", "model_id", "model_revision", "adapter_source"):
        _require_nonempty_string(processor[key], f"corpus config processor.{key}")
    _require_int(processor["profile_version"], "corpus config processor.profile_version", minimum=1)
    if (
        processor["max_sequence_length"] != MAX_SEQUENCE_LENGTH
        or processor["skip_first"] != SKIP_FIRST
    ):
        _fail("corpus processor sequence/skip contract is invalid")
    if (
        processor["require_contiguous_audio_span"] is not True
        or processor["require_nonempty_stock_positions"] is not True
    ):
        _fail("corpus processor eligibility flags are invalid")
    from .models import DEFAULT_MODEL_PROFILE

    profile = DEFAULT_MODEL_PROFILE
    expected_processor = {
        "profile_key": profile.key,
        "profile_version": profile.version,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "adapter_source": profile.adapter_source,
        "max_sequence_length": profile.max_sequence_length,
        "skip_first": profile.skip_first,
        "require_contiguous_audio_span": True,
        "require_nonempty_stock_positions": True,
    }
    if not _same_json(processor, expected_processor):
        _fail("corpus processor identity differs from the fixed model profile")

    transcript = _require_mapping(value["transcript"], "corpus config transcript")
    expected_transcript = {
        "required": True,
        "normalization": "none_utf8_exact",
        "fit": False,
    }
    _require_exact_fields(transcript, frozenset(expected_transcript), "corpus config transcript")
    if not _same_json(transcript, expected_transcript):
        _fail("corpus transcript contract is invalid")
    _validate_json_value(value, "corpus config")
    return dict(value)


def corpus_config_digest(config: Mapping[str, Any]) -> str:
    return _config_digest(validate_corpus_config(config))


def source_pool_path(config: Mapping[str, Any], source_pool_sha256: str) -> str:
    config_sha = corpus_config_digest(config)
    pool_sha = _require_sha256(source_pool_sha256, "source pool digest")
    return f"audio-source-pools/{config_sha}/{pool_sha}/pool.json"


def corpus_paths(config: Mapping[str, Any], ordered_corpus_sha256: str) -> dict[str, str]:
    config_sha = corpus_config_digest(config)
    ordered_sha = _require_sha256(ordered_corpus_sha256, "ordered corpus digest")
    root = f"audio-corpora/{config_sha}/{ordered_sha}"
    return {
        "envelope": f"{root}/envelope.json",
        "rows": f"{root}/rows.jsonl",
        "attempt_ledger": f"{root}/attempt-ledger.jsonl",
    }


def audio_blob_path(audio_sha256: str) -> str:
    return f"audio-blobs/{_require_sha256(audio_sha256, 'audio digest')}.flac"


def make_pair_id(row: Mapping[str, Any]) -> str:
    """Bind source coordinates, exact transcript, encoded audio, and decoded PCM."""

    value = _require_mapping(row, "pair row")
    required = (
        "dataset",
        "revision",
        "config",
        "split",
        "source_id",
        "speaker_id",
        "chapter_id",
        "transcript",
        "transcript_sha256",
        "audio_sha256",
        "decoded_pcm_sha256",
    )
    missing = [key for key in required if key not in value]
    if missing:
        _fail(f"pair row is missing identity fields {missing}")
    payload = {key: value[key] for key in required}
    metadata = {key: payload[key] for key in _SOURCE_METADATA_FIELDS}
    _validate_source_metadata(metadata, "pair source metadata")
    if payload["transcript_sha256"] != transcript_sha256(payload["transcript"]):
        _fail("pair transcript digest does not match exact transcript")
    _require_sha256(payload["audio_sha256"], "pair audio_sha256")
    _require_sha256(payload["decoded_pcm_sha256"], "pair decoded_pcm_sha256")
    return _sha256_bytes(_canonical_json_bytes(payload))


def validate_corpus_row(
    row: Mapping[str, Any],
    *,
    expected_index: int | None = None,
    volume_root: str | pathlib.Path | None = None,
    require_file: bool = False,
) -> dict[str, Any]:
    value = _require_mapping(row, "corpus row")
    _require_exact_fields(value, _CORPUS_ROW_FIELDS, "corpus row")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != CORPUS_ROW_KIND:
        _fail("corpus row kind/schema version is invalid")
    selection_index = _require_int(
        value["selection_index"], "corpus row selection_index", minimum=0
    )
    stratum_index = _require_int(value["stratum_index"], "corpus row stratum_index", minimum=0)
    if expected_index is not None and selection_index != expected_index:
        _fail(f"corpus row {expected_index} has selection_index={selection_index}")
    if selection_index >= CORPUS_SIZE:
        _fail(f"corpus row selection_index {selection_index} is outside the 1,000-row corpus")
    expected_coordinates = STRATA[selection_index % len(STRATA)]
    if (value["config"], value["split"]) != expected_coordinates:
        _fail(
            f"corpus row {selection_index} has stratum {(value['config'], value['split'])!r}, "
            f"expected {expected_coordinates!r}"
        )
    if stratum_index != selection_index // len(STRATA):
        _fail(
            f"corpus row {selection_index} has stratum_index={stratum_index}, "
            f"expected {selection_index // len(STRATA)}"
        )
    metadata = {key: value[key] for key in _SOURCE_METADATA_FIELDS}
    _validate_source_metadata(metadata, f"corpus row {selection_index} source metadata")
    for key in (
        "pair_id",
        "transcript_sha256",
        "audio_sha256",
        "decoded_pcm_sha256",
        "processor_input_ids_sha256",
        "fit_input_ids_sha256",
    ):
        _require_sha256(value[key], f"corpus row {selection_index}.{key}")
    if value["transcript_sha256"] != transcript_sha256(value["transcript"]):
        _fail(f"corpus row {selection_index} transcript digest changed")
    if value["pair_id"] != make_pair_id(value):
        _fail(f"corpus row {selection_index} pair_id changed")
    if value["sampling_rate"] != SAMPLE_RATE:
        _fail(f"corpus row {selection_index} sampling_rate must be {SAMPLE_RATE}")
    num_samples = _require_int(
        value["num_samples"], f"corpus row {selection_index}.num_samples", minimum=1
    )
    expected_duration = round(num_samples / SAMPLE_RATE, 6)
    duration = _require_finite_number(
        value["duration_seconds"], f"corpus row {selection_index}.duration_seconds"
    )
    if (
        duration != expected_duration
        or not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS
    ):
        _fail(
            f"corpus row {selection_index} duration/sample count is invalid: "
            f"duration={duration}, samples={num_samples}"
        )
    if value["volume_path"] != audio_blob_path(value["audio_sha256"]):
        _fail(f"corpus row {selection_index} volume_path is not content-addressed by audio bytes")
    if value["audio_start"] != AUDIO_START:
        _fail(f"corpus row {selection_index} audio_start must be {AUDIO_START}")
    n_audio_tokens = _require_int(
        value["n_audio_tokens"], f"corpus row {selection_index}.n_audio_tokens", minimum=1
    )
    sliced = _require_int(
        value["sliced_seq_len"], f"corpus row {selection_index}.sliced_seq_len", minimum=1
    )
    processor_length = _require_int(
        value["processor_seq_len"], f"corpus row {selection_index}.processor_seq_len", minimum=1
    )
    valid_positions = _require_int(
        value["n_valid_positions"], f"corpus row {selection_index}.n_valid_positions", minimum=1
    )
    if sliced != AUDIO_START + n_audio_tokens:
        _fail(f"corpus row {selection_index} audio span is not contiguous through sliced_seq_len")
    if sliced > MAX_SEQUENCE_LENGTH:
        _fail(f"corpus row {selection_index} sliced_seq_len exceeds {MAX_SEQUENCE_LENGTH}")
    if processor_length != sliced + CLOSING_TOKEN_COUNT:
        _fail(f"corpus row {selection_index} complete processor framing is invalid")
    if valid_positions != sliced - SKIP_FIRST - 1:
        _fail(f"corpus row {selection_index} stock-valid position count is invalid")

    if require_file and volume_root is None:
        _fail("volume_root is required when corpus row file validation is requested")
    if volume_root is not None and require_file:
        path = _physical_path(
            volume_root, value["volume_path"], f"corpus row {selection_index} volume_path"
        )
        if not path.is_file():
            _fail(f"corpus row {selection_index} staged audio is missing at {path}")
        if _sha256_file(path) != value["audio_sha256"]:
            _fail(f"corpus row {selection_index} staged audio bytes changed at {path}")
        try:
            import soundfile as sf

            info = sf.info(path)
        except Exception as exc:
            raise AudioFitContractError(
                f"corpus row {selection_index} staged audio cannot be inspected at {path}"
            ) from exc
        if info.channels != 1 or info.samplerate != SAMPLE_RATE or int(info.frames) != num_samples:
            _fail(
                f"corpus row {selection_index} staged audio framing differs "
                "from its bounded mono 16 kHz identity"
            )
        try:
            pcm, rate = sf.read(
                path,
                frames=num_samples,
                dtype="int16",
                always_2d=False,
            )
        except Exception as exc:
            raise AudioFitContractError(
                f"corpus row {selection_index} staged audio cannot be decoded at {path}"
            ) from exc
        if rate != SAMPLE_RATE or getattr(pcm, "ndim", None) != 1:
            _fail(f"corpus row {selection_index} staged audio is not native mono 16 kHz")
        if int(pcm.shape[0]) != num_samples:
            _fail(f"corpus row {selection_index} decoded sample count changed")
        if decoded_pcm_sha256(pcm) != value["decoded_pcm_sha256"]:
            _fail(f"corpus row {selection_index} decoded PCM changed")
    return dict(value)


def validate_corpus_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    volume_root: str | pathlib.Path | None = None,
    require_files: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        _fail("corpus rows must be a sequence")
    if len(rows) != CORPUS_SIZE:
        _fail(f"corpus has {len(rows)} rows, expected exactly {CORPUS_SIZE}")
    validated: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    source_ids: set[tuple[str, str, str]] = set()
    speakers: set[int] = set()
    for index, raw in enumerate(rows):
        row = validate_corpus_row(
            raw,
            expected_index=index,
            volume_root=volume_root,
            require_file=require_files,
        )
        if row["pair_id"] in pair_ids:
            _fail(f"duplicate pair_id at corpus row {index}: {row['pair_id']}")
        pair_ids.add(row["pair_id"])
        source_key = (row["config"], row["split"], row["source_id"])
        if source_key in source_ids:
            _fail(f"duplicate source identity at corpus row {index}: {source_key!r}")
        source_ids.add(source_key)
        if row["speaker_id"] in speakers:
            _fail(f"duplicate speaker_id at corpus row {index}: {row['speaker_id']}")
        speakers.add(row["speaker_id"])
        validated.append(row)
    return validated


def validate_restored_row(
    expected: Mapping[str, Any], restored: Mapping[str, Any]
) -> dict[str, Any]:
    """Require complete source, transcript, waveform, processor, and fit identity equality."""

    left = validate_corpus_row(expected)
    right = validate_corpus_row(restored)
    for key in _CORPUS_ROW_FIELDS:
        if left[key] != right[key]:
            _fail(
                f"restored corpus row {left['selection_index']} changed {key}: "
                f"expected {left[key]!r}, got {right[key]!r}"
            )
    return right


def canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        _fail("canonical JSONL rows must be a sequence")
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        mapping = _require_mapping(row, f"canonical JSONL row {index}")
        digest.update(_canonical_json_bytes(dict(mapping)))
        digest.update(b"\n")
    return digest.hexdigest()


def ordered_corpus_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical row identities in fit order; reordering changes the digest."""

    validated = validate_corpus_rows(rows)
    row_digests = [_sha256_bytes(_canonical_json_bytes(row)) for row in validated]
    return _sha256_bytes(_canonical_json_bytes(row_digests))


def validate_attempt_ledger(
    ledger: Sequence[Mapping[str, Any]],
    *,
    corpus_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = validate_corpus_config(corpus_config)
    if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
        _fail("attempt ledger must be a sequence")
    maximum = config["selection"]["max_attempts_per_stratum"]
    stratum_attempts = {name: 0 for name, _ in STRATA}
    selected = {name: 0 for name, _ in STRATA}
    selected_speakers: set[int] = set()
    selected_pairs: set[str] = set()
    attempted_sources: set[tuple[str, str, str]] = set()
    validated: list[dict[str, Any]] = []
    current_stratum_position = -1
    last_speaker_key: dict[str, tuple[str, int] | None] = {name: None for name, _ in STRATA}
    last_utterance_rank: dict[str, str | None] = {name: None for name, _ in STRATA}
    for index, raw in enumerate(ledger):
        attempt = _require_mapping(raw, f"attempt ledger row {index}")
        _require_exact_fields(attempt, _ATTEMPT_ROW_FIELDS, f"attempt ledger row {index}")
        if attempt["schema_version"] != SCHEMA_VERSION or attempt["kind"] != ATTEMPT_ROW_KIND:
            _fail(f"attempt ledger row {index} kind/schema version is invalid")
        _require_int(
            attempt["attempt_index"],
            f"attempt ledger row {index}.attempt_index",
            minimum=0,
        )
        if attempt["attempt_index"] != index:
            _fail(f"attempt ledger row {index} has attempt_index={attempt['attempt_index']!r}")
        coordinates = (attempt["config"], attempt["split"])
        if coordinates not in STRATA or attempt["stratum"] != attempt["config"]:
            _fail(f"attempt ledger row {index} has invalid stratum coordinates")
        name = attempt["config"]
        if selected[name] == STRATUM_SIZE:
            _fail(f"attempt ledger continues {name} after filling its selection quota")
        stratum_position = STRATA.index(coordinates)
        if stratum_position < current_stratum_position:
            _fail(f"attempt ledger row {index} returns to an earlier stratum")
        if stratum_position > current_stratum_position:
            if stratum_position != current_stratum_position + 1:
                _fail(f"attempt ledger row {index} skips the canonical stratum order")
            if (
                current_stratum_position >= 0
                and selected[STRATA[current_stratum_position][0]] != STRATUM_SIZE
            ):
                _fail(f"attempt ledger row {index} advances before filling its prior stratum")
            current_stratum_position = stratum_position
        _require_int(
            attempt["stratum_attempt_index"],
            f"attempt ledger row {index}.stratum_attempt_index",
            minimum=0,
        )
        if attempt["stratum_attempt_index"] != stratum_attempts[name]:
            _fail(
                f"attempt ledger row {index} has stratum_attempt_index="
                f"{attempt['stratum_attempt_index']!r}, expected {stratum_attempts[name]}"
            )
        stratum_attempts[name] += 1
        if stratum_attempts[name] > maximum:
            _fail(f"attempt ledger exceeds {maximum} bounded attempts for {name}")
        speaker = _require_int(
            attempt["speaker_id"], f"attempt ledger row {index}.speaker_id", minimum=0
        )
        source_id = _require_nonempty_string(
            attempt["source_id"], f"attempt ledger row {index}.source_id"
        )
        source_key = (name, attempt["split"], source_id)
        if source_key in attempted_sources:
            _fail(f"attempt ledger repeats source identity {source_key!r}")
        attempted_sources.add(source_key)
        expected_speaker_rank = metadata_rank(
            AUDIO_SELECTION_SEED,
            DATASET_ID,
            LIBRISPEECH_REVISION,
            name,
            attempt["split"],
            speaker,
        )
        expected_utterance_rank = metadata_rank(
            AUDIO_SELECTION_SEED,
            DATASET_ID,
            LIBRISPEECH_REVISION,
            name,
            attempt["split"],
            speaker,
            source_id,
        )
        if attempt["speaker_rank_sha256"] != expected_speaker_rank:
            _fail(f"attempt ledger row {index} speaker rank changed")
        if attempt["utterance_rank_sha256"] != expected_utterance_rank:
            _fail(f"attempt ledger row {index} utterance rank changed")
        speaker_key = (expected_speaker_rank, speaker)
        previous_speaker_key = last_speaker_key[name]
        if previous_speaker_key is None or speaker_key > previous_speaker_key:
            last_speaker_key[name] = speaker_key
            last_utterance_rank[name] = expected_utterance_rank
        elif speaker_key == previous_speaker_key:
            previous_utterance_rank = last_utterance_rank[name]
            if (
                previous_utterance_rank is None
                or expected_utterance_rank <= previous_utterance_rank
            ):
                _fail(f"attempt ledger row {index} is outside canonical utterance-rank order")
            last_utterance_rank[name] = expected_utterance_rank
        else:
            _fail(f"attempt ledger row {index} is outside canonical speaker-rank order")
        outcome = attempt["outcome"]
        if outcome == "selected":
            if attempt["reason"] is not None:
                _fail(f"selected attempt ledger row {index} must have reason=null")
            pair_id = _require_sha256(attempt["pair_id"], f"attempt ledger row {index}.pair_id")
            if speaker in selected_speakers:
                _fail(f"attempt ledger selects duplicate speaker_id {speaker}")
            if pair_id in selected_pairs:
                _fail(f"attempt ledger selects duplicate pair_id {pair_id}")
            selected_speakers.add(speaker)
            selected_pairs.add(pair_id)
            selected[name] += 1
        elif outcome == "rejected":
            if attempt["pair_id"] is not None:
                _fail(f"rejected attempt ledger row {index} must have pair_id=null")
            reason = _require_nonempty_string(
                attempt["reason"], f"attempt ledger row {index}.reason"
            )
            if reason not in REJECTION_REASONS:
                _fail(f"attempt ledger row {index} has unsupported rejection reason {reason!r}")
            if speaker in selected_speakers and reason != "speaker_already_selected":
                _fail(
                    f"attempt ledger row {index} retries a selected speaker without "
                    "speaker_already_selected exclusion"
                )
            if speaker not in selected_speakers and reason == "speaker_already_selected":
                _fail(f"attempt ledger row {index} falsely excludes an unselected speaker")
        else:
            _fail(f"attempt ledger row {index} has invalid outcome {outcome!r}")
        validated.append(dict(attempt))
    expected_counts = {name: STRATUM_SIZE for name, _ in STRATA}
    if selected != expected_counts:
        _fail(f"attempt ledger selected counts are {selected}, expected {expected_counts}")
    return validated


def validate_attempt_ledger_against_source_pool(
    ledger: Sequence[Mapping[str, Any]],
    source_pool: Mapping[str, Any],
    *,
    corpus_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Prove the ledger is the exact ranked traversal of the frozen source pool."""

    config = validate_corpus_config(corpus_config)
    pool = validate_source_pool_record(source_pool, config)
    validated = validate_attempt_ledger(
        ledger,
        corpus_config=config,
    )
    selected_speakers: set[int] = set()
    pool_ranks = _source_pool_ranks_validated(pool["rows"])
    for config_name, split in STRATA:
        attempts = [
            attempt
            for attempt in validated
            if (attempt["config"], attempt["split"]) == (config_name, split)
        ]
        ranks = [
            rank for rank in pool_ranks if (rank["config"], rank["split"]) == (config_name, split)
        ]
        by_speaker: dict[int, list[Mapping[str, Any]]] = {}
        speaker_ranks: dict[int, str] = {}
        for rank in ranks:
            speaker = rank["speaker_id"]
            by_speaker.setdefault(speaker, []).append(rank)
            speaker_ranks[speaker] = rank["speaker_rank_sha256"]
        ranked_speakers = sorted(
            by_speaker,
            key=lambda speaker: (speaker_ranks[speaker], speaker),
        )
        attempt_index = 0
        selected_count = 0
        for speaker in ranked_speakers:
            if selected_count == STRATUM_SIZE:
                break
            utterances = sorted(
                by_speaker[speaker],
                key=lambda rank: (
                    rank["utterance_rank_sha256"],
                    rank["source_id"],
                ),
            )
            for rank in utterances:
                if attempt_index >= len(attempts):
                    _fail(
                        f"attempt ledger omits ranked {config_name}/{split} "
                        f"candidate {rank['source_id']}"
                    )
                attempt = attempts[attempt_index]
                expected_identity = {
                    key: rank[key]
                    for key in (
                        "config",
                        "split",
                        "speaker_id",
                        "source_id",
                        "speaker_rank_sha256",
                        "utterance_rank_sha256",
                    )
                }
                observed_identity = {key: attempt[key] for key in expected_identity}
                if observed_identity != expected_identity:
                    _fail(
                        f"attempt ledger skips or changes ranked candidate "
                        f"{config_name}/{split}/{rank['source_id']}"
                    )
                attempt_index += 1
                if speaker in selected_speakers:
                    if (
                        attempt["outcome"] != "rejected"
                        or attempt["reason"] != "speaker_already_selected"
                    ):
                        _fail(
                            f"attempt ledger does not deterministically exclude "
                            f"already-selected speaker {speaker}"
                        )
                    break
                if attempt["outcome"] == "selected":
                    selected_speakers.add(speaker)
                    selected_count += 1
                    break
            else:
                continue
        if selected_count != STRATUM_SIZE:
            _fail(
                f"source-pool traversal selected {selected_count}/{STRATUM_SIZE} "
                f"for {config_name}/{split}"
            )
        if attempt_index != len(attempts):
            _fail(
                f"attempt ledger has {len(attempts) - attempt_index} unconsumed "
                f"{config_name}/{split} entries"
            )
    return validated


def validate_corpus_envelope(
    envelope: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    *,
    volume_root: str | pathlib.Path | None = None,
    require_files: bool = False,
) -> dict[str, Any]:
    value = _require_mapping(envelope, "corpus envelope")
    _require_exact_fields(value, _CORPUS_ENVELOPE_FIELDS, "corpus envelope")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != CORPUS_ENVELOPE_KIND:
        _fail("corpus envelope kind/schema version is invalid")
    config = validate_corpus_config(_require_mapping(value["config"], "corpus envelope config"))
    config_sha = corpus_config_digest(config)
    if value["corpus_config_sha256"] != config_sha:
        _fail("corpus envelope config digest changed")
    source_pool_sha = _require_sha256(
        value["source_pool_sha256"],
        "corpus envelope source_pool_sha256",
    )
    expected_pool_path = source_pool_path(config, source_pool_sha)
    if value["source_pool_path"] != expected_pool_path:
        _fail("corpus envelope source_pool_path is not content-addressed")
    _relative_path(value["source_pool_path"], "corpus envelope source_pool_path")
    _require_sha256(
        value["source_pool_file_sha256"],
        "corpus envelope source_pool_file_sha256",
    )
    validated_rows = validate_corpus_rows(
        rows, volume_root=volume_root, require_files=require_files
    )
    validated_ledger = validate_attempt_ledger(
        ledger,
        corpus_config=config,
    )
    expected_counts = {name: STRATUM_SIZE for name, _ in STRATA}
    _require_int(value["row_count"], "corpus envelope row_count", minimum=1)
    if value["row_count"] != CORPUS_SIZE or value["stratum_counts"] != expected_counts:
        _fail("corpus envelope row/stratum counts are invalid")
    ordered_sha = ordered_corpus_digest(validated_rows)
    if value["ordered_corpus_sha256"] != ordered_sha:
        _fail("corpus envelope ordered row digest changed")
    paths = corpus_paths(config, ordered_sha)
    if value["rows_path"] != paths["rows"]:
        _fail("corpus envelope rows_path is not content-addressed")
    if value["attempt_ledger_path"] != paths["attempt_ledger"]:
        _fail("corpus envelope attempt_ledger_path is not content-addressed")
    if value["rows_sha256"] != canonical_jsonl_sha256(validated_rows):
        _fail("corpus envelope canonical rows JSONL digest changed")
    if value["attempt_ledger_sha256"] != canonical_jsonl_sha256(validated_ledger):
        _fail("corpus envelope canonical attempt JSONL digest changed")
    _require_int(
        value["attempt_count"],
        "corpus envelope attempt_count",
        minimum=CORPUS_SIZE,
    )
    if value["attempt_count"] != len(validated_ledger):
        _fail("corpus envelope attempt_count changed")
    if value["max_attempts_per_stratum"] != config["selection"]["max_attempts_per_stratum"]:
        _fail("corpus envelope attempt bound changed")
    if value["audio_root"] != "audio-blobs":
        _fail("corpus envelope audio_root must be 'audio-blobs'")

    selected_by_stratum: dict[str, list[tuple[int, str, str]]] = {name: [] for name, _ in STRATA}
    for attempt in validated_ledger:
        if attempt["outcome"] == "selected":
            selected_by_stratum[attempt["config"]].append(
                (
                    attempt["speaker_id"],
                    attempt["source_id"],
                    attempt["pair_id"],
                )
            )
    rows_by_stratum: dict[str, list[tuple[int, str, str]]] = {name: [] for name, _ in STRATA}
    for row in validated_rows:
        rows_by_stratum[row["config"]].append((row["speaker_id"], row["source_id"], row["pair_id"]))
    if selected_by_stratum != rows_by_stratum:
        _fail("attempt ledger selected identities do not match alternating corpus rows")
    return dict(value)


def corpus_artifact(
    envelope: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_corpus_envelope(envelope, rows, ledger)
    config = validated["config"]
    paths = corpus_paths(config, validated["ordered_corpus_sha256"])
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": CORPUS_ARTIFACT_KIND,
        "config": config,
        "source_sha256": config["source_sha256"],
        "lock_sha256": config["lock_sha256"],
        "corpus_config_sha256": validated["corpus_config_sha256"],
        "source_pool_sha256": validated["source_pool_sha256"],
        "source_pool_path": validated["source_pool_path"],
        "source_pool_file_sha256": validated["source_pool_file_sha256"],
        "row_count": validated["row_count"],
        "stratum_counts": validated["stratum_counts"],
        "ordered_corpus_sha256": validated["ordered_corpus_sha256"],
        "envelope_path": paths["envelope"],
        "envelope_sha256": _sha256_bytes(_canonical_json_bytes(validated) + b"\n"),
        "rows_path": validated["rows_path"],
        "rows_sha256": validated["rows_sha256"],
        "attempt_ledger_path": validated["attempt_ledger_path"],
        "attempt_ledger_sha256": validated["attempt_ledger_sha256"],
        "attempt_count": validated["attempt_count"],
        "audio_root": validated["audio_root"],
    }
    return _validate_corpus_artifact(artifact)


def _validate_corpus_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(artifact, "corpus artifact")
    _require_exact_fields(value, _CORPUS_ARTIFACT_FIELDS, "corpus artifact")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != CORPUS_ARTIFACT_KIND:
        _fail("corpus artifact kind/schema version is invalid")
    config = validate_corpus_config(_require_mapping(value["config"], "corpus artifact config"))
    for key in (
        "source_sha256",
        "lock_sha256",
        "corpus_config_sha256",
        "source_pool_sha256",
        "source_pool_file_sha256",
        "ordered_corpus_sha256",
        "envelope_sha256",
        "rows_sha256",
        "attempt_ledger_sha256",
    ):
        _require_sha256(value[key], f"corpus artifact {key}")
    if value["corpus_config_sha256"] != corpus_config_digest(config):
        _fail("corpus artifact config digest changed")
    if (
        value["source_sha256"] != config["source_sha256"]
        or value["lock_sha256"] != config["lock_sha256"]
    ):
        _fail("corpus artifact source/lock identity differs from its config")
    _require_int(value["row_count"], "corpus artifact row_count", minimum=1)
    if value["row_count"] != CORPUS_SIZE:
        _fail("corpus artifact row_count is invalid")
    if value["stratum_counts"] != {name: STRATUM_SIZE for name, _ in STRATA}:
        _fail("corpus artifact stratum_counts are invalid")
    _require_int(value["attempt_count"], "corpus artifact attempt_count", minimum=CORPUS_SIZE)
    expected_pool_path = source_pool_path(config, value["source_pool_sha256"])
    if value["source_pool_path"] != expected_pool_path:
        _fail("corpus artifact source_pool_path is not content-addressed")
    _relative_path(value["source_pool_path"], "corpus artifact source_pool_path")
    config_sha = value["corpus_config_sha256"]
    ordered_sha = value["ordered_corpus_sha256"]
    root = f"audio-corpora/{config_sha}/{ordered_sha}"
    expected_paths = {
        "envelope_path": f"{root}/envelope.json",
        "rows_path": f"{root}/rows.jsonl",
        "attempt_ledger_path": f"{root}/attempt-ledger.jsonl",
    }
    for key, expected in expected_paths.items():
        if value[key] != expected:
            _fail(f"corpus artifact {key} is not content-addressed")
        _relative_path(value[key], f"corpus artifact {key}")
    if value["audio_root"] != "audio-blobs":
        _fail("corpus artifact audio_root is invalid")
    return dict(value)


def build_fit_config(
    corpus: Mapping[str, Any],
    profile: Any,
    *,
    source_digest: str,
    lock_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable waveform-only stock-JLens fit configuration."""

    corpus_record = _validate_corpus_artifact(corpus)
    source_digest = _require_sha256(source_digest, "fit source digest")
    lock_sha256 = _require_sha256(lock_sha256, "fit lock digest")
    if (
        corpus_record["source_sha256"] != source_digest
        or corpus_record["lock_sha256"] != lock_sha256
    ):
        _fail("fit source/lock identity does not match the sealed corpus")
    processor = corpus_record["config"]["processor"]
    expected_processor = {
        "profile_key": profile.key,
        "profile_version": profile.version,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "adapter_source": profile.adapter_source,
        "max_sequence_length": profile.max_sequence_length,
        "skip_first": profile.skip_first,
        "require_contiguous_audio_span": True,
        "require_nonempty_stock_positions": True,
    }
    if not _same_json(processor, expected_processor):
        _fail("fit model profile differs from the processor-bound corpus")
    runtime_record = _validate_runtime_config(runtime)
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": FIT_CONFIG_KIND,
        "source_sha256": source_digest,
        "lock_sha256": lock_sha256,
        "corpus": corpus_record,
        "model": {
            "profile_key": profile.key,
            "profile_version": profile.version,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "adapter_source": profile.adapter_source,
            "max_sequence_length": profile.max_sequence_length,
        },
        "runtime": runtime_record,
        "fit": {
            "estimator": "anthropic_jlens_fit_stock",
            "jlens_revision": JLENS_REVISION,
            "input": "waveform_only",
            "transcript_fit": False,
            "expected_count": CORPUS_SIZE,
            "prefix_count": PREFIX_COUNT,
            "source_layers": list(profile.source_layers),
            "target_layer": profile.target_layer,
            "skip_first": profile.skip_first,
            "d_model": profile.d_model,
            "dimension_batch_size": profile.dimension_batch_size,
            "max_sequence_length": profile.max_sequence_length,
            "checkpoint_every": 5,
            "resume": True,
            "compile": False,
            "checkpoint_dtype": "float32",
            "lens_dtype": "float16",
        },
    }
    return validate_fit_config(config)


def _validate_runtime_config(runtime: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(runtime, "fit runtime")
    fields = frozenset(
        {
            "python_version",
            "torch_version",
            "transformers_version",
            "datasets_version",
            "jlens_revision",
            "cuda_version",
        }
    )
    _require_exact_fields(value, fields, "fit runtime")
    for key in (
        "python_version",
        "torch_version",
        "transformers_version",
        "datasets_version",
    ):
        _require_nonempty_string(value[key], f"fit runtime.{key}")
    if value["jlens_revision"] != JLENS_REVISION:
        _fail("fit runtime JLens revision changed")
    if value["cuda_version"] is not None:
        _require_nonempty_string(value["cuda_version"], "fit runtime.cuda_version")
    return dict(value)


def validate_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(config, "fit config")
    fields = frozenset(
        {
            "schema_version",
            "kind",
            "source_sha256",
            "lock_sha256",
            "corpus",
            "model",
            "runtime",
            "fit",
        }
    )
    _require_exact_fields(value, fields, "fit config")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != FIT_CONFIG_KIND:
        _fail("fit config kind/schema version is invalid")
    source_sha = _require_sha256(value["source_sha256"], "fit config source_sha256")
    lock_sha = _require_sha256(value["lock_sha256"], "fit config lock_sha256")
    corpus = _validate_corpus_artifact(_require_mapping(value["corpus"], "fit config corpus"))
    if corpus["source_sha256"] != source_sha or corpus["lock_sha256"] != lock_sha:
        _fail("fit config source/lock identity does not match corpus artifact")
    model = _require_mapping(value["model"], "fit config model")
    model_fields = frozenset(
        {
            "profile_key",
            "profile_version",
            "model_id",
            "model_revision",
            "adapter_source",
            "max_sequence_length",
        }
    )
    _require_exact_fields(model, model_fields, "fit config model")
    for key in ("profile_key", "model_id", "model_revision", "adapter_source"):
        _require_nonempty_string(model[key], f"fit config model.{key}")
    _require_int(model["profile_version"], "fit config model.profile_version", minimum=1)
    if model["max_sequence_length"] != MAX_SEQUENCE_LENGTH:
        _fail("fit config model max_sequence_length changed")
    from .models import DEFAULT_MODEL_PROFILE

    profile = DEFAULT_MODEL_PROFILE
    expected_model = {
        "profile_key": profile.key,
        "profile_version": profile.version,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "adapter_source": profile.adapter_source,
        "max_sequence_length": profile.max_sequence_length,
    }
    if not _same_json(model, expected_model):
        _fail("fit config model identity differs from the fixed model profile")
    _validate_runtime_config(_require_mapping(value["runtime"], "fit config runtime"))

    fit = _require_mapping(value["fit"], "fit config fit")
    fit_fields = frozenset(
        {
            "estimator",
            "jlens_revision",
            "input",
            "transcript_fit",
            "expected_count",
            "prefix_count",
            "source_layers",
            "target_layer",
            "skip_first",
            "d_model",
            "dimension_batch_size",
            "max_sequence_length",
            "checkpoint_every",
            "resume",
            "compile",
            "checkpoint_dtype",
            "lens_dtype",
        }
    )
    _require_exact_fields(fit, fit_fields, "fit config fit")
    fixed = {
        "estimator": "anthropic_jlens_fit_stock",
        "jlens_revision": JLENS_REVISION,
        "input": "waveform_only",
        "transcript_fit": False,
        "expected_count": CORPUS_SIZE,
        "prefix_count": PREFIX_COUNT,
        "skip_first": SKIP_FIRST,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "checkpoint_every": 5,
        "resume": True,
        "compile": False,
        "checkpoint_dtype": "float32",
        "lens_dtype": "float16",
        "source_layers": list(profile.source_layers),
        "target_layer": profile.target_layer,
        "d_model": profile.d_model,
        "dimension_batch_size": profile.dimension_batch_size,
    }
    for key, expected in fixed.items():
        if not _same_json(fit[key], expected):
            _fail(f"fit config {key} must be {expected!r}")
    if not isinstance(fit["source_layers"], list) or not fit["source_layers"]:
        _fail("fit config source_layers must be a nonempty list")
    if any(not _is_int(layer) or layer < 0 for layer in fit["source_layers"]):
        _fail("fit config source_layers must contain nonnegative integers")
    if len(set(fit["source_layers"])) != len(fit["source_layers"]):
        _fail("fit config source_layers contains duplicates")
    _require_int(fit["target_layer"], "fit config target_layer", minimum=0)
    _require_int(fit["d_model"], "fit config d_model", minimum=1)
    _require_int(fit["dimension_batch_size"], "fit config dimension_batch_size", minimum=1)
    _validate_json_value(value, "fit config")
    return dict(value)


def fit_config_digest(config: Mapping[str, Any]) -> str:
    return _config_digest(validate_fit_config(config))


def run_paths(config: Mapping[str, Any]) -> dict[str, str]:
    digest = fit_config_digest(config)
    root = f"audio-fit-runs/{digest}"
    return {
        "manifest": f"{root}/run.json",
        "checkpoint": f"{root}/working-checkpoint.pt",
        "snapshot_dir": f"{root}/prefix-500",
        "stability_dir": f"{root}/stability",
        "lens_dir": f"{root}/lens",
        "gate_dir": f"{root}/gates",
    }


def gate_record(config: Mapping[str, Any], gate: str) -> dict[str, Any]:
    """Build the deterministic proof record for one required execution gate."""

    validated_config = validate_fit_config(config)
    if gate not in REQUIRED_GATES:
        _fail(f"unsupported audio-fit gate {gate!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_RECORD_KIND,
        "gate": gate,
        "status": "passed",
        "fit_config_sha256": fit_config_digest(validated_config),
        "ordered_corpus_sha256": validated_config["corpus"]["ordered_corpus_sha256"],
    }


def gate_path(config: Mapping[str, Any], gate: str) -> str:
    record = gate_record(config, gate)
    return f"{run_paths(config)['gate_dir']}/{record['gate']}.json"


def gate_artifact(
    path: str | pathlib.Path,
    config: Mapping[str, Any],
    gate: str,
) -> dict[str, Any]:
    expected_record = gate_record(config, gate)
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        _fail(f"audio-fit gate {gate!r} is missing at {file_path}")
    observed_record = _read_json(file_path, f"audio-fit gate {gate}")
    if not _same_json(observed_record, expected_record):
        _fail(f"audio-fit gate {gate!r} content or identity changed")
    digest = _sha256_file(file_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_ARTIFACT_KIND,
        "gate": gate,
        "fit_config_sha256": expected_record["fit_config_sha256"],
        "ordered_corpus_sha256": expected_record["ordered_corpus_sha256"],
        "relative_path": gate_path(config, gate),
        "sha256": digest,
        "bytes": file_path.stat().st_size,
    }


def validate_gate_artifact(
    artifact: Mapping[str, Any],
    path: str | pathlib.Path,
    config: Mapping[str, Any],
    gate: str,
) -> dict[str, Any]:
    value = _require_mapping(artifact, f"audio-fit gate artifact {gate}")
    _require_exact_fields(
        value,
        _GATE_ARTIFACT_FIELDS,
        f"audio-fit gate artifact {gate}",
    )
    recomputed = gate_artifact(path, config, gate)
    if not _same_json(dict(value), recomputed):
        _fail(f"audio-fit gate artifact {gate!r} changed")
    expected_parts = pathlib.PurePosixPath(recomputed["relative_path"]).parts
    if tuple(pathlib.Path(path).parts[-len(expected_parts) :]) != expected_parts:
        _fail(f"audio-fit gate {gate!r} is not stored at {recomputed['relative_path']}")
    return recomputed


def checkpoint_identity(config: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, Any]:
    validated_config = validate_fit_config(config)
    expected_paths = run_paths(validated_config)
    if not _same_json(dict(paths), expected_paths):
        _fail("checkpoint paths do not match the content-addressed fit config")
    fit = validated_config["fit"]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_IDENTITY_KIND,
        "role": "working_checkpoint",
        "fit_config_sha256": fit_config_digest(validated_config),
        "ordered_corpus_sha256": validated_config["corpus"]["ordered_corpus_sha256"],
        "path": expected_paths["checkpoint"],
        "expected_count": fit["expected_count"],
        "source_layers": fit["source_layers"],
        "target_layer": fit["target_layer"],
        "skip_first": fit["skip_first"],
        "d_model": fit["d_model"],
        "dtype": "float32",
        "snapshot_dir": expected_paths["snapshot_dir"],
        "stability_dir": expected_paths["stability_dir"],
        "lens_dir": expected_paths["lens_dir"],
    }
    return validate_checkpoint_identity(identity)


def validate_checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(identity, "checkpoint identity")
    _require_exact_fields(value, _CHECKPOINT_IDENTITY_FIELDS, "checkpoint identity")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != CHECKPOINT_IDENTITY_KIND:
        _fail("checkpoint identity kind/schema version is invalid")
    if value["role"] != "working_checkpoint" or value["dtype"] != "float32":
        _fail("checkpoint identity role/dtype is invalid")
    _require_sha256(value["fit_config_sha256"], "checkpoint identity fit_config_sha256")
    _require_sha256(value["ordered_corpus_sha256"], "checkpoint identity ordered_corpus_sha256")
    _require_int(
        value["expected_count"],
        "checkpoint identity expected_count",
        minimum=1,
    )
    if value["expected_count"] != CORPUS_SIZE:
        _fail(f"checkpoint identity expected_count must be {CORPUS_SIZE}")
    _require_int(
        value["skip_first"],
        "checkpoint identity skip_first",
        minimum=0,
    )
    if value["skip_first"] != SKIP_FIRST:
        _fail(f"checkpoint identity skip_first must be {SKIP_FIRST}")
    _require_int(value["target_layer"], "checkpoint identity target_layer", minimum=0)
    _require_int(value["d_model"], "checkpoint identity d_model", minimum=1)
    if not isinstance(value["source_layers"], list) or not value["source_layers"]:
        _fail("checkpoint identity source_layers must be a nonempty list")
    if any(not _is_int(layer) or layer < 0 for layer in value["source_layers"]):
        _fail("checkpoint identity source_layers are invalid")
    root = f"audio-fit-runs/{value['fit_config_sha256']}"
    expected_paths = {
        "path": f"{root}/working-checkpoint.pt",
        "snapshot_dir": f"{root}/prefix-500",
        "stability_dir": f"{root}/stability",
        "lens_dir": f"{root}/lens",
    }
    for key, expected in expected_paths.items():
        if value[key] != expected:
            _fail(f"checkpoint identity {key} is not content-addressed")
        _relative_path(value[key], f"checkpoint identity {key}")
    return dict(value)


def _identity_digest(identity: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(validate_checkpoint_identity(identity)))


def validate_checkpoint_state(
    path: str | pathlib.Path,
    identity: Mapping[str, Any],
    *,
    maximum_count: int,
    exact_count: int | None = None,
) -> dict[str, Any]:
    """Validate stock mutable checkpoint state against immutable external identity."""

    import torch

    expected = validate_checkpoint_identity(identity)
    maximum = _require_int(maximum_count, "checkpoint maximum_count", minimum=0)
    if maximum > expected["expected_count"]:
        _fail("checkpoint maximum_count exceeds immutable expected_count")
    if exact_count is not None:
        exact = _require_int(exact_count, "checkpoint exact_count", minimum=0)
        if exact > maximum:
            _fail("checkpoint exact_count exceeds maximum_count")
    checkpoint_path = pathlib.Path(path)
    maximum_bytes = (
        len(expected["source_layers"]) * expected["d_model"] * expected["d_model"] * 4
        + 32 * 1024 * 1024
    )
    try:
        checkpoint_bytes = checkpoint_path.stat().st_size
    except OSError as exc:
        raise AudioFitContractError(f"cannot inspect fit checkpoint at {checkpoint_path}") from exc
    if checkpoint_bytes > maximum_bytes:
        _fail(
            f"fit checkpoint at {checkpoint_path} is {checkpoint_bytes} bytes, "
            f"above the bounded maximum {maximum_bytes}"
        )
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AudioFitContractError(f"invalid fit checkpoint at {path}") from exc
    if not isinstance(loaded, Mapping):
        _fail(f"fit checkpoint root at {path} must be an object")
    state = dict(loaded)
    for key in ("source_layers", "target_layer", "skip_first"):
        if state.get(key) != expected[key]:
            _fail(f"checkpoint {key} does not match immutable identity")
    n_done = state.get("n_done")
    next_idx = state.get("next_idx")
    if (
        not _is_int(n_done)
        or not _is_int(next_idx)
        or n_done != next_idx
        or n_done < 0
        or n_done > maximum
        or (exact_count is not None and n_done != exact_count)
    ):
        _fail(
            f"checkpoint counts {(n_done, next_idx)} violate exact-success bounds "
            f"maximum={maximum}, exact={exact_count}"
        )
    stamped_fit = state.get("fit_config_sha256")
    if stamped_fit != expected["fit_config_sha256"]:
        _fail("checkpoint is not bound to the immutable fit-config digest")
    stamped_corpus = state.get("ordered_corpus_sha256")
    if stamped_corpus != expected["ordered_corpus_sha256"]:
        _fail("checkpoint is not bound to the immutable ordered corpus")
    sums = state.get("jacobian_sum")
    expected_layers = set(expected["source_layers"])
    if not isinstance(sums, dict) or set(sums) != expected_layers:
        _fail("checkpoint Jacobian layers do not match immutable identity")
    shape = (expected["d_model"], expected["d_model"])
    for layer, tensor in sums.items():
        if (
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not bool(torch.isfinite(tensor).all())
        ):
            _fail(f"checkpoint layer {layer} is not a finite fp32 {shape} running sum")
    return state


def checkpoint_artifact(
    path: str | pathlib.Path,
    identity: Mapping[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    expected = validate_checkpoint_identity(identity)
    if expected_count not in {PREFIX_COUNT, CORPUS_SIZE}:
        _fail(f"checkpoint artifact count must be {PREFIX_COUNT} or {CORPUS_SIZE}")
    state = validate_checkpoint_state(
        path,
        expected,
        maximum_count=expected_count,
        exact_count=expected_count,
    )
    file_sha = _sha256_file(path)
    if expected_count == PREFIX_COUNT:
        kind = PREFIX_SNAPSHOT_KIND
        role = "prefix_500"
        relative = f"{expected['snapshot_dir']}/{file_sha}.pt"
    else:
        kind = FINAL_CHECKPOINT_KIND
        role = "final_checkpoint"
        relative = expected["path"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "role": role,
        "fit_config_sha256": expected["fit_config_sha256"],
        "ordered_corpus_sha256": expected["ordered_corpus_sha256"],
        "checkpoint_identity_sha256": _identity_digest(expected),
        "relative_path": relative,
        "sha256": file_sha,
        "bytes": pathlib.Path(path).stat().st_size,
        "n_done": state["n_done"],
        "next_idx": state["next_idx"],
        "dtype": "float32",
        "d_model": expected["d_model"],
        "source_layers": expected["source_layers"],
        "target_layer": expected["target_layer"],
        "skip_first": expected["skip_first"],
    }


def validate_checkpoint_artifact(
    artifact: Mapping[str, Any],
    path: str | pathlib.Path,
    identity: Mapping[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    value = _require_mapping(artifact, "checkpoint artifact")
    _require_exact_fields(value, _CHECKPOINT_ARTIFACT_FIELDS, "checkpoint artifact")
    recomputed = checkpoint_artifact(path, identity, expected_count)
    if not _same_json(dict(value), recomputed):
        _fail(f"checkpoint artifact metadata changed for count {expected_count}")
    portable_path = recomputed["relative_path"]
    path_parts = pathlib.Path(path).parts
    expected_parts = pathlib.PurePosixPath(portable_path).parts
    if (
        len(path_parts) < len(expected_parts)
        or tuple(path_parts[-len(expected_parts) :]) != expected_parts
    ):
        _fail(f"checkpoint artifact is not stored at {portable_path}")
    return recomputed


def reconstruct_layer_means(
    prefix_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    layer: int,
) -> tuple[Any, Any, Any]:
    """Return bounded fp32 first-half, full, and disjoint second-half means."""

    import torch

    if prefix_state.get("n_done") != PREFIX_COUNT or final_state.get("n_done") != CORPUS_SIZE:
        _fail("mean reconstruction requires exact 500/1,000 checkpoint states")
    try:
        prefix_sum = prefix_state["jacobian_sum"][layer]
        final_sum = final_state["jacobian_sum"][layer]
    except (KeyError, TypeError) as exc:
        raise AudioFitContractError(f"mean reconstruction is missing layer {layer}") from exc
    if prefix_sum.dtype != torch.float32 or final_sum.dtype != torch.float32:
        _fail(f"mean reconstruction layer {layer} is not fp32")
    first = prefix_sum.detach().to(device="cpu", dtype=torch.float32).clone().div_(PREFIX_COUNT)
    full = final_sum.detach().to(device="cpu", dtype=torch.float32).clone().div_(CORPUS_SIZE)
    second = (
        final_sum.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
        .sub_(prefix_sum.detach().to(device="cpu", dtype=torch.float32))
        .div_(CORPUS_SIZE - PREFIX_COUNT)
    )
    if not bool(
        torch.isfinite(first).all() and torch.isfinite(full).all() and torch.isfinite(second).all()
    ):
        _fail(f"reconstructed layer {layer} means are nonfinite")
    return first, full, second


def _fp64_norm(value: Any, *, chunk_size: int = 262_144) -> float:
    """Reduce a finite fp32 tensor in bounded fp64 chunks."""

    import torch

    flat = value.reshape(-1)
    squares: list[float] = []
    for chunk in flat.split(chunk_size):
        promoted = chunk.to(dtype=torch.float64)
        squares.append(float(torch.dot(promoted, promoted)))
    return math.sqrt(math.fsum(squares))


def _fp64_dot(left: Any, right: Any, *, chunk_size: int = 262_144) -> float:
    """Reduce a finite fp32 dot product in bounded fp64 chunks."""

    import torch

    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    products: list[float] = []
    for left_chunk, right_chunk in zip(
        left_flat.split(chunk_size),
        right_flat.split(chunk_size),
        strict=True,
    ):
        promoted_left = left_chunk.to(dtype=torch.float64)
        promoted_right = right_chunk.to(dtype=torch.float64)
        products.append(float(torch.dot(promoted_left, promoted_right)))
    return math.fsum(products)


def _fp64_difference_norm(
    left: Any,
    right: Any,
    *,
    chunk_size: int = 262_144,
) -> float:
    """Reduce ``||left - right||`` without a persistent fp64 matrix."""

    import torch

    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    squares: list[float] = []
    for left_chunk, right_chunk in zip(
        left_flat.split(chunk_size),
        right_flat.split(chunk_size),
        strict=True,
    ):
        difference = left_chunk.to(dtype=torch.float64)
        difference.sub_(right_chunk.to(dtype=torch.float64))
        squares.append(float(torch.dot(difference, difference)))
    return math.sqrt(math.fsum(squares))


def stability_from_checkpoints(
    prefix_path: str | pathlib.Path,
    final_path: str | pathlib.Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute only the preregistered identity-centered cosine and relative L2."""

    expected = validate_checkpoint_identity(identity)
    prefix_state = validate_checkpoint_state(
        prefix_path,
        expected,
        maximum_count=PREFIX_COUNT,
        exact_count=PREFIX_COUNT,
    )
    final_state = validate_checkpoint_state(
        final_path,
        expected,
        maximum_count=CORPUS_SIZE,
        exact_count=CORPUS_SIZE,
    )
    layers: dict[str, dict[str, float]] = {}
    for layer in expected["source_layers"]:
        first, full, second = reconstruct_layer_means(prefix_state, final_state, layer)
        relative_denominator = _fp64_norm(first)
        if relative_denominator == 0.0 or not math.isfinite(relative_denominator):
            _fail(f"layer {layer} first-half mean has invalid relative-L2 denominator")
        relative_l2 = _fp64_difference_norm(full, first) / relative_denominator
        first.diagonal().sub_(1.0)
        second.diagonal().sub_(1.0)
        first_centered_norm = _fp64_norm(first)
        second_centered_norm = _fp64_norm(second)
        cosine_denominator = first_centered_norm * second_centered_norm
        if cosine_denominator == 0.0 or not math.isfinite(cosine_denominator):
            _fail(f"layer {layer} identity-centered cosine denominator is invalid")
        cosine = _fp64_dot(first, second) / cosine_denominator
        if not math.isfinite(cosine) or not math.isfinite(relative_l2):
            _fail(f"layer {layer} stability diagnostics are nonfinite")
        layers[str(layer)] = {
            "identity_centered_split_half_cosine": cosine,
            "first_half_to_full_relative_l2": relative_l2,
        }
        del first, full, second
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": STABILITY_REPORT_KIND,
        "fit_config_sha256": expected["fit_config_sha256"],
        "ordered_corpus_sha256": expected["ordered_corpus_sha256"],
        "prefix_checkpoint_sha256": _sha256_file(prefix_path),
        "final_checkpoint_sha256": _sha256_file(final_path),
        "first_half_count": PREFIX_COUNT,
        "second_half_count": CORPUS_SIZE - PREFIX_COUNT,
        "full_count": CORPUS_SIZE,
        "cosine_centering": "subtract_identity",
        "relative_l2_reference": "first_half_to_full",
        "layers": layers,
    }
    return validate_stability_report(report, expected)


def validate_stability_report(
    report: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    value = _require_mapping(report, "stability report")
    expected = validate_checkpoint_identity(identity)
    _require_exact_fields(value, _STABILITY_REPORT_FIELDS, "stability report")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != STABILITY_REPORT_KIND:
        _fail("stability report kind/schema version is invalid")
    if value["fit_config_sha256"] != expected["fit_config_sha256"]:
        _fail("stability report fit-config digest changed")
    if value["ordered_corpus_sha256"] != expected["ordered_corpus_sha256"]:
        _fail("stability report corpus digest changed")
    _require_sha256(value["prefix_checkpoint_sha256"], "stability prefix checkpoint digest")
    _require_sha256(value["final_checkpoint_sha256"], "stability final checkpoint digest")
    _require_int(
        value["first_half_count"],
        "stability report first_half_count",
        minimum=1,
    )
    _require_int(
        value["second_half_count"],
        "stability report second_half_count",
        minimum=1,
    )
    _require_int(
        value["full_count"],
        "stability report full_count",
        minimum=1,
    )
    if (
        value["first_half_count"] != PREFIX_COUNT
        or value["second_half_count"] != CORPUS_SIZE - PREFIX_COUNT
        or value["full_count"] != CORPUS_SIZE
    ):
        _fail("stability report half/full counts are invalid")
    if value["cosine_centering"] != "subtract_identity":
        _fail("stability report cosine centering changed")
    if value["relative_l2_reference"] != "first_half_to_full":
        _fail("stability report relative-L2 definition changed")
    layers = _require_mapping(value["layers"], "stability report layers")
    expected_layer_keys = {str(layer) for layer in expected["source_layers"]}
    if set(layers) != expected_layer_keys:
        _fail("stability report layers do not match checkpoint identity")
    for layer, raw in layers.items():
        layer_record = _require_mapping(raw, f"stability layer {layer}")
        _require_exact_fields(layer_record, _STABILITY_LAYER_FIELDS, f"stability layer {layer}")
        cosine = _require_finite_number(
            layer_record["identity_centered_split_half_cosine"],
            f"stability layer {layer} identity-centered cosine",
        )
        if cosine < -1.000001 or cosine > 1.000001:
            _fail(f"stability layer {layer} cosine is outside [-1, 1]")
        relative = _require_finite_number(
            layer_record["first_half_to_full_relative_l2"],
            f"stability layer {layer} relative L2",
        )
        if relative < 0:
            _fail(f"stability layer {layer} relative L2 is negative")
    return dict(value)


def stability_artifact(report: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    expected = validate_checkpoint_identity(identity)
    validated = validate_stability_report(report, expected)
    content = _canonical_json_bytes(validated) + b"\n"
    digest = _sha256_bytes(content)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": STABILITY_ARTIFACT_KIND,
        "role": "stability",
        "fit_config_sha256": expected["fit_config_sha256"],
        "ordered_corpus_sha256": expected["ordered_corpus_sha256"],
        "relative_path": f"{expected['stability_dir']}/{digest}.json",
        "sha256": digest,
        "bytes": len(content),
    }


def validate_stability_artifact(
    artifact: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    value = _require_mapping(artifact, "stability artifact")
    _require_exact_fields(value, _STABILITY_ARTIFACT_FIELDS, "stability artifact")
    expected_identity = validate_checkpoint_identity(identity)
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != STABILITY_ARTIFACT_KIND
        or value["role"] != "stability"
    ):
        _fail("stability artifact kind/role/schema version is invalid")
    if (
        value["fit_config_sha256"] != expected_identity["fit_config_sha256"]
        or value["ordered_corpus_sha256"] != expected_identity["ordered_corpus_sha256"]
    ):
        _fail("stability artifact immutable identity changed")
    digest = _require_sha256(value["sha256"], "stability artifact sha256")
    if value["relative_path"] != f"{expected_identity['stability_dir']}/{digest}.json":
        _fail("stability artifact path is not content-addressed")
    _require_int(value["bytes"], "stability artifact bytes", minimum=1)
    if path is not None:
        file_path = pathlib.Path(path)
        if not file_path.is_file():
            _fail(f"stability artifact is missing at {file_path}")
        report = _read_json(file_path, "stability report")
        expected_artifact = stability_artifact(
            validate_stability_report(report, expected_identity), expected_identity
        )
        if (
            not _same_json(dict(value), expected_artifact)
            or _sha256_file(file_path) != digest
            or file_path.stat().st_size != value["bytes"]
        ):
            _fail("stability artifact content or metadata changed")
        expected_parts = pathlib.PurePosixPath(value["relative_path"]).parts
        if tuple(file_path.parts[-len(expected_parts) :]) != expected_parts:
            _fail(f"stability artifact is not stored at {value['relative_path']}")
    return dict(value)


def _profile_for_identity(identity: Mapping[str, Any]) -> Any:
    from .models import DEFAULT_MODEL_PROFILE

    expected = validate_checkpoint_identity(identity)
    return replace(
        DEFAULT_MODEL_PROFILE,
        source_layers=tuple(expected["source_layers"]),
        target_layer=expected["target_layer"],
        skip_first=expected["skip_first"],
        d_model=expected["d_model"],
    )


def lens_artifact(
    path: str | pathlib.Path,
    identity: Mapping[str, Any],
    final_checkpoint_path: str | pathlib.Path,
) -> dict[str, Any]:
    """Bind the serialized fp16 lens exactly to the final fp32 running sums."""

    import torch

    expected = validate_checkpoint_identity(identity)
    profile = _profile_for_identity(expected)
    from .fitting import validate_runtime_lens_file

    lens_path = pathlib.Path(path)
    maximum_lens_bytes = (
        len(expected["source_layers"]) * expected["d_model"] * expected["d_model"] * 2
        + 32 * 1024 * 1024
    )
    try:
        lens_bytes = lens_path.stat().st_size
    except OSError as exc:
        raise AudioFitContractError(f"cannot inspect serialized lens at {lens_path}") from exc
    if lens_bytes > maximum_lens_bytes:
        _fail(
            f"serialized lens at {lens_path} is {lens_bytes} bytes, "
            f"above the bounded maximum {maximum_lens_bytes}"
        )
    validate_runtime_lens_file(lens_path, CORPUS_SIZE, profile=profile)
    checkpoint_state = validate_checkpoint_state(
        final_checkpoint_path,
        expected,
        maximum_count=CORPUS_SIZE,
        exact_count=CORPUS_SIZE,
    )
    try:
        lens_state = torch.load(lens_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AudioFitContractError(f"invalid serialized lens at {path}") from exc
    jacobians = lens_state.get("J") if isinstance(lens_state, Mapping) else None
    if not isinstance(jacobians, Mapping):
        _fail("serialized lens has no Jacobian mapping")
    if (
        not _is_int(lens_state.get("n_prompts"))
        or lens_state["n_prompts"] != CORPUS_SIZE
        or not _is_int(lens_state.get("d_model"))
        or lens_state["d_model"] != expected["d_model"]
    ):
        _fail("serialized lens prompt/model counts are not strict integers")
    for layer in expected["source_layers"]:
        expected_jacobian = (
            checkpoint_state["jacobian_sum"][layer]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .div(CORPUS_SIZE)
            .to(dtype=torch.float16)
        )
        if not torch.equal(jacobians[layer], expected_jacobian):
            _fail(
                f"serialized lens layer {layer} does not equal the final "
                "checkpoint mean after fp16 conversion"
            )
        del expected_jacobian
    digest = _sha256_file(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LENS_ARTIFACT_KIND,
        "role": "lens",
        "fit_config_sha256": expected["fit_config_sha256"],
        "ordered_corpus_sha256": expected["ordered_corpus_sha256"],
        "final_checkpoint_sha256": _sha256_file(final_checkpoint_path),
        "relative_path": f"{expected['lens_dir']}/{digest}.pt",
        "sha256": digest,
        "bytes": pathlib.Path(path).stat().st_size,
        "dtype": "float16",
        "n_prompts": CORPUS_SIZE,
        "d_model": expected["d_model"],
        "source_layers": expected["source_layers"],
    }


def validate_lens_artifact(
    artifact: Mapping[str, Any],
    path: str | pathlib.Path,
    identity: Mapping[str, Any],
    final_checkpoint_path: str | pathlib.Path,
) -> dict[str, Any]:
    value = _require_mapping(artifact, "lens artifact")
    _require_exact_fields(value, _LENS_ARTIFACT_FIELDS, "lens artifact")
    recomputed = lens_artifact(path, identity, final_checkpoint_path)
    if not _same_json(dict(value), recomputed):
        _fail("lens artifact content or metadata changed")
    expected_parts = pathlib.PurePosixPath(recomputed["relative_path"]).parts
    file_parts = pathlib.Path(path).parts
    if tuple(file_parts[-len(expected_parts) :]) != expected_parts:
        _fail(f"lens artifact is not stored at {recomputed['relative_path']}")
    return recomputed


def _read_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > MAX_JSON_BYTES:
            _fail(f"{label} at {path} is {size} bytes, above the bounded maximum {MAX_JSON_BYTES}")
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AudioFitContractError(f"cannot load {label} at {path}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} at {path} must contain an object")
    return value


def _read_jsonl_bounded(
    path: pathlib.Path,
    label: str,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Load JSONL while rejecting the first record beyond its protocol bound."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            line_number = 0
            while True:
                line = handle.readline(MAX_JSONL_LINE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_JSONL_LINE_BYTES:
                    _fail(
                        f"{label} line {line_number} at {path} exceeds {MAX_JSONL_LINE_BYTES} bytes"
                    )
                if not line.strip():
                    continue
                if len(rows) == max_rows:
                    _fail(f"{label} at {path} exceeds {max_rows} records")
                value = json.loads(line)
                if not isinstance(value, dict):
                    _fail(f"{label} line {line_number} at {path} must contain an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AudioFitContractError(f"cannot load {label} at {path}") from exc
    return rows


def validate_completed_run(
    record: Mapping[str, Any], *, volume_root: str | pathlib.Path
) -> dict[str, Any]:
    """Fail closed on every public corpus, checkpoint, report, and lens binding."""

    value = _require_mapping(record, "completed run")
    _require_exact_fields(value, _COMPLETED_RUN_FIELDS, "completed run")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != COMPLETED_RUN_KIND
        or value["status"] != "complete"
    ):
        _fail("completed run kind/status/schema version is invalid")
    config = validate_fit_config(_require_mapping(value["config"], "completed run config"))
    digest = fit_config_digest(config)
    if value["fit_config_sha256"] != digest:
        _fail("completed run fit-config digest changed")
    paths = run_paths(config)
    if not _same_json(value["paths"], paths):
        _fail("completed run content-addressed paths changed")
    corpus = _validate_corpus_artifact(_require_mapping(value["corpus"], "completed run corpus"))
    if not _same_json(corpus, config["corpus"]):
        _fail("completed run corpus differs from immutable fit config")
    identity = checkpoint_identity(config, paths)
    if not _same_json(value["checkpoint_identity"], identity):
        _fail("completed run checkpoint identity changed")

    root = pathlib.Path(volume_root)
    envelope_path = _physical_path(root, corpus["envelope_path"], "completed corpus envelope_path")
    rows_path = _physical_path(root, corpus["rows_path"], "completed corpus rows_path")
    ledger_path = _physical_path(
        root, corpus["attempt_ledger_path"], "completed corpus attempt_ledger_path"
    )
    source_pool_path_value = _physical_path(
        root,
        corpus["source_pool_path"],
        "completed corpus source_pool_path",
    )
    if (
        not source_pool_path_value.is_file()
        or not envelope_path.is_file()
        or not rows_path.is_file()
        or not ledger_path.is_file()
    ):
        _fail("completed run source-pool/envelope/rows/ledger artifact is missing")
    if _sha256_file(envelope_path) != corpus["envelope_sha256"]:
        _fail("completed run corpus envelope bytes changed")
    if _sha256_file(rows_path) != corpus["rows_sha256"]:
        _fail("completed run corpus rows bytes changed")
    if _sha256_file(ledger_path) != corpus["attempt_ledger_sha256"]:
        _fail("completed run attempt ledger bytes changed")
    if _sha256_file(source_pool_path_value) != corpus["source_pool_file_sha256"]:
        _fail("completed run frozen source-pool bytes changed")
    envelope = _read_json(envelope_path, "completed corpus envelope")
    rows = _read_jsonl_bounded(
        rows_path,
        "completed corpus rows",
        max_rows=CORPUS_SIZE,
    )
    maximum_attempts = len(STRATA) * corpus["config"]["selection"]["max_attempts_per_stratum"]
    ledger = _read_jsonl_bounded(
        ledger_path,
        "completed corpus attempt ledger",
        max_rows=maximum_attempts,
    )
    source_pool = validate_source_pool_record(
        _read_json(source_pool_path_value, "completed source pool"),
        corpus["config"],
    )
    if source_pool["source_pool_sha256"] != corpus["source_pool_sha256"]:
        _fail("completed run frozen source-pool identity changed")
    validate_attempt_ledger_against_source_pool(
        ledger,
        source_pool,
        corpus_config=corpus["config"],
    )
    validate_corpus_envelope(
        envelope,
        rows,
        ledger,
        volume_root=root,
        require_files=True,
    )
    if not _same_json(corpus_artifact(envelope, rows, ledger), corpus):
        _fail("completed run corpus artifact metadata changed")

    gate_artifacts = _require_mapping(value["gates"], "completed run gates")
    if set(gate_artifacts) != set(REQUIRED_GATES):
        _fail(
            "completed run gate set is invalid; "
            f"expected {sorted(REQUIRED_GATES)}, got {sorted(gate_artifacts)}"
        )
    for gate in REQUIRED_GATES:
        gate_value = _require_mapping(
            gate_artifacts[gate],
            f"completed run gate {gate}",
        )
        gate_file = _physical_path(
            root,
            gate_path(config, gate),
            f"completed run gate {gate} path",
        )
        validate_gate_artifact(
            gate_value,
            gate_file,
            config,
            gate,
        )

    prefix_record = _require_mapping(value["prefix_snapshot"], "completed prefix snapshot")
    prefix_path = _physical_path(
        root, prefix_record.get("relative_path"), "completed prefix snapshot path"
    )
    validate_checkpoint_artifact(prefix_record, prefix_path, identity, PREFIX_COUNT)
    checkpoint_record = _require_mapping(value["checkpoint"], "completed final checkpoint")
    checkpoint_path = _physical_path(
        root, checkpoint_record.get("relative_path"), "completed checkpoint path"
    )
    validate_checkpoint_artifact(checkpoint_record, checkpoint_path, identity, CORPUS_SIZE)

    stability_record = _require_mapping(value["stability"], "completed stability artifact")
    stability_path = _physical_path(
        root, stability_record.get("relative_path"), "completed stability path"
    )
    validate_stability_artifact(stability_record, identity, path=stability_path)
    report = _read_json(stability_path, "completed stability report")
    recomputed_report = stability_from_checkpoints(prefix_path, checkpoint_path, identity)
    if not _same_json(report, recomputed_report):
        _fail("completed stability report does not match bound checkpoints")

    lens_record = _require_mapping(value["lens"], "completed lens artifact")
    lens_path = _physical_path(root, lens_record.get("relative_path"), "completed lens path")
    validate_lens_artifact(
        lens_record,
        lens_path,
        identity,
        checkpoint_path,
    )
    return dict(value)
