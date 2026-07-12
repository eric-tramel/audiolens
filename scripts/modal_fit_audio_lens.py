"""Stage and fit the source-bound 1,000-example audio Jacobian lens on Modal.

The local process only constructs the Modal app and dispatches explicitly selected
stages.  LibriSpeech access, Gemma processor execution, model loading, replay,
and fitting all happen inside Modal functions.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import hashlib
import io
import json
import os
import pathlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
VOLUME_NAME = "audiolens-vol"
APP_NAME = "audiolens-audio-fit"
MODEL_GPU = "H100"
SOURCE_TIMEOUT_SECONDS = 12 * 60 * 60
PROCESSOR_TIMEOUT_SECONDS = 12 * 60 * 60
FIT_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024
SOURCE_CENSUS_CHUNK_SIZE = 10_000
BF16_BATCH_RELATIVE_L2_MAX = 0.05
BF16_BATCH_COSINE_MIN = 1.0 - 0.5 * BF16_BATCH_RELATIVE_L2_MAX**2

AUDIO_FIT_SOURCE_RELATIVES = (
    "pyproject.toml",
    "uv.lock",
    "src/audiolens/__init__.py",
    "src/audiolens/fitting.py",
    "src/audiolens/audio_fitting.py",
    "src/audiolens/models/__init__.py",
    "src/audiolens/models/base.py",
    "src/audiolens/models/gemma4.py",
    "scripts/modal_fit_audio_lens.py",
)


def _source_digest() -> str | None:
    if all((REPO_ROOT / relative).is_file() for relative in AUDIO_FIT_SOURCE_RELATIVES):
        digest = hashlib.sha256()
        for relative in AUDIO_FIT_SOURCE_RELATIVES:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update((REPO_ROOT / relative).read_bytes())
        return digest.hexdigest()
    return os.environ.get("AUDIOLENS_AUDIO_FIT_SOURCE_DIGEST")


def _lock_digest() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    if lock.is_file():
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    return os.environ.get("AUDIOLENS_LOCK_SHA256")


SOURCE_DIGEST = _source_digest()
LOCK_SHA256 = _lock_digest()
_HAS_LOCAL_PROJECT = all(
    (REPO_ROOT / relative).is_file() for relative in AUDIO_FIT_SOURCE_RELATIVES
)
_DEPLOY_MODAL = _HAS_LOCAL_PROJECT and os.environ.get("AUDIOLENS_DISABLE_MODAL") != "1"

if _DEPLOY_MODAL:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libsndfile1")
        .uv_sync(
            uv_project_dir=str(REPO_ROOT),
            frozen=True,
            groups=["fit"],
            gpu=MODEL_GPU,
        )
        .env(
            {
                "HF_HOME": f"{VOL_MOUNT}/hf",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "AUDIOLENS_AUDIO_FIT_SOURCE_DIGEST": SOURCE_DIGEST or "",
                "AUDIOLENS_LOCK_SHA256": LOCK_SHA256 or "",
            }
        )
        .add_local_python_source("audiolens")
    )
    app = modal.App(APP_NAME, image=image)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
else:
    image = None
    app = None
    vol = None


def _modal_cpu_function(*, timeout: int, model_secret: bool = False):
    def decorate(function):
        if app is None or vol is None:
            return function
        kwargs: dict[str, Any] = {
            "cpu": 4.0,
            "memory": 16_384,
            "timeout": timeout,
            "volumes": {VOL_MOUNT: vol},
        }
        if model_secret:
            kwargs["secrets"] = [modal.Secret.from_name("huggingface")]
        return app.function(**kwargs)(function)

    return decorate


def _modal_gpu_function(function):
    if app is None or vol is None:
        return function
    return app.function(
        gpu=MODEL_GPU,
        timeout=FIT_TIMEOUT_SECONDS,
        volumes={VOL_MOUNT: vol},
        secrets=[modal.Secret.from_name("huggingface")],
    )(function)


def _modal_local_entrypoint(function):
    if app is None:
        return function
    return app.local_entrypoint()(function)


def _commit_volume() -> None:
    global vol
    if vol is None:
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    vol.commit()


def _reload_volume() -> None:
    """Refresh a worker mount after a separately deployed function commits."""

    global vol
    if vol is None:
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    vol.reload()


def _call_deployed_function(name: str, **kwargs: Any) -> Any:
    """Call a separately deployed Modal function from an orchestration worker."""

    function = modal.Function.from_name(APP_NAME, name)
    return function.remote(**kwargs)


def _required_source_identity() -> tuple[str, str]:
    if SOURCE_DIGEST is None or len(SOURCE_DIGEST) != 64:
        raise RuntimeError("audio-fit source digest is unavailable")
    if LOCK_SHA256 is None or len(LOCK_SHA256) != 64:
        raise RuntimeError("uv.lock digest is unavailable")
    return SOURCE_DIGEST, LOCK_SHA256


def _profile():
    from audiolens.models import DEFAULT_MODEL_PROFILE

    return DEFAULT_MODEL_PROFILE


def _corpus_config() -> dict[str, Any]:
    from audiolens.audio_fitting import build_corpus_config

    source_digest, lock_sha256 = _required_source_identity()
    return build_corpus_config(
        _profile(),
        source_digest=source_digest,
        lock_sha256=lock_sha256,
    )


def _runtime_config() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import torch

    from audiolens.audio_fitting import JLENS_REVISION

    return {
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "datasets_version": importlib.metadata.version("datasets"),
        "jlens_revision": JLENS_REVISION,
        "cuda_version": torch.version.cuda,
    }


def _default_source_loader(
    config: str,
    split: str,
    *,
    metadata_only: bool,
):
    """Load the pinned public source with dataset authentication disabled."""

    from datasets import Audio, load_dataset

    from audiolens.audio_fitting import DATASET_ID, LIBRISPEECH_REVISION

    dataset = load_dataset(
        DATASET_ID,
        config,
        split=split,
        revision=LIBRISPEECH_REVISION,
        streaming=metadata_only,
        trust_remote_code=False,
        token=False,
    )
    if metadata_only:
        return dataset.select_columns(["id", "speaker_id", "chapter_id", "text"])
    return dataset.cast_column("audio", Audio(decode=False))


def _default_processor_loader():
    from audiolens.models import DEFAULT_MODEL_KEY, load_audio_processor

    return load_audio_processor(DEFAULT_MODEL_KEY)


def _default_model_loader():
    from audiolens.models import DEFAULT_MODEL_KEY, load_model_runtime

    return load_model_runtime(DEFAULT_MODEL_KEY, device_map="cuda")


def _source_metadata(source: Mapping[str, Any], config: str, split: str) -> dict[str, Any]:
    from audiolens.audio_fitting import DATASET_ID, LIBRISPEECH_REVISION
    from audiolens.models import AudioFitContractError

    source_id = source.get("id")
    transcript = source.get("text")
    speaker_id = source.get("speaker_id")
    chapter_id = source.get("chapter_id")
    if not isinstance(source_id, str) or not source_id:
        raise AudioFitContractError(f"{config}/{split} source has no exact string id")
    if not isinstance(transcript, str) or not transcript or not transcript.strip():
        raise AudioFitContractError(f"{config}/{split}/{source_id} has no exact transcript")
    try:
        speaker = int(speaker_id)
        chapter = int(chapter_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioFitContractError(
            f"{config}/{split}/{source_id} has invalid speaker/chapter identity"
        ) from exc
    if isinstance(speaker_id, bool) or isinstance(chapter_id, bool) or speaker < 0 or chapter < 0:
        raise AudioFitContractError(
            f"{config}/{split}/{source_id} has invalid speaker/chapter identity"
        )
    return {
        "dataset": DATASET_ID,
        "revision": LIBRISPEECH_REVISION,
        "config": config,
        "split": split,
        "source_id": source_id,
        "speaker_id": speaker,
        "chapter_id": chapter,
        "transcript": transcript,
    }


def _collect_source_metadata(
    source_loader: Callable[..., Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    from audiolens.audio_fitting import STRATA

    rows: list[dict[str, Any]] = []
    for config, split in STRATA:
        print(
            f"enumerating complete metadata pool for {config}/{split}",
            flush=True,
        )
        source = source_loader(config, split, metadata_only=True)
        stratum_count = 0
        for candidate in source:
            rows.append(_source_metadata(candidate, config, split))
            stratum_count += 1
            if stratum_count % 10_000 == 0:
                print(
                    f"{config}/{split}: enumerated {stratum_count:,} metadata rows",
                    flush=True,
                )
        print(
            f"{config}/{split}: metadata enumeration complete ({stratum_count:,} rows)",
            flush=True,
        )
    stratum_order = {coordinates: index for index, coordinates in enumerate(STRATA)}
    rows.sort(
        key=lambda row: (
            stratum_order[(row["config"], row["split"])],
            row["speaker_id"],
            row["source_id"],
        )
    )
    return rows


def _collect_source_metadata_durable(
    source_loader: Callable[..., Iterable[Mapping[str, Any]]],
    *,
    root: pathlib.Path,
    corpus_config: Mapping[str, Any],
    commit: Callable[[], None] | None,
) -> list[dict[str, Any]]:
    """Checkpoint the complete public census in restart-safe 10k-row shards."""

    from audiolens.audio_fitting import (
        SOURCE_POOL_COUNTS,
        SCHEMA_VERSION,
        STRATA,
        corpus_config_digest,
    )
    from audiolens.models import AudioFitContractError

    census_root = root / "audio-source-census" / corpus_config_digest(corpus_config)
    if census_root.exists() and not census_root.is_dir():
        raise AudioFitContractError(
            f"source census root exists but is not a directory: {census_root}"
        )
    rows: list[dict[str, Any]] = []
    for config, split in STRATA:
        expected_count = SOURCE_POOL_COUNTS[config]
        stratum_root = census_root / f"{config}-{split.replace('.', '-')}"
        if stratum_root.exists() and not stratum_root.is_dir():
            raise AudioFitContractError(
                f"source census root exists but is not a directory: {stratum_root}"
            )
        completion_path = stratum_root / "complete.json"
        completion_record = {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "split": split,
            "row_count": expected_count,
        }
        completed_proof = completion_path.exists()
        if (
            completed_proof
            and _load_json(
                completion_path,
                f"{config}/{split} source census completion",
            )
            != completion_record
        ):
            raise AudioFitContractError(f"source census completion changed at {completion_path}")
        stratum_rows: list[dict[str, Any]] = []
        for shard in sorted(stratum_root.glob("*.jsonl")):
            expected_name = f"{len(stratum_rows):06d}.jsonl"
            if shard.name != expected_name:
                raise AudioFitContractError(f"source census shard sequence changed at {shard}")
            maximum = min(
                SOURCE_CENSUS_CHUNK_SIZE,
                expected_count - len(stratum_rows),
            )
            if maximum <= 0:
                raise AudioFitContractError(f"source census has excess shard {shard}")
            shard_rows = _load_jsonl(
                shard,
                f"{config}/{split} source census shard",
                max_rows=maximum,
            )
            if len(shard_rows) != maximum:
                raise AudioFitContractError(
                    f"source census shard {shard} has {len(shard_rows)} rows, expected {maximum}"
                )
            stratum_rows.extend(shard_rows)

        completed = len(stratum_rows)
        if completed_proof and completed != expected_count:
            raise AudioFitContractError(
                f"source census completion binds {completed} rows, expected {expected_count}"
            )
        if completed < expected_count or not completed_proof:
            print(
                f"resuming complete metadata pool for {config}/{split} "
                f"at row {completed:,}/{expected_count:,}",
                flush=True,
            )
            source = source_loader(config, split, metadata_only=True)
            if completed:
                skip = getattr(source, "skip", None)
                source = (
                    skip(completed) if callable(skip) else itertools.islice(source, completed, None)
                )
            iterator = iter(source)
            buffer: list[dict[str, Any]] = []
            shard_start = completed
            while completed < expected_count:
                try:
                    candidate = next(iterator)
                except StopIteration as exc:
                    raise AudioFitContractError(
                        f"{config}/{split} ended at {completed:,} rows; expected {expected_count:,}"
                    ) from exc
                buffer.append(_source_metadata(candidate, config, split))
                completed += 1
                if len(buffer) == SOURCE_CENSUS_CHUNK_SIZE or completed == expected_count:
                    shard = stratum_root / f"{shard_start:06d}.jsonl"
                    _write_immutable_bytes(shard, _canonical_jsonl_bytes(buffer))
                    stratum_rows.extend(buffer)
                    buffer = []
                    shard_start = completed
                    if commit is not None:
                        commit()
                    print(
                        f"{config}/{split}: sealed {completed:,}/{expected_count:,} metadata rows",
                        flush=True,
                    )
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise AudioFitContractError(
                    f"{config}/{split} exceeds pinned count {expected_count:,}"
                )
            _write_immutable_bytes(
                completion_path,
                _canonical_bytes(completion_record),
            )
            if commit is not None:
                commit()
        else:
            print(
                f"{config}/{split}: restored complete census ({completed:,} rows)",
                flush=True,
            )
        rows.extend(stratum_rows)

    stratum_order = {coordinates: index for index, coordinates in enumerate(STRATA)}
    rows.sort(
        key=lambda row: (
            stratum_order[(row["config"], row["split"])],
            row["speaker_id"],
            row["source_id"],
        )
    )
    return rows


def _source_pool_record(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from audiolens.audio_fitting import (
        SCHEMA_VERSION,
        SOURCE_POOL_KIND,
        STRATA,
        corpus_config_digest,
        source_pool_digest,
    )

    digest = source_pool_digest(rows)
    counts = {
        name: sum(row["config"] == name and row["split"] == split for row in rows)
        for name, split in STRATA
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_POOL_KIND,
        "config": dict(config),
        "corpus_config_sha256": corpus_config_digest(config),
        "source_pool_sha256": digest,
        "row_count": len(rows),
        "stratum_counts": counts,
        "rows": [dict(row) for row in rows],
    }


def _validate_source_pool_record(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    from audiolens.audio_fitting import validate_source_pool_record

    return validate_source_pool_record(record, config)


def _reusable_source_metadata(
    root: pathlib.Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Reuse a fully validated census when only bound implementation bytes changed."""

    from audiolens.audio_fitting import validate_source_pool_record
    from audiolens.models import AudioFitContractError

    expected_config = dict(config)
    candidates: dict[str, list[dict[str, Any]]] = {}
    pool_root = root / "audio-source-pools"
    for path in sorted(pool_root.glob("*/*/pool.json")):
        record = _load_json(path, "reusable source pool")
        record_config = record.get("config")
        if not isinstance(record_config, Mapping):
            continue
        comparable = dict(record_config)
        comparable["source_sha256"] = expected_config["source_sha256"]
        if comparable != expected_config:
            continue
        validated = validate_source_pool_record(record, record_config)
        digest = validated["source_pool_sha256"]
        candidates.setdefault(digest, validated["rows"])
    if not candidates:
        return None
    if len(candidates) != 1:
        raise AudioFitContractError("reusable source pools disagree on complete pinned metadata")
    digest, rows = next(iter(candidates.items()))
    print(
        f"reusing validated complete source census {digest} ({len(rows):,} rows)",
        flush=True,
    )
    return [dict(row) for row in rows]


def _canonical_bytes(value: Any) -> bytes:
    from audiolens.fitting import canonical_json_bytes

    return canonical_json_bytes(value) + b"\n"


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    from audiolens.fitting import canonical_json_bytes

    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_immutable_bytes(path: pathlib.Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"immutable content-addressed artifact differs at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    from audiolens.models import AudioFitContractError

    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise AudioFitContractError(f"{label} at {path} exceeds {MAX_JSON_BYTES} bytes")
        content = path.read_bytes()
        if len(content) > MAX_JSON_BYTES:
            raise AudioFitContractError(f"{label} at {path} exceeds {MAX_JSON_BYTES} bytes")
        value = json.loads(content.decode("utf-8"))
    except AudioFitContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioFitContractError(f"cannot load {label} at {path}") from exc
    if not isinstance(value, dict):
        raise AudioFitContractError(f"{label} root is not an object at {path}")
    return value


def _load_jsonl(
    path: pathlib.Path,
    label: str,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    from audiolens.models import AudioFitContractError

    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
        raise ValueError("max_rows must be a nonnegative integer")
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "rb") as source:
            line_number = 0
            while True:
                raw_line = source.readline(MAX_JSONL_LINE_BYTES + 1)
                if not raw_line:
                    break
                line_number += 1
                if len(raw_line) > MAX_JSONL_LINE_BYTES:
                    raise AudioFitContractError(
                        f"{label} line {line_number} at {path} exceeds {MAX_JSONL_LINE_BYTES} bytes"
                    )
                if not raw_line.strip():
                    continue
                if len(rows) >= max_rows:
                    raise AudioFitContractError(
                        f"{label} exceeds its bounded maximum of {max_rows} records"
                    )
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AudioFitContractError(
                        f"invalid {label} JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise AudioFitContractError(
                        f"{label} record at {path}:{line_number} is not an object"
                    )
                rows.append(value)
    except AudioFitContractError:
        raise
    except OSError as exc:
        raise AudioFitContractError(f"cannot load {label} at {path}") from exc
    return rows


def _resolve_digest_child(
    parent: pathlib.Path,
    explicit_digest: str | None,
    filename: str,
    label: str,
) -> pathlib.Path:
    from audiolens.models import AudioFitContractError

    if explicit_digest:
        if len(explicit_digest) != 64 or any(
            character not in "0123456789abcdef" for character in explicit_digest
        ):
            raise AudioFitContractError(f"{label} digest must be lowercase 64-hex")
        path = parent / explicit_digest / filename
        if not path.is_file():
            raise AudioFitContractError(f"{label} is missing at {path}")
        return path
    candidates = sorted(path for path in parent.glob(f"*/{filename}") if path.is_file())
    if len(candidates) != 1:
        raise AudioFitContractError(
            f"expected exactly one {label} below {parent}, found {len(candidates)}; pass its digest explicitly"
        )
    return candidates[0]


def _rank_audio_source_impl(
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    source_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from audiolens.audio_fitting import source_pool_path
    from audiolens.fitting import sha256_file

    config = _corpus_config()
    loader = _default_source_loader if source_loader is None else source_loader
    root = pathlib.Path(volume_root)
    metadata = _reusable_source_metadata(root, config)
    if metadata is None:
        metadata = _collect_source_metadata_durable(
            loader,
            root=root,
            corpus_config=config,
            commit=commit,
        )
    record = _source_pool_record(config, metadata)
    relative_path = source_pool_path(config, record["source_pool_sha256"])
    path = root / relative_path
    _write_immutable_bytes(path, _canonical_bytes(record))
    _validate_source_pool_record(_load_json(path, "source pool"), config)
    pool_file_sha256 = sha256_file(path)
    if commit is not None:
        commit()
    return {
        "source_pool_path": relative_path,
        "source_pool_sha256": record["source_pool_sha256"],
        "source_pool_file_sha256": pool_file_sha256,
        "row_count": record["row_count"],
        "stratum_counts": record["stratum_counts"],
    }


def _load_source_pool(
    volume_root: str | pathlib.Path,
    config: Mapping[str, Any],
    source_pool_sha256: str | None,
) -> tuple[dict[str, Any], str, str]:
    from audiolens.audio_fitting import corpus_config_digest, source_pool_path
    from audiolens.fitting import sha256_file

    root = pathlib.Path(volume_root)
    config_sha = corpus_config_digest(config)
    path = _resolve_digest_child(
        root / "audio-source-pools" / config_sha,
        source_pool_sha256,
        "pool.json",
        "source pool",
    )
    record = _validate_source_pool_record(_load_json(path, "source pool"), config)
    if path.parent.name != record["source_pool_sha256"]:
        raise RuntimeError("source-pool directory does not match its content digest")
    relative_path = source_pool_path(config, record["source_pool_sha256"])
    if path != root / relative_path:
        raise RuntimeError("source pool is not at its content-addressed path")
    return record, relative_path, sha256_file(path)


class _IneligibleAudio(RuntimeError):
    """A deterministic, ledger-safe corpus-selection rejection."""


def _audio_blob(source: Mapping[str, Any]) -> bytes:
    audio = source.get("audio")
    if not isinstance(audio, Mapping):
        raise _IneligibleAudio("missing_audio")
    blob = audio.get("bytes")
    if blob is None:
        candidate_path = audio.get("path")
        if isinstance(candidate_path, str):
            path = pathlib.Path(candidate_path)
            if path.is_file():
                try:
                    if path.stat().st_size > MAX_AUDIO_BYTES:
                        raise _IneligibleAudio("decode_error")
                    blob = path.read_bytes()
                except _IneligibleAudio:
                    raise
                except OSError as exc:
                    raise _IneligibleAudio("missing_audio_bytes") from exc
    if not isinstance(blob, bytes) or not blob:
        raise _IneligibleAudio("missing_audio_bytes")
    if len(blob) > MAX_AUDIO_BYTES:
        raise _IneligibleAudio("decode_error")
    return blob


def _prepare_candidate(
    metadata: Mapping[str, Any],
    source: Mapping[str, Any],
    processor: Any,
    *,
    selection_index: int,
    stratum_index: int,
    volume_root: pathlib.Path,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    from audiolens.audio_fitting import (
        CORPUS_ROW_KIND,
        MAX_DURATION_SECONDS,
        MIN_DURATION_SECONDS,
        SAMPLE_RATE,
        SCHEMA_VERSION,
        audio_blob_path,
        decoded_pcm_sha256,
        input_ids_sha256,
        make_pair_id,
        transcript_sha256,
    )
    from audiolens.fitting import sha256_bytes
    from audiolens.models import AudioFitContractError, prepare_audio

    restored_metadata = _source_metadata(source, metadata["config"], metadata["split"])
    if restored_metadata != metadata:
        raise AudioFitContractError(
            f"source metadata drift for {metadata['config']}/{metadata['split']}/{metadata['source_id']}"
        )
    blob = _audio_blob(source)
    try:
        info = sf.info(io.BytesIO(blob))
    except Exception as exc:
        raise _IneligibleAudio("decode_error") from exc
    if info.channels != 1 or info.samplerate != SAMPLE_RATE:
        raise _IneligibleAudio("not_native_mono_16khz")
    num_samples = int(info.frames)
    minimum_samples = int(MIN_DURATION_SECONDS * SAMPLE_RATE)
    maximum_samples = int(MAX_DURATION_SECONDS * SAMPLE_RATE)
    if not minimum_samples <= num_samples <= maximum_samples:
        raise _IneligibleAudio("duration_out_of_range")
    try:
        pcm, sampling_rate = sf.read(
            io.BytesIO(blob),
            frames=num_samples,
            dtype="int16",
            always_2d=False,
        )
        waveform, float_rate = sf.read(
            io.BytesIO(blob),
            frames=num_samples,
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise _IneligibleAudio("decode_error") from exc
    if (
        sampling_rate != SAMPLE_RATE
        or float_rate != SAMPLE_RATE
        or pcm.ndim != 1
        or waveform.ndim != 1
    ):
        raise _IneligibleAudio("not_native_mono_16khz")
    if not bool(np.isfinite(waveform).all()):
        raise _IneligibleAudio("nonfinite_waveform")
    if num_samples != int(pcm.shape[0]) or num_samples != int(waveform.shape[0]):
        raise _IneligibleAudio("decoded_sample_count_mismatch")
    duration = round(num_samples / SAMPLE_RATE, 6)

    with tempfile.NamedTemporaryFile(suffix=".flac") as temporary:
        temporary.write(blob)
        temporary.flush()
        try:
            prepared = prepare_audio(processor, temporary.name, profile=_profile())
        except AudioFitContractError as exc:
            raise _IneligibleAudio("processor_contract") from exc

    processor_ids = prepared.input_ids
    fit_ids = processor_ids[:, : prepared.layout.stop]
    audio_sha = sha256_bytes(blob)
    row = {
        "schema_version": SCHEMA_VERSION,
        "kind": CORPUS_ROW_KIND,
        "selection_index": selection_index,
        "stratum_index": stratum_index,
        **dict(metadata),
        "pair_id": "",
        "transcript_sha256": transcript_sha256(metadata["transcript"]),
        "audio_sha256": audio_sha,
        "decoded_pcm_sha256": decoded_pcm_sha256(pcm),
        "sampling_rate": SAMPLE_RATE,
        "num_samples": num_samples,
        "duration_seconds": duration,
        "volume_path": audio_blob_path(audio_sha),
        "audio_start": prepared.layout.audio_start,
        "n_audio_tokens": prepared.layout.n_audio_tokens,
        "processor_seq_len": int(processor_ids.shape[1]),
        "sliced_seq_len": prepared.layout.stop,
        "n_valid_positions": prepared.layout.n_valid_positions,
        "processor_input_ids_sha256": input_ids_sha256(processor_ids),
        "fit_input_ids_sha256": input_ids_sha256(fit_ids),
    }
    row["pair_id"] = make_pair_id(row)
    destination = volume_root / row["volume_path"]
    _write_immutable_bytes(destination, blob)
    return row


def _source_accessor(dataset: Any) -> Callable[[str], Mapping[str, Any]]:
    from audiolens.models import AudioFitContractError

    if hasattr(dataset, "column_names") and hasattr(dataset, "__getitem__"):
        try:
            identifiers = dataset["id"]
            index = {str(source_id): position for position, source_id in enumerate(identifiers)}
        except Exception:
            index = {}
        if index:
            if len(index) != len(identifiers):
                raise AudioFitContractError("dataset contains duplicate source IDs")

            def get(source_id: str) -> Mapping[str, Any]:
                try:
                    row = dataset[index[source_id]]
                except (KeyError, IndexError) as exc:
                    raise AudioFitContractError(
                        f"ranked source ID {source_id!r} is absent from the pinned source"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise AudioFitContractError(f"source row {source_id!r} is not an object")
                return row

            return get

    rows: dict[str, Mapping[str, Any]] = {}
    for candidate in dataset:
        if not isinstance(candidate, Mapping):
            raise AudioFitContractError("source loader yielded a non-object row")
        source_id = candidate.get("id")
        if not isinstance(source_id, str) or source_id in rows:
            raise AudioFitContractError("source loader yielded an invalid/duplicate source ID")
        rows[source_id] = candidate

    def get(source_id: str) -> Mapping[str, Any]:
        try:
            return rows[source_id]
        except KeyError as exc:
            raise AudioFitContractError(
                f"ranked source ID {source_id!r} is absent from the pinned source"
            ) from exc

    return get


def _ranked_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[tuple[str, str, str], Mapping[str, Any]],
    config_name: str,
    split: str,
) -> list[tuple[str, int, list[tuple[str, Mapping[str, Any]]]]]:
    from audiolens.models import AudioFitContractError

    by_speaker: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    speaker_ranks: dict[int, str] = {}
    for row in rows:
        if row["config"] != config_name or row["split"] != split:
            continue
        key = (config_name, split, row["source_id"])
        try:
            rank = frozen[key]
        except KeyError as exc:
            raise AudioFitContractError(f"source pool has no frozen rank for {key!r}") from exc
        speaker_id = row["speaker_id"]
        if rank["speaker_id"] != speaker_id:
            raise AudioFitContractError(f"source pool speaker changed for {row['source_id']}")
        observed_speaker_rank = speaker_ranks.setdefault(speaker_id, rank["speaker_rank_sha256"])
        if observed_speaker_rank != rank["speaker_rank_sha256"]:
            raise AudioFitContractError(f"source pool speaker rank changed for {speaker_id}")
        by_speaker.setdefault(speaker_id, []).append((rank["utterance_rank_sha256"], row))
    ranked = [
        (
            speaker_ranks[speaker_id],
            speaker_id,
            sorted(
                utterances,
                key=lambda item: (item[0], item[1]["source_id"]),
            ),
        )
        for speaker_id, utterances in by_speaker.items()
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def _select_corpus(
    pool: Mapping[str, Any],
    config: Mapping[str, Any],
    processor: Any,
    source_loader: Callable[..., Any],
    *,
    volume_root: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from audiolens.audio_fitting import (
        ATTEMPT_ROW_KIND,
        SCHEMA_VERSION,
        STRATA,
        STRATUM_SIZE,
        source_pool_ranks,
    )
    from audiolens.models import AudioFitContractError

    selected_by_stratum: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in STRATA}
    ledger: list[dict[str, Any]] = []
    selected_speakers: set[int] = set()
    maximum = config["selection"]["max_attempts_per_stratum"]
    stratum_offsets = {name: index for index, (name, _) in enumerate(STRATA)}
    ranks = source_pool_ranks(pool["rows"])
    frozen = {(rank["config"], rank["split"], rank["source_id"]): rank for rank in ranks}
    if len(frozen) != len(ranks):
        raise AudioFitContractError("source pool has duplicate frozen rank identities")

    for config_name, split in STRATA:
        source = source_loader(config_name, split, metadata_only=False)
        get_source = _source_accessor(source)
        ranked_speakers = _ranked_pool_rows(pool["rows"], frozen, config_name, split)
        stratum_attempt_index = 0
        for speaker_rank, speaker_id, utterances in ranked_speakers:
            if len(selected_by_stratum[config_name]) == STRATUM_SIZE:
                break
            for utterance_rank, metadata in utterances:
                if stratum_attempt_index >= maximum:
                    raise AudioFitContractError(
                        f"{config_name}/{split} reached bounded attempt limit {maximum}"
                    )
                attempt: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": ATTEMPT_ROW_KIND,
                    "attempt_index": len(ledger),
                    "stratum": config_name,
                    "stratum_attempt_index": stratum_attempt_index,
                    "config": config_name,
                    "split": split,
                    "speaker_id": speaker_id,
                    "source_id": metadata["source_id"],
                    "speaker_rank_sha256": speaker_rank,
                    "utterance_rank_sha256": utterance_rank,
                    "outcome": "rejected",
                    "reason": None,
                    "pair_id": None,
                }
                stratum_attempt_index += 1
                if speaker_id in selected_speakers:
                    attempt["reason"] = "speaker_already_selected"
                    ledger.append(attempt)
                    break
                try:
                    source_row = get_source(metadata["source_id"])
                    stratum_index = len(selected_by_stratum[config_name])
                    selection_index = stratum_index * len(STRATA) + stratum_offsets[config_name]
                    row = _prepare_candidate(
                        metadata,
                        source_row,
                        processor,
                        selection_index=selection_index,
                        stratum_index=stratum_index,
                        volume_root=volume_root,
                    )
                except _IneligibleAudio as exc:
                    attempt["reason"] = str(exc)
                    ledger.append(attempt)
                    continue
                attempt["outcome"] = "selected"
                attempt["reason"] = None
                attempt["pair_id"] = row["pair_id"]
                ledger.append(attempt)
                selected_by_stratum[config_name].append(row)
                selected_speakers.add(speaker_id)
                if len(selected_by_stratum[config_name]) % 25 == 0:
                    print(
                        f"{config_name}/{split}: selected "
                        f"{len(selected_by_stratum[config_name])}/{STRATUM_SIZE} "
                        f"after {stratum_attempt_index} attempts"
                    )
                break
        if len(selected_by_stratum[config_name]) != STRATUM_SIZE:
            raise AudioFitContractError(
                f"{config_name}/{split} selected only "
                f"{len(selected_by_stratum[config_name])}/{STRATUM_SIZE}"
            )

    rows: list[dict[str, Any]] = []
    for stratum_index in range(STRATUM_SIZE):
        for config_name, _split in STRATA:
            row = selected_by_stratum[config_name][stratum_index]
            if row["selection_index"] != len(rows):
                raise AssertionError("alternating selection index construction failed")
            rows.append(row)
    return rows, ledger


def _build_corpus_envelope(
    config: Mapping[str, Any],
    source_pool_sha256: str,
    source_pool_path: str,
    source_pool_file_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from audiolens.audio_fitting import (
        CORPUS_ENVELOPE_KIND,
        CORPUS_SIZE,
        SCHEMA_VERSION,
        STRATA,
        STRATUM_SIZE,
        canonical_jsonl_sha256,
        corpus_config_digest,
        corpus_paths,
        ordered_corpus_digest,
    )

    ordered_sha = ordered_corpus_digest(rows)
    paths = corpus_paths(config, ordered_sha)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CORPUS_ENVELOPE_KIND,
        "config": dict(config),
        "corpus_config_sha256": corpus_config_digest(config),
        "source_pool_sha256": source_pool_sha256,
        "source_pool_path": source_pool_path,
        "source_pool_file_sha256": source_pool_file_sha256,
        "row_count": CORPUS_SIZE,
        "stratum_counts": {name: STRATUM_SIZE for name, _ in STRATA},
        "ordered_corpus_sha256": ordered_sha,
        "rows_path": paths["rows"],
        "rows_sha256": canonical_jsonl_sha256(rows),
        "attempt_ledger_path": paths["attempt_ledger"],
        "attempt_ledger_sha256": canonical_jsonl_sha256(ledger),
        "attempt_count": len(ledger),
        "max_attempts_per_stratum": config["selection"]["max_attempts_per_stratum"],
        "audio_root": "audio-blobs",
    }


def _load_corpus(
    volume_root: str | pathlib.Path,
    config: Mapping[str, Any],
    ordered_corpus_sha256: str | None,
    *,
    require_files: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from audiolens.audio_fitting import (
        corpus_artifact,
        CORPUS_SIZE,
        STRATA,
        corpus_config_digest,
        corpus_paths,
        source_pool_path,
        validate_attempt_ledger_against_source_pool,
        validate_corpus_envelope,
    )
    from audiolens.fitting import sha256_file
    from audiolens.models import AudioFitContractError

    root = pathlib.Path(volume_root)
    config_sha = corpus_config_digest(config)
    envelope_path = _resolve_digest_child(
        root / "audio-corpora" / config_sha,
        ordered_corpus_sha256,
        "envelope.json",
        "audio corpus",
    )
    envelope = _load_json(envelope_path, "corpus envelope")
    expected_corpus_paths = corpus_paths(config, envelope_path.parent.name)
    if (
        envelope.get("rows_path") != expected_corpus_paths["rows"]
        or envelope.get("attempt_ledger_path") != expected_corpus_paths["attempt_ledger"]
    ):
        raise AudioFitContractError("corpus envelope rows/ledger paths are not content-addressed")
    rows_path = root / expected_corpus_paths["rows"]
    ledger_path = root / expected_corpus_paths["attempt_ledger"]
    if sha256_file(rows_path) != envelope.get("rows_sha256"):
        raise AudioFitContractError("bound corpus rows file SHA-256 changed")
    if sha256_file(ledger_path) != envelope.get("attempt_ledger_sha256"):
        raise AudioFitContractError("bound attempt-ledger file SHA-256 changed")
    rows = _load_jsonl(
        rows_path,
        "corpus rows",
        max_rows=CORPUS_SIZE,
    )
    ledger = _load_jsonl(
        ledger_path,
        "attempt ledger",
        max_rows=(len(STRATA) * config["selection"]["max_attempts_per_stratum"]),
    )
    expected_pool_path = source_pool_path(config, envelope.get("source_pool_sha256", ""))
    if envelope.get("source_pool_path") != expected_pool_path:
        raise AudioFitContractError(
            "corpus envelope does not bind its content-addressed source pool"
        )
    pool_path = root / expected_pool_path
    pool_file_sha256 = sha256_file(pool_path)
    if envelope.get("source_pool_file_sha256") != pool_file_sha256:
        raise AudioFitContractError("bound source-pool file SHA-256 changed")
    pool = _validate_source_pool_record(_load_json(pool_path, "bound source pool"), config)
    if pool["source_pool_sha256"] != envelope["source_pool_sha256"]:
        raise AudioFitContractError("bound source-pool identity changed")
    validate_attempt_ledger_against_source_pool(
        ledger,
        pool,
        corpus_config=config,
    )
    validated = validate_corpus_envelope(
        envelope,
        rows,
        ledger,
        volume_root=root,
        require_files=require_files,
    )
    artifact = corpus_artifact(validated, rows, ledger)
    if sha256_file(envelope_path) != artifact["envelope_sha256"]:
        raise AudioFitContractError("bound corpus envelope file SHA-256 changed")
    if envelope_path != root / artifact["envelope_path"]:
        raise RuntimeError("corpus envelope is not at its content-addressed path")
    return validated, rows, ledger, artifact


def _stage_audio_corpus_impl(
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    source_pool_sha256: str | None = None,
    source_loader: Callable[..., Any] | None = None,
    processor_loader: Callable[[], Any] | None = None,
    commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Select and atomically seal the processor-valid 500/500 corpus."""

    from audiolens.audio_fitting import (
        CORPUS_SIZE,
        STRATA,
        corpus_artifact,
        corpus_config_digest,
        corpus_paths,
        validate_corpus_envelope,
        validate_attempt_ledger_against_source_pool,
    )

    root = pathlib.Path(volume_root)
    config = _corpus_config()
    pool, pool_relative_path, pool_file_sha256 = _load_source_pool(root, config, source_pool_sha256)
    corpus_parent = root / "audio-corpora" / corpus_config_digest(config)
    if corpus_parent.exists() and not corpus_parent.is_dir():
        raise RuntimeError(f"corpus root exists but is not a directory: {corpus_parent}")
    existing = sorted(corpus_parent.glob("*/envelope.json"))
    if existing:
        if len(existing) != 1:
            raise RuntimeError(
                "multiple sealed corpora exist for this immutable config; select one explicitly in later gates"
            )
        envelope, rows, ledger, artifact = _load_corpus(
            root, config, existing[0].parent.name, require_files=True
        )
        if (
            envelope["source_pool_sha256"] != pool["source_pool_sha256"]
            or envelope["source_pool_path"] != pool_relative_path
            or envelope["source_pool_file_sha256"] != pool_file_sha256
        ):
            raise RuntimeError("sealed corpus binds a different source pool")
        return artifact

    loader = _default_source_loader if source_loader is None else source_loader
    load_processor = _default_processor_loader if processor_loader is None else processor_loader
    processor = load_processor()
    rows, ledger = _select_corpus(
        pool,
        config,
        processor,
        loader,
        volume_root=root,
    )
    validate_attempt_ledger_against_source_pool(
        ledger,
        pool,
        corpus_config=config,
    )
    envelope = _build_corpus_envelope(
        config,
        pool["source_pool_sha256"],
        pool_relative_path,
        pool_file_sha256,
        rows,
        ledger,
    )
    validate_corpus_envelope(
        envelope,
        rows,
        ledger,
        volume_root=root,
        require_files=True,
    )
    paths = corpus_paths(config, envelope["ordered_corpus_sha256"])
    _write_immutable_bytes(root / paths["rows"], _canonical_jsonl_bytes(rows))
    _write_immutable_bytes(root / paths["attempt_ledger"], _canonical_jsonl_bytes(ledger))
    _write_immutable_bytes(root / paths["envelope"], _canonical_bytes(envelope))
    validate_corpus_envelope(
        _load_json(root / paths["envelope"], "sealed corpus envelope"),
        _load_jsonl(
            root / paths["rows"],
            "sealed corpus rows",
            max_rows=CORPUS_SIZE,
        ),
        _load_jsonl(
            root / paths["attempt_ledger"],
            "sealed attempt ledger",
            max_rows=(len(STRATA) * config["selection"]["max_attempts_per_stratum"]),
        ),
        volume_root=root,
        require_files=True,
    )
    artifact = corpus_artifact(envelope, rows, ledger)
    if commit is not None:
        commit()
    return artifact


def _replay_audio_selection_impl(
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    source_pool_sha256: str | None = None,
    ordered_corpus_sha256: str | None = None,
    source_loader: Callable[..., Any] | None = None,
    processor_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Replay metadata ranking and selection on fresh ephemeral storage."""

    from audiolens.audio_fitting import validate_corpus_envelope, validate_restored_row
    from audiolens.audio_fitting import validate_attempt_ledger_against_source_pool
    from audiolens.models import AudioFitContractError

    root = pathlib.Path(volume_root)
    config = _corpus_config()
    (
        committed_pool,
        committed_pool_path,
        committed_pool_file_sha256,
    ) = _load_source_pool(root, config, source_pool_sha256)
    envelope, committed_rows, committed_ledger, _artifact = _load_corpus(
        root, config, ordered_corpus_sha256, require_files=True
    )
    if (
        envelope["source_pool_sha256"] != committed_pool["source_pool_sha256"]
        or envelope["source_pool_path"] != committed_pool_path
        or envelope["source_pool_file_sha256"] != committed_pool_file_sha256
    ):
        raise AudioFitContractError(
            "selected replay source pool differs from the sealed corpus binding"
        )
    loader = _default_source_loader if source_loader is None else source_loader
    fresh_metadata = _collect_source_metadata(loader)
    fresh_pool = _source_pool_record(config, fresh_metadata)
    if fresh_pool != committed_pool:
        raise AudioFitContractError("fresh metadata pool/ranking does not replay exactly")
    load_processor = _default_processor_loader if processor_loader is None else processor_loader
    processor = load_processor()
    with tempfile.TemporaryDirectory(prefix="audiolens-selection-replay-") as temporary:
        replay_root = pathlib.Path(temporary)
        replay_rows, replay_ledger = _select_corpus(
            fresh_pool,
            config,
            processor,
            loader,
            volume_root=replay_root,
        )
        validate_attempt_ledger_against_source_pool(
            replay_ledger,
            fresh_pool,
            corpus_config=config,
        )
        replay_envelope = _build_corpus_envelope(
            config,
            fresh_pool["source_pool_sha256"],
            committed_pool_path,
            committed_pool_file_sha256,
            replay_rows,
            replay_ledger,
        )
        validate_corpus_envelope(
            replay_envelope,
            replay_rows,
            replay_ledger,
            volume_root=replay_root,
            require_files=True,
        )
        if replay_envelope != envelope:
            raise AudioFitContractError("fresh selection envelope does not replay exactly")
        if replay_ledger != committed_ledger:
            raise AudioFitContractError("fresh bounded attempt ledger does not replay exactly")
        for expected, restored in zip(committed_rows, replay_rows, strict=True):
            validate_restored_row(expected, restored)
    return {
        "source_pool_sha256": fresh_pool["source_pool_sha256"],
        "ordered_corpus_sha256": envelope["ordered_corpus_sha256"],
        "row_count": envelope["row_count"],
        "attempt_count": envelope["attempt_count"],
        "replayed": True,
    }


def _restore_audio_sources_impl(
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    ordered_corpus_sha256: str | None = None,
    source_loader: Callable[..., Any] | None = None,
    commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Restore selected public FLACs with dataset token=False and verify identity."""

    from audiolens.audio_fitting import STRATA, validate_corpus_rows
    from audiolens.fitting import sha256_bytes
    from audiolens.models import AudioFitContractError

    root = pathlib.Path(volume_root)
    config = _corpus_config()
    envelope, rows, _ledger, _artifact = _load_corpus(
        root, config, ordered_corpus_sha256, require_files=False
    )
    loader = _default_source_loader if source_loader is None else source_loader
    restored = 0
    for config_name, split in STRATA:
        source = loader(config_name, split, metadata_only=False)
        get_source = _source_accessor(source)
        for row in rows:
            if row["config"] != config_name or row["split"] != split:
                continue
            candidate = get_source(row["source_id"])
            metadata = _source_metadata(candidate, config_name, split)
            expected_metadata = {
                key: row[key]
                for key in (
                    "dataset",
                    "revision",
                    "config",
                    "split",
                    "source_id",
                    "speaker_id",
                    "chapter_id",
                    "transcript",
                )
            }
            if metadata != expected_metadata:
                raise AudioFitContractError(
                    f"restored source metadata changed for {row['source_id']}"
                )
            blob = _audio_blob(candidate)
            if sha256_bytes(blob) != row["audio_sha256"]:
                raise AudioFitContractError(f"restored source audio changed for {row['source_id']}")
            destination = root / row["volume_path"]
            if not destination.is_file() or destination.read_bytes() != blob:
                if destination.exists() and not destination.is_file():
                    raise AudioFitContractError(
                        f"audio content path is not a file at {destination}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f"{destination.name}.restore.{os.getpid()}")
                temporary.write_bytes(blob)
                os.replace(temporary, destination)
            restored += 1
    validate_corpus_rows(rows, volume_root=root, require_files=True)
    if commit is not None:
        commit()
    return {
        "ordered_corpus_sha256": envelope["ordered_corpus_sha256"],
        "row_count": len(rows),
        "restored_source_rows": restored,
        "dataset_token": False,
    }


def _fit_context(
    volume_root: str | pathlib.Path,
    ordered_corpus_sha256: str | None,
    *,
    require_files: bool = True,
) -> dict[str, Any]:
    from audiolens.audio_fitting import (
        build_fit_config,
        checkpoint_identity,
        fit_config_digest,
        run_paths,
    )

    root = pathlib.Path(volume_root)
    corpus_config = _corpus_config()
    envelope, rows, ledger, corpus = _load_corpus(
        root,
        corpus_config,
        ordered_corpus_sha256,
        require_files=require_files,
    )
    source_digest, lock_sha256 = _required_source_identity()
    fit_config = build_fit_config(
        corpus,
        _profile(),
        source_digest=source_digest,
        lock_sha256=lock_sha256,
        runtime=_runtime_config(),
    )
    paths = run_paths(fit_config)
    identity = checkpoint_identity(fit_config, paths)
    return {
        "root": root,
        "envelope": envelope,
        "rows": rows,
        "ledger": ledger,
        "corpus": corpus,
        "config": fit_config,
        "fit_config_sha256": fit_config_digest(fit_config),
        "paths": paths,
        "identity": identity,
    }


def _require_gates(
    context: Mapping[str, Any],
    gates: Sequence[str],
) -> dict[str, dict[str, Any]]:
    from audiolens.audio_fitting import (
        REQUIRED_GATES,
        gate_artifact,
        gate_path,
        validate_gate_artifact,
    )
    from audiolens.models import AudioFitContractError

    required = tuple(gates)
    if required != REQUIRED_GATES[: len(required)]:
        raise AudioFitContractError(
            f"gate requirement is not an exact protocol prefix: {required!r}"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for gate in required:
        path = context["root"] / gate_path(context["config"], gate)
        artifact = gate_artifact(path, context["config"], gate)
        artifacts[gate] = validate_gate_artifact(
            artifact,
            path,
            context["config"],
            gate,
        )
    return artifacts


def _write_gate(
    context: Mapping[str, Any],
    gate: str,
) -> dict[str, Any]:
    from audiolens.audio_fitting import (
        REQUIRED_GATES,
        gate_artifact,
        gate_path,
        gate_record,
        validate_gate_artifact,
    )
    from audiolens.models import AudioFitContractError

    if gate not in REQUIRED_GATES:
        raise AudioFitContractError(f"unsupported audio-fit gate {gate!r}")
    gate_index = REQUIRED_GATES.index(gate)
    _require_gates(context, REQUIRED_GATES[:gate_index])
    path = context["root"] / gate_path(context["config"], gate)
    _write_immutable_bytes(path, _canonical_bytes(gate_record(context["config"], gate)))
    artifact = gate_artifact(path, context["config"], gate)
    return validate_gate_artifact(
        artifact,
        path,
        context["config"],
        gate,
    )


def _pending_run_record(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": context["config"]["schema_version"],
        "kind": "source_bound_audio_jlens_run",
        "status": "pending",
        "fit_config_sha256": context["fit_config_sha256"],
        "config": context["config"],
        "paths": context["paths"],
        "corpus": context["corpus"],
        "checkpoint_identity": context["identity"],
    }


def _ensure_run_record(context: Mapping[str, Any]) -> dict[str, Any]:
    from audiolens.audio_fitting import validate_completed_run
    from audiolens.models import AudioFitContractError

    root = context["root"]
    manifest_path = root / context["paths"]["manifest"]
    pending = _pending_run_record(context)
    if not manifest_path.exists():
        _write_immutable_bytes(manifest_path, _canonical_bytes(pending))
        return pending
    record = _load_json(manifest_path, "audio fit run")
    if record.get("status") == "complete":
        validated = validate_completed_run(record, volume_root=root)
        expected_bindings = {
            "fit_config_sha256": context["fit_config_sha256"],
            "config": context["config"],
            "paths": context["paths"],
            "corpus": context["corpus"],
            "checkpoint_identity": context["identity"],
        }
        for key, expected in expected_bindings.items():
            if validated.get(key) != expected:
                raise AudioFitContractError(f"completed run {key} differs from current context")
        return validated
    if record != pending:
        raise AudioFitContractError(f"pending run identity changed at {manifest_path}")
    return record


def _prefix_snapshot_candidates(context: Mapping[str, Any]) -> list[pathlib.Path]:
    from audiolens.models import AudioFitContractError

    directory = context["root"] / context["paths"]["snapshot_dir"]
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise AudioFitContractError(
            f"prefix snapshot path exists but is not a directory: {directory}"
        )
    return sorted(path for path in directory.glob("*.pt") if path.is_file())


def _stamp_checkpoint_state(
    state: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    from audiolens.audio_fitting import validate_checkpoint_identity
    from audiolens.models import AudioFitContractError

    expected = validate_checkpoint_identity(identity)
    stamped = dict(state)
    for key in ("fit_config_sha256", "ordered_corpus_sha256"):
        value = expected[key]
        if key in stamped and stamped[key] != value:
            raise AudioFitContractError(f"{label} {key} conflicts with immutable identity")
        stamped[key] = value
    return stamped


def _stamp_checkpoint_identity(
    path: pathlib.Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from audiolens.models import AudioFitContractError

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AudioFitContractError(f"invalid fit checkpoint at {path}") from exc
    if not isinstance(loaded, Mapping):
        raise AudioFitContractError(f"fit checkpoint root at {path} is not an object")
    state = _stamp_checkpoint_state(loaded, identity, label="checkpoint")
    temporary = path.with_name(f"{path.name}.stamp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return state


@contextlib.contextmanager
def _stamped_jlens_checkpoint_writer(
    identity: Mapping[str, Any],
) -> Iterator[None]:
    """Bind every stock periodic checkpoint write to the immutable run."""

    import jlens.fitting as jlens_fitting

    from audiolens.models import AudioFitContractError

    original = getattr(jlens_fitting, "_atomic_save", None)
    if not callable(original):
        raise AudioFitContractError("pinned JLens has no callable periodic checkpoint writer")

    def write_stamped_checkpoint(obj: Any, path: str) -> Any:
        if not isinstance(obj, Mapping):
            raise AudioFitContractError("stock JLens checkpoint state is not an object")
        return original(
            _stamp_checkpoint_state(obj, identity, label="stock JLens checkpoint"),
            path,
        )

    jlens_fitting._atomic_save = write_stamped_checkpoint
    try:
        yield
    finally:
        jlens_fitting._atomic_save = original


def _ensure_prefix_snapshot(
    context: Mapping[str, Any], current_count: int
) -> dict[str, Any] | None:
    from audiolens.audio_fitting import (
        PREFIX_COUNT,
        checkpoint_artifact,
        validate_checkpoint_artifact,
    )
    from audiolens.models import AudioFitContractError

    candidates = _prefix_snapshot_candidates(context)
    identity = context["identity"]
    if current_count < PREFIX_COUNT:
        if candidates:
            raise AudioFitContractError(
                "prefix snapshot exists before the working checkpoint reached 500"
            )
        return None
    if current_count == PREFIX_COUNT:
        working = context["root"] / context["paths"]["checkpoint"]
        expected = checkpoint_artifact(working, identity, PREFIX_COUNT)
        destination = context["root"] / expected["relative_path"]
        if candidates and candidates != [destination]:
            raise AudioFitContractError("unexpected immutable prefix snapshot set")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
            shutil.copyfile(working, temporary)
            os.replace(temporary, destination)
        validate_checkpoint_artifact(expected, destination, identity, PREFIX_COUNT)
        return expected
    if len(candidates) != 1:
        raise AudioFitContractError(
            "working checkpoint passed 500 without exactly one immutable prefix snapshot"
        )
    artifact = checkpoint_artifact(candidates[0], identity, PREFIX_COUNT)
    expected_path = context["root"] / artifact["relative_path"]
    if candidates[0] != expected_path:
        raise AudioFitContractError(
            "immutable prefix snapshot is not at its content-addressed path"
        )
    validate_checkpoint_artifact(artifact, candidates[0], identity, PREFIX_COUNT)
    return artifact


def _prove_reordered_rejection(context: Mapping[str, Any]) -> None:
    from audiolens.audio_fitting import validate_corpus_envelope
    from audiolens.models import AudioFitContractError

    reordered = list(context["rows"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    try:
        validate_corpus_envelope(context["envelope"], reordered, context["ledger"])
    except AudioFitContractError:
        return
    raise AssertionError("reordered corpus unexpectedly passed pre-allocation validation")


def _preflight_audio_fit_impl(
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    ordered_corpus_sha256: str | None = None,
    prove_reordered_rejection: bool = False,
    commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from audiolens.audio_fitting import CORPUS_SIZE, validate_checkpoint_state

    context = _fit_context(volume_root, ordered_corpus_sha256)
    record = _ensure_run_record(context)
    if record.get("status") == "complete":
        return {
            "status": "complete",
            "current_count": CORPUS_SIZE,
            "context": context,
            "record": record,
        }
    checkpoint = context["root"] / context["paths"]["checkpoint"]
    current_count = 0
    if checkpoint.exists():
        state = validate_checkpoint_state(
            checkpoint,
            context["identity"],
            maximum_count=CORPUS_SIZE,
        )
        current_count = state["n_done"]
    prefix = _ensure_prefix_snapshot(context, current_count)
    if prove_reordered_rejection:
        _prove_reordered_rejection(context)
    if commit is not None:
        commit()
    return {
        "status": "pending",
        "current_count": current_count,
        "prefix_snapshot": prefix,
        "context": context,
        "record": record,
    }


def _processor_replay_impl(
    context: Mapping[str, Any],
    *,
    processor_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    from audiolens.audio_fitting import input_ids_sha256
    from audiolens.models import AudioFitContractError, prepare_audio

    load_processor = _default_processor_loader if processor_loader is None else processor_loader
    processor = load_processor()
    rows = context["rows"]
    for index, row in enumerate(rows):
        prepared = prepare_audio(
            processor,
            context["root"] / row["volume_path"],
            profile=_profile(),
        )
        processor_ids = prepared.input_ids
        observed = {
            "processor_input_ids_sha256": input_ids_sha256(processor_ids),
            "fit_input_ids_sha256": input_ids_sha256(processor_ids[:, : prepared.layout.stop]),
            "audio_start": prepared.layout.audio_start,
            "n_audio_tokens": prepared.layout.n_audio_tokens,
            "processor_seq_len": int(processor_ids.shape[1]),
            "sliced_seq_len": prepared.layout.stop,
            "n_valid_positions": prepared.layout.n_valid_positions,
        }
        expected = {key: row[key] for key in observed}
        if observed != expected:
            raise AudioFitContractError(
                f"processor replay changed corpus row {index}: "
                f"expected={expected}, observed={observed}"
            )
        if (index + 1) % 25 == 0:
            print(
                f"processor replay validated {index + 1}/{len(rows)} rows",
                flush=True,
            )
    return {
        "row_count": len(rows),
        "processor_replayed": True,
    }


def _selected_gradient_rows(target, sources, valid_positions, dimensions):
    import torch

    per_source: list[list[Any]] = [[] for _ in sources]
    for index, dimension in enumerate(dimensions):
        cotangent = torch.zeros_like(target)
        cotangent[0, valid_positions, dimension] = 1.0
        gradients = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=cotangent,
            retain_graph=index < len(dimensions) - 1,
        )
        for result, gradient in zip(per_source, gradients, strict=True):
            result.append(gradient[0, valid_positions].float().mean(0).detach().cpu())
    return [torch.stack(result) for result in per_source]


def _batch_diagnostics(actual, expected) -> dict[str, float]:
    import torch

    actual_float = actual.float()
    expected_float = expected.float()
    close = torch.isclose(actual_float, expected_float, rtol=1e-2, atol=1e-2)
    denominator = float(torch.linalg.vector_norm(expected_float))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("replay reference activation has nonfinite/zero norm")
    mismatch_fraction = float(1.0 - close.float().mean())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_float.flatten(), expected_float.flatten(), dim=0
        )
    )
    relative_l2 = float(torch.linalg.vector_norm(actual_float - expected_float) / denominator)
    if not all(math.isfinite(value) for value in (mismatch_fraction, cosine, relative_l2)):
        raise RuntimeError("batched replay diagnostics contain nonfinite values")
    return {
        "mismatch_fraction": mismatch_fraction,
        "cosine": cosine,
        "relative_l2": relative_l2,
    }


def _validate_batched_replay_bounds(
    batch: Mapping[str, Mapping[str, float]],
) -> tuple[float, float]:
    """Apply mutually consistent direction and relative-error bf16 bounds."""

    worst_cosine = min(value["cosine"] for value in batch.values())
    worst_relative_l2 = max(value["relative_l2"] for value in batch.values())
    if worst_cosine < BF16_BATCH_COSINE_MIN or worst_relative_l2 > BF16_BATCH_RELATIVE_L2_MAX:
        raise RuntimeError(
            "batched prepared replay exceeds bf16 parity bounds: "
            f"cosine={worst_cosine}, relative_l2={worst_relative_l2}"
        )
    return worst_cosine, worst_relative_l2


def _validate_one_prepared_replay(runtime: Any, path: str) -> dict[str, Any]:
    import torch
    from jlens.hooks import ActivationRecorder

    profile = runtime.profile
    audio_model = runtime.audio_lens_model
    layers = [*profile.source_layers, profile.target_layer]
    with torch.no_grad(), ActivationRecorder(runtime.layers, at=layers) as full_recorder:
        replay_ids = audio_model.encode(path)
    full_activations = {
        layer: full_recorder.activations[layer].detach().clone() for layer in layers
    }
    prepared = runtime.prepare_audio(path)
    with torch.no_grad(), ActivationRecorder(runtime.layers, at=layers) as replay_recorder:
        audio_model.forward(replay_ids)
    for layer in layers:
        torch.testing.assert_close(
            replay_recorder.activations[layer],
            full_activations[layer],
            rtol=1e-2,
            atol=1e-2,
        )
    full_logits = runtime.unembed(full_activations[profile.target_layer].float())
    replay_logits = runtime.unembed(replay_recorder.activations[profile.target_layer].float())
    torch.testing.assert_close(replay_logits, full_logits, rtol=1e-2, atol=5e-2)

    batch_size = profile.dimension_batch_size
    with torch.no_grad(), ActivationRecorder(runtime.layers, at=layers) as batch_recorder:
        audio_model.forward(replay_ids.expand(batch_size, -1))
    batch = {
        str(layer): _batch_diagnostics(
            batch_recorder.activations[layer][0:1],
            replay_recorder.activations[layer],
        )
        for layer in layers
    }
    if any(
        not math.isfinite(metric)
        for diagnostics in batch.values()
        for metric in diagnostics.values()
    ):
        raise RuntimeError("batched prepared replay produced nonfinite metrics")
    worst_cosine, worst_relative_l2 = _validate_batched_replay_bounds(batch)
    del batch_recorder, replay_recorder, full_activations

    sources = list(
        dict.fromkeys((profile.source_layers[0], profile.read_layer, profile.read_layers[-1]))
    )
    record_at = [*sources, profile.target_layer]
    valid = prepared.layout.valid_mask.nonzero(as_tuple=True)[0].to(prepared.audio_positions.device)
    if valid.numel() == 0:
        raise RuntimeError("prepared replay has no stock-valid positions")
    dimensions = min(8, profile.d_model)
    selected_dimensions = (
        torch.linspace(
            0,
            profile.d_model - 1,
            dimensions,
            device=valid.device,
        )
        .long()
        .tolist()
    )
    with (
        torch.enable_grad(),
        ActivationRecorder(
            runtime.layers,
            at=record_at,
            start_graph_at=profile.source_layers[0],
        ) as recorder,
    ):
        runtime.forward_audio(prepared)
        full_rows = _selected_gradient_rows(
            recorder.activations[profile.target_layer],
            [recorder.activations[layer] for layer in sources],
            valid,
            selected_dimensions,
        )
    with (
        torch.enable_grad(),
        ActivationRecorder(
            runtime.layers,
            at=record_at,
            start_graph_at=profile.source_layers[0],
        ) as recorder,
    ):
        audio_model.forward(replay_ids)
        replay_rows = _selected_gradient_rows(
            recorder.activations[profile.target_layer],
            [recorder.activations[layer] for layer in sources],
            valid,
            selected_dimensions,
        )
    gradient: dict[str, dict[str, float]] = {}
    for layer, actual, expected in zip(sources, replay_rows, full_rows, strict=True):
        denominator = float(torch.linalg.vector_norm(expected))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                f"prepared gradient replay layer {layer} has nonfinite/zero reference norm"
            )
        cosine = float(
            torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        )
        relative_l2 = float(torch.linalg.vector_norm(actual - expected) / denominator)
        if not math.isfinite(cosine) or not math.isfinite(relative_l2):
            raise RuntimeError(f"prepared gradient replay layer {layer} produced nonfinite metrics")
        if cosine < 0.999 or relative_l2 > 0.02:
            raise RuntimeError(
                f"prepared gradient replay mismatch at layer {layer}: "
                f"cosine={cosine}, relative_l2={relative_l2}"
            )
        gradient[str(layer)] = {
            "cosine": cosine,
            "relative_l2": relative_l2,
        }
    return {
        "batch_size": batch_size,
        "worst_batch_cosine": worst_cosine,
        "worst_batch_relative_l2": worst_relative_l2,
        "gradient": gradient,
    }


def _representative_replay_parity(
    runtime: Any,
    rows: Sequence[Mapping[str, Any]],
    volume_root: pathlib.Path,
) -> dict[str, Any]:
    representatives = (rows[0], rows[1])
    if [row["config"] for row in representatives] != ["clean", "other"]:
        raise RuntimeError("representative replay rows are not first clean/other")
    evidence: dict[str, Any] = {}
    for row in representatives:
        path = str(volume_root / row["volume_path"])
        evidence[row["config"]] = {
            "pair_id": row["pair_id"],
            **_validate_one_prepared_replay(runtime, path),
        }
    return evidence


def _atomic_copy_content(source: pathlib.Path, destination: pathlib.Path) -> None:
    if destination.exists():
        from audiolens.fitting import sha256_file

        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"content-addressed destination differs at {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _save_content_addressed_lens(
    lens: Any,
    context: Mapping[str, Any],
    final_checkpoint_path: pathlib.Path,
) -> dict[str, Any]:
    import torch

    from audiolens.audio_fitting import lens_artifact, validate_lens_artifact

    with tempfile.TemporaryDirectory(prefix="audiolens-final-lens-") as temporary:
        candidate = pathlib.Path(temporary) / "lens.pt"
        lens.save(str(candidate), dtype=torch.float16)
        artifact = lens_artifact(candidate, context["identity"], final_checkpoint_path)
        destination = context["root"] / artifact["relative_path"]
        _atomic_copy_content(candidate, destination)
    validate_lens_artifact(
        artifact,
        destination,
        context["identity"],
        final_checkpoint_path,
    )
    return artifact


def _complete_audio_run(
    context: Mapping[str, Any],
    *,
    commit: Callable[[], None] | None,
) -> dict[str, Any]:
    from audiolens.audio_fitting import (
        COMPLETED_RUN_KIND,
        CORPUS_SIZE,
        PREFIX_COUNT,
        SCHEMA_VERSION,
        checkpoint_artifact,
        REQUIRED_GATES,
        stability_artifact,
        stability_from_checkpoints,
        validate_completed_run,
        validate_stability_artifact,
    )
    from audiolens.fitting import lens_from_fit_checkpoint

    root = context["root"]
    identity = context["identity"]
    gate_artifacts = _require_gates(context, REQUIRED_GATES)
    working = root / context["paths"]["checkpoint"]
    final_artifact = checkpoint_artifact(working, identity, CORPUS_SIZE)
    prefix = _ensure_prefix_snapshot(context, CORPUS_SIZE)
    if prefix is None:
        raise AssertionError("completed fit has no prefix snapshot")
    prefix_path = root / prefix["relative_path"]
    report = stability_from_checkpoints(prefix_path, working, identity)
    report_artifact = stability_artifact(report, identity)
    report_path = root / report_artifact["relative_path"]
    _write_immutable_bytes(report_path, _canonical_bytes(report))
    validate_stability_artifact(report_artifact, identity, path=report_path)
    lens = lens_from_fit_checkpoint(working, CORPUS_SIZE, profile=_profile())
    lens_record = _save_content_addressed_lens(lens, context, working)
    del lens
    if commit is not None:
        commit()
    completed = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPLETED_RUN_KIND,
        "status": "complete",
        "fit_config_sha256": context["fit_config_sha256"],
        "config": context["config"],
        "paths": context["paths"],
        "corpus": context["corpus"],
        "checkpoint_identity": identity,
        "prefix_snapshot": checkpoint_artifact(prefix_path, identity, PREFIX_COUNT),
        "checkpoint": final_artifact,
        "stability": report_artifact,
        "lens": lens_record,
        "gates": gate_artifacts,
    }
    validated = validate_completed_run(completed, volume_root=root)
    manifest_path = root / context["paths"]["manifest"]
    temporary = manifest_path.with_name(f"{manifest_path.name}.complete.{os.getpid()}")
    temporary.write_bytes(_canonical_bytes(validated))
    os.replace(temporary, manifest_path)
    if commit is not None:
        commit()
    return validated


def _fit_audio_lens_impl(
    *,
    requested_prefix: int,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    ordered_corpus_sha256: str | None = None,
    model_loader: Callable[[], Any] | None = None,
    commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resume stock JLens only after all stale state has passed CPU-safe checks."""

    import jlens

    from audiolens.audio_fitting import (
        CORPUS_SIZE,
        PREFIX_COUNT,
        REQUIRED_GATES,
        validate_checkpoint_state,
    )
    from audiolens.fitting import validate_lens
    from audiolens.models import AudioFitContractError

    allowed_prefixes = {10, 20, CORPUS_SIZE}
    if requested_prefix not in allowed_prefixes:
        raise ValueError(
            f"requested prefix must be one of {sorted(allowed_prefixes)}, got {requested_prefix}"
        )
    preflight = _preflight_audio_fit_impl(
        volume_root=volume_root,
        ordered_corpus_sha256=ordered_corpus_sha256,
        commit=commit,
    )
    context = preflight["context"]
    required_gates = REQUIRED_GATES[:4] if requested_prefix in {10, 20} else REQUIRED_GATES
    _require_gates(context, required_gates)
    if preflight["status"] == "complete":
        return preflight["record"]
    current_count = preflight["current_count"]
    if current_count > requested_prefix:
        return {
            "status": "pending",
            "fit_config_sha256": context["fit_config_sha256"],
            "current_count": current_count,
            "requested_prefix": requested_prefix,
            "already_past_prefix": True,
        }
    if current_count == CORPUS_SIZE:
        return _complete_audio_run(context, commit=commit)
    if current_count == requested_prefix:
        return {
            "status": "pending",
            "fit_config_sha256": context["fit_config_sha256"],
            "current_count": current_count,
            "requested_prefix": requested_prefix,
            "already_at_prefix": True,
        }

    loader = _default_model_loader if model_loader is None else model_loader
    runtime = loader()
    profile = runtime.profile
    fit = context["config"]["fit"]
    if (
        list(profile.source_layers) != fit["source_layers"]
        or profile.target_layer != fit["target_layer"]
        or profile.skip_first != fit["skip_first"]
        or profile.d_model != fit["d_model"]
        or profile.dimension_batch_size != fit["dimension_batch_size"]
    ):
        raise AudioFitContractError("loaded model geometry differs from immutable fit config")

    checkpoint = context["root"] / context["paths"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    prompts = [str(context["root"] / row["volume_path"]) for row in context["rows"]]
    milestones: list[int]
    if requested_prefix == CORPUS_SIZE and current_count < PREFIX_COUNT:
        milestones = [PREFIX_COUNT, CORPUS_SIZE]
    else:
        milestones = [requested_prefix]
    started = time.monotonic()
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for milestone in milestones:
        if current_count >= milestone:
            continue
        # Stock JLens atomically replaces the working file every five successes.
        # After a complete requested milestone returns, the file is atomically
        # bound to this run below before structural validation and commit.
        with _stamped_jlens_checkpoint_writer(context["identity"]):
            lens = jlens.fit(
                runtime.audio_lens_model,
                prompts=prompts[:milestone],
                source_layers=fit["source_layers"],
                target_layer=fit["target_layer"],
                dim_batch=fit["dimension_batch_size"],
                max_seq_len=fit["max_sequence_length"],
                skip_first=fit["skip_first"],
                checkpoint_path=str(checkpoint),
                checkpoint_every=fit["checkpoint_every"],
                resume=fit["resume"],
            )
        validate_lens(lens, milestone, profile=profile)
        _stamp_checkpoint_identity(checkpoint, context["identity"])
        state = validate_checkpoint_state(
            checkpoint,
            context["identity"],
            maximum_count=milestone,
            exact_count=milestone,
        )
        if state["n_done"] != milestone or state["next_idx"] != milestone:
            raise AudioFitContractError(
                f"stock JLens checkpoint did not reach exact prefix {milestone}"
            )
        current_count = milestone
        if milestone == PREFIX_COUNT:
            snapshot = _ensure_prefix_snapshot(context, current_count)
            if snapshot is None:
                raise AssertionError("failed to bind immutable 500-example snapshot")
        if commit is not None:
            commit()
    elapsed = time.monotonic() - started
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    if current_count == CORPUS_SIZE:
        return _complete_audio_run(context, commit=commit)
    return {
        "status": "pending",
        "fit_config_sha256": context["fit_config_sha256"],
        "current_count": current_count,
        "requested_prefix": requested_prefix,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "projected_full_seconds": elapsed
        * CORPUS_SIZE
        / max(1, requested_prefix - preflight["current_count"]),
    }


@_modal_cpu_function(timeout=SOURCE_TIMEOUT_SECONDS)
def rank_audio_source() -> str:
    result = _rank_audio_source_impl(commit=_commit_volume)
    return json.dumps(result, indent=2, sort_keys=True)


@_modal_cpu_function(timeout=PROCESSOR_TIMEOUT_SECONDS, model_secret=True)
def stage_audio_corpus(source_pool_sha256: str = "") -> str:
    result = _stage_audio_corpus_impl(
        source_pool_sha256=source_pool_sha256 or None,
        commit=_commit_volume,
    )
    return json.dumps(result, indent=2, sort_keys=True)


@_modal_cpu_function(timeout=PROCESSOR_TIMEOUT_SECONDS, model_secret=True)
def replay_audio_selection(source_pool_sha256: str = "", ordered_corpus_sha256: str = "") -> str:
    result = _replay_audio_selection_impl(
        source_pool_sha256=source_pool_sha256 or None,
        ordered_corpus_sha256=ordered_corpus_sha256 or None,
    )
    context = _fit_context(
        VOL_MOUNT,
        ordered_corpus_sha256 or result["ordered_corpus_sha256"],
    )
    result["gate"] = _write_gate(context, "selection_replay")
    _commit_volume()
    return json.dumps(result, indent=2, sort_keys=True)


@_modal_cpu_function(timeout=SOURCE_TIMEOUT_SECONDS)
def restore_audio_sources(ordered_corpus_sha256: str = "") -> str:
    context = _fit_context(
        VOL_MOUNT,
        ordered_corpus_sha256 or None,
        require_files=False,
    )
    _require_gates(context, ("selection_replay",))
    result = _restore_audio_sources_impl(
        ordered_corpus_sha256=ordered_corpus_sha256 or None,
        commit=None,
    )
    result["gate"] = _write_gate(context, "source_restore")
    _commit_volume()
    return json.dumps(result, indent=2, sort_keys=True)


@_modal_cpu_function(
    timeout=PROCESSOR_TIMEOUT_SECONDS,
    model_secret=True,
)
def preflight_audio_fit(ordered_corpus_sha256: str = "") -> str:
    result = _preflight_audio_fit_impl(
        ordered_corpus_sha256=ordered_corpus_sha256 or None,
        prove_reordered_rejection=True,
        commit=None,
    )
    context = result["context"]
    _require_gates(
        context,
        ("selection_replay", "source_restore"),
    )
    processor_evidence = _processor_replay_impl(context)
    processor_gate = _write_gate(context, "processor_replay")
    _commit_volume()
    payload = {
        "status": result["status"],
        "current_count": result["current_count"],
        "fit_config_sha256": context["fit_config_sha256"],
        "manifest": context["paths"]["manifest"],
        "reordered_rejection_before_gpu": True,
        "processor_replay": processor_evidence,
        "gate": processor_gate,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


@_modal_gpu_function
def _validate_audio_replay_gpu(ordered_corpus_sha256: str = "") -> str:
    preflight = _preflight_audio_fit_impl(ordered_corpus_sha256=ordered_corpus_sha256 or None)
    _require_gates(
        preflight["context"],
        ("selection_replay", "source_restore", "processor_replay"),
    )
    runtime = _default_model_loader()
    evidence = _representative_replay_parity(
        runtime,
        preflight["context"]["rows"],
        preflight["context"]["root"],
    )
    return json.dumps(evidence, indent=2, sort_keys=True)


@_modal_cpu_function(timeout=FIT_TIMEOUT_SECONDS)
def validate_audio_replay(ordered_corpus_sha256: str = "") -> str:
    preflight = _preflight_audio_fit_impl(
        ordered_corpus_sha256=ordered_corpus_sha256 or None,
        prove_reordered_rejection=True,
        commit=None,
    )
    context = preflight["context"]
    _require_gates(
        context,
        ("selection_replay", "source_restore", "processor_replay"),
    )
    evidence = json.loads(
        _call_deployed_function(
            "_validate_audio_replay_gpu",
            ordered_corpus_sha256=ordered_corpus_sha256,
        )
    )
    gate = _write_gate(context, "decoder_replay")
    _commit_volume()
    return json.dumps(
        {"replay": evidence, "gate": gate},
        indent=2,
        sort_keys=True,
    )


@_modal_gpu_function
def _fit_audio_lens_gpu(
    requested_prefix: int,
    ordered_corpus_sha256: str = "",
) -> str:
    record = _fit_audio_lens_impl(
        requested_prefix=requested_prefix,
        ordered_corpus_sha256=ordered_corpus_sha256 or None,
        commit=_commit_volume,
    )
    return json.dumps(record, indent=2, sort_keys=True)


@_modal_cpu_function(timeout=FIT_TIMEOUT_SECONDS)
def smoke_audio_fit(ordered_corpus_sha256: str = "") -> str:
    """Prove a separately persisted balanced 10-to-20 resume transition."""

    from audiolens.audio_fitting import REQUIRED_GATES, gate_path
    from audiolens.models import AudioFitContractError

    digest = ordered_corpus_sha256 or None
    before = _preflight_audio_fit_impl(
        ordered_corpus_sha256=digest,
        prove_reordered_rejection=True,
        commit=None,
    )
    context = before["context"]
    _require_gates(context, REQUIRED_GATES[:4])
    smoke_path = context["root"] / gate_path(context["config"], "smoke_resume")
    if smoke_path.exists():
        gates = _require_gates(context, REQUIRED_GATES)
        return json.dumps(
            {
                "status": before["status"],
                "current_count": before["current_count"],
                "smoke_already_proven": True,
                "gate": gates["smoke_resume"],
            },
            indent=2,
            sort_keys=True,
        )
    if before["current_count"] >= 20:
        raise AudioFitContractError(
            "checkpoint reached/passed 20 without a valid smoke-resume gate"
        )

    transitions: list[dict[str, Any]] = []
    if before["current_count"] < 10:
        transitions.append(
            json.loads(
                _call_deployed_function(
                    "_fit_audio_lens_gpu",
                    requested_prefix=10,
                    ordered_corpus_sha256=ordered_corpus_sha256,
                )
            )
        )
        _reload_volume()
    middle = _preflight_audio_fit_impl(
        ordered_corpus_sha256=digest,
        prove_reordered_rejection=True,
        commit=None,
    )
    _require_gates(middle["context"], REQUIRED_GATES[:4])
    if middle["current_count"] != 10:
        raise AudioFitContractError(
            "smoke resume requires an exact persisted 10-example checkpoint"
        )
    transitions.append(
        json.loads(
            _call_deployed_function(
                "_fit_audio_lens_gpu",
                requested_prefix=20,
                ordered_corpus_sha256=ordered_corpus_sha256,
            )
        )
    )
    _reload_volume()
    after = _preflight_audio_fit_impl(
        ordered_corpus_sha256=digest,
        prove_reordered_rejection=True,
        commit=None,
    )
    _require_gates(after["context"], REQUIRED_GATES[:4])
    if after["current_count"] != 20:
        raise AudioFitContractError("balanced smoke did not resume to exact prefix 20")
    gate = _write_gate(after["context"], "smoke_resume")
    _commit_volume()
    return json.dumps(
        {
            "status": after["status"],
            "current_count": after["current_count"],
            "first_prefix": 10,
            "resumed_prefix": 20,
            "balanced_clean_other": [10, 10],
            "reordered_rejection_before_gpu": True,
            "transitions": transitions,
            "gate": gate,
        },
        indent=2,
        sort_keys=True,
    )


@_modal_cpu_function(timeout=FIT_TIMEOUT_SECONDS)
def fit_audio_lens(ordered_corpus_sha256: str = "") -> str:
    """CPU-gated idempotent orchestration; allocate H100 only if required."""

    digest = ordered_corpus_sha256 or None
    preflight = _preflight_audio_fit_impl(
        ordered_corpus_sha256=digest,
        prove_reordered_rejection=True,
        commit=_commit_volume,
    )
    from audiolens.audio_fitting import REQUIRED_GATES

    _require_gates(preflight["context"], REQUIRED_GATES)
    if preflight["status"] == "complete":
        return json.dumps(preflight["record"], indent=2, sort_keys=True)
    if preflight["current_count"] == 1_000:
        completed = _complete_audio_run(preflight["context"], commit=_commit_volume)
        return json.dumps(completed, indent=2, sort_keys=True)
    return _call_deployed_function(
        "_fit_audio_lens_gpu",
        requested_prefix=1_000,
        ordered_corpus_sha256=ordered_corpus_sha256,
    )


@_modal_local_entrypoint
def main(
    rank_source_only: bool = False,
    stage_corpus_only: bool = False,
    selection_replay_only: bool = False,
    restore_source_only: bool = False,
    preflight_only: bool = False,
    replay_parity_only: bool = False,
    smoke_only: bool = False,
    fit: bool = False,
    source_pool_sha256: str = "",
    ordered_corpus_sha256: str = "",
):
    actions = {
        "--rank-source-only": rank_source_only,
        "--stage-corpus-only": stage_corpus_only,
        "--selection-replay-only": selection_replay_only,
        "--restore-source-only": restore_source_only,
        "--preflight-only": preflight_only,
        "--replay-parity-only": replay_parity_only,
        "--smoke-only": smoke_only,
        "--fit": fit,
    }
    selected = [name for name, enabled in actions.items() if enabled]
    if len(selected) != 1:
        raise SystemExit("select exactly one staged action: " + ", ".join(actions))
    if rank_source_only:
        print(rank_audio_source.remote())
    elif stage_corpus_only:
        print(stage_audio_corpus.remote(source_pool_sha256=source_pool_sha256))
    elif selection_replay_only:
        print(
            replay_audio_selection.remote(
                source_pool_sha256=source_pool_sha256,
                ordered_corpus_sha256=ordered_corpus_sha256,
            )
        )
    elif restore_source_only:
        print(restore_audio_sources.remote(ordered_corpus_sha256=ordered_corpus_sha256))
    elif preflight_only:
        print(preflight_audio_fit.remote(ordered_corpus_sha256=ordered_corpus_sha256))
    elif replay_parity_only:
        print(validate_audio_replay.remote(ordered_corpus_sha256=ordered_corpus_sha256))
    elif smoke_only:
        print(smoke_audio_fit.remote(ordered_corpus_sha256=ordered_corpus_sha256))
    else:
        print(fit_audio_lens.remote(ordered_corpus_sha256=ordered_corpus_sha256))
