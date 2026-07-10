"""Reproducible mixed text/audio Jacobian-lens fitting helpers.

Generic waveform, corpus, resume, checkpoint, and lens validation live here.
Model-specific processor framing and prepared decoder replay live in
``audiolens.models.gemma4``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from typing import Any

import torch
from .models import (
    DEFAULT_MODEL_PROFILE,
    AudioFitContractError,
    ModelProfile,
    PreparedAudio,
    prepare_audio,
)

MODEL_REVISION = DEFAULT_MODEL_PROFILE.model_revision
LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

FIT_SEED = 20260709

MANIFEST_FIELDS = {
    "dataset",
    "revision",
    "config",
    "split",
    "selection_index",
    "id",
    "speaker_id",
    "chapter_id",
    "transcript",
    "duration_seconds",
    "sampling_rate",
    "audio_sha256",
    "audio_start",
    "n_audio_tokens",
    "sliced_seq_len",
    "n_valid_positions",
    "volume_path",
}




def canonical_json_bytes(value: Any) -> bytes:
    """Stable UTF-8 JSON used for content-addressed experiment identity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    """Digest only immutable inputs; output hashes live outside ``config``."""
    return sha256_bytes(canonical_json_bytes(config))


def atomic_write_json(path: str | pathlib.Path, value: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(tmp, path)


def load_jsonl(path: str | pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_canonical_jsonl(path: str | pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        for row in rows:
            f.write(canonical_json_bytes(row) + b"\n")
    os.replace(tmp, path)


def paired_resume_prefix(
    path: str | pathlib.Path,
    config: dict[str, Any],
    expected_clips: list[str],
) -> set[str]:
    """Read a strict sorted paired prefix, repairing only a torn final write."""
    path = pathlib.Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    valid_bytes = 0
    with open(path, "rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AudioFitContractError(
                    f"paired result row {index} is invalid JSON"
                ) from exc
            validate_paired_result_row(row, config)
            clip = row["clip"]
            if index >= len(expected_clips) or clip != expected_clips[index]:
                raise AudioFitContractError(
                    f"paired result row {index} is {clip}, expected sorted clip "
                    f"{expected_clips[index] if index < len(expected_clips) else '<none>'}"
                )
            if clip in done:
                raise AudioFitContractError(f"duplicate paired result clip {clip}")
            done.add(clip)
            valid_bytes += len(line)
    if valid_bytes < path.stat().st_size:
        with open(path, "r+b") as stream:
            stream.truncate(valid_bytes)
    return done


def describe_audio_file(path: str | pathlib.Path) -> dict[str, Any]:
    """Validate and hash one staged waveform without invoking a model processor."""

    import numpy as np
    import soundfile as sf

    path = pathlib.Path(path)
    if not path.is_file():
        raise AudioFitContractError(f"missing staged audio {path}")
    try:
        info = sf.info(path)
    except sf.LibsndfileError as exc:
        raise AudioFitContractError(f"cannot decode staged audio {path}") from exc
    duration = info.frames / info.samplerate
    if info.samplerate != 16_000 or info.channels != 1:
        raise AudioFitContractError(
            f"{path} must be mono 16 kHz, got {info.channels}ch at {info.samplerate} Hz"
        )
    if not 2.0 <= duration <= 4.0:
        raise AudioFitContractError(f"{path} duration {duration} is out of range")
    audio, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    if sampling_rate != 16_000 or audio.ndim != 1 or not bool(np.isfinite(audio).all()):
        raise AudioFitContractError(f"{path} does not contain finite mono 16 kHz audio")
    return {
        "duration_seconds": round(duration, 6),
        "sampling_rate": int(info.samplerate),
        "audio_sha256": sha256_file(path),
    }


def describe_audio_sample(
    processor: Any,
    path: str | pathlib.Path,
    *,
    profile: ModelProfile = DEFAULT_MODEL_PROFILE,
) -> PreparedAudio:
    """Validate a waveform and attach its model-specific prepared layout."""

    waveform_fields = describe_audio_file(path)
    prepared = prepare_audio(processor, path, profile=profile)
    prepared.manifest_fields = {**waveform_fields, **prepared.manifest_fields}
    return prepared


def audit_audio_manifest_files(
    rows: list[dict[str, Any]],
    processor: Any,
    *,
    profile: ModelProfile = DEFAULT_MODEL_PROFILE,
) -> None:
    """Recompute every derived field without constructing model weights."""

    for index, row in enumerate(rows):
        sample = describe_audio_sample(
            processor, row["volume_path"], profile=profile
        )
        actual = {key: row.get(key) for key in sample.manifest_fields}
        if actual != sample.manifest_fields:
            raise AudioFitContractError(
                f"row {index} descriptor mismatch: manifest={actual}, "
                f"recomputed={sample.manifest_fields}"
            )




def audit_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    profile: ModelProfile = DEFAULT_MODEL_PROFILE,
) -> None:
    """Fail unless rows are the exact fixed 64-clean/64-other fit design."""
    if len(rows) != 128:
        raise AudioFitContractError(f"manifest has {len(rows)} rows, expected 128")
    ids = [str(row.get("id")) for row in rows]
    speakers = [str(row.get("speaker_id")) for row in rows]
    if len(set(ids)) != 128 or len(set(speakers)) != 128:
        raise AudioFitContractError("manifest IDs and speakers must each be unique")
    strata = {("clean", "train.100"): 0, ("other", "train.500"): 0}
    for index, row in enumerate(rows):
        missing = MANIFEST_FIELDS - set(row)
        if missing:
            raise AudioFitContractError(f"row {index} missing fields {sorted(missing)}")
        if row["selection_index"] != index:
            raise AudioFitContractError(
                f"row {index} has selection_index={row['selection_index']}; manifest reordered"
            )
        key = (row["config"], row["split"])
        expected_key = ("clean", "train.100") if index < 64 else ("other", "train.500")
        if key != expected_key:
            raise AudioFitContractError(
                f"row {index} has stratum {key}, expected ordered stratum {expected_key}"
            )
        if key not in strata:
            raise AudioFitContractError(f"row {index} has unexpected stratum {key}")
        strata[key] += 1
        if row["dataset"] != "openslr/librispeech_asr":
            raise AudioFitContractError(f"row {index} has wrong dataset")
        if row["revision"] != LIBRISPEECH_REVISION:
            raise AudioFitContractError(f"row {index} has wrong dataset revision")
        if row["sampling_rate"] != 16_000:
            raise AudioFitContractError(f"row {index} is not 16 kHz")
        duration = float(row["duration_seconds"])
        if not 2.0 <= duration <= 4.0:
            raise AudioFitContractError(f"row {index} duration {duration} is out of range")
        if not str(row["transcript"]).strip():
            raise AudioFitContractError(f"row {index} has an empty transcript")
        if not (0 < int(row["n_valid_positions"]) < int(row["n_audio_tokens"])):
            raise AudioFitContractError(f"row {index} has invalid position counts")
        if int(row["sliced_seq_len"]) > profile.max_sequence_length:
            raise AudioFitContractError(
                f"row {index} exceeds max sequence length "
                f"{profile.max_sequence_length}"
            )
        digest = str(row["audio_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AudioFitContractError(f"row {index} has invalid audio SHA-256")
    if any(count != 64 for count in strata.values()):
        raise AudioFitContractError(f"manifest strata are {strata}, expected 64 each")


def validate_paired_result_row(row: dict[str, Any], config: dict[str, Any]) -> None:
    """Validate one complete content-addressed paired evaluation record."""
    from audiolens import parse_ravdess_name

    expected_lenses = set(config["lenses"])
    expected_layers = {str(layer) for layer in config["read_layers"]}
    expected_clusters = set(config["anchors"]["clusters"])
    topk = int(config["topk"])
    if "curiosity" in expected_clusters:
        raise AudioFitContractError(
            "paired production-anchor results must not contain curiosity"
        )
    clip = row.get("clip")
    meta = row.get("meta")
    if not isinstance(clip, str) or not clip.endswith(".wav"):
        raise AudioFitContractError("paired record has an invalid clip name")
    if not isinstance(meta, dict) or not all(
        isinstance(meta.get(key), str) and meta[key]
        for key in ("actor", "statement", "emotion", "intensity")
    ):
        raise AudioFitContractError(f"{clip}: invalid RAVDESS metadata")
    parsed = parse_ravdess_name(pathlib.Path(clip).stem)
    if parsed is None or meta != parsed:
        raise AudioFitContractError(f"{clip}: metadata does not match the clip name")
    n_audio = row.get("n_audio_tokens")
    seq_len = row.get("seq_len")
    if (
        not isinstance(n_audio, int)
        or isinstance(n_audio, bool)
        or not isinstance(seq_len, int)
        or isinstance(seq_len, bool)
        or not 0 < n_audio <= seq_len
    ):
        raise AudioFitContractError(f"{clip}: invalid audio/sequence lengths")
    readouts = row.get("readouts")
    if not isinstance(readouts, dict) or set(readouts) != expected_lenses:
        raise AudioFitContractError(f"{clip}: incomplete lens pair")
    for readout in readouts.values():
        layers = readout.get("layers") if isinstance(readout, dict) else None
        if not isinstance(layers, dict) or set(layers) != expected_layers:
            raise AudioFitContractError(f"{clip}: layer mismatch")
        for layer in layers.values():
            mass = layer.get("anchor_mass") if isinstance(layer, dict) else None
            if not isinstance(mass, dict) or set(mass) != expected_clusters or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in mass.values()
            ):
                raise AudioFitContractError(f"{clip}: invalid anchor masses")
            top_ids = layer.get("topk_ids")
            top_tokens = layer.get("topk_toks")
            if (
                not isinstance(top_ids, list)
                or len(top_ids) != topk
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in top_ids)
                or not isinstance(top_tokens, list)
                or len(top_tokens) != topk
                or not all(isinstance(value, str) for value in top_tokens)
            ):
                raise AudioFitContractError(f"{clip}: invalid top-k payload")


def lens_from_fit_checkpoint(
    path: str | pathlib.Path,
    expected_count: int,
    *,
    profile: ModelProfile,
):
    """Reconstruct an fp32 :class:`JacobianLens` mean from a fit checkpoint."""
    import jlens

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("n_done") != expected_count or state.get("next_idx") != expected_count:
        raise AudioFitContractError(
            f"checkpoint counts {(state.get('n_done'), state.get('next_idx'))} "
            f"do not equal {expected_count}"
        )
    expected_layers = list(profile.source_layers)
    if state.get("source_layers") != expected_layers:
        raise AudioFitContractError("checkpoint source layers do not match")
    if state.get("target_layer") != profile.target_layer:
        raise AudioFitContractError("checkpoint target layer does not match")
    if state.get("skip_first") != profile.skip_first:
        raise AudioFitContractError("checkpoint skip_first does not match")
    jacobian_sum = state.get("jacobian_sum")
    if not isinstance(jacobian_sum, dict) or set(jacobian_sum) != set(
        profile.source_layers
    ):
        raise AudioFitContractError("checkpoint Jacobian layers do not match")
    expected_shape = (profile.d_model, profile.d_model)
    if any(
        not torch.is_tensor(value)
        or value.shape != expected_shape
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
        for value in jacobian_sum.values()
    ):
        raise AudioFitContractError(
            "checkpoint running sums do not match finite fp32 profile geometry"
        )
    jacobians = {
        layer: value / expected_count for layer, value in jacobian_sum.items()
    }
    return jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=expected_count,
        d_model=profile.d_model,
    )


def fit_checkpoint_metadata(
    path: str | pathlib.Path,
    expected_count: int,
    *,
    profile: ModelProfile,
) -> dict[str, Any]:
    """Validate and summarize a durable fp32 running-sum checkpoint."""
    lens = lens_from_fit_checkpoint(path, expected_count, profile=profile)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "kind": "fp32_running_sum_checkpoint",
        "dtype": "float32",
        "n_done": state["n_done"],
        "next_idx": state["next_idx"],
        "d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "target_layer": state["target_layer"],
        "skip_first": state["skip_first"],
    }


def validate_lens(
    lens: Any,
    expected_count: int,
    *,
    profile: ModelProfile,
) -> None:
    if lens.n_prompts != expected_count:
        raise AudioFitContractError(
            f"lens has n_prompts={lens.n_prompts}, expected {expected_count}"
        )
    expected_layers = list(profile.source_layers)
    if lens.source_layers != expected_layers or lens.d_model != profile.d_model:
        raise AudioFitContractError("lens layers/d_model do not match the model profile")
    expected_shape = (profile.d_model, profile.d_model)
    if set(lens.jacobians) != set(profile.source_layers):
        raise AudioFitContractError("lens Jacobian layers do not match the model profile")
    for layer, value in lens.jacobians.items():
        if (
            not torch.is_tensor(value)
            or value.shape != expected_shape
            or not bool(torch.isfinite(value).all())
        ):
            raise AudioFitContractError(f"lens layer {layer} is invalid")


def validate_runtime_lens_file(
    path: str | pathlib.Path,
    expected_count: int,
    *,
    profile: ModelProfile,
) -> None:
    """Verify the serialized fp16 contract before JLens casts it back to fp32."""
    import jlens

    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("n_prompts") != expected_count
        or state.get("d_model") != profile.d_model
    ):
        raise AudioFitContractError(f"runtime lens metadata is invalid at {path}")
    if state.get("source_layers") != list(profile.source_layers):
        raise AudioFitContractError(f"runtime lens layers are invalid at {path}")
    jacobians = state.get("J")
    expected_shape = (profile.d_model, profile.d_model)
    if (
        not isinstance(jacobians, dict)
        or set(jacobians) != set(profile.source_layers)
        or any(
            not torch.is_tensor(value)
            or value.shape != expected_shape
            or value.dtype != torch.float16
            for value in jacobians.values()
        )
    ):
        raise AudioFitContractError(f"runtime lens is not serialized as fp16 at {path}")
    validate_lens(
        jlens.JacobianLens.load(path), expected_count, profile=profile
    )
