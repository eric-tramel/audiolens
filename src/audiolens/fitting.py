"""Reproducible mixed text/audio Jacobian-lens fitting helpers.

The audio adapter is intentionally Gemma-4-specific.  Gemma replaces repeated
audio placeholder IDs with continuous audio-tower embeddings before entering
``model.language_model``.  JLens only needs derivatives between decoder
residual layers, so we run that tower once, detach the exact decoder-boundary
inputs, and replay the decoder for JLens's expanded gradient batches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import torch

MODEL_REVISION = "70af34e20bd4b7a91f0de6b22675850c43922a03"
LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

FIT_SEED = 20260709
FIT_MAX_SEQ_LEN = 128
FIT_SKIP_FIRST = 16
FIT_DIM_BATCH = 128
FIT_SOURCE_LAYERS = list(range(34))
FIT_TARGET_LAYER = 34

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


class AudioFitContractError(RuntimeError):
    """A non-skippable violation of the pinned Gemma audio-fit contract.

    Stock :func:`jlens.fit` catches ``ValueError`` and skips that prompt, so
    adapter/data contract failures deliberately use a different exception.
    """


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


@dataclass(frozen=True)
class AudioLayout:
    """The audio-only prefix retained for the stock JLens position mask."""

    audio_start: int
    n_audio_tokens: int
    stop: int
    n_valid_positions: int
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DescribedAudioSample:
    """Canonical staged-file descriptor plus its exact processor inputs."""

    manifest_fields: dict[str, Any]
    model_inputs: Any
    layout: AudioLayout


def validate_audio_layout(
    input_ids: torch.Tensor,
    audio_id: int,
    *,
    max_length: int = FIT_MAX_SEQ_LEN,
    skip_first: int = FIT_SKIP_FIRST,
) -> AudioLayout:
    """Validate one contiguous audio span and the stock JLens fit positions.

    The returned ``stop`` is one past the final audio slot.  Removing the
    closing chat tokens is safe only for the causal Gemma configuration,
    which :class:`PreparedAudioLensModel` checks separately.
    """
    from jlens.fitting import valid_position_mask

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise AudioFitContractError(
            f"expected input_ids [1, seq], got {tuple(input_ids.shape)}"
        )
    positions = (input_ids[0] == audio_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise AudioFitContractError("no audio soft-token positions")
    expected = torch.arange(
        int(positions[0]), int(positions[-1]) + 1, device=positions.device
    )
    if not torch.equal(positions, expected):
        raise AudioFitContractError("audio soft-token positions are not contiguous")
    stop = int(positions[-1]) + 1
    if stop > max_length:
        raise AudioFitContractError(
            f"audio prefix has {stop} positions, above max_length={max_length}"
        )
    try:
        valid = valid_position_mask(stop, skip_first=skip_first)
    except ValueError as exc:
        raise AudioFitContractError(str(exc)) from exc
    selected = input_ids[0, :stop][valid.to(input_ids.device)]
    if selected.numel() == 0 or not bool((selected == audio_id).all()):
        raise AudioFitContractError(
            "stock JLens valid-position mask is not exclusively nonempty audio"
        )
    return AudioLayout(
        audio_start=int(positions[0]),
        n_audio_tokens=int(positions.numel()),
        stop=stop,
        n_valid_positions=int(valid.sum()),
        valid_mask=valid,
    )


def describe_audio_sample(processor, path: str | pathlib.Path, audio_id: int) -> DescribedAudioSample:
    """Describe one exact audio file using the same path-based input used by fitting."""
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
    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": str(path)}]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    layout = validate_audio_layout(inputs["input_ids"], audio_id)
    full_len = int(inputs["input_ids"].shape[1])
    if layout.audio_start != 5 or full_len - layout.stop != 3:
        raise AudioFitContractError(
            f"{path} has unexpected framing: audio_start={layout.audio_start}, "
            f"closing_tokens={full_len - layout.stop}"
        )
    return DescribedAudioSample(
        manifest_fields={
            "duration_seconds": round(duration, 6),
            "sampling_rate": int(info.samplerate),
            "audio_sha256": sha256_file(path),
            "audio_start": layout.audio_start,
            "n_audio_tokens": layout.n_audio_tokens,
            "sliced_seq_len": layout.stop,
            "n_valid_positions": layout.n_valid_positions,
        },
        model_inputs=inputs,
        layout=layout,
    )


def audit_audio_manifest_files(rows: list[dict[str, Any]], processor, audio_id: int) -> None:
    """Recompute every derived manifest field from each exact staged file."""
    for index, row in enumerate(rows):
        sample = describe_audio_sample(processor, row["volume_path"], audio_id)
        actual = {key: row.get(key) for key in sample.manifest_fields}
        if actual != sample.manifest_fields:
            raise AudioFitContractError(
                f"row {index} descriptor mismatch: manifest={actual}, "
                f"recomputed={sample.manifest_fields}"
            )


def crop_attention_mapping(
    mapping: dict[str, torch.Tensor | None], stop: int
) -> dict[str, torch.Tensor | None]:
    """Crop Gemma's exact eager full/sliding masks to an audio-only prefix."""
    expected = {"full_attention", "sliding_attention"}
    if set(mapping) != expected:
        raise AudioFitContractError(
            f"unexpected attention mapping keys {sorted(mapping)}; expected {sorted(expected)}"
        )
    cropped: dict[str, torch.Tensor | None] = {}
    for name, value in mapping.items():
        if value is None:
            cropped[name] = None
            continue
        if not torch.is_tensor(value) or value.ndim < 2:
            raise AudioFitContractError(
                f"{name} mask must be None or a Tensor with query/key axes"
            )
        if value.shape[-2] < stop or value.shape[-1] < stop:
            raise AudioFitContractError(
                f"{name} mask {tuple(value.shape)} is shorter than stop={stop}"
            )
        cropped[name] = value[..., :stop, :stop].detach().clone()
    return cropped


def expand_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Zero-copy expand a captured batch-one tensor."""
    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise AudioFitContractError(
            f"captured tensor must have batch dimension 1, got {tuple(tensor.shape)}"
        )
    return tensor.expand(batch_size, *tensor.shape[1:])


@dataclass
class PreparedDecoderInputs:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    per_layer_inputs: torch.Tensor
    attention_mask: dict[str, torch.Tensor | None]
    position_ids: torch.Tensor
    layout: AudioLayout


class PreparedAudioLensModel:
    """Gemma-4 audio-path adapter implementing JLens's ``LensModel`` protocol."""

    def __init__(self, hf, processor):
        import jlens

        self.hf = hf
        self.processor = processor
        self.base = jlens.from_hf(hf, processor.tokenizer, force_bos=False)
        self.n_layers = self.base.n_layers
        self.d_model = self.base.d_model
        self.layers = self.base.layers
        self.tokenizer = self.base.tokenizer
        self._language_model = hf.model.language_model
        if self.base._text_module is not self._language_model:  # noqa: SLF001
            raise AudioFitContractError("JLens and Gemma resolved different text decoders")
        text_config = hf.config.get_text_config()
        if text_config.use_bidirectional_attention is not None:
            raise AudioFitContractError(
                "suffix slicing requires causal-only use_bidirectional_attention=None"
            )
        self.audio_id = int(hf.config.audio_token_id)
        self._prepared: PreparedDecoderInputs | None = None

    @property
    def input_device(self) -> torch.device:
        return self.base.input_device

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.base.unembed(residual)

    def encode(self, path: str, *, max_length: int = FIT_MAX_SEQ_LEN) -> torch.Tensor:
        sample = describe_audio_sample(self.processor, path, self.audio_id)
        inputs = sample.model_inputs.to(self.input_device)
        layout = sample.layout
        if layout.stop > max_length:
            raise AudioFitContractError(
                f"audio prefix has {layout.stop} positions, above max_length={max_length}"
            )

        calls: list[dict[str, Any]] = []

        def capture(_module, _args, kwargs):
            calls.append(dict(kwargs))

        handle = self._language_model.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            with torch.no_grad():
                self.hf.model(**inputs, use_cache=False)
        finally:
            handle.remove()
        if len(calls) != 1:
            raise AudioFitContractError(
                f"expected one language-model call during preparation, got {len(calls)}"
            )
        kwargs = calls[0]
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise AudioFitContractError("prepared decoder call unexpectedly used a cache")
        embeds = kwargs.get("inputs_embeds")
        ple = kwargs.get("per_layer_inputs")
        masks = kwargs.get("attention_mask")
        positions = kwargs.get("position_ids")
        if not torch.is_tensor(embeds) or not torch.is_tensor(ple):
            raise AudioFitContractError("decoder call did not contain Gemma inputs_embeds/PLE")
        if not isinstance(masks, dict) or not torch.is_tensor(positions):
            raise AudioFitContractError("decoder call did not contain eager masks/position_ids")
        stop = layout.stop
        sliced_ids = inputs["input_ids"][:, :stop].detach().clone()
        self._prepared = PreparedDecoderInputs(
            input_ids=sliced_ids,
            inputs_embeds=embeds[:, :stop].detach().clone(),
            per_layer_inputs=ple[:, :stop].detach().clone(),
            attention_mask=crop_attention_mapping(masks, stop),
            position_ids=positions[:, :stop].detach().clone(),
            layout=layout,
        )
        return sliced_ids

    def forward(self, input_ids: torch.Tensor):
        prepared = self._prepared
        if prepared is None:
            raise AudioFitContractError("forward called before encode")
        if input_ids.ndim != 2 or input_ids.shape[1:] != prepared.input_ids.shape[1:]:
            raise AudioFitContractError(
                f"replay IDs shape {tuple(input_ids.shape)} does not match "
                f"prepared {tuple(prepared.input_ids.shape)}"
            )
        expected = prepared.input_ids.expand(input_ids.shape[0], -1)
        if not torch.equal(input_ids, expected):
            raise AudioFitContractError("replay IDs do not match the prepared sample")
        batch = input_ids.shape[0]
        masks = {
            name: None if value is None else expand_batch(value, batch)
            for name, value in prepared.attention_mask.items()
        }
        return self._language_model(
            inputs_embeds=expand_batch(prepared.inputs_embeds, batch),
            per_layer_inputs=expand_batch(prepared.per_layer_inputs, batch),
            attention_mask=masks,
            position_ids=expand_batch(prepared.position_ids, batch),
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )


def audit_manifest_rows(rows: list[dict[str, Any]]) -> None:
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
        if int(row["sliced_seq_len"]) > FIT_MAX_SEQ_LEN:
            raise AudioFitContractError(f"row {index} exceeds max sequence length")
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


def lens_from_fit_checkpoint(path: str | pathlib.Path, expected_count: int):
    """Reconstruct an fp32 :class:`JacobianLens` mean from a fit checkpoint."""
    import jlens

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("n_done") != expected_count or state.get("next_idx") != expected_count:
        raise AudioFitContractError(
            f"checkpoint counts {(state.get('n_done'), state.get('next_idx'))} "
            f"do not equal {expected_count}"
        )
    if state.get("source_layers") != FIT_SOURCE_LAYERS:
        raise AudioFitContractError("checkpoint source layers do not match")
    if state.get("target_layer") != FIT_TARGET_LAYER:
        raise AudioFitContractError("checkpoint target layer does not match")
    if state.get("skip_first") != FIT_SKIP_FIRST:
        raise AudioFitContractError("checkpoint skip_first does not match")
    jacobian_sum = state.get("jacobian_sum")
    if not isinstance(jacobian_sum, dict) or set(jacobian_sum) != set(FIT_SOURCE_LAYERS):
        raise AudioFitContractError("checkpoint Jacobian layers do not match")
    shapes = {tuple(value.shape) for value in jacobian_sum.values()}
    if (
        len(shapes) != 1
        or any(value.dtype != torch.float32 for value in jacobian_sum.values())
        or not all(bool(torch.isfinite(value).all()) for value in jacobian_sum.values())
    ):
        raise AudioFitContractError("checkpoint running sums are not finite fp32 matrices")
    jacobians = {
        layer: value.float() / expected_count
        for layer, value in jacobian_sum.items()
    }
    return jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=expected_count,
        d_model=next(iter(jacobians.values())).shape[0],
    )


def fit_checkpoint_metadata(path: str | pathlib.Path, expected_count: int) -> dict[str, Any]:
    """Validate and summarize a durable fp32 running-sum checkpoint."""
    lens = lens_from_fit_checkpoint(path, expected_count)
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


def validate_lens(lens, expected_count: int, *, d_model: int = 1536) -> None:
    if lens.n_prompts != expected_count:
        raise AudioFitContractError(
            f"lens has n_prompts={lens.n_prompts}, expected {expected_count}"
        )
    if lens.source_layers != FIT_SOURCE_LAYERS or lens.d_model != d_model:
        raise AudioFitContractError("lens layers/d_model do not match the fit contract")
    for layer, value in lens.jacobians.items():
        if value.shape != (d_model, d_model) or not bool(torch.isfinite(value).all()):
            raise AudioFitContractError(f"lens layer {layer} is invalid")


def validate_runtime_lens_file(
    path: str | pathlib.Path, expected_count: int, *, d_model: int = 1536
) -> None:
    """Verify the serialized fp16 contract before JLens casts it back to fp32."""
    import jlens

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("n_prompts") != expected_count or state.get("d_model") != d_model:
        raise AudioFitContractError(f"runtime lens metadata is invalid at {path}")
    if state.get("source_layers") != FIT_SOURCE_LAYERS:
        raise AudioFitContractError(f"runtime lens layers are invalid at {path}")
    jacobians = state.get("J")
    if not isinstance(jacobians, dict) or any(
        value.dtype != torch.float16 for value in jacobians.values()
    ):
        raise AudioFitContractError(f"runtime lens is not serialized as fp16 at {path}")
    validate_lens(jlens.JacobianLens.load(path), expected_count, d_model=d_model)
