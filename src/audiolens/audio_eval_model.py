"""Evaluation-only audio preparation beside the source-bound fit adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .models.base import AudioFitContractError


@dataclass(frozen=True, slots=True)
class AudioEvaluationLayout:
    """Complete decoder layout retained for audio-only evaluation."""

    audio_start: int
    n_audio_tokens: int
    audio_stop: int
    sequence_length: int


@dataclass(frozen=True, slots=True, eq=False)
class PreparedAudioEvaluation:
    """Validated evaluation record with exact processor framing evidence."""

    model_inputs: Any = field(repr=False)
    input_ids: Any = field(repr=False)
    audio_positions: Any = field(repr=False)
    last_processor_valid_audio_position: int
    response_position: int
    prefix_framing_ids: tuple[int, ...]
    suffix_framing_ids: tuple[int, ...]
    layout: AudioEvaluationLayout
    manifest_fields: Mapping[str, Any]


EVALUATION_PREFIX_FRAMING_IDS = (2, 105, 2364, 107, 256000)
EVALUATION_SUFFIX_FRAMING_IDS = (258883, 106, 107, 105, 4368, 107)


def prepare_audio_evaluation(
    processor: Any,
    path: str | Path,
    max_sequence_length: int = 512,
) -> PreparedAudioEvaluation:
    """Prepare one complete audio-only assistant prefill for evaluation."""

    import torch

    from .models.gemma4 import resolve_audio_token_id

    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length < 1
    ):
        raise AudioFitContractError("max_sequence_length must be a positive integer")
    messages = [{"role": "user", "content": [{"type": "audio", "audio": str(path)}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        truncation=False,
    )
    if not isinstance(inputs, Mapping):
        raise AudioFitContractError("processor did not return a model-input mapping")
    required_audio_inputs = ("input_features", "input_features_mask")
    missing_audio_inputs = [name for name in required_audio_inputs if inputs.get(name) is None]
    if missing_audio_inputs:
        raise AudioFitContractError(
            f"processor omitted required audio model inputs {missing_audio_inputs}"
        )
    for name in required_audio_inputs:
        value = inputs[name]
        if (
            not torch.is_tensor(value)
            or value.ndim < 2
            or value.shape[0] != 1
            or value.numel() == 0
        ):
            raise AudioFitContractError(f"processor {name} is not a nonempty batch-one Tensor")
        if not bool(torch.isfinite(value).all()):
            raise AudioFitContractError(f"processor {name} contains nonfinite values")
    if not bool(inputs["input_features_mask"].bool().any()):
        raise AudioFitContractError("processor input_features_mask selects no audio")
    forbidden_inputs = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    )
    present_forbidden = [name for name in forbidden_inputs if inputs.get(name) is not None]
    if present_forbidden:
        raise AudioFitContractError(
            f"audio-only processor returned image/video inputs {present_forbidden}"
        )
    try:
        input_ids = inputs["input_ids"]
    except (KeyError, TypeError) as exc:
        raise AudioFitContractError("processor did not return input_ids") from exc
    if (
        not torch.is_tensor(input_ids)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
    ):
        shape = tuple(input_ids.shape) if torch.is_tensor(input_ids) else type(input_ids).__name__
        raise AudioFitContractError(f"expected input_ids [1, seq], got {shape}")
    if (
        input_ids.dtype == torch.bool
        or torch.is_floating_point(input_ids)
        or torch.is_complex(input_ids)
    ):
        raise AudioFitContractError("processor input_ids must contain integer token IDs")
    sequence_length = int(input_ids.shape[1])
    if sequence_length > max_sequence_length:
        raise AudioFitContractError(
            f"complete audio sequence has {sequence_length} positions, "
            f"above max_length={max_sequence_length}"
        )
    audio_id = resolve_audio_token_id(processor, processor.tokenizer)
    audio_positions = (input_ids[0] == audio_id).nonzero(as_tuple=True)[0]
    if audio_positions.numel() == 0:
        raise AudioFitContractError("no audio soft-token positions")
    expected_positions = torch.arange(
        int(audio_positions[0]),
        int(audio_positions[-1]) + 1,
        device=audio_positions.device,
    )
    if not torch.equal(audio_positions, expected_positions):
        raise AudioFitContractError("audio soft-token positions are not contiguous")
    audio_start = int(audio_positions[0])
    audio_stop = int(audio_positions[-1]) + 1
    prefix_framing_ids = tuple(int(value) for value in input_ids[0, :audio_start].tolist())
    suffix_framing_ids = tuple(int(value) for value in input_ids[0, audio_stop:].tolist())
    if (
        prefix_framing_ids != EVALUATION_PREFIX_FRAMING_IDS
        or suffix_framing_ids != EVALUATION_SUFFIX_FRAMING_IDS
    ):
        raise AudioFitContractError(
            f"{path} has unexpected evaluation framing: "
            f"prefix={prefix_framing_ids}, suffix={suffix_framing_ids}"
        )
    last_audio_position = audio_stop - 1
    response_position = sequence_length - 1
    layout = AudioEvaluationLayout(
        audio_start=audio_start,
        n_audio_tokens=int(audio_positions.numel()),
        audio_stop=audio_stop,
        sequence_length=sequence_length,
    )
    manifest_fields = MappingProxyType(
        {
            "audio_start": layout.audio_start,
            "n_audio_tokens": layout.n_audio_tokens,
            "audio_stop": layout.audio_stop,
            "sequence_length": layout.sequence_length,
            "max_sequence_length": max_sequence_length,
            "last_processor_valid_audio_position": last_audio_position,
            "response_position": response_position,
            "prefix_framing_ids": prefix_framing_ids,
            "suffix_framing_ids": suffix_framing_ids,
        }
    )
    return PreparedAudioEvaluation(
        model_inputs=inputs,
        input_ids=input_ids,
        audio_positions=audio_positions.detach().clone(),
        last_processor_valid_audio_position=last_audio_position,
        response_position=response_position,
        prefix_framing_ids=prefix_framing_ids,
        suffix_framing_ids=suffix_framing_ids,
        layout=layout,
        manifest_fields=manifest_fields,
    )


__all__ = [
    "EVALUATION_PREFIX_FRAMING_IDS",
    "EVALUATION_SUFFIX_FRAMING_IDS",
    "AudioEvaluationLayout",
    "PreparedAudioEvaluation",
    "prepare_audio_evaluation",
]
