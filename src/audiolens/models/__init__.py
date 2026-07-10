"""Audio-model profiles, adapter contracts, and built-in family runtimes."""

from __future__ import annotations
from dataclasses import dataclass

from types import MappingProxyType
from typing import Any, Callable

from .base import (
    AudioFitContractError,
    AudioLayout,
    AudioModelRuntime,
    ModelProfile,
    PreparedAudio,
    UnknownModelProfileError,
    audio_residuals,
)
from .gemma4 import (
    GEMMA4_PROFILE,
    GemmaAudioRuntime,
    GemmaPreparedAudioLensModel,
    crop_attention_mapping,
    expand_batch,
    prepare_audio,
    resolve_audio_token_id,
    validate_audio_layout,
)
from .gemma4 import load_audio_processor as _load_gemma4_processor
from .gemma4 import load_model_runtime as _load_gemma4_runtime

DEFAULT_MODEL_KEY = GEMMA4_PROFILE.key
DEFAULT_MODEL_PROFILE = GEMMA4_PROFILE

@dataclass(frozen=True, slots=True)
class _ModelRegistration:
    profile: ModelProfile
    load_processor: Callable[[ModelProfile], Any]
    load_runtime: Callable[..., AudioModelRuntime]


_REGISTRATIONS = MappingProxyType(
    {
        GEMMA4_PROFILE.key: _ModelRegistration(
            profile=GEMMA4_PROFILE,
            load_processor=_load_gemma4_processor,
            load_runtime=_load_gemma4_runtime,
        )
    }
)


def _get_registration(key: str) -> _ModelRegistration:
    try:
        return _REGISTRATIONS[key]
    except KeyError:
        available = ", ".join(sorted(_REGISTRATIONS))
        raise UnknownModelProfileError(
            f"unknown audio model profile {key!r}; available: {available}"
        ) from None


def get_model_profile(key: str = DEFAULT_MODEL_KEY) -> ModelProfile:
    """Return a built-in profile without importing any ML dependencies."""

    return _get_registration(key).profile


def load_audio_processor(key: str = DEFAULT_MODEL_KEY) -> Any:
    """Load only the selected family processor without constructing weights."""

    registration = _get_registration(key)
    return registration.load_processor(registration.profile)


def load_model_runtime(
    key: str = DEFAULT_MODEL_KEY,
    *,
    device: str | None = None,
    device_map: Any | None = None,
) -> AudioModelRuntime:
    """Load the selected built-in audio-model runtime."""

    registration = _get_registration(key)
    return registration.load_runtime(
        registration.profile, device=device, device_map=device_map
    )


__all__ = [
    "DEFAULT_MODEL_KEY",
    "DEFAULT_MODEL_PROFILE",
    "GEMMA4_PROFILE",
    "AudioFitContractError",
    "AudioLayout",
    "AudioModelRuntime",
    "GemmaAudioRuntime",
    "GemmaPreparedAudioLensModel",
    "ModelProfile",
    "PreparedAudio",
    "UnknownModelProfileError",
    "audio_residuals",
    "crop_attention_mapping",
    "expand_batch",
    "get_model_profile",
    "load_audio_processor",
    "load_model_runtime",
    "prepare_audio",
    "resolve_audio_token_id",
    "validate_audio_layout",
]
