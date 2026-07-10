"""Import-light contracts shared by Audiolens model-family adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn

    from jlens.protocol import LensModel


class AudioFitContractError(RuntimeError):
    """A non-skippable audio preparation or replay contract violation.

    Stock :func:`jlens.fit` catches ``ValueError`` and skips that prompt, so
    adapter and data-contract failures deliberately use a different base type.
    """


class UnknownModelProfileError(KeyError):
    """Raised when a caller selects an unregistered model profile."""


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Immutable execution identity and JLens geometry for one audio model."""

    key: str
    version: int
    slug: str
    model_id: str
    model_revision: str
    adapter_source: str
    d_model: int
    source_layers: tuple[int, ...]
    target_layer: int
    max_sequence_length: int
    skip_first: int
    dimension_batch_size: int
    read_layer: int
    read_layers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AudioLayout:
    """Decoder layout used to expose only audio positions to JLens fitting."""

    audio_start: int
    n_audio_tokens: int
    stop: int
    n_valid_positions: int
    valid_mask: Any = field(repr=False)


@dataclass(slots=True, eq=False)
class PreparedAudio:
    """Opaque model inputs plus explicit decoder-aligned audio positions."""

    model_inputs: Any = field(repr=False)
    input_ids: Any = field(repr=False)
    audio_positions: Any = field(repr=False)
    layout: AudioLayout
    manifest_fields: dict[str, Any]


class AudioModelRuntime(Protocol):
    """Narrow runtime surface consumed by fit and readout orchestration."""

    profile: ModelProfile
    processor: Any
    model: Any
    tokenizer: Any
    layers: "Sequence[nn.Module]"
    text_lens_model: "LensModel"
    audio_lens_model: "LensModel"

    def prepare_audio(self, path: str | Path) -> PreparedAudio: ...

    def unembed(self, residual: Any) -> Any: ...

    def forward_audio(self, prepared: PreparedAudio) -> Any: ...


def audio_residuals(
    activations: dict[int, Any], prepared: PreparedAudio, layer: int
) -> Any:
    """Select decoder residuals at adapter-provided audio positions."""

    return activations[layer][0].index_select(0, prepared.audio_positions)
