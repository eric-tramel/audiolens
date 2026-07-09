"""Import-light audio-model profiles and the canonical Gemma runtime adapter.

Profile lookup and the value types in this module do not import Torch,
Transformers, or JLens. Heavy dependencies are loaded only by preparation or
runtime construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from jlens.protocol import LensModel


DEFAULT_MODEL_KEY = "gemma-4-e2b-it"


class AudioFitContractError(RuntimeError):
    """A non-skippable audio preparation or replay contract violation.

    Stock :func:`jlens.fit` catches ``ValueError`` and skips that prompt, so
    adapter and data contract failures deliberately use a different exception.
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
    layers: "Sequence[Any]"
    text_lens_model: "LensModel"
    audio_lens_model: "LensModel"

    def unembed(self, residual: "torch.Tensor") -> "torch.Tensor": ...

    def prepare_audio(self, path: str | Path) -> PreparedAudio: ...

    def forward_audio(self, prepared: PreparedAudio) -> Any: ...

def audio_residuals(
    activations: dict[int, Any], prepared: PreparedAudio, layer: int
) -> Any:
    """Select decoder residuals at the adapter-provided audio positions."""

    return activations[layer][0].index_select(0, prepared.audio_positions)



_DEFAULT_PROFILE = ModelProfile(
    key=DEFAULT_MODEL_KEY,
    version=1,
    slug="gemma-4-E2B-it",
    model_id="google/gemma-4-E2B-it",
    model_revision="70af34e20bd4b7a91f0de6b22675850c43922a03",
    adapter_source="audiolens.models:GemmaAudioRuntime",
    d_model=1536,
    source_layers=tuple(range(34)),
    target_layer=34,
    max_sequence_length=128,
    skip_first=16,
    dimension_batch_size=128,
    read_layer=29,
    read_layers=(23, 29, 33),
)
_PROFILES = MappingProxyType({_DEFAULT_PROFILE.key: _DEFAULT_PROFILE})
DEFAULT_MODEL_PROFILE = _DEFAULT_PROFILE


def get_model_profile(key: str = DEFAULT_MODEL_KEY) -> ModelProfile:
    """Return a built-in profile without importing any ML dependencies."""

    try:
        return _PROFILES[key]
    except KeyError:
        available = ", ".join(sorted(_PROFILES))
        raise UnknownModelProfileError(
            f"unknown audio model profile {key!r}; available: {available}"
        ) from None


def resolve_audio_token_id(config: Any, tokenizer: Any) -> int:
    """Resolve Gemma's audio soft-token ID from its pinned runtime contract."""

    audio_id = getattr(config, "audio_token_id", None)
    if audio_id is None:
        audio_id = tokenizer.convert_tokens_to_ids("<audio_soft_token>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    try:
        resolved = int(audio_id)
        unknown = None if unknown_id is None else int(unknown_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioFitContractError(
            "audio soft-token id is not an integer"
        ) from exc
    if (
        isinstance(audio_id, bool)
        or resolved < 0
        or (unknown is not None and resolved == unknown)
    ):
        raise AudioFitContractError("could not resolve the audio soft-token id")
    return resolved


def validate_audio_layout(
    input_ids: Any,
    audio_id: int,
    *,
    profile: ModelProfile = DEFAULT_MODEL_PROFILE,
) -> AudioLayout:
    """Validate Gemma's contiguous audio span and stock JLens fit positions."""

    import torch
    from jlens.fitting import valid_position_mask

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        shape = tuple(input_ids.shape) if torch.is_tensor(input_ids) else type(input_ids).__name__
        raise AudioFitContractError(f"expected input_ids [1, seq], got {shape}")
    positions = (input_ids[0] == audio_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise AudioFitContractError("no audio soft-token positions")
    expected = torch.arange(
        int(positions[0]), int(positions[-1]) + 1, device=positions.device
    )
    if not torch.equal(positions, expected):
        raise AudioFitContractError("audio soft-token positions are not contiguous")
    stop = int(positions[-1]) + 1
    if stop > profile.max_sequence_length:
        raise AudioFitContractError(
            f"audio prefix has {stop} positions, above "
            f"max_length={profile.max_sequence_length}"
        )
    try:
        valid = valid_position_mask(stop, skip_first=profile.skip_first)
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


def prepare_audio(
    processor: Any,
    path: str | Path,
    *,
    profile: ModelProfile = DEFAULT_MODEL_PROFILE,
) -> PreparedAudio:
    """Prepare one Gemma audio path without constructing model weights."""

    import torch

    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": str(path)}]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    try:
        input_ids = inputs["input_ids"]
    except (KeyError, TypeError) as exc:
        raise AudioFitContractError("processor did not return input_ids") from exc
    if not torch.is_tensor(input_ids):
        raise AudioFitContractError("processor input_ids are not a Tensor")
    audio_id = resolve_audio_token_id(processor, processor.tokenizer)
    layout = validate_audio_layout(input_ids, audio_id, profile=profile)
    full_length = int(input_ids.shape[1])
    if layout.audio_start != 5 or full_length - layout.stop != 3:
        raise AudioFitContractError(
            f"{path} has unexpected framing: audio_start={layout.audio_start}, "
            f"closing_tokens={full_length - layout.stop}"
        )
    positions = (input_ids[0] == audio_id).nonzero(as_tuple=True)[0].detach().clone()
    return PreparedAudio(
        model_inputs=inputs,
        input_ids=input_ids,
        audio_positions=positions,
        layout=layout,
        manifest_fields={
            "audio_start": layout.audio_start,
            "n_audio_tokens": layout.n_audio_tokens,
            "sliced_seq_len": layout.stop,
            "n_valid_positions": layout.n_valid_positions,
        },
    )


def crop_attention_mapping(
    mapping: dict[str, Any], stop: int
) -> dict[str, Any]:
    """Crop Gemma's exact eager full/sliding masks to a decoder prefix."""

    import torch

    expected = {"full_attention", "sliding_attention"}
    if set(mapping) != expected:
        raise AudioFitContractError(
            f"unexpected attention mapping keys {sorted(mapping)}; expected {sorted(expected)}"
        )
    cropped: dict[str, Any] = {}
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


def expand_batch(tensor: Any, batch_size: int) -> Any:
    """Zero-copy expand a captured batch-one tensor."""

    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise AudioFitContractError(
            f"captured tensor must have batch dimension 1, got {tuple(tensor.shape)}"
        )
    return tensor.expand(batch_size, *tensor.shape[1:])


@dataclass(slots=True, eq=False)
class _PreparedDecoderInputs:
    input_ids: Any = field(repr=False)
    inputs_embeds: Any = field(repr=False)
    per_layer_inputs: Any = field(repr=False)
    attention_mask: dict[str, Any] = field(repr=False)
    position_ids: Any = field(repr=False)


class GemmaPreparedAudioLensModel:
    """Stateful prepared-audio implementation of JLens's ``LensModel``.

    Each ``encode`` replaces the retained sample. A successful encoding may be
    replayed repeatedly by ``forward`` until the next encoding begins.
    """

    def __init__(self, runtime: GemmaAudioRuntime, base: Any):
        self.runtime = runtime
        self.base = base
        self.n_layers = base.n_layers
        self.d_model = base.d_model
        self.layers = base.layers
        self.tokenizer = base.tokenizer
        self._language_model = runtime.model.model.language_model
        if base._text_module is not self._language_model:  # noqa: SLF001
            raise AudioFitContractError("JLens and Gemma resolved different text decoders")
        text_config = runtime.model.config.get_text_config()
        if text_config.use_bidirectional_attention is not None:
            raise AudioFitContractError(
                "suffix slicing requires causal-only use_bidirectional_attention=None"
            )
        self._prepared: _PreparedDecoderInputs | None = None

    @property
    def input_device(self) -> Any:
        return self.base.input_device

    def unembed(self, residual: Any) -> Any:
        return self.base.unembed(residual)

    def encode(self, path: str, *, max_length: int | None = None) -> Any:
        import torch

        self._prepared = None
        prepared = self.runtime.prepare_audio(path)
        limit = self.runtime.profile.max_sequence_length if max_length is None else max_length
        if prepared.layout.stop > limit:
            raise AudioFitContractError(
                f"audio prefix has {prepared.layout.stop} positions, above max_length={limit}"
            )

        calls: list[dict[str, Any]] = []

        def capture(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            calls.append(dict(kwargs))

        handle = self._language_model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                self.runtime.forward_audio(prepared)
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
        full_length = int(prepared.input_ids.shape[1])
        sliced_ids = prepared.input_ids[:, : prepared.layout.stop].detach().clone()
        captured = _PreparedDecoderInputs(
            input_ids=sliced_ids,
            inputs_embeds=embeds[:, :full_length].detach().clone(),
            per_layer_inputs=ple[:, :full_length].detach().clone(),
            attention_mask=crop_attention_mapping(masks, full_length),
            position_ids=positions[:, :full_length].detach().clone(),
        )
        self._prepared = captured
        return sliced_ids

    def forward(self, input_ids: Any) -> Any:
        import torch

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
        batch_size = input_ids.shape[0]
        masks = {
            name: None if value is None else expand_batch(value, batch_size)
            for name, value in prepared.attention_mask.items()
        }
        return self._language_model(
            inputs_embeds=expand_batch(prepared.inputs_embeds, batch_size),
            per_layer_inputs=expand_batch(prepared.per_layer_inputs, batch_size),
            attention_mask=masks,
            position_ids=expand_batch(prepared.position_ids, batch_size),
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )


class GemmaAudioRuntime:
    """Canonical pinned Gemma runtime shared by fitting and audio readout."""

    def __init__(
        self,
        profile: ModelProfile,
        processor: Any,
        model: Any,
        text_tokenizer: Any,
    ) -> None:
        import jlens

        self.profile = profile
        self.processor = processor
        self.model = model
        self.tokenizer = processor.tokenizer
        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False
        self.text_lens_model = jlens.from_hf(model, text_tokenizer, force_bos=True)
        audio_base = jlens.from_hf(model, self.tokenizer, force_bos=False)
        self.layers = audio_base.layers
        self.audio_lens_model = GemmaPreparedAudioLensModel(self, audio_base)

    @property
    def input_device(self) -> Any:
        return self.audio_lens_model.input_device

    def unembed(self, residual: Any) -> Any:
        return self.audio_lens_model.unembed(residual)

    def prepare_audio(self, path: str | Path) -> PreparedAudio:
        prepared = prepare_audio(self.processor, path, profile=self.profile)
        inputs = prepared.model_inputs
        if hasattr(inputs, "to"):
            moved = inputs.to(self.input_device)
        elif isinstance(inputs, dict):
            moved = {
                name: value.to(self.input_device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
        else:
            raise AudioFitContractError("processor inputs cannot be moved to the model device")
        prepared.model_inputs = moved
        prepared.input_ids = moved["input_ids"]
        prepared.audio_positions = prepared.audio_positions.to(self.input_device)
        return prepared

    def forward_audio(self, prepared: PreparedAudio) -> Any:
        return self.model.model(**prepared.model_inputs, use_cache=False)


def load_audio_processor(key: str = DEFAULT_MODEL_KEY) -> Any:
    """Load only the pinned processor for lightweight tokenizer/preparation use."""

    import transformers

    profile = get_model_profile(key)
    return transformers.AutoProcessor.from_pretrained(
        profile.model_id, revision=profile.model_revision
    )


def load_model_runtime(
    key: str = DEFAULT_MODEL_KEY,
    *,
    device: str | None = None,
    device_map: Any | None = None,
) -> AudioModelRuntime:
    """Load pinned Gemma weights and construct the canonical runtime adapter."""

    import torch
    import transformers

    if device is not None and device_map is not None:
        raise AudioFitContractError("device and device_map are mutually exclusive")
    profile = get_model_profile(key)
    processor = load_audio_processor(key)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(
        profile.model_id, revision=profile.model_revision
    )
    model_kwargs: dict[str, Any] = {
        "revision": profile.model_revision,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        profile.model_id, **model_kwargs
    )
    if device_map is None:
        resolved_device = device
        if resolved_device is None:
            resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(resolved_device)
    model = model.eval()
    return GemmaAudioRuntime(profile, processor, model, text_tokenizer)
