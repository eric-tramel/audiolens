import inspect
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import audiolens
from audiolens import models
from audiolens.models import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_PROFILE,
    AudioFitContractError,
    GemmaPreparedAudioLensModel,
    PreparedAudio,
    UnknownModelProfileError,
    audio_residuals,
    get_model_profile,
    load_audio_processor,
    load_model_runtime,
    prepare_audio,
)


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        assert token == "<audio_soft_token>"
        return 9


class _Processor:
    audio_token_id = 9

    def __init__(self, input_ids=None):
        self.tokenizer = _Tokenizer()
        self.input_ids = input_ids
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": self.input_ids.clone()}


def _framed_ids(audio_tokens=50):
    return torch.tensor(
        [[2, 105, 2364, 107, 256000] + [9] * audio_tokens + [258883, 106, 107]]
    )


def test_profile_lookup_is_pure_immutable_and_complete():
    profile = get_model_profile()
    assert profile is DEFAULT_MODEL_PROFILE
    assert profile.key == DEFAULT_MODEL_KEY
    assert profile.model_id == "google/gemma-4-E2B-it"
    assert profile.model_revision == "70af34e20bd4b7a91f0de6b22675850c43922a03"
    assert profile.source_layers == tuple(range(34))
    assert profile.target_layer == 34
    assert profile.d_model == 1536
    assert profile.read_layer == 29
    assert profile.read_layers == (23, 29, 33)
    assert not hasattr(profile, "__dict__")
    with pytest.raises(FrozenInstanceError):
        profile.d_model = 1


def test_profile_lookup_imports_no_ml_runtime_dependencies():
    code = (
        "import sys; from audiolens.models import get_model_profile; "
        "assert get_model_profile().d_model == 1536; "
        "assert not ({'torch', 'transformers', 'jlens'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_profile_fails_closed_without_value_error():
    assert not issubclass(UnknownModelProfileError, ValueError)
    with pytest.raises(UnknownModelProfileError, match="unknown audio model profile"):
        get_model_profile("not-registered")


def test_malformed_audio_token_id_is_non_skippable_contract_failure():
    config = types.SimpleNamespace(audio_token_id="not-an-integer")
    with pytest.raises(AudioFitContractError, match="not an integer") as caught:
        models.resolve_audio_token_id(config, _Tokenizer())
    assert not isinstance(caught.value, ValueError)


def test_prepared_audio_is_mutable_slotted_identity_value():
    first = PreparedAudio({}, torch.tensor([[1]]), torch.tensor([0]), object(), {})
    second = PreparedAudio({}, torch.tensor([[1]]), torch.tensor([0]), object(), {})
    assert first != second
    assert not hasattr(first, "__dict__")
    first.manifest_fields = {"n_audio_tokens": 1}
    assert first.manifest_fields == {"n_audio_tokens": 1}
    assert "tensor" not in repr(first)


def test_processor_only_loader_is_pinned_and_never_constructs_weights(monkeypatch):
    calls = []
    sentinel = object()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return sentinel

    class ForbiddenModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("processor-only loading constructed model weights")

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=AutoProcessor,
        AutoModelForImageTextToText=ForbiddenModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    assert load_audio_processor() is sentinel
    assert calls == [
        (
            DEFAULT_MODEL_PROFILE.model_id,
            {"revision": DEFAULT_MODEL_PROFILE.model_revision},
        )
    ]


def test_runtime_loader_uses_qualified_recipe_and_pinned_identity(monkeypatch):
    processor = object()
    tokenizer = object()
    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True
            return self

        def to(self, device):
            raise AssertionError(f"device_map model unexpectedly moved to {device}")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForImageTextToText=AutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(models, "load_audio_processor", lambda key: processor)
    monkeypatch.setattr(
        models,
        "GemmaAudioRuntime",
        lambda profile, actual_processor, model, text_tokenizer: types.SimpleNamespace(
            profile=profile,
            processor=actual_processor,
            model=model,
            text_tokenizer=text_tokenizer,
        ),
    )

    runtime = load_model_runtime(device_map="cuda")
    expected_identity = (
        DEFAULT_MODEL_PROFILE.model_id,
        {"revision": DEFAULT_MODEL_PROFILE.model_revision},
    )
    assert calls["tokenizer"] == expected_identity
    assert calls["model"] == (
        DEFAULT_MODEL_PROFILE.model_id,
        {
            "revision": DEFAULT_MODEL_PROFILE.model_revision,
            "dtype": torch.bfloat16,
            "device_map": "cuda",
            "attn_implementation": "eager",
        },
    )
    assert calls["eval"] is True
    assert runtime.profile is DEFAULT_MODEL_PROFILE
    assert runtime.processor is processor
    assert runtime.text_tokenizer is tokenizer


def test_runtime_loader_rejects_conflicting_placement_without_value_error():
    with pytest.raises(AudioFitContractError, match="mutually exclusive") as caught:
        load_model_runtime(device="cpu", device_map="cuda")
    assert not isinstance(caught.value, ValueError)


def test_gemma_processor_preparation_returns_explicit_decoder_positions():
    processor = _Processor(_framed_ids())
    prepared = prepare_audio(processor, "clip.wav")
    assert torch.equal(prepared.audio_positions, torch.arange(5, 55))
    assert prepared.layout.audio_start == 5
    assert prepared.layout.n_audio_tokens == 50
    assert prepared.layout.stop == 55
    assert prepared.input_ids.shape == (1, 58)
    assert prepared.manifest_fields == {
        "audio_start": 5,
        "n_audio_tokens": 50,
        "sliced_seq_len": 55,
        "n_valid_positions": 38,
    }
    messages, kwargs = processor.calls[0]
    assert messages == [
        {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}]}
    ]
    assert kwargs == {"tokenize": True, "return_dict": True, "return_tensors": "pt"}


def test_gemma_preparation_rejects_wrong_exact_framing_non_skippably():
    processor = _Processor(torch.tensor([[2, 105, 107, 256000] + [9] * 50 + [1, 2, 3]]))
    with pytest.raises(AudioFitContractError, match="unexpected framing") as caught:
        prepare_audio(processor, "clip.wav")
    assert not isinstance(caught.value, ValueError)


def test_explicit_positions_support_non_token_derived_alternate_runtime():
    profile = replace(
        DEFAULT_MODEL_PROFILE,
        key="fake-position-model",
        slug="fake-position-model",
        d_model=3,
        source_layers=(0, 2),
        target_layer=3,
        read_layer=2,
        read_layers=(0, 2),
    )

    class FakeRuntime:
        def __init__(self):
            self.profile = profile

        def prepare_audio(self, path):
            assert path == "alternate.wav"
            ids = torch.tensor([[41, 41, 41, 41, 41]])
            return PreparedAudio(
                model_inputs={"opaque": path},
                input_ids=ids,
                audio_positions=torch.tensor([1, 4]),
                layout=object(),
                manifest_fields={},
            )

    prepared = FakeRuntime().prepare_audio("alternate.wav")
    activations = {profile.read_layer: torch.arange(15).view(1, 5, 3)}
    assert torch.equal(
        audio_residuals(activations, prepared, profile.read_layer),
        torch.tensor([[3, 4, 5], [12, 13, 14]]),
    )
    assert torch.unique(prepared.input_ids).numel() == 1


def test_replay_rejects_mismatched_ids_and_failed_encode_clears_state():
    adapter = object.__new__(GemmaPreparedAudioLensModel)
    adapter._prepared = types.SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
    with pytest.raises(AudioFitContractError, match="do not match"):
        adapter.forward(torch.tensor([[1, 3]]))

    class FailingRuntime:
        profile = DEFAULT_MODEL_PROFILE

        def prepare_audio(self, path):
            raise AudioFitContractError(f"cannot prepare {path}")

    adapter.runtime = FailingRuntime()
    with pytest.raises(AudioFitContractError, match="cannot prepare"):
        adapter.encode("bad.wav")
    assert adapter._prepared is None


def test_replay_validates_shape_and_remains_repeatable_after_encode():
    adapter = object.__new__(GemmaPreparedAudioLensModel)
    prepared = types.SimpleNamespace(
        input_ids=torch.tensor([[1, 2]]),
        inputs_embeds=torch.arange(6).view(1, 2, 3),
        per_layer_inputs=torch.arange(6).view(1, 2, 3),
        attention_mask={
            "full_attention": torch.ones(1, 1, 2, 2),
            "sliding_attention": None,
        },
        position_ids=torch.tensor([[0, 1]]),
    )
    adapter._prepared = prepared
    calls = []

    def language_model(**kwargs):
        calls.append(kwargs)
        return kwargs["inputs_embeds"]

    adapter._language_model = language_model
    with pytest.raises(AudioFitContractError, match="shape"):
        adapter.forward(torch.tensor([[1, 2, 3]]))
    first = adapter.forward(torch.tensor([[1, 2], [1, 2]]))
    second = adapter.forward(torch.tensor([[1, 2]]))
    assert first.shape == (2, 2, 3)
    assert second.shape == (1, 2, 3)
    assert len(calls) == 2
    assert adapter._prepared is prepared


def test_root_loader_delegates_and_preserves_signature_and_tuple(monkeypatch):
    runtime = types.SimpleNamespace(
        processor=object(), model=object(), text_lens_model=object()
    )
    lens = object()
    calls = []

    def fake_runtime_loader(key, *, device=None):
        calls.append((key, device))
        return runtime

    fake_jlens = types.SimpleNamespace(
        JacobianLens=types.SimpleNamespace(
            from_pretrained=lambda path: lens if path == "lens.pt" else None
        )
    )
    monkeypatch.setattr(audiolens, "load_model_runtime", fake_runtime_loader)
    monkeypatch.setitem(sys.modules, "jlens", fake_jlens)

    assert str(inspect.signature(audiolens.load_lensed_model)) == (
        "(lens_path: 'str', *, device: 'str | None' = None)"
    )
    loaded = audiolens.load_lensed_model("lens.pt", device="cpu")
    assert loaded == (
        runtime.processor,
        runtime.model,
        runtime.text_lens_model,
        lens,
    )
    assert calls == [(DEFAULT_MODEL_KEY, "cpu")]
    assert audiolens.MODEL_ID == DEFAULT_MODEL_PROFILE.model_id
    assert audiolens.READ_LAYER == DEFAULT_MODEL_PROFILE.read_layer
