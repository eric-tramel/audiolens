from dataclasses import FrozenInstanceError

import pytest
import torch

from audiolens.audio_eval_model import (
    AudioEvaluationLayout,
    PreparedAudioEvaluation,
    prepare_audio_evaluation,
)
from audiolens.models import DEFAULT_MODEL_PROFILE
from audiolens.models.base import AudioFitContractError
from audiolens.models.gemma4 import prepare_audio


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        assert token == "<audio_soft_token>"
        return 9


class _Processor:
    audio_token_id = 9

    def __init__(self, input_ids, **model_inputs):
        self.tokenizer = _Tokenizer()
        self.input_ids = input_ids
        self.model_inputs = {
            "input_features": torch.ones(1, 4, 3),
            "input_features_mask": torch.ones(1, 4, dtype=torch.bool),
            **model_inputs,
        }
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": self.input_ids.clone(),
            **self.model_inputs,
        }


def _evaluation_ids(audio_tokens=50):
    return torch.tensor(
        [[2, 105, 2364, 107, 256000] + [9] * audio_tokens + [258883, 106, 107, 105, 4368, 107]]
    )


def _fit_ids(audio_tokens=50):
    return torch.tensor([[2, 105, 2364, 107, 256000] + [9] * audio_tokens + [258883, 106, 107]])


def test_evaluation_preparation_records_exact_validated_framing_and_positions():
    processor = _Processor(_evaluation_ids())
    prepared = prepare_audio_evaluation(processor, "spoken.wav")

    assert isinstance(prepared, PreparedAudioEvaluation)
    assert not hasattr(prepared, "__dict__")
    assert "tensor" not in repr(prepared)
    with pytest.raises(FrozenInstanceError):
        prepared.response_position = 0
    assert prepared.model_inputs["input_ids"] is prepared.input_ids
    assert torch.equal(prepared.audio_positions, torch.arange(5, 55))
    assert prepared.audio_positions.data_ptr() != prepared.input_ids.data_ptr()
    assert prepared.last_processor_valid_audio_position == 54
    assert prepared.response_position == 60
    assert prepared.prefix_framing_ids == (2, 105, 2364, 107, 256000)
    assert prepared.suffix_framing_ids == (258883, 106, 107, 105, 4368, 107)
    assert prepared.layout == AudioEvaluationLayout(5, 50, 55, 61)
    assert dict(prepared.manifest_fields) == {
        "audio_start": 5,
        "n_audio_tokens": 50,
        "audio_stop": 55,
        "sequence_length": 61,
        "max_sequence_length": 512,
        "last_processor_valid_audio_position": 54,
        "response_position": 60,
        "prefix_framing_ids": (2, 105, 2364, 107, 256000),
        "suffix_framing_ids": (258883, 106, 107, 105, 4368, 107),
    }
    with pytest.raises(TypeError):
        prepared.manifest_fields["response_position"] = 0
    assert processor.calls == [
        (
            [{"role": "user", "content": [{"type": "audio", "audio": "spoken.wav"}]}],
            {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "truncation": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("model_inputs", "message"),
    [
        ({"input_features": None}, "omitted required audio model inputs"),
        ({"input_features_mask": None}, "omitted required audio model inputs"),
        ({"input_features": torch.ones(2, 4, 3)}, "nonempty batch-one Tensor"),
        ({"input_features": torch.empty(1, 0, 3)}, "nonempty batch-one Tensor"),
        (
            {"input_features": torch.tensor([[[float("nan")]]])},
            "input_features contains nonfinite",
        ),
        (
            {"input_features_mask": torch.tensor([[float("inf")]])},
            "input_features_mask contains nonfinite",
        ),
        (
            {"input_features_mask": torch.zeros(1, 4, dtype=torch.bool)},
            "selects no audio",
        ),
        ({"pixel_values": torch.ones(1, 3, 4, 4)}, "image/video inputs"),
    ],
)
def test_evaluation_preparation_requires_finite_audio_only_inputs(model_inputs, message):
    processor = _Processor(_evaluation_ids(), **model_inputs)
    with pytest.raises(AudioFitContractError, match=message):
        prepare_audio_evaluation(processor, "spoken.wav")


@pytest.mark.parametrize(
    ("input_ids", "message"),
    [
        (torch.ones(2, 12, dtype=torch.long), r"expected input_ids \[1, seq\]"),
        (_evaluation_ids().to(torch.float32), "integer token IDs"),
        (
            torch.tensor([[2, 105, 2364, 107, 256000, 9, 10, 9, 258883, 106, 107, 105, 4368, 107]]),
            "not contiguous",
        ),
        (
            torch.tensor([[2, 105, 2364, 107, 256000, 10, 10, 258883, 106, 107, 105, 4368, 107]]),
            "no audio soft-token positions",
        ),
        (
            torch.tensor([[2, 105, 2364, 107, 256000, 9, 258883, 106, 107, 105, 999, 107]]),
            "unexpected evaluation framing",
        ),
    ],
)
def test_evaluation_preparation_rejects_malformed_ids(input_ids, message):
    processor = _Processor(input_ids)
    with pytest.raises(AudioFitContractError, match=message) as caught:
        prepare_audio_evaluation(processor, "spoken.wav")
    assert not isinstance(caught.value, ValueError)


def test_evaluation_preparation_rejects_nonmapping_processor_output():
    processor = _Processor(_evaluation_ids())
    processor.apply_chat_template = lambda *args, **kwargs: []
    with pytest.raises(AudioFitContractError, match="model-input mapping"):
        prepare_audio_evaluation(processor, "spoken.wav")


def test_evaluation_preparation_enforces_no_truncation_and_512_boundary():
    at_boundary = prepare_audio_evaluation(_Processor(_evaluation_ids(501)), "512.wav")
    assert at_boundary.layout.sequence_length == 512
    assert at_boundary.response_position == 511
    assert at_boundary.last_processor_valid_audio_position == 505

    with pytest.raises(AudioFitContractError, match=r"513 positions.*max_length=512"):
        prepare_audio_evaluation(_Processor(_evaluation_ids(502)), "513.wav")
    with pytest.raises(AudioFitContractError, match="positive integer"):
        prepare_audio_evaluation(_Processor(_evaluation_ids()), "bad-max.wav", True)


def test_existing_fit_preparation_and_profile_contract_remain_unchanged():
    processor = _Processor(_fit_ids())
    prepared = prepare_audio(processor, "fit.wav")

    assert DEFAULT_MODEL_PROFILE.max_sequence_length == 128
    assert DEFAULT_MODEL_PROFILE.target_layer == 34
    assert prepared.layout.audio_start == 5
    assert prepared.layout.stop == 55
    assert prepared.layout.n_valid_positions == 38
    assert prepared.input_ids.shape == (1, 58)
    assert processor.calls[0][1] == {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
