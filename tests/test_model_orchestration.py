import pathlib
import sys
from dataclasses import replace

import pytest
import torch

from audiolens.fitting import config_digest
from audiolens.models import DEFAULT_MODEL_PROFILE
from audiolens.models.base import AudioFitContractError, PreparedAudio

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

import audio_readout  # noqa: E402
import modal_audio_eval  # noqa: E402
import modal_fit_mixed_lens  # noqa: E402


def _alternate_profile():
    return replace(
        DEFAULT_MODEL_PROFILE,
        key="alternate-audio",
        version=7,
        slug="alternate-audio-model",
        model_id="test/alternate-audio",
        model_revision="alternate-revision",
        adapter_source="tests:AlternateAudioRuntime",
        d_model=3,
        source_layers=(0, 2),
        target_layer=3,
        max_sequence_length=16,
        skip_first=2,
        dimension_batch_size=4,
        read_layer=2,
        read_layers=(0, 2),
    )


def test_fit_profile_identity_drives_digest_tag_and_resume_guard(tmp_path):
    profile = _alternate_profile()
    identity = modal_fit_mixed_lens._fit_profile_config(profile)
    assert identity == {
        "profile_key": "alternate-audio",
        "profile_version": 7,
        "profile_slug": "alternate-audio-model",
        "adapter_source": "tests:AlternateAudioRuntime",
        "model": {
            "id": "test/alternate-audio",
            "revision": "alternate-revision",
        },
    }

    geometry = modal_fit_mixed_lens._fit_geometry_config(profile)
    assert geometry == {
        "source_layers": [0, 2],
        "target_layer": 3,
        "skip_first": 2,
        "max_seq_len": 16,
        "dim_batch": 4,
        "model_dtype": "bfloat16",
        "artifact_dtype": "float16",
    }
    assert modal_fit_mixed_lens._fit_geometry_config(DEFAULT_MODEL_PROFILE) == {
        "source_layers": list(range(34)),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
        "model_dtype": "bfloat16",
        "artifact_dtype": "float16",
    }
    config = {"schema_version": 2, **identity, "fit": geometry}
    digest = config_digest(config)
    assert modal_fit_mixed_lens._fit_run_tag(profile, digest) == (
        f"alternate-audio-model-mixed-{digest[:12]}"
    )
    default_digest = "a" * 64
    assert modal_fit_mixed_lens._fit_run_tag(
        DEFAULT_MODEL_PROFILE, default_digest
    ) == "gemma-4-E2B-it-mixed-aaaaaaaaaaaa"

    run_path = tmp_path / "run.json"
    modal_fit_mixed_lens._ensure_run_json(run_path, config)
    changed = {**config, "profile_version": 8}
    assert config_digest(changed) != digest
    with pytest.raises(AudioFitContractError, match="run config mismatch"):
        modal_fit_mixed_lens._ensure_run_json(run_path, changed)


def test_evaluation_identity_and_readout_use_alternate_profile_positions():
    profile = _alternate_profile()
    assert modal_audio_eval._evaluation_profile_config(profile) == {
        "profile_key": "alternate-audio",
        "profile_version": 7,
        "profile_slug": "alternate-audio-model",
        "adapter_source": "tests:AlternateAudioRuntime",
        "model": {
            "id": "test/alternate-audio",
            "revision": "alternate-revision",
        },
        "read_layers": [0, 2],
    }
    assert audio_readout._readout_identity(profile) == (
        2,
        "lenses/alternate-audio-model_jacobian_lens.pt",
    )

    prepared = PreparedAudio(
        model_inputs={"opaque": True},
        input_ids=torch.full((1, 5), 41),
        audio_positions=torch.tensor([1, 4]),
        layout=object(),
        manifest_fields={},
    )
    activations = {2: torch.arange(15).view(1, 5, 3)}
    assert torch.equal(
        audio_readout.audio_residuals(activations, prepared, 2),
        torch.tensor([[3, 4, 5], [12, 13, 14]]),
    )
