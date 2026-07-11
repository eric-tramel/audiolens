from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from audiolens import audio_workspace_eval as audio_eval
from audiolens.audio_eval_model import (
    EVALUATION_PREFIX_FRAMING_IDS,
    EVALUATION_SUFFIX_FRAMING_IDS,
)


ACTIVE_PER_DISTRIBUTION = 5
VOCABULARY_SIZE = 128


def _active_coordinates() -> set[tuple[str, str]]:
    result = set()
    for distribution in audio_eval.DISTRIBUTIONS:
        names = [name for slug, name in audio_eval.EXPECTED_COORDINATES if slug == distribution]
        result.update((distribution, name) for name in names[:ACTIVE_PER_DISTRIBUTION])
    return result


def _concept_id(coordinate: tuple[str, str]) -> str:
    return f"{coordinate[0]}/{coordinate[1]}/0"


def _record_rank(
    signal: str,
    *,
    position: str,
    layer: int,
    rank_variant: str,
) -> int:
    in_candidate = layer in audio_eval.CANDIDATE_LAYERS
    if rank_variant == "candidate":
        if signal == "negative":
            return VOCABULARY_SIZE
        if signal == "audio_only" and position == audio_eval.RESPONSE_POSITION:
            return VOCABULARY_SIZE
        return 1 if in_candidate else VOCABULARY_SIZE
    if rank_variant == "logit":
        return 1 if signal == "negative" and in_candidate else VOCABULARY_SIZE
    if rank_variant in {"transposed", "permuted"}:
        if signal == "control" and in_candidate:
            return 1
        return VOCABULARY_SIZE
    raise AssertionError(rank_variant)


def _score_records(signal: str = "positive") -> list[dict[str, Any]]:
    active = _active_coordinates()
    pools = {
        distribution: {
            _concept_id(coordinate) for coordinate in active if coordinate[0] == distribution
        }
        for distribution in audio_eval.DISTRIBUTIONS
    }
    records = []
    for coordinate in audio_eval.EXPECTED_COORDINATES:
        distribution, name = coordinate
        own = [_concept_id(coordinate)] if coordinate in active else []
        for voice in audio_eval.TTS_VARIANTS:
            positions = {}
            position_indices = {
                audio_eval.AUDIO_POSITION: len(EVALUATION_PREFIX_FRAMING_IDS) + 6,
                audio_eval.RESPONSE_POSITION: (
                    len(EVALUATION_PREFIX_FRAMING_IDS) + 7 + len(EVALUATION_SUFFIX_FRAMING_IDS) - 1
                ),
            }
            for position in audio_eval.POSITIONS:
                position_index = position_indices[position]
                layers = {}
                for layer in audio_eval.LENS_LAYERS:
                    concept_ranks = {
                        rank_variant: {
                            concept_id: _record_rank(
                                signal,
                                position=position,
                                layer=layer,
                                rank_variant=rank_variant,
                            )
                            for concept_id in own
                        }
                        for rank_variant in audio_eval.RANK_VARIANTS
                    }
                    own_candidate_rank = (
                        _record_rank(
                            signal,
                            position=position,
                            layer=layer,
                            rank_variant="candidate",
                        )
                        if own
                        else VOCABULARY_SIZE
                    )
                    label_pool_ranks = {
                        concept_id: (own_candidate_rank if concept_id in own else VOCABULARY_SIZE)
                        for concept_id in pools[distribution]
                    }
                    layer_record: dict[str, Any] = {
                        "concept_ranks": concept_ranks,
                        "candidate_label_pool_ranks": label_pool_ranks,
                    }
                    if position == audio_eval.RESPONSE_POSITION:
                        if layer in audio_eval.MOTOR_LAYERS:
                            actual_rank, divergence = 1, 0.01
                        else:
                            actual_rank, divergence = VOCABULARY_SIZE, 0.50
                        layer_record["motor"] = {
                            "actual_token_id": 7,
                            "actual_token_ranks": {
                                "candidate": actual_rank,
                                "logit": VOCABULARY_SIZE,
                            },
                            "candidate_logit_top1_agreement": False,
                            "candidate_logit_js_nats": divergence,
                        }
                    layers[str(layer)] = layer_record
                positions[position] = {"index": position_index, "layers": layers}
            records.append(
                {
                    "distribution": distribution,
                    "name": name,
                    "variant": voice,
                    "included_in_metrics": bool(own),
                    "eligible_concept_ids": own,
                    "vocabulary_size": VOCABULARY_SIZE,
                    "positions": positions,
                    "actual_output": {
                        "layer": 34,
                        "position": audio_eval.RESPONSE_POSITION,
                        "position_index": position_indices[audio_eval.RESPONSE_POSITION],
                        "token_id": 7,
                    },
                }
            )
    return records


def _eligibility() -> dict[str, Any]:
    active = _active_coordinates()
    items = []
    for coordinate in audio_eval.EXPECTED_COORDINATES:
        concept_ids = [_concept_id(coordinate)] if coordinate in active else []
        authored = "addition" if coordinate[0] == "order-ops" else "concept"
        items.append(
            {
                "distribution": coordinate[0],
                "name": coordinate[1],
                "included_in_metrics": bool(concept_ids),
                "eligible_concept_ids": concept_ids,
                "concepts": [
                    {
                        "concept_id": concept_id,
                        "authored": authored,
                        "forms": list(
                            audio_eval.allowed_forms(
                                coordinate[0],
                                authored,
                            )
                        ),
                        "allowed_token_ids": [7],
                    }
                    for concept_id in concept_ids
                ],
            }
        )
    return audio_eval.seal_mapping({"items": items}, "eligibility_sha256")


def _spoken_items() -> list[dict[str, Any]]:
    items = []
    for index, (distribution, name) in enumerate(audio_eval.EXPECTED_COORDINATES):
        language = audio_eval.language_for_coordinate(distribution, name)
        script = f"{language} {name} sealed spoken prefix"
        items.append(
            {
                "coordinate_index": index,
                "distribution": distribution,
                "name": name,
                "language": language,
                "script": script,
                "script_sha256": audio_eval.sha256_bytes(script.encode()),
                "source_item_sha256": audio_eval.canonical_sha256(
                    {"distribution": distribution, "name": name}
                ),
                "intermediates": ["addition" if distribution == "order-ops" else "concept"],
                "target_excluded": True,
            }
        )
    return items


def _source_identity() -> dict[str, Any]:
    return {
        "git_revision": "5" * 40,
        "source_sha256": "6" * 64,
        "lock_sha256": "7" * 64,
    }


def _runtime_environment(*, include_cuda: bool) -> dict[str, Any]:
    result = {
        "python": "3.12.0",
        "platform": "test-platform",
        "packages": {package: "1.0" for package in audio_eval.RUNTIME_PACKAGES},
        "modal_image_id": "im-test",
    }
    if include_cuda:
        result.update(cuda="12.8", device="H100")
    return result


def _processor_preparation(
    stimulus: dict[str, Any],
) -> dict[str, Any]:
    observations = []
    for observation in stimulus["observations"]:
        observations.append(
            {
                "observation_index": observation["observation_index"],
                "distribution": observation["distribution"],
                "name": observation["name"],
                "variant": observation["variant"],
                "normalized_wav_sha256": observation["normalized_wav_sha256"],
                "audio_start": len(EVALUATION_PREFIX_FRAMING_IDS),
                "n_audio_tokens": 7,
                "audio_stop": len(EVALUATION_PREFIX_FRAMING_IDS) + 7,
                "sequence_length": (
                    len(EVALUATION_PREFIX_FRAMING_IDS) + 7 + len(EVALUATION_SUFFIX_FRAMING_IDS)
                ),
                "max_sequence_length": 512,
                audio_eval.AUDIO_POSITION: len(EVALUATION_PREFIX_FRAMING_IDS) + 6,
                audio_eval.RESPONSE_POSITION: (
                    len(EVALUATION_PREFIX_FRAMING_IDS) + 7 + len(EVALUATION_SUFFIX_FRAMING_IDS) - 1
                ),
                "prefix_framing_ids": list(EVALUATION_PREFIX_FRAMING_IDS),
                "suffix_framing_ids": list(EVALUATION_SUFFIX_FRAMING_IDS),
                "model_inputs": [
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                        "sha256": str(offset) * 64,
                    }
                    for offset, (name, dtype, shape) in enumerate(
                        (
                            ("input_features", "torch.float32", [1, 4, 3]),
                            ("input_features_mask", "torch.bool", [1, 4]),
                            ("input_ids", "torch.int64", [1, 18]),
                        ),
                        start=1,
                    )
                ],
            }
        )
    body = {
        "kind": "audio_workspace_processor_preparation",
        "model": {
            "id": audio_eval.MODEL_ID,
            "revision": audio_eval.MODEL_REVISION,
        },
        "max_sequence_length": 512,
        "observations": observations,
    }
    return audio_eval.seal_mapping(body, "preparation_sha256")


def _preregistration_runtime(stimulus: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": _runtime_environment(include_cuda=True),
        "processor_preparation": _processor_preparation(stimulus),
    }


def _stimulus_manifest() -> dict[str, Any]:
    items = _spoken_items()
    observations = []
    for index in range(audio_eval.EXPECTED_OBSERVATION_COUNT):
        item_index = index // 2
        item = items[item_index]
        voice = audio_eval.TTS_VARIANTS[index % 2]
        phonemes = f"phonemes-{index}"
        observations.append(
            {
                "observation_index": index,
                "coordinate_index": item_index,
                "distribution": item["distribution"],
                "name": item["name"],
                "variant": voice,
                "language": item["language"],
                "script_sha256": item["script_sha256"],
                "wav_relative_path": f"wavs/{index:03d}-{voice}.wav",
                "source_wav_sha256": audio_eval.canonical_sha256(["source-wav", index]),
                "source_pcm_sha256": audio_eval.canonical_sha256(["source-pcm", index]),
                "normalized_wav_sha256": audio_eval.canonical_sha256(["wav", index]),
                "decoded_pcm_sha256": audio_eval.canonical_sha256(["pcm", index]),
                "source_sample_rate": audio_eval.TTS_SAMPLE_RATE,
                "sample_rate": audio_eval.NORMALIZED_SAMPLE_RATE,
                "sample_count": 16_000 + index,
                "duration_seconds": (16_000 + index) / 16_000,
                "argv": audio_eval.tts_argv(item["language"], voice, item["script"]),
                "phonemes": phonemes,
                "phoneme_sha256": audio_eval.sha256_bytes(phonemes.encode()),
            }
        )
    fit_rows = [
        {
            "audio_sha256": audio_eval.canonical_sha256(["fit-wav", index]),
            "decoded_pcm_sha256": audio_eval.canonical_sha256(["fit-pcm", index]),
            "transcript": f"fit transcript {index}",
        }
        for index in range(1_000)
    ]
    overlap = audio_eval.audit_fit_overlap(
        observations,
        items,
        fit_rows,
        fit_manifest_sha256="f" * 64,
    )
    return audio_eval.build_stimulus_manifest(
        items=items,
        observations=observations,
        overlap_audit=overlap,
        espeak_binary_sha256="3" * 64,
        espeak_voices_sha256="4" * 64,
        source_identity=_source_identity(),
        runtime_identity=_runtime_environment(include_cuda=False),
    )


def _calibration(stimulus: dict[str, Any]) -> dict[str, Any]:
    items = stimulus["items"]
    item_by_coordinate = {(item["distribution"], item["name"]): item for item in items}
    cells = []
    for coordinate in audio_eval.calibration_coordinates(items):
        item = item_by_coordinate[(coordinate["distribution"], coordinate["name"])]
        reference = item["script"]
        cells.append(
            {
                **coordinate,
                "reference": reference,
                "transcript": reference,
                "normalized_reference": audio_eval.normalize_transcript(reference),
                "normalized_transcript": audio_eval.normalize_transcript(reference),
                "cer": 0.0,
            }
        )
    return audio_eval.build_calibration(stimulus_manifest=stimulus, cells=cells)


def _preregistration() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stimulus = _stimulus_manifest()
    calibration = _calibration(stimulus)
    preregistration = audio_eval.build_preregistration(
        stimulus_manifest=stimulus,
        expected_items=_spoken_items(),
        calibration=calibration,
        artifact_identity={
            "fit_config_sha256": audio_eval.AUDIO_FIT_CONFIG_SHA256,
            "lens_sha256": audio_eval.AUDIO_LENS_SHA256,
            "lens_dtype": "float16",
            "lens_layers": list(audio_eval.LENS_LAYERS),
            "completed_run_manifest_sha256": "7" * 64,
        },
        historical_text_report=_text_report(),
        eligibility=_eligibility(),
        source_identity=_source_identity(),
        runtime_identity=_preregistration_runtime(stimulus),
    )
    return preregistration, stimulus, calibration


def _text_report() -> dict[str, Any]:
    active = _active_coordinates()
    items = []
    for coordinate in audio_eval.EXPECTED_COORDINATES:
        own = [_concept_id(coordinate)] if coordinate in active else []
        layers = {}
        for layer in audio_eval.LENS_LAYERS:
            candidate = 1 if layer in audio_eval.CANDIDATE_LAYERS else VOCABULARY_SIZE
            layers[str(layer)] = {
                "concept_ranks": {
                    "candidate": {concept_id: candidate for concept_id in own},
                    "logit": {concept_id: VOCABULARY_SIZE for concept_id in own},
                    "transposed": {concept_id: VOCABULARY_SIZE for concept_id in own},
                    "permuted": {concept_id: VOCABULARY_SIZE for concept_id in own},
                }
            }
        items.append(
            {
                "distribution": coordinate[0],
                "name": coordinate[1],
                "included_in_metrics": bool(own),
                "eligible_concept_ids": own,
                "layers": layers,
            }
        )
    body = {
        "kind": "workspace_jlens_evaluation",
        "status": "complete",
        "adjudication": {"status": "no_band"},
        "items": items,
    }
    return {
        **body,
        "workspace_report_sha256": audio_eval.canonical_sha256(body),
    }


def _interval(
    *,
    point: float = 0.02,
    lower: float = 1e-12,
    median: float = 0.1,
    upper: float = 0.2,
) -> dict[str, Any]:
    replicates = [lower, median, upper]
    return {
        "point": point,
        "lower_95": lower,
        "median": median,
        "upper_95": upper,
        "replicates": replicates,
        "replicate_sha256": audio_eval.canonical_sha256(replicates),
    }


def _passing_evidence() -> dict[str, Any]:
    return {
        "semantic_vs_logit": {
            "cell": f"all_five.{audio_eval.AUDIO_POSITION}",
            "candidate_minus_logit": _interval(),
            "label_max_stat_plus_one_p": 0.01,
            "distribution_point_deltas": {
                distribution: 0.0 for distribution in audio_eval.DISTRIBUTIONS
            },
            "variant_point_deltas": {variant: 0.0 for variant in audio_eval.TTS_VARIANTS},
        },
        "structural_controls": {
            "candidate_minus_transposed": _interval(),
            "candidate_minus_permuted": _interval(),
        },
        "fixed_region_localization": {
            "candidate_minus_early": _interval(),
            "candidate_minus_motor": _interval(),
            "historical_secondary_non_adjudicating": {},
        },
        "response_boundary_corroboration": {
            "scope": "four_non_multilingual_distributions",
            "candidate_minus_logit": _interval(),
            "candidate_minus_early": _interval(),
            "candidate_minus_motor": _interval(),
            "label_max_stat_plus_one_p": 0.01,
        },
        "motor_transition": {
            "reference": "L34_unmodified_actual_next_token_argmax",
            "motor_agreement_motor_minus_candidate": _interval(),
            "motor_js_motor_minus_candidate": _interval(
                point=-0.2,
                lower=-0.3,
                median=-0.2,
                upper=-1e-12,
            ),
        },
        "common_item_text_comparison": {"text_status": "no_band"},
    }


def test_physical_audio_artifact_chain_is_validated_before_identity_is_returned(
    monkeypatch,
):
    from audiolens import audio_fitting

    validated = {
        "fit_config_sha256": audio_eval.AUDIO_FIT_CONFIG_SHA256,
        "lens": {
            "fit_config_sha256": audio_eval.AUDIO_FIT_CONFIG_SHA256,
            "sha256": audio_eval.AUDIO_LENS_SHA256,
            "dtype": "float16",
            "source_layers": list(audio_eval.LENS_LAYERS),
            "d_model": audio_eval.D_MODEL,
        },
    }
    calls = []

    def fake_validate(record, *, volume_root):
        calls.append((record, volume_root))
        return copy.deepcopy(validated)

    completed_run_bytes = audio_eval.canonical_json_bytes({"sealed": True})
    completed_run_sha256 = audio_eval.sha256_bytes(completed_run_bytes)
    monkeypatch.setattr(audio_fitting, "validate_completed_run", fake_validate)
    identity = audio_eval.validate_audio_artifact_chain(
        {"sealed": True},
        completed_run_manifest_bytes=completed_run_bytes,
        volume_root="/vol",
        completed_run_manifest_sha256=completed_run_sha256,
    )
    assert calls == [({"sealed": True}, "/vol")]
    assert identity == {
        "fit_config_sha256": audio_eval.AUDIO_FIT_CONFIG_SHA256,
        "lens_sha256": audio_eval.AUDIO_LENS_SHA256,
        "lens_dtype": "float16",
        "lens_layers": list(audio_eval.LENS_LAYERS),
        "completed_run_manifest_sha256": completed_run_sha256,
    }
    for field, wrong in (
        ("fit_config_sha256", "0" * 64),
        ("sha256", "0" * 64),
        ("dtype", "float32"),
        ("source_layers", list(range(33))),
        ("d_model", 1024),
    ):
        forged = copy.deepcopy(validated)
        if field == "fit_config_sha256":
            forged[field] = wrong
        else:
            forged["lens"][field] = wrong
        monkeypatch.setattr(
            audio_fitting,
            "validate_completed_run",
            lambda record, *, volume_root, forged=forged: forged,
        )
        with pytest.raises(
            audio_eval.AudioWorkspaceEvalContractError,
            match="exact final audio lens",
        ):
            audio_eval.validate_audio_artifact_chain(
                {"sealed": True},
                completed_run_manifest_bytes=completed_run_bytes,
                volume_root="/vol",
                completed_run_manifest_sha256=completed_run_sha256,
            )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="bytes or SHA",
    ):
        audio_eval.validate_audio_artifact_chain(
            {"sealed": True},
            completed_run_manifest_bytes=completed_run_bytes + b" ",
            volume_root="/vol",
            completed_run_manifest_sha256=completed_run_sha256,
        )
    integer_bytes = audio_eval.canonical_json_bytes({"sealed": 1})
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="mapping does not match",
    ):
        audio_eval.validate_audio_artifact_chain(
            {"sealed": True},
            completed_run_manifest_bytes=integer_bytes,
            volume_root="/vol",
            completed_run_manifest_sha256=audio_eval.sha256_bytes(integer_bytes),
        )


def test_frozen_coordinates_configs_statuses_and_no_alternate_choices():
    assert len(audio_eval.EXPECTED_COORDINATES) == audio_eval.EXPECTED_ITEM_COUNT
    assert len(set(audio_eval.EXPECTED_COORDINATES)) == audio_eval.EXPECTED_ITEM_COUNT
    assert audio_eval.canonical_sha256(audio_eval.EXPECTED_COORDINATES) == (
        audio_eval.EXPECTED_COORDINATES_SHA256
    )
    assert {
        slug: sum(coordinate[0] == slug for coordinate in audio_eval.EXPECTED_COORDINATES)
        for slug in audio_eval.DISTRIBUTIONS
    } == audio_eval.PUBLICATION_COUNTS
    assert ("multilingual", "filipino-opposite-up") not in audio_eval.EXPECTED_COORDINATES
    assert ("multilingual", "irish-opposite-big") not in audio_eval.EXPECTED_COORDINATES
    assert "typo" not in audio_eval.DISTRIBUTIONS
    assert audio_eval.LENS_LAYERS == tuple(range(34))
    assert audio_eval.RESIDUAL_LAYERS == tuple(range(35))
    assert (
        *audio_eval.EARLY_LAYERS,
        *audio_eval.CANDIDATE_LAYERS,
        *audio_eval.MOTOR_LAYERS,
    ) == audio_eval.LENS_LAYERS
    assert audio_eval.TTS_VARIANTS == ("m1", "f1")
    assert len(audio_eval.LANGUAGE_TO_ESPEAK) == 34
    assert audio_eval.BOOTSTRAP_REPLICATES == audio_eval.PERMUTATION_REPLICATES == 10_000
    assert audio_eval.SCIENTIFIC_STATUSES == (
        "validated_fixed_band_synthetic_speech_readout",
        "no_fixed_band_synthetic_speech_readout",
        "inconclusive_synthetic_stimulus",
        "invalid_protocol_or_artifact",
    )
    protocol = audio_eval.frozen_protocol()
    assert protocol["claims"]["alternate_band_search"] is False
    assert protocol["inference"]["partial_scores"] is False
    assert protocol["inference"]["max_sequence_length"] == 512
    assert protocol["statistics"]["label_max_stat_cells"] == [
        {"scope": "all_five", "position": audio_eval.AUDIO_POSITION},
        {"scope": "all_five", "position": audio_eval.RESPONSE_POSITION},
        {"scope": "non_multilingual", "position": audio_eval.AUDIO_POSITION},
        {"scope": "non_multilingual", "position": audio_eval.RESPONSE_POSITION},
    ]
    assert protocol["controls"]["permuted"]["output_basis_sha256"] == (
        audio_eval.CONTROL_OUTPUT_BASIS_SHA256
    )


def test_spoken_boundaries_tts_argv_languages_and_allowed_forms_are_exact():
    assert (
        audio_eval.spoken_script(
            "association", {"prompt": "A complete sentence.", "intermediates": ["x"]}
        )
        == "A complete sentence."
    )
    assert (
        audio_eval.spoken_script("multihop", {"prompt": "answer is ", "target": "SECRET"})
        == "answer is "
    )
    assert (
        audio_eval.spoken_script("poetry", {"prompt": "header\nrhyme line\ncontinuation"})
        == "header\nrhyme line\n"
    )
    assert audio_eval.language_for_coordinate("association", "grief") == "en-us"
    assert audio_eval.language_for_coordinate("multilingual", "chinese-color-banana") == "cmn"
    assert audio_eval.tts_argv("en-us", "m1", "literal script") == [
        "espeak-ng",
        "-s",
        "180",
        "-p",
        "50",
        "-a",
        "100",
        "-g",
        "10",
        "-v",
        "en-us+m1",
        "--stdout",
        "literal script",
    ]
    assert audio_eval.allowed_forms("order-ops", "multiplication") == (
        "multiplication",
        "*",
        "×",
        "times",
        "multiply",
    )
    assert audio_eval.allowed_forms("multihop", "Brazil") == ("Brazil",)
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="unregistered"):
        audio_eval.allowed_forms("order-ops", "power")


def test_canonical_json_hashes_control_permutation_and_golden_leaf_equivalence():
    value = {"β": [3, 2, 1], "a": {"z": True}}
    assert audio_eval.canonical_json_bytes(value) == b'{"a":{"z":true},"\xce\xb2":[3,2,1]}'
    assert audio_eval.canonical_sha256(value) == audio_eval.canonical_sha256(copy.deepcopy(value))
    sealed = audio_eval.seal_mapping({"kind": "test", "value": value}, "sha256")
    assert audio_eval.validate_seal(sealed, "sha256", "test") == sealed
    value["a"]["z"] = False
    assert sealed["value"]["a"]["z"] is True
    assert [audio_eval.control_source_layer(layer) for layer in (0, 16, 17, 33)] == [17, 33, 0, 16]
    first_control = audio_eval.control_identity()
    assert first_control == audio_eval.control_identity()
    assert sorted(first_control["output_basis"]) == list(range(audio_eval.D_MODEL))
    assert first_control["output_basis_sha256"] == audio_eval.canonical_sha256(
        first_control["output_basis"]
    )
    assert first_control["output_basis_sha256"] == (audio_eval.CONTROL_OUTPUT_BASIS_SHA256)

    curve = {1: 0.0, 2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}
    canonical_leaf = 1.0 - math.log(2.0) / (2.0 * math.log(100.0))
    assert audio_eval.log_k_auc(curve) == pytest.approx(canonical_leaf)
    assert audio_eval.full_vocabulary_rank([0.0, 4.0, 4.0, 3.0], [1, 2]) == 1
    assert audio_eval.full_vocabulary_rank([4.0, 4.0, 3.0], [2]) == 3
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="nonfinite"):
        audio_eval.full_vocabulary_rank([0.0, math.nan], [0])
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="frozen k grid",
    ):
        audio_eval.log_k_auc({**curve, 200: 1.0})


def test_overlap_normalization_and_all_three_overlap_classes_reject():
    assert audio_eval.normalize_transcript(" Café,\tWORLD!! ") == "café world"
    assert audio_eval.character_error_rate("abcdefghij", "ab") == pytest.approx(0.8)
    stimulus = _stimulus_manifest()
    observations = stimulus["observations"]
    spoken = stimulus["items"]
    fit_rows = [
        {
            "audio_sha256": audio_eval.canonical_sha256(["other-wav", index]),
            "decoded_pcm_sha256": audio_eval.canonical_sha256(["other-pcm", index]),
            "transcript": f"different fit transcript {index}",
        }
        for index in range(1_000)
    ]
    clean = audio_eval.audit_fit_overlap(
        observations,
        spoken,
        fit_rows,
        fit_manifest_sha256="6" * 64,
    )
    assert clean["waveform_overlap_count"] == 0
    mutations = (
        ("audio_sha256", observations[0]["source_wav_sha256"]),
        ("decoded_pcm_sha256", observations[0]["source_pcm_sha256"]),
        ("transcript", spoken[0]["script"].swapcase()),
    )
    for key, overlap in mutations:
        overlapping_rows = copy.deepcopy(fit_rows)
        overlapping_rows[0][key] = overlap
        with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="overlap"):
            audio_eval.audit_fit_overlap(
                observations,
                spoken,
                overlapping_rows,
                fit_manifest_sha256="6" * 64,
            )
    missing_identity = copy.deepcopy(observations)
    del missing_identity[0]["source_wav_sha256"]
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="source_wav_sha256",
    ):
        audio_eval.audit_fit_overlap(
            missing_identity,
            spoken,
            fit_rows,
            fit_manifest_sha256="6" * 64,
        )


def test_exact_record_schema_two_positions_l0_l34_and_mutations_reject():
    records = _score_records()
    stimulus = _stimulus_manifest()
    runtime_identity = _preregistration_runtime(stimulus)
    validated = audio_eval.validate_score_records(
        records,
        eligibility=_eligibility(),
        runtime_identity=runtime_identity,
    )
    assert len(validated) == audio_eval.EXPECTED_OBSERVATION_COUNT
    assert [record["variant"] for record in validated[:4]] == ["m1", "f1", "m1", "f1"]
    for record in (validated[0], validated[-1]):
        assert set(record["positions"]) == set(audio_eval.POSITIONS)
        assert all(
            set(record["positions"][position]["layers"]) == {str(layer) for layer in range(34)}
            for position in audio_eval.POSITIONS
        )
        assert record["actual_output"] == {
            "layer": 34,
            "position": audio_eval.RESPONSE_POSITION,
            "position_index": record["positions"][audio_eval.RESPONSE_POSITION]["index"],
            "token_id": 7,
        }

    missing = copy.deepcopy(records)
    missing.pop()
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match=str(audio_eval.EXPECTED_OBSERVATION_COUNT),
    ):
        audio_eval.validate_score_records(missing)
    reordered = copy.deepcopy(records)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="order"):
        audio_eval.validate_score_records(reordered)
    unknown = copy.deepcopy(records)
    unknown[0]["debug"] = True
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="schema"):
        audio_eval.validate_score_records(unknown)
    missing_layer = copy.deepcopy(records)
    del missing_layer[0]["positions"][audio_eval.AUDIO_POSITION]["layers"]["0"]
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="L0-L33"):
        audio_eval.validate_score_records(missing_layer)
    bad_l34 = copy.deepcopy(records)
    bad_l34[0]["actual_output"]["layer"] = 33
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="L34"):
        audio_eval.validate_score_records(bad_l34)
    wrong_motor_token = copy.deepcopy(records)
    wrong_motor_token[0]["positions"][audio_eval.RESPONSE_POSITION]["layers"]["0"]["motor"][
        "actual_token_id"
    ] = 8
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="L34 actual token",
    ):
        audio_eval.validate_score_records(wrong_motor_token)
    nonfinite = copy.deepcopy(records)
    nonfinite[0]["positions"][audio_eval.RESPONSE_POSITION]["layers"]["0"]["motor"][
        "candidate_logit_js_nats"
    ] = math.nan
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="JS"):
        audio_eval.validate_score_records(nonfinite)
    wrong_pool = copy.deepcopy(records)
    own = wrong_pool[0]["eligible_concept_ids"][0]
    wrong_pool[0]["positions"][audio_eval.AUDIO_POSITION]["layers"]["0"][
        "candidate_label_pool_ranks"
    ][own] = 2
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="pool rank"):
        audio_eval.validate_score_records(wrong_pool)
    out_of_vocabulary = _eligibility()
    out_of_vocabulary["items"][0]["concepts"][0]["allowed_token_ids"] = [VOCABULARY_SIZE]
    out_of_vocabulary = audio_eval.seal_mapping(
        out_of_vocabulary,
        "eligibility_sha256",
    )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="preregistered eligibility",
    ):
        audio_eval.validate_score_records(
            records,
            eligibility=out_of_vocabulary,
        )
    wrong_preparation = copy.deepcopy(runtime_identity)
    preparation = wrong_preparation["processor_preparation"]
    preparation["observations"][0]["audio_start"] = 3
    preparation["observations"][0]["audio_stop"] = 10
    preparation["observations"][0][audio_eval.AUDIO_POSITION] = 9
    wrong_preparation["processor_preparation"] = audio_eval.seal_mapping(
        preparation, "preparation_sha256"
    )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="processor preparation observation geometry changed",
    ):
        audio_eval.validate_score_records(
            records,
            eligibility=_eligibility(),
            runtime_identity=wrong_preparation,
        )


def test_fair_per_layer_reducer_is_width_invariant_and_historical_min_is_secondary():
    record = _score_records()[0]
    one = audio_eval.fair_item_curve(
        record,
        position=audio_eval.AUDIO_POSITION,
        rank_variant="candidate",
        layers=(13,),
    )
    nineteen = audio_eval.fair_item_curve(
        record,
        position=audio_eval.AUDIO_POSITION,
        rank_variant="candidate",
        layers=audio_eval.CANDIDATE_LAYERS,
    )
    duplicated = audio_eval.fair_item_curve(
        record,
        position=audio_eval.AUDIO_POSITION,
        rank_variant="candidate",
        layers=(*audio_eval.CANDIDATE_LAYERS, *audio_eval.CANDIDATE_LAYERS),
    )
    assert one == nineteen == duplicated

    planted = copy.deepcopy(record)
    own = planted["eligible_concept_ids"][0]
    for layer in audio_eval.CANDIDATE_LAYERS[1:]:
        planted["positions"][audio_eval.AUDIO_POSITION]["layers"][str(layer)]["concept_ranks"][
            "candidate"
        ][own] = VOCABULARY_SIZE
    fair_auc = audio_eval.log_k_auc(
        audio_eval.fair_item_curve(
            planted,
            position=audio_eval.AUDIO_POSITION,
            rank_variant="candidate",
            layers=audio_eval.CANDIDATE_LAYERS,
        )
    )
    historical_auc = audio_eval.log_k_auc(
        audio_eval._historical_item_curve(
            planted, audio_eval.AUDIO_POSITION, "candidate", audio_eval.CANDIDATE_LAYERS
        )
    )
    assert historical_auc == 1.0
    assert fair_auc < historical_auc


def test_planted_positive_negative_and_control_signals_have_expected_summaries():
    positive = audio_eval.build_summaries(_score_records("positive"))
    candidate = positive["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["candidate"][
        "aggregate"
    ]["log_k_auc"]
    logit = positive["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["logit"][
        "aggregate"
    ]["log_k_auc"]
    early = positive["primary"][audio_eval.AUDIO_POSITION]["early_l0_l12"]["candidate"][
        "aggregate"
    ]["log_k_auc"]
    assert candidate == 1.0
    assert candidate > logit
    assert candidate > early

    negative = audio_eval.build_summaries(_score_records("negative"))
    negative_candidate = negative["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"][
        "candidate"
    ]["aggregate"]["log_k_auc"]
    negative_logit = negative["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["logit"][
        "aggregate"
    ]["log_k_auc"]
    assert negative_candidate < negative_logit

    control = audio_eval.build_summaries(_score_records("control"))
    assert (
        control["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["candidate"][
            "aggregate"
        ]["log_k_auc"]
        == control["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["transposed"][
            "aggregate"
        ]["log_k_auc"]
    )


def test_bundled_bootstrap_and_four_cell_label_max_stat_are_deterministic_and_effective():
    records = _score_records("positive")
    first = audio_eval.bundled_item_bootstrap(records, replicates=25, seed=17)
    second = audio_eval.bundled_item_bootstrap(copy.deepcopy(records), replicates=25, seed=17)
    assert first == second
    for name, contrast in first["contrasts"].items():
        assert contrast["replicate_sha256"] == audio_eval.canonical_sha256(contrast["replicates"])
        if "motor_js" in name:
            assert contrast["upper_95"] < 0
        else:
            assert contrast["lower_95"] > 0

    perm_one = audio_eval.label_permutation_max_stat(records, replicates=200, seed=19)
    perm_two = audio_eval.label_permutation_max_stat(
        copy.deepcopy(records), replicates=200, seed=19
    )
    assert perm_one == perm_two
    assert tuple(perm_one["cells"]) == audio_eval.LABEL_MAX_STAT_CELLS
    assert len(perm_one["max_null"]) == 200
    assert perm_one["effective_replicates"] > 0
    assert perm_one["max_null_sha256"] == audio_eval.canonical_sha256(perm_one["max_null"])
    for cell, values in perm_one["cell_null"].items():
        assert perm_one["cell_null_sha256"][cell] == (audio_eval.canonical_sha256(values))
    assert set(perm_one["plus_one_p_values"]) == set(audio_eval.LABEL_MAX_STAT_CELLS)
    assert all(value >= 1 / 201 for value in perm_one["plus_one_p_values"].values())
    assert perm_one["plus_one_p_values"][f"all_five.{audio_eval.AUDIO_POSITION}"] <= 0.01


def test_calibration_macro_and_cell_thresholds_are_inclusive_and_exact():
    cells = audio_eval.EXPECTED_CALIBRATION_CELL_COUNT
    assert audio_eval.calibration_status([0.35] * cells)["status"] == "passed"
    assert audio_eval.calibration_status([0.0] * (cells - 1) + [0.80])["status"] == "passed"
    assert audio_eval.calibration_status([math.nextafter(0.35, 1.0)] * cells)["status"] == "failed"
    assert (
        audio_eval.calibration_status([0.0] * (cells - 1) + [math.nextafter(0.80, 1.0)])["status"]
        == "failed"
    )


def test_each_adjudication_criterion_is_independently_required():
    mutations = [
        lambda value: value["semantic_vs_logit"]["candidate_minus_logit"].update(
            point=math.nextafter(0.02, 0.0)
        ),
        lambda value: value["semantic_vs_logit"]["candidate_minus_logit"].update(lower_95=0.0),
        lambda value: value["semantic_vs_logit"].update(
            label_max_stat_plus_one_p=math.nextafter(0.01, 1.0)
        ),
        lambda value: value["semantic_vs_logit"]["distribution_point_deltas"].update(
            association=-1e-12
        ),
        lambda value: value["semantic_vs_logit"]["variant_point_deltas"].update(m1=-1e-12),
        lambda value: value["structural_controls"]["candidate_minus_transposed"].update(
            lower_95=0.0
        ),
        lambda value: value["structural_controls"]["candidate_minus_permuted"].update(lower_95=0.0),
        lambda value: value["fixed_region_localization"]["candidate_minus_early"].update(
            lower_95=0.0
        ),
        lambda value: value["fixed_region_localization"]["candidate_minus_motor"].update(
            lower_95=0.0
        ),
        lambda value: value["response_boundary_corroboration"]["candidate_minus_logit"].update(
            lower_95=0.0
        ),
        lambda value: value["response_boundary_corroboration"]["candidate_minus_early"].update(
            lower_95=0.0
        ),
        lambda value: value["response_boundary_corroboration"]["candidate_minus_motor"].update(
            lower_95=0.0
        ),
        lambda value: value["motor_transition"]["motor_agreement_motor_minus_candidate"].update(
            lower_95=0.0
        ),
        lambda value: value["motor_transition"]["motor_js_motor_minus_candidate"].update(
            upper_95=0.0
        ),
    ]
    assert len(mutations) == len(audio_eval.CRITERIA)
    for criterion, mutate in zip(audio_eval.CRITERIA, mutations, strict=True):
        evidence = _passing_evidence()
        mutate(evidence)
        result = audio_eval.adjudicate(evidence)
        assert result["status"] == audio_eval.NO_READOUT_STATUS
        assert criterion in result["failed_criteria"]


def test_adjudication_threshold_boundaries_every_status_and_audio_only_failure():
    passing = _passing_evidence()
    result = audio_eval.adjudicate(passing)
    assert result["status"] == audio_eval.VALIDATED_STATUS
    assert result["failed_criteria"] == []
    assert result["searched_alternate_bands"] is False
    assert (
        audio_eval.determine_status(
            protocol_valid=True, calibration_passed=True, complete=True, evidence=passing
        )
        == audio_eval.VALIDATED_STATUS
    )
    assert (
        audio_eval.determine_status(
            protocol_valid=True, calibration_passed=False, complete=True, evidence=None
        )
        == audio_eval.INCONCLUSIVE_STIMULUS_STATUS
    )
    assert (
        audio_eval.determine_status(
            protocol_valid=False, calibration_passed=False, complete=True, evidence=None
        )
        == audio_eval.INVALID_PROTOCOL_STATUS
    )

    failed = copy.deepcopy(passing)
    failed["semantic_vs_logit"]["candidate_minus_logit"]["point"] = math.nextafter(0.02, 0.0)
    assert audio_eval.adjudicate(failed)["status"] == audio_eval.NO_READOUT_STATUS
    failed = copy.deepcopy(passing)
    failed["semantic_vs_logit"]["candidate_minus_logit"]["lower_95"] = 0.0
    assert audio_eval.CRITERIA[1] in audio_eval.adjudicate(failed)["failed_criteria"]
    failed = copy.deepcopy(passing)
    failed["semantic_vs_logit"]["label_max_stat_plus_one_p"] = math.nextafter(0.01, 1.0)
    assert audio_eval.CRITERIA[2] in audio_eval.adjudicate(failed)["failed_criteria"]
    failed = copy.deepcopy(passing)
    failed["motor_transition"]["motor_js_motor_minus_candidate"]["upper_95"] = 0.0
    assert audio_eval.CRITERIA[-1] in audio_eval.adjudicate(failed)["failed_criteria"]
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="pending"):
        audio_eval.determine_status(
            protocol_valid=True, calibration_passed=True, complete=False, evidence=passing
        )
    missing_evidence = copy.deepcopy(passing)
    del missing_evidence["motor_transition"]["reference"]
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="schema",
    ):
        audio_eval.adjudicate(missing_evidence)
    unknown_evidence = copy.deepcopy(passing)
    unknown_evidence["semantic_vs_logit"]["alternate_band"] = True
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="schema",
    ):
        audio_eval.adjudicate(unknown_evidence)

    audio_only_records = _score_records("audio_only")
    bootstrap = audio_eval.bundled_item_bootstrap(audio_only_records, replicates=20, seed=7)
    assert (
        bootstrap["contrasts"][f"all_five.{audio_eval.AUDIO_POSITION}.candidate_minus_logit"][
            "lower_95"
        ]
        > 0
    )
    assert (
        bootstrap["contrasts"][
            f"non_multilingual.{audio_eval.RESPONSE_POSITION}.candidate_minus_logit"
        ]["lower_95"]
        == 0
    )


def test_stimulus_calibration_preregistration_round_trip_and_resealed_mutations_reject():
    preregistration, stimulus, calibration = _preregistration()
    assert audio_eval.validate_stimulus_manifest(stimulus) == stimulus
    assert (
        audio_eval.validate_stimulus_manifest(
            stimulus,
            expected_items=_spoken_items(),
        )
        == stimulus
    )
    assert len(stimulus["items"]) == audio_eval.EXPECTED_ITEM_COUNT
    assert len(stimulus["observations"]) == audio_eval.EXPECTED_OBSERVATION_COUNT
    assert calibration["status"] == "passed"
    assert len(calibration["cells"]) == audio_eval.EXPECTED_CALIBRATION_CELL_COUNT
    assert audio_eval.validate_calibration(calibration, stimulus) == calibration
    assert (
        audio_eval.validate_preregistration(
            preregistration,
            stimulus_manifest=stimulus,
            expected_items=_spoken_items(),
            calibration=calibration,
        )
        == preregistration
    )

    unsealed = copy.deepcopy(stimulus)
    unsealed["observations"][0]["sample_count"] += 1
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="digest"):
        audio_eval.validate_stimulus_manifest(unsealed)
    resealed = copy.deepcopy(stimulus)
    resealed["observations"][0]["variant"] = "alternate"
    resealed = audio_eval.seal_mapping(resealed, "stimulus_manifest_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="order"):
        audio_eval.validate_stimulus_manifest(resealed)
    unknown = copy.deepcopy(stimulus)
    unknown["debug"] = True
    unknown = audio_eval.seal_mapping(unknown, "stimulus_manifest_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="schema"):
        audio_eval.validate_stimulus_manifest(unknown)
    bad_source = copy.deepcopy(stimulus)
    bad_source["source_identity"]["debug"] = True
    bad_source = audio_eval.seal_mapping(bad_source, "stimulus_manifest_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="source identity schema",
    ):
        audio_eval.validate_stimulus_manifest(bad_source)
    duplicate_path = copy.deepcopy(stimulus)
    duplicate_path["observations"][1]["wav_relative_path"] = duplicate_path["observations"][0][
        "wav_relative_path"
    ]
    duplicate_path = audio_eval.seal_mapping(duplicate_path, "stimulus_manifest_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="WAV path identity",
    ):
        audio_eval.validate_stimulus_manifest(duplicate_path)
    nontext_phonemes = copy.deepcopy(stimulus)
    nontext_phonemes["observations"][0]["phonemes"] = 123
    nontext_phonemes["observations"][0]["phoneme_sha256"] = audio_eval.sha256_bytes(b"123")
    nontext_phonemes = audio_eval.seal_mapping(nontext_phonemes, "stimulus_manifest_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="observation hashes",
    ):
        audio_eval.validate_stimulus_manifest(nontext_phonemes)

    script_mutation = copy.deepcopy(stimulus)
    mutated_script = "resealed alternate confirmatory script"
    mutated_script_sha = audio_eval.sha256_bytes(mutated_script.encode())
    script_mutation["items"][0]["script"] = mutated_script
    script_mutation["items"][0]["script_sha256"] = mutated_script_sha
    for observation in script_mutation["observations"][:2]:
        observation["script_sha256"] = mutated_script_sha
        observation["argv"] = audio_eval.tts_argv(
            observation["language"],
            observation["variant"],
            mutated_script,
        )
    script_mutation["overlap_audit"]["stimulus_transcript_set_sha256"] = (
        audio_eval.canonical_sha256(
            sorted(
                {
                    audio_eval.normalize_transcript(item["script"])
                    for item in script_mutation["items"]
                }
            )
        )
    )
    script_mutation = audio_eval.seal_mapping(
        script_mutation,
        "stimulus_manifest_sha256",
    )
    assert audio_eval.validate_stimulus_manifest(script_mutation) == script_mutation
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="scripts"):
        audio_eval.validate_stimulus_manifest(
            script_mutation,
            expected_items=_spoken_items(),
        )

    bad_calibration = copy.deepcopy(calibration)
    bad_calibration["cells"][0]["cer"] = 0.1
    bad_calibration = audio_eval.seal_mapping(bad_calibration, "calibration_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="CER"):
        audio_eval.validate_calibration(bad_calibration, stimulus)

    bad_prereg = copy.deepcopy(preregistration)
    bad_prereg["artifact_identity"]["lens_sha256"] = "0" * 64
    bad_prereg = audio_eval.seal_mapping(bad_prereg, "preregistration_sha256")
    bad_calibration_scalar = copy.deepcopy(calibration)
    bad_calibration_scalar["macro_cer"] = "0.0"
    bad_calibration_scalar = audio_eval.seal_mapping(bad_calibration_scalar, "calibration_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="calibration result",
    ):
        audio_eval.validate_calibration(bad_calibration_scalar, stimulus)
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="final audio lens"):
        audio_eval.validate_preregistration(bad_prereg)
    bad_forms = copy.deepcopy(preregistration)
    bad_forms["eligibility"]["items"][0]["concepts"][0]["forms"].append("alternate")
    bad_forms["eligibility"] = audio_eval.seal_mapping(
        bad_forms["eligibility"],
        "eligibility_sha256",
    )
    bad_forms = audio_eval.seal_mapping(
        bad_forms,
        "preregistration_sha256",
    )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="authored forms",
    ):
        audio_eval.validate_preregistration(bad_forms)
    wrong_authored = copy.deepcopy(preregistration)
    concept = wrong_authored["eligibility"]["items"][0]["concepts"][0]
    concept["authored"] = "invented"
    concept["forms"] = ["invented"]
    wrong_authored["eligibility"] = audio_eval.seal_mapping(
        wrong_authored["eligibility"], "eligibility_sha256"
    )
    wrong_authored = audio_eval.seal_mapping(wrong_authored, "preregistration_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="canonical stimulus intermediate",
    ):
        audio_eval.validate_preregistration(
            wrong_authored,
            stimulus_manifest=stimulus,
            expected_items=_spoken_items(),
            calibration=calibration,
        )


def test_common_item_text_summary_recomputes_and_rejects_mutation():
    text_report = _text_report()
    summary = audio_eval.recompute_common_item_text_summary(text_report)
    assert summary["n_common_items"] == audio_eval.EXPECTED_ITEM_COUNT
    assert summary["coordinates_sha256"] == audio_eval.EXPECTED_COORDINATES_SHA256
    assert summary["text_status"] == "no_band"
    assert summary["canonical_text_verdict_unchanged"] is True
    assert (
        summary["primary_fair_region"]["candidate_l13_l31"]["candidate"]["aggregate"]["log_k_auc"]
        == 1.0
    )
    assert audio_eval.validate_common_item_text_summary(summary, text_report) == summary
    mutated = copy.deepcopy(summary)
    mutated["primary_fair_region"]["candidate_l13_l31"]["candidate"]["aggregate"]["log_k_auc"] = 0.5
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="recompute"):
        audio_eval.validate_common_item_text_summary(mutated, text_report)
    forged_report = copy.deepcopy(text_report)
    forged_report["items"][0]["layers"]["13"]["concept_ranks"]["candidate"][
        _concept_id(audio_eval.EXPECTED_COORDINATES[0])
    ] = 2
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="content digest",
    ):
        audio_eval.recompute_common_item_text_summary(forged_report)


def test_complete_report_self_hash_round_trip_and_unsealed_resealed_unknown_mutations(monkeypatch):
    preregistration, _, _ = _preregistration()
    records = _score_records("positive")
    text_report = _text_report()
    bootstrap = audio_eval.bundled_item_bootstrap(records, replicates=20, seed=13)
    permutation = audio_eval.label_permutation_max_stat(records, replicates=100, seed=14)
    monkeypatch.setattr(audio_eval, "bundled_item_bootstrap", lambda rows: bootstrap)
    monkeypatch.setattr(audio_eval, "label_permutation_max_stat", lambda rows: permutation)

    report = audio_eval.build_report(
        preregistration=preregistration,
        records=records,
        historical_text_report=text_report,
    )
    assert report["status"] == audio_eval.VALIDATED_STATUS
    records[0]["actual_output"]["token_id"] = 8
    assert report["records"][0]["actual_output"]["token_id"] == 7
    assert (
        audio_eval.validate_report(
            report,
            preregistration=preregistration,
            historical_text_report=_text_report(),
        )
        == report
    )
    body = dict(report)
    claimed = body.pop("report_sha256")
    assert claimed == audio_eval.canonical_sha256(body)
    failed_preregistration = copy.deepcopy(preregistration)
    failed_preregistration["calibration_status"] = "failed"
    failed_preregistration = audio_eval.seal_mapping(
        failed_preregistration, "preregistration_sha256"
    )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="failed calibration",
    ):
        audio_eval.validate_report(
            report,
            preregistration=failed_preregistration,
            historical_text_report=text_report,
        )

    forged_text = copy.deepcopy(report)
    forged_summary = forged_text["common_item_text_summary"]
    forged_summary["primary_fair_region"]["candidate_l13_l31"]["candidate"]["aggregate"][
        "log_k_auc"
    ] = 0.5
    summary_body = dict(forged_summary)
    summary_body.pop("common_item_text_summary_sha256")
    forged_summary["common_item_text_summary_sha256"] = audio_eval.canonical_sha256(summary_body)
    forged_text["evidence"]["common_item_text_comparison"] = copy.deepcopy(forged_summary)
    forged_text = audio_eval.seal_mapping(forged_text, "report_sha256")
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="recompute",
    ):
        audio_eval.validate_report(
            forged_text,
            preregistration=preregistration,
            historical_text_report=_text_report(),
        )

    unsealed = copy.deepcopy(report)
    unsealed["summaries"]["primary"][audio_eval.AUDIO_POSITION]["candidate_l13_l31"]["candidate"][
        "aggregate"
    ]["log_k_auc"] = 0.5
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="digest"):
        audio_eval.validate_report(
            unsealed,
            preregistration=preregistration,
            historical_text_report=_text_report(),
        )
    resealed = audio_eval.seal_mapping(unsealed, "report_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="summaries"):
        audio_eval.validate_report(
            resealed,
            preregistration=preregistration,
            historical_text_report=_text_report(),
        )
    unknown = copy.deepcopy(report)
    unknown["alternate_band"] = [14, 15]
    unknown = audio_eval.seal_mapping(unknown, "report_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="schema"):
        audio_eval.validate_report(
            unknown,
            preregistration=preregistration,
            historical_text_report=_text_report(),
        )

    eligibility_mutation = copy.deepcopy(preregistration)
    eligibility_mutation["eligibility"]["items"][0]["eligible_concept_ids"] = []
    eligibility_mutation["eligibility"]["items"][0]["concepts"] = []
    eligibility_mutation["eligibility"]["items"][0]["included_in_metrics"] = False
    eligibility_mutation["eligibility"] = audio_eval.seal_mapping(
        eligibility_mutation["eligibility"], "eligibility_sha256"
    )
    eligibility_mutation = audio_eval.seal_mapping(eligibility_mutation, "preregistration_sha256")
    with pytest.raises(audio_eval.AudioWorkspaceEvalContractError, match="identity"):
        audio_eval.validate_report(
            report,
            preregistration=eligibility_mutation,
            historical_text_report=_text_report(),
        )


def _load_modal_audio_workspace_eval(monkeypatch):
    import importlib.util
    import pathlib
    import sys
    import uuid

    monkeypatch.setenv("AUDIOLENS_DISABLE_MODAL", "1")
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "modal_audio_workspace_eval.py"
    name = f"modal_audio_workspace_eval_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_deployment_module_is_import_light_and_has_only_four_explicit_modes(
    monkeypatch,
):
    import inspect

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    assert deployment.app is None
    assert deployment.image is None
    assert callable(deployment.preregister_experiment)
    assert callable(deployment.smoke_experiment)
    assert callable(deployment.evaluate_experiment)
    assert callable(deployment.validate_report_experiment)
    assert list(inspect.signature(deployment.main).parameters) == [
        "preregister",
        "smoke",
        "evaluate",
        "validate_report",
        "preregistration",
        "sha256",
    ]
    assert "limit" not in inspect.signature(deployment.main).parameters
    assert "debug" not in inspect.signature(deployment.main).parameters


def test_deployment_bounded_reader_rejects_symlinks_and_oversize_files(
    monkeypatch,
    tmp_path,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    source = tmp_path / "source.bin"
    source.write_bytes(b"bound")
    assert deployment._read_bounded_bytes(source, label="test", maximum=5) == b"bound"
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(deployment.ModalAudioWorkspaceEvalError, match="unsafe"):
        deployment._read_bounded_bytes(link, label="test", maximum=5)
    with pytest.raises(deployment.ModalAudioWorkspaceEvalError, match="byte size"):
        deployment._read_bounded_bytes(source, label="test", maximum=4)


def test_deployment_archive_guard_accepts_internal_symlink(monkeypatch):
    import tarfile

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    internal = tarfile.TarInfo("espeak-ng-1.52.0/src/include/espeak/speak_lib.h")
    internal.type = tarfile.SYMTYPE
    internal.linkname = "../espeak-ng/speak_lib.h"
    deployment._validate_espeak_archive_member(internal)

    escaping = tarfile.TarInfo("espeak-ng-1.52.0/src/include/escape.h")
    escaping.type = tarfile.SYMTYPE
    escaping.linkname = "../../../../etc/passwd"
    with pytest.raises(RuntimeError, match="unsafe eSpeak archive link"):
        deployment._validate_espeak_archive_member(escaping)


def test_deployment_literal_espeak_argv_never_uses_a_shell(monkeypatch):
    import types

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        output = b"RIFF-fake" if "--stdout" in argv else b"f eI k"
        return types.SimpleNamespace(stdout=output)

    wav, phonemes, recorded = deployment._run_espeak(
        "/opt/espeak-ng/bin/espeak-ng",
        "fr",
        "m1",
        "Bonjour",
        runner=runner,
    )
    assert wav == b"RIFF-fake"
    assert phonemes == b"f eI k"
    assert recorded == [
        "espeak-ng",
        "-s",
        "180",
        "-p",
        "50",
        "-a",
        "100",
        "-g",
        "10",
        "-v",
        "fr+m1",
        "--stdout",
        "Bonjour",
    ]
    assert calls[0][0] == [
        "/opt/espeak-ng/bin/espeak-ng",
        *recorded[1:],
    ]
    assert all(call[1] == {"check": True, "capture_output": True} for call in calls)


def test_deployment_fake_smoke_duplicates_inference_and_never_uses_final_lens(
    monkeypatch,
    tmp_path,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    rows = [
        {
            "name": f"nonpublication-{index}",
            "prompt": f"smoke prompt {index}",
            "intermediates": ["smoke"],
        }
        for index in range(52)
    ]
    calls = []

    def inference(items):
        calls.append(copy.deepcopy(list(items)))
        concept_id = "nonconfirmatory-smoke/concept"
        position_indices = {
            "last_processor_valid_audio_position": 5,
            "response_position": 9,
        }
        positions = {}
        for position_name, position_index in position_indices.items():
            layers = {}
            for layer in range(34):
                candidate_rank = layer + 1
                layer_record = {
                    "concept_ranks": {
                        control: {concept_id: candidate_rank + offset}
                        for offset, control in enumerate(
                            ("candidate", "logit", "transposed", "permuted")
                        )
                    },
                    "candidate_label_pool_ranks": {concept_id: candidate_rank},
                }
                if position_name == "response_position":
                    layer_record["motor"] = {
                        "actual_token_id": 17,
                        "actual_token_ranks": {
                            "candidate": candidate_rank,
                            "logit": candidate_rank + 1,
                        },
                        "candidate_logit_top1_agreement": True,
                        "candidate_logit_js_nats": 0.0,
                    }
                layers[str(layer)] = layer_record
            positions[position_name] = {
                "index": position_index,
                "layers": layers,
            }
        score = {
            "vocabulary_size": 128,
            "positions": positions,
            "actual_output": {
                "layer": 34,
                "position": "response_position",
                "position_index": 9,
                "token_id": 17,
            },
        }
        return [
            {
                "namespace": "nonconfirmatory-smoke",
                "name": item["name"],
                "score": score,
                "audio": {
                    "source_sample_rate": 22_050,
                    "source_sample_count": 10,
                    "source_decoded_pcm_sha256": "1" * 64,
                    "sample_rate": 16_000,
                    "sample_count": 8,
                    "duration_seconds": 8 / 16_000,
                    "decoded_pcm_sha256": "2" * 64,
                },
            }
            for item in items
        ]

    monkeypatch.setattr(deployment, "_decode_fixture", lambda spec, raw: rows)
    monkeypatch.setattr(
        deployment,
        "_require_source_identity",
        lambda: {
            "git_revision": "a" * 40,
            "source_sha256": "b" * 64,
            "lock_sha256": "c" * 64,
        },
    )
    published = []
    monkeypatch.setattr(
        deployment,
        "_write_content_addressed_json",
        lambda root, report: (
            published.append(copy.deepcopy(report)) or (tmp_path / "smoke.json", "d" * 64)
        ),
    )
    commits = []
    monkeypatch.setattr(deployment, "_commit_volume", lambda: commits.append(True))

    result = deployment._smoke_impl(
        fixture_loader=lambda: {"association": b"pinned"},
        inference=inference,
    )
    assert calls[0] == calls[1]
    assert len(calls) == 2
    assert all(item["name"].startswith("nonconfirmatory/") for item in calls[0])
    assert result["records"] == 2
    assert result["sha256"] == "d" * 64
    assert published[0]["uses_final_lens"] is False
    assert published[0]["uses_confirmatory_items"] is False
    assert set(published[0]["records"][0]["score"]["positions"]) == {
        "last_processor_valid_audio_position",
        "response_position",
    }
    assert set(published[0]["records"][0]["score"]["positions"]["response_position"]["layers"]) == {
        str(layer) for layer in range(34)
    }
    assert set(
        published[0]["records"][0]["score"]["positions"]["response_position"]["layers"]["0"][
            "concept_ranks"
        ]
    ) == {"candidate", "logit", "transposed", "permuted"}
    assert commits == [True]


def test_deployment_confirmatory_failure_publishes_no_score_artifact(
    monkeypatch,
    tmp_path,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    preregistration = {"preregistration_sha256": "a" * 64}
    monkeypatch.setattr(
        deployment,
        "_validate_preregistration_file",
        lambda path, sha256: preregistration,
    )
    monkeypatch.setattr(
        deployment,
        "_require_runtime_environment",
        lambda value: None,
    )
    monkeypatch.setattr(
        deployment,
        "_validate_physical_stimuli",
        lambda value: ({"items": []}, tmp_path / "manifest.json", {}),
    )
    monkeypatch.setattr(
        deployment,
        "_bind_items_to_text_report",
        lambda items, text: None,
    )
    monkeypatch.setattr(
        deployment,
        "_score_confirmatory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected scoring interruption")
        ),
    )
    writes = []
    monkeypatch.setattr(
        deployment,
        "_atomic_immutable_write",
        lambda path, payload: writes.append((path, payload)),
    )
    commits = []
    monkeypatch.setattr(deployment, "_commit_volume", lambda: commits.append(True))

    with pytest.raises(RuntimeError, match="injected scoring interruption"):
        deployment._evaluate_impl(
            "/vol/preregistration.json",
            "a" * 64,
            text_loader=lambda: {},
        )
    assert writes == []
    assert commits == []


def test_deployment_complete_report_validates_before_one_atomic_publish(
    monkeypatch,
    tmp_path,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    preregistration = {"preregistration_sha256": "a" * 64}
    monkeypatch.setattr(
        deployment,
        "_validate_preregistration_file",
        lambda path, sha256: preregistration,
    )
    monkeypatch.setattr(
        deployment,
        "_require_runtime_environment",
        lambda value: None,
    )
    monkeypatch.setattr(
        deployment,
        "_validate_physical_stimuli",
        lambda value: ({"items": []}, tmp_path / "manifest.json", {}),
    )
    monkeypatch.setattr(
        deployment,
        "_bind_items_to_text_report",
        lambda items, text: None,
    )
    monkeypatch.setattr(
        deployment,
        "_score_confirmatory",
        lambda *args, **kwargs: ([{"complete": True}], {"physical": True}),
    )
    validated = []

    def report_validator(report, bound_preregistration, historical_text_report):
        assert report == {"built": True}
        assert bound_preregistration is preregistration
        validated.append(True)
        return {
            "status": audio_eval.NO_READOUT_STATUS,
            "report_sha256": "e" * 64,
        }

    writes = []

    def atomic_write(path, payload):
        assert validated == [True]
        writes.append((path, payload))
        return path

    monkeypatch.setattr(deployment, "REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(deployment, "_atomic_immutable_write", atomic_write)
    commits = []
    monkeypatch.setattr(deployment, "_commit_volume", lambda: commits.append(True))
    result = deployment._evaluate_impl(
        "/vol/preregistration.json",
        "a" * 64,
        report_builder=lambda *args: {"built": True},
        report_validator=report_validator,
        text_loader=lambda: {},
    )
    assert len(writes) == 1
    assert writes[0][0] == tmp_path / "reports" / f"{'e' * 64}.json"
    assert commits == [True]
    assert result == {
        "mode": "evaluate",
        "status": audio_eval.NO_READOUT_STATUS,
        "path": str(tmp_path / "reports" / f"{'e' * 64}.json"),
        "sha256": "e" * 64,
        "records": audio_eval.EXPECTED_OBSERVATION_COUNT,
    }


def test_deployment_fake_staging_writes_exact_wavs_and_pure_manifest(
    monkeypatch,
    tmp_path,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    items = _spoken_items()
    counter = {"value": 0}

    monkeypatch.setattr(
        deployment,
        "_engine_identity",
        lambda binary, runner: {
            "version": "1.52.0",
            "binary_sha256": "1" * 64,
            "voices_sha256": "2" * 64,
        },
    )

    def fake_espeak(binary, language, variant, script, runner):
        index = counter["value"]
        counter["value"] += 1
        source = f"source-{index}".encode()
        phonemes = f"phoneme-{index}".encode()
        return source, phonemes, audio_eval.tts_argv(language, variant, script)

    def fake_normalize(source):
        normalized = b"normalized-" + source
        return normalized, {
            "source_sample_rate": 22_050,
            "source_sample_count": 10,
            "source_decoded_pcm_sha256": deployment._sha256_bytes(source),
            "sample_rate": 16_000,
            "sample_count": 8,
            "duration_seconds": 8 / 16_000,
            "decoded_pcm_sha256": deployment._sha256_bytes(normalized),
        }

    monkeypatch.setattr(deployment, "_run_espeak", fake_espeak)
    monkeypatch.setattr(deployment, "_normalize_wav", fake_normalize)
    monkeypatch.setattr(deployment, "STIMULUS_ROOT", str(tmp_path / "stimuli"))
    manifest, path = deployment._stage_stimuli(
        items,
        [
            {
                "transcript": f"held out fit transcript {index}",
                "audio_sha256": "3" * 64,
                "decoded_pcm_sha256": "4" * 64,
            }
            for index in range(1_000)
        ],
        "5" * 64,
        {
            "git_revision": "a" * 40,
            "source_sha256": "b" * 64,
            "lock_sha256": "c" * 64,
        },
        {
            "python": "3.12.7",
            "platform": "test-platform",
            "packages": {package: "test-version" for package in audio_eval.RUNTIME_PACKAGES},
            "modal_image_id": "im-test",
        },
        binary="/fake/espeak-ng",
        runner=lambda *args, **kwargs: None,
    )
    assert len(manifest["observations"]) == audio_eval.EXPECTED_OBSERVATION_COUNT
    assert len(list((path.parent / "wavs").glob("*.wav"))) == audio_eval.EXPECTED_OBSERVATION_COUNT
    assert (
        audio_eval.validate_stimulus_manifest(
            manifest,
            expected_items=items,
        )
        == manifest
    )


def test_deployment_eligibility_binds_authored_forms_and_token_ids(monkeypatch):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)

    class Tokenizer:
        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [sum(text.encode("utf-8"))]}

    item = {
        "distribution": "order-ops",
        "name": "test-coordinate",
        "intermediates": ["subtraction"],
    }
    eligibility = deployment._eligibility([item], Tokenizer())
    concept = eligibility["items"][0]["concepts"][0]
    assert set(concept) == {
        "concept_id",
        "authored",
        "forms",
        "allowed_token_ids",
    }
    assert concept["authored"] == "subtraction"
    assert concept["forms"] == list(audio_eval.allowed_forms("order-ops", "subtraction"))
    assert len(concept["allowed_token_ids"]) == len(set(concept["allowed_token_ids"]))
    assert eligibility["items"][0]["eligible_concept_ids"] == [concept["concept_id"]]


def test_deployment_preparation_identity_binds_model_input_tensor_bytes(monkeypatch):
    import types

    import torch

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    input_ids = torch.tensor(
        [list(EVALUATION_PREFIX_FRAMING_IDS) + [9] * 7 + list(EVALUATION_SUFFIX_FRAMING_IDS)]
    )
    model_inputs = {
        "input_features": torch.ones(1, 4, 3),
        "input_features_mask": torch.ones(1, 4, dtype=torch.bool),
        "input_ids": input_ids,
    }
    layout = types.SimpleNamespace(
        audio_start=5,
        n_audio_tokens=7,
        audio_stop=12,
        sequence_length=18,
    )
    manifest_fields = {
        "audio_start": 5,
        "n_audio_tokens": 7,
        "audio_stop": 12,
        "sequence_length": 18,
        "max_sequence_length": 512,
        audio_eval.AUDIO_POSITION: 11,
        audio_eval.RESPONSE_POSITION: 17,
        "prefix_framing_ids": EVALUATION_PREFIX_FRAMING_IDS,
        "suffix_framing_ids": EVALUATION_SUFFIX_FRAMING_IDS,
    }
    prepared = types.SimpleNamespace(
        model_inputs=model_inputs,
        layout=layout,
        last_processor_valid_audio_position=11,
        response_position=17,
        prefix_framing_ids=EVALUATION_PREFIX_FRAMING_IDS,
        suffix_framing_ids=EVALUATION_SUFFIX_FRAMING_IDS,
        manifest_fields=manifest_fields,
    )
    observation = {
        "observation_index": 0,
        "distribution": "association",
        "name": "grief",
        "variant": "m1",
        "normalized_wav_sha256": "a" * 64,
    }
    first = deployment._preparation_record(observation, prepared)
    model_inputs["input_features"][0, 0, 0] = 2
    second = deployment._preparation_record(observation, prepared)
    assert first["model_inputs"] != second["model_inputs"]
    assert first["model_inputs"][0]["name"] == "input_features"
    assert first["model_inputs"][0]["sha256"] != second["model_inputs"][0]["sha256"]


def test_deployment_dispatch_accepts_only_the_four_strict_modes(monkeypatch):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    calls = []

    def invoke(name):
        return lambda **kwargs: calls.append((name, kwargs)) or name

    common = {
        "preregister_call": invoke("preregister"),
        "smoke_call": invoke("smoke"),
        "evaluate_call": invoke("evaluate"),
        "validate_call": invoke("validate"),
    }
    assert (
        deployment._dispatch(
            preregister=True,
            smoke=False,
            evaluate=False,
            validate_report="",
            preregistration="",
            sha256="",
            **common,
        )
        == "preregister"
    )
    assert (
        deployment._dispatch(
            preregister=False,
            smoke=True,
            evaluate=False,
            validate_report="",
            preregistration="",
            sha256="",
            **common,
        )
        == "smoke"
    )
    assert (
        deployment._dispatch(
            preregister=False,
            smoke=False,
            evaluate=True,
            validate_report="",
            preregistration="/vol/prereg.json",
            sha256="a" * 64,
            **common,
        )
        == "evaluate"
    )
    assert (
        deployment._dispatch(
            preregister=False,
            smoke=False,
            evaluate=False,
            validate_report="/vol/report.json",
            preregistration="",
            sha256="b" * 64,
            **common,
        )
        == "validate"
    )
    with pytest.raises(SystemExit, match="exactly one"):
        deployment._dispatch(
            preregister=True,
            smoke=True,
            evaluate=False,
            validate_report="",
            preregistration="",
            sha256="",
            **common,
        )
    with pytest.raises(SystemExit, match="requires"):
        deployment._dispatch(
            preregister=False,
            smoke=False,
            evaluate=True,
            validate_report="",
            preregistration="/vol/prereg.json",
            sha256="",
            **common,
        )
    assert [name for name, _ in calls] == [
        "preregister",
        "smoke",
        "evaluate",
        "validate",
    ]


def test_deployment_source_and_runtime_drift_fail_before_scoring(
    monkeypatch,
    tmp_path,
):
    import json

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    digest = "a" * 64
    monkeypatch.setattr(deployment, "PREREGISTRATION_ROOT", str(tmp_path))
    record = {
        "preregistration_sha256": digest,
        "source_identity": {"source": "old"},
        "calibration_status": "passed",
    }
    (tmp_path / f"{digest}.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        audio_eval,
        "validate_preregistration",
        lambda value: value,
    )
    monkeypatch.setattr(
        deployment,
        "_require_source_identity",
        lambda: {"source": "current"},
    )
    with pytest.raises(
        deployment.ModalAudioWorkspaceEvalError,
        match="source identity drifted",
    ):
        deployment._validate_preregistration_file(
            str(tmp_path / f"{digest}.json"),
            digest,
        )

    preparation_body = {
        "kind": "audio_workspace_processor_preparation",
        "model": {
            "id": deployment.MODEL_ID,
            "revision": deployment.MODEL_REVISION,
        },
        "max_sequence_length": deployment.MAX_SEQUENCE_LENGTH,
        "observations": [{} for _ in range(audio_eval.EXPECTED_OBSERVATION_COUNT)],
    }
    preregistration = {
        "runtime_identity": {
            "environment": {"runtime": "sealed"},
            "processor_preparation": {
                **preparation_body,
                "preparation_sha256": deployment._sha256_json(preparation_body),
            },
        }
    }
    monkeypatch.setattr(
        deployment,
        "_runtime_identity",
        lambda include_cuda: {"runtime": "drifted"},
    )
    with pytest.raises(
        deployment.ModalAudioWorkspaceEvalError,
        match="runtime environment drifted",
    ):
        deployment._require_runtime_environment(preregistration)


def test_deployment_config_drift_aborts_before_any_score_or_publish(
    monkeypatch,
):
    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    touched = []
    monkeypatch.setattr(
        deployment,
        "_validate_preregistration_file",
        lambda path, sha: (_ for _ in ()).throw(
            audio_eval.AudioWorkspaceEvalContractError("final audio lens config changed")
        ),
    )
    monkeypatch.setattr(
        deployment,
        "_validate_physical_stimuli",
        lambda value: touched.append("physical"),
    )
    monkeypatch.setattr(
        deployment,
        "_score_confirmatory",
        lambda *args, **kwargs: touched.append("score"),
    )
    monkeypatch.setattr(
        deployment,
        "_atomic_immutable_write",
        lambda *args: touched.append("publish"),
    )
    with pytest.raises(
        audio_eval.AudioWorkspaceEvalContractError,
        match="config changed",
    ):
        deployment._evaluate_impl("/vol/prereg.json", "a" * 64)
    assert touched == []


def test_deployment_independent_validator_reloads_every_bound_artifact(
    monkeypatch,
    tmp_path,
):
    import json

    deployment = _load_modal_audio_workspace_eval(monkeypatch)
    report_sha = "e" * 64
    preregistration_sha = "a" * 64
    run_bytes = b"completed-run-manifest"
    artifact_identity = {
        "completed_run_manifest_sha256": deployment._sha256_bytes(run_bytes),
        "fit_config_sha256": deployment.FINAL_AUDIO_FIT_CONFIG_SHA256,
        "lens_sha256": deployment.FINAL_AUDIO_LENS_SHA256,
        "lens_dtype": "float16",
        "lens_layers": list(range(34)),
    }
    report = {
        "report_sha256": report_sha,
        "preregistration_sha256": preregistration_sha,
        "status": audio_eval.NO_READOUT_STATUS,
        "records": [{"complete": True}],
    }
    report_root = tmp_path / "reports"
    report_root.mkdir()
    report_path = report_root / f"{report_sha}.json"
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(deployment, "REPORT_ROOT", str(report_root))
    monkeypatch.setattr(
        deployment,
        "PREREGISTRATION_ROOT",
        str(tmp_path / "preregistrations"),
    )
    calls = []
    preregistration = {
        "preregistration_sha256": preregistration_sha,
        "artifact_identity": artifact_identity,
    }
    text_report = {"canonical": True}

    def physical_validator(value, *, verify_synthesis):
        assert value is preregistration
        assert verify_synthesis is True
        calls.append("physical")
        return (
            {"stimulus_manifest_sha256": "c" * 64, "items": []},
            tmp_path / "manifest.json",
            {},
        )

    monkeypatch.setattr(
        deployment,
        "_bind_items_to_text_report",
        lambda items, text: calls.append("bind"),
    )
    result = deployment._validate_report_impl(
        str(report_path),
        report_sha,
        preregistration_loader=lambda path, sha: calls.append("preregistration") or preregistration,
        text_loader=lambda: calls.append("text") or text_report,
        physical_validator=physical_validator,
        completed_run_loader=lambda: calls.append("run") or {"run": True},
        completed_run_bytes=lambda: run_bytes,
        artifact_validator=lambda run, **kwargs: calls.append("artifact") or artifact_identity,
        report_validator=lambda candidate, prereg, text: calls.append("report") or report,
        report_builder=lambda prereg, records, text, physical: calls.append("rebuild") or report,
    )
    assert result["independently_reproduced"] is True
    assert result["sha256"] == report_sha
    assert calls == [
        "preregistration",
        "text",
        "physical",
        "bind",
        "run",
        "artifact",
        "rebuild",
        "report",
    ]
