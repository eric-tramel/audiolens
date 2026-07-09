"""Planted-signal test of the eval analyzer: per-voice and speech-generic
biases must divide out through the neutral baseline, and planted own-cluster
signal must be recovered exactly."""

import copy
import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

import analyze_audio_eval as an  # noqa: E402

from audiolens import ACTED_TO_CLUSTER, EMOTION_ANCHORS  # noqa: E402

CLUSTERS = list(EMOTION_ANCHORS)
EMO_CODE = {"neutral": "01", "calm": "02", "happy": "03", "sad": "04",
            "angry": "05", "fearful": "06", "disgust": "07", "surprised": "08"}


def _record(actor: str, emotion: str, intensity: str, rep: str, mass: dict) -> dict:
    stem = f"03-01-{EMO_CODE[emotion]}-{'01' if intensity == 'normal' else '02'}-01-{rep}-{actor}"
    return {
        "clip": stem + ".wav",
        "meta": {"emotion": emotion, "intensity": intensity,
                 "statement": "Kids are talking by the door",
                 "rep": rep, "actor": actor},
        "n_audio_tokens": 90,
        "seq_len": 98,
        "layers": {"29": {"anchor_mass": mass, "topk_ids": [1], "topk_toks": ["x"]}},
    }


def _synthetic(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two actors with different per-voice bias, speech-generic curiosity
    bias, planted 2x (normal) / 3x (strong) own-cluster lift."""
    rows = []
    for actor, voice_bias in (("01", 1.0), ("02", 1.7)):
        base = {e: 0.01 * voice_bias for e in CLUSTERS}
        base["curiosity"] *= 5  # speech-generic bias, present in every clip
        for rep in ("01", "02"):
            rows.append(_record(actor, "neutral", "normal", rep, dict(base)))
        for acted, cluster in ACTED_TO_CLUSTER.items():
            for intensity, lift in (("normal", 2.0), ("strong", 3.0)):
                mass = dict(base)
                mass[cluster] *= lift
                rows.append(_record(actor, acted, intensity, "01", mass))
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_planted_signal_recovered(tmp_path):
    records = an.load_records(_synthetic(tmp_path))
    lifts = an.lifts_by_layer(records, "29")
    assert len(lifts) == 2 * len(ACTED_TO_CLUSTER) * 2  # actors x acted x intensity

    for intensity, expected in (("normal", 2.0), ("strong", 3.0)):
        mat = an.matrix([row for row in lifts if row["intensity"] == intensity])
        for acted, own in ACTED_TO_CLUSTER.items():
            row = mat[acted]
            assert abs(row[own] - expected) < 1e-9, (acted, own, row[own])
            assert max(row, key=row.get) == own  # argmax lands on own cluster
            for e in CLUSTERS:  # everything else divides out to 1.0
                if e != own:
                    assert abs(row[e] - 1.0) < 1e-9


def test_truncated_tail_skipped(tmp_path):
    path = _synthetic(tmp_path)
    with open(path, "a") as f:
        f.write('{"clip": "03-01-03-01-01-01-01.wav", "meta": {"emo')  # torn write
    records = an.load_records(path)
    assert all("layers" in r for r in records)


def _paired_record(
    actor: str,
    emotion: str,
    intensity: str,
    rep: str,
    baseline_mass: dict,
    candidate_mass: dict,
) -> dict:
    row = _record(actor, emotion, intensity, rep, baseline_mass)
    layers = row.pop("layers")
    row["readouts"] = {
        "text400": {"layers": layers},
        "mixed528": {
            "layers": {
                "29": {
                    "anchor_mass": candidate_mass,
                    "topk_ids": [1],
                    "topk_toks": ["x"],
                }
            }
        },
    }
    return row


def _paired_synthetic() -> list[dict]:
    rows = []
    for actor, voice_bias in (("01", 1.0), ("02", 1.7)):
        neutral = {emotion: 0.01 * voice_bias for emotion in CLUSTERS}
        for rep in ("01", "02"):
            rows.append(
                _paired_record(actor, "neutral", "normal", rep, neutral, neutral)
            )
        for acted, cluster in ACTED_TO_CLUSTER.items():
            for intensity, baseline_lift, candidate_lift in (
                ("normal", 2.0, 3.0),
                ("strong", 4.0, 9.0),
            ):
                baseline = dict(neutral)
                candidate = dict(neutral)
                baseline[cluster] *= baseline_lift
                candidate[cluster] *= candidate_lift
                rows.append(
                    _paired_record(
                        actor, acted, intensity, "01", baseline, candidate
                    )
                )
    return rows


def test_paired_estimands_and_duplicate_actor_weighting():
    rows = _paired_synthetic()
    result = an.paired_estimands(
        rows,
        layer="29",
        clusters=CLUSTERS,
        acted_to_cluster=ACTED_TO_CLUSTER,
    )
    expected_own = (math.log(3 / 2) + math.log(9 / 4)) / 2
    assert abs(result["own_cluster_log_lift_delta"] - expected_own) < 1e-12
    assert abs(result["strong_minus_normal_log_lift_delta"] - math.log(3 / 2)) < 1e-12
    # Both actors have different voice baselines but identical lifts; resampling
    # one actor twice proves baselines are actor-local and duplicate blocks work.
    duplicate = an.paired_estimands(
        rows,
        layer="29",
        clusters=CLUSTERS,
        acted_to_cluster=ACTED_TO_CLUSTER,
        actor_draw=["01", "01"],
    )
    assert duplicate == result


def test_actor_block_bootstrap_is_seeded_and_recomputes_neutral_baselines():
    rows = _paired_synthetic()
    kwargs = {
        "layer": "29",
        "clusters": CLUSTERS,
        "acted_to_cluster": ACTED_TO_CLUSTER,
        "seed": 42,
        "n_replicates": 50,
    }
    first = an.actor_block_bootstrap(rows, **kwargs)
    second = an.actor_block_bootstrap(rows, **kwargs)
    assert first == second
    for metric in first.values():
        assert metric["ci_low"] <= metric["estimate"] <= metric["ci_high"]


def test_paired_no_improvement_is_a_valid_zero_result():
    rows = _paired_synthetic()
    for row in rows:
        row["readouts"]["mixed528"] = copy.deepcopy(row["readouts"]["text400"])
    result = an.paired_estimands(
        rows,
        layer="29",
        clusters=CLUSTERS,
        acted_to_cluster=ACTED_TO_CLUSTER,
    )
    assert result == {
        "own_cluster_log_lift_delta": 0.0,
        "strong_minus_normal_log_lift_delta": 0.0,
    }


def test_paired_validation_rejects_incomplete_pair_and_curiosity():
    rows = _paired_synthetic()
    production_clusters = [cluster for cluster in CLUSTERS if cluster != "curiosity"]
    for row in rows:
        for readout in row["readouts"].values():
            readout["layers"]["29"]["anchor_mass"].pop("curiosity", None)
    metadata = {
        "config": {
            "lenses": {"text400": {}, "mixed528": {}},
            "read_layers": [29],
            "anchors": {"clusters": production_clusters},
            "topk": 1,
            "n_clips": len(rows),
        },
        "completed": True,
        "n_records": len(rows),
    }
    an.validate_paired_records(rows, metadata)
    broken = copy.deepcopy(rows)
    broken[0]["readouts"].pop("mixed528")
    with pytest.raises(ValueError, match="incomplete lens pair"):
        an.validate_paired_records(broken, metadata)
    metadata["config"]["anchors"]["clusters"].append("curiosity")
    with pytest.raises(ValueError, match="curiosity"):
        an.validate_paired_records(rows, metadata)
