"""Planted-signal test of the eval analyzer: per-voice and speech-generic
biases must divide out through the neutral baseline, and planted own-cluster
signal must be recovered exactly."""

import json
import pathlib
import sys

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
