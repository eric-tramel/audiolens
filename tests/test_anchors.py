"""load_anchors / anchor_fingerprint — no tokenizer, no model."""

from __future__ import annotations

import pytest

from audiolens import (
    EMOTION_ANCHORS,
    anchor_fingerprint,
    load_anchors,
    load_default_anchors,
)


def test_default_is_builtin_copy():
    anchors, colors = load_anchors(None)
    assert anchors == EMOTION_ANCHORS
    assert colors == {}
    anchors["joy"].append("mutated")
    assert "mutated" not in EMOTION_ANCHORS["joy"]


def test_packaged_multilingual_anchors_have_expected_active_clusters():
    anchors, colors = load_default_anchors()
    expected_clusters = {
        "sadness",
        "surprise",
        "joy",
        "disgust",
        "fear",
        "anger",
        "neutral",
    }

    assert set(anchors) == expected_clusters
    assert all(anchors[emotion] for emotion in expected_clusters)
    assert "alegría" in anchors["joy"]
    assert "恐怖" in anchors["fear"]
    assert colors == {}


def test_yaml_list_and_dict_forms(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(
        "joy: [happy, glad]\n"
        "dread:\n  words: [dread, doom]\n  color: dark_orange\n"
    )
    anchors, colors = load_anchors(str(p))
    assert anchors == {"joy": ["happy", "glad"], "dread": ["dread", "doom"]}
    assert colors == {"dread": "dark_orange"}


@pytest.mark.parametrize("content", ["[]", "joy: 3", "joy:\n  color: red"])
def test_yaml_bad_shapes_raise(tmp_path, content):
    p = tmp_path / "bad.yaml"
    p.write_text(content)
    with pytest.raises(ValueError):
        load_anchors(str(p))


def test_fingerprint_ignores_order_and_dupes():
    a = {"joy": ["happy", "glad"], "fear": ["dread"]}
    b = {"fear": ["dread"], "joy": ["glad", "happy", "glad"]}
    assert anchor_fingerprint(a) == anchor_fingerprint(b)
    assert anchor_fingerprint(a) != anchor_fingerprint({"joy": ["happy"], "fear": ["dread"]})
