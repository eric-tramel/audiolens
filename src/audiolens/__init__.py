"""Shared audiolens constants and helpers: mood anchors, RAVDESS metadata.

One canonical copy for the local scripts and the Modal containers (which get
this package via ``add_local_python_source``). Keep this module import-light:
no torch/transformers at module level.
"""

from __future__ import annotations

# Mood anchors from jlens-mood (vendored to avoid a cross-project dep;
# sync manually if the clusters change).
EMOTION_ANCHORS = {
    "sadness": ["sad", "sadness", "grief", "sorrow", "cry", "crying", "tears",
                "mourning", "lonely", "despair", "misery", "mourn", "ache", "anguish"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished", "amazed",
                 "sudden", "unexpected", "startled", "stunned", "disbelief", "marvel",
                 "wow", "abrupt"],
    "joy": ["joy", "happy", "happiness", "delight", "laugh", "laughter", "smile",
            "celebrate", "cheerful", "wonderful", "glad", "thrilled", "bliss"],
    "disgust": ["disgust", "disgusting", "gross", "nausea", "filthy", "vile",
                "rotten", "foul", "nasty", "slime", "rot", "sewage", "mold"],
    "fear": ["fear", "afraid", "terror", "panic", "dread", "scared", "anxious",
             "anxiety", "horror", "threat", "worried", "frightened"],
    "anger": ["anger", "angry", "rage", "furious", "fury", "outrage", "hate",
              "hatred", "resentment", "hostile", "mad", "irritated"],
    "curiosity": ["curious", "curiosity", "wonder", "wondering", "intrigued",
                  "fascinated", "fascinating", "explore", "mystery", "puzzle",
                  "inquiry", "interested", "discover", "investigate"],
    "neutral": ["bland", "boring", "bored", "boredom", "dull", "mundane",
                "ordinary", "routine", "tedious", "meh"],
}

# RAVDESS filename fields: modality-vocal-emotion-intensity-statement-rep-actor
RAVDESS_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

# acted RAVDESS emotion -> our anchor cluster (calm has no exact cluster;
# neutral prosody is the baseline, not a row). curiosity has no acted source.
ACTED_TO_CLUSTER = {
    "happy": "joy", "sad": "sadness", "angry": "anger", "fearful": "fear",
    "disgust": "disgust", "surprised": "surprise", "calm": "neutral",
}


def parse_ravdess_name(stem: str) -> dict[str, str] | None:
    """Decode a RAVDESS file stem into named fields; None if not RAVDESS."""
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    return {
        "emotion": RAVDESS_EMOTION.get(parts[2], parts[2]),
        "intensity": {"01": "normal", "02": "strong"}.get(parts[3], parts[3]),
        "statement": RAVDESS_STATEMENT.get(parts[4], parts[4]),
        "rep": parts[5],
        "actor": parts[6],
    }


def variant_token_ids(tok, word: str) -> list[int]:
    """Ids of ``word``'s single-token ' word' / ' Word' variants (0, 1, or 2).
    Both variants count: dropping one shifts the anchor masses."""
    out = []
    for v in (f" {word}", f" {word.capitalize()}"):
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            out.append(enc[0])
    return out


def single_token_id(tok, word: str) -> int | None:
    """First single-token variant id of ``word``, else None (vetting predicate)."""
    ids = variant_token_ids(tok, word)
    return ids[0] if ids else None


def anchor_token_ids(tok) -> dict[str, list[int]]:
    """All single-token variant ids for each anchor cluster."""
    return {
        emotion: [t for w in words for t in variant_token_ids(tok, w)]
        for emotion, words in EMOTION_ANCHORS.items()
    }


def resolve_audio_token_id(config, tok) -> int:
    """The audio soft-token id marking audio positions in the prefill."""
    audio_id = getattr(config, "audio_token_id", None)
    if audio_id is None:
        audio_id = tok.convert_tokens_to_ids("<audio_soft_token>")
    if audio_id is None or audio_id == tok.unk_token_id:
        raise RuntimeError("could not resolve the audio soft-token id")
    return audio_id


def mood_readout(lens_logits, anchor_ids: dict[str, list[int]], topk: int = 10):
    """The canonical measurement, shared by the local smoke script and the
    Modal eval so the two surfaces cannot drift: from ``[n_positions, vocab]``
    lens logits, return (per-cluster softmax mass averaged over the span,
    span-mean top-k token ids)."""
    probs = lens_logits.softmax(-1)
    mass = {e: probs[:, ids].sum(-1).mean().item() for e, ids in anchor_ids.items()}
    top_ids = lens_logits.mean(0).topk(topk).indices.tolist()
    return mass, top_ids
