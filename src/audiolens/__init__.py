"""Shared audiolens constants and helpers: mood anchors, RAVDESS metadata.

One canonical copy for the local scripts and the Modal containers (which get
this package via ``add_local_python_source``). Keep this module import-light:
no torch/transformers at module level.
"""

from __future__ import annotations

MODEL_ID = "google/gemma-4-E2B-it"
READ_LAYER = 29  # readout resolves late on gemma-4-E2B (see scripts/sanity_check.py)

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


# Word-boundary markers across tokenizer families: SentencePiece uses ▁,
# byte-level BPE uses Ġ, some vocabularies store a literal leading space.
_BOUNDARY_MARKERS = ("▁", "Ġ", " ")


def _vocab_surface_index(tok) -> dict[str, list[int]]:
    """casefolded surface form -> all token ids whose vocab string renders to
    it, with any leading word-boundary marker stripped.

    Probing with ``tok.encode(" word")`` finds only the variants you thought
    to spell out; scanning the vocabulary itself captures every whitespace
    and case variant the tokenizer actually has (" sad", "sad", " Sad",
    "SAD", …), whatever its marker convention. Byte-level BPE stores
    non-ASCII text byte-mangled, so non-Latin words in such vocabularies
    would additionally need a per-token decode pass — not needed for the
    SentencePiece-family tokenizers we target.
    """
    index: dict[str, list[int]] = {}
    for token_str, tid in tok.get_vocab().items():
        surface = token_str
        if surface and surface[0] in _BOUNDARY_MARKERS:
            surface = surface[1:]
        if surface:
            index.setdefault(surface.casefold(), []).append(tid)
    for ids in index.values():
        ids.sort()
    return index


_INDEX_CACHE: dict[int, dict[str, list[int]]] = {}


def variant_token_ids(tok, word: str) -> list[int]:
    """Ids of every whole-token variant of ``word`` in this tokenizer's
    vocabulary — all whitespace-marker and case forms. All variants count:
    dropping one shifts the anchor masses."""
    index = _INDEX_CACHE.get(id(tok))
    if index is None:
        index = _INDEX_CACHE[id(tok)] = _vocab_surface_index(tok)
    return list(index.get(word.casefold(), ()))


def single_token_id(tok, word: str) -> int | None:
    """First single-token variant id of ``word``, else None (vetting predicate)."""
    ids = variant_token_ids(tok, word)
    return ids[0] if ids else None


def load_anchors(path: str | None = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """The anchor vocabulary: ``{emotion: [words]}`` plus optional display
    colors, from a YAML file or the built-in :data:`EMOTION_ANCHORS`.

    YAML schema — each emotion maps to either a plain word list or
    ``{words: [...], color: "<rich style>"}``::

        joy: [joy, happy, delight]
        dread:
          words: [dread, doom, foreboding]
          color: dark_orange

    The vocabulary is part of the measurement: baselines record its
    :func:`anchor_fingerprint`, and lifts refuse to mix vocabularies.
    """
    if path is None:
        return {e: list(ws) for e, ws in EMOTION_ANCHORS.items()}, {}
    import pathlib

    import yaml

    data = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a mapping of emotion -> words")
    anchors: dict[str, list[str]] = {}
    colors: dict[str, str] = {}
    for emotion, spec in data.items():
        words = spec
        if isinstance(spec, dict):
            words = spec.get("words")
            if spec.get("color"):
                colors[emotion] = str(spec["color"])
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise ValueError(f"{path}: {emotion!r} needs a list of words")
        anchors[emotion] = words
    return anchors, colors


def anchor_fingerprint(anchors: dict[str, list[str]]) -> str:
    """Stable id of an anchor vocabulary (order- and duplicate-insensitive).
    Baselines record it so a lift is never computed against a baseline
    measured with different anchors."""
    import hashlib
    import json

    canon = json.dumps({e: sorted(set(ws)) for e, ws in sorted(anchors.items())})
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def anchor_token_ids(
    tok, anchors: dict[str, list[str]] | None = None
) -> dict[str, list[int]]:
    """All single-token variant ids for each anchor cluster."""
    if anchors is None:
        anchors = EMOTION_ANCHORS
    ids = {
        # dict.fromkeys: dedupe (multilingual lists share surface forms
        # across languages) while keeping first-seen order
        emotion: list(dict.fromkeys(t for w in words for t in variant_token_ids(tok, w)))
        for emotion, words in anchors.items()
    }
    empty = [e for e, ts in ids.items() if not ts]
    if empty:
        raise ValueError(
            f"anchor clusters with no single-token variants: {empty} — "
            "every cluster needs at least one word the tokenizer keeps whole"
        )
    return ids


def resolve_audio_token_id(config, tok) -> int:
    """The audio soft-token id marking audio positions in the prefill."""
    audio_id = getattr(config, "audio_token_id", None)
    if audio_id is None:
        audio_id = tok.convert_tokens_to_ids("<audio_soft_token>")
    if audio_id is None or audio_id == tok.unk_token_id:
        raise RuntimeError("could not resolve the audio soft-token id")
    return audio_id


def load_lensed_model(lens_path: str, *, device: str | None = None):
    """Load the audio model + processor + fitted lens as one bundle.

    Returns ``(processor, hf, lens_model, lens)``: the raw HF model (for the
    multimodal forward) and the jlens wrapper (``.layers`` for hooks,
    ``.unembed``). One loading recipe for the scripts here and downstream
    consumers (moodmic).
    """
    import torch
    import transformers

    import jlens

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16  # matches the fit dtype
    ).to(device).eval()
    lens_model = jlens.from_hf(hf, processor.tokenizer)
    lens = jlens.JacobianLens.from_pretrained(lens_path)
    return processor, hf, lens_model, lens


def mood_readout_per_position(lens_logits, anchor_ids: dict[str, list[int]]):
    """Per-POSITION cluster scores from ``[n_positions, vocab]`` lens logits:
    ``{emotion: [score per position]}``, each score the mean softmax mass per
    anchor token. Normalizing by the resolved cluster size happens here,
    dynamically, so clusters of different richness — and different tokenizers
    resolving different subsets of the vocabulary — stay comparable."""
    probs = lens_logits.softmax(-1)
    return {
        e: (probs[:, ids].sum(-1) / len(ids)).tolist()
        for e, ids in anchor_ids.items()
    }


def mood_readout(lens_logits, anchor_ids: dict[str, list[int]], topk: int = 10):
    """The canonical span-level measurement, shared by the local smoke script
    and the Modal eval so the two surfaces cannot drift: the position-mean of
    :func:`mood_readout_per_position`, plus span-mean top-k token ids."""
    per_pos = mood_readout_per_position(lens_logits, anchor_ids)
    mass = {e: sum(v) / len(v) for e, v in per_pos.items()}
    top_ids = lens_logits.mean(0).topk(topk).indices.tolist()
    return mass, top_ids
