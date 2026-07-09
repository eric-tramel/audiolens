"""Lens readout over audio token positions — the first audio-side test.

Feeds RAVDESS clips (fixed sentences, acted emotional prosody) through the
selected audio model and reads the text-fit lens at its prepared audio positions.
Two questions, in order:

1. Does a text-fit lens read out anything text-like over audio positions
   at all? (top-k tokens over the audio span)
2. With the words held constant, do the mood-anchor scores move with the
   acted prosody? (per-emotion anchor mass, compared across clips)

jlens.JacobianLens.apply() is text-only, so this drives the pieces directly:
full multimodal forward with ActivationRecorder on the decoder blocks, then
lens.transport() + unembed at the audio positions.

    uv run python scripts/audio_readout.py data/ravdess/Actor_01/03-01-*-01-01-01-01.wav
"""

from __future__ import annotations

import argparse
import pathlib

import torch

import jlens
from jlens.hooks import ActivationRecorder

from audiolens import (
    anchor_token_ids,
    mood_readout,
    parse_ravdess_name,
)
from audiolens.models import audio_residuals, get_model_profile, load_model_runtime

def clip_label(path: pathlib.Path) -> str:
    meta = parse_ravdess_name(path.stem)
    if meta is None:
        return path.name
    return f"{meta['emotion']:<9} actor {meta['actor']} \"{meta['statement']}\""


def _readout_identity(profile) -> tuple[int, str]:
    """Canonical single read layer and default lens path for a profile."""
    return profile.read_layer, f"lenses/{profile.slug}_jacobian_lens.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="+", type=pathlib.Path)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    profile = get_model_profile()
    read_layer, lens_path = _readout_identity(profile)
    runtime = load_model_runtime(profile.key, device=device)
    tok = runtime.tokenizer
    lens = jlens.JacobianLens.from_pretrained(lens_path)
    print(f"lens: {lens}  read layer: L{read_layer}\n")

    anchors = anchor_token_ids(tok)

    for wav in args.wavs:
        prepared = runtime.prepare_audio(wav)
        positions = prepared.audio_positions
        if positions.numel() == 0:
            raise RuntimeError(f"{wav}: no audio soft tokens in the prefill")

        with torch.no_grad(), ActivationRecorder(
            runtime.layers, at=[read_layer]
        ) as rec:
            runtime.forward_audio(prepared)
        residual = audio_residuals(rec.activations, prepared, read_layer).float()

        lens_logits = runtime.unembed(
            lens.transport(residual, read_layer)
        ).float().cpu()

        print(f"== {clip_label(wav)}  ({positions.numel()} audio tokens, "
              f"seq {prepared.input_ids.shape[1]}) ==")

        mass, top_ids = mood_readout(lens_logits, anchors, topk=args.topk)

        # 1. text-likeness: top-k of the mean lens logits over the audio span
        toks = ", ".join(repr(tok.decode([i])) for i in top_ids)
        print(f"  span-mean top-{args.topk}: {toks}")

        # 2. mood: anchor mass over the span, shown normalized across emotions
        total = sum(mass.values())
        ranked = sorted(mass.items(), key=lambda kv: -kv[1])
        print("  mood: " + "  ".join(f"{e}={m / total:.2f}" for e, m in ranked))
        print()


if __name__ == "__main__":
    main()
