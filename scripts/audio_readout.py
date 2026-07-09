"""Lens readout over audio token positions — the first audio-side test.

Feeds RAVDESS clips (fixed sentences, acted emotional prosody) through
gemma-4-E2B-it and reads the text-fit lens at the audio soft-token positions.
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
import transformers

import jlens
from jlens.hooks import ActivationRecorder

from audiolens import (
    anchor_token_ids,
    mood_readout,
    parse_ravdess_name,
    resolve_audio_token_id,
)

MODEL_ID = "google/gemma-4-E2B-it"
OUR_LENS = "lenses/gemma-4-E2B-it_jacobian_lens.pt"
READ_LAYER = 29  # readout resolves late on gemma-4-E2B (see sanity_check.py)


def clip_label(path: pathlib.Path) -> str:
    meta = parse_ravdess_name(path.stem)
    if meta is None:
        return path.name
    return f"{meta['emotion']:<9} actor {meta['actor']} \"{meta['statement']}\""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="+", type=pathlib.Path)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16  # matches the fit dtype
    ).to(device).eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(OUR_LENS)
    print(f"lens: {lens}  read layer: L{READ_LAYER}\n")

    anchors = anchor_token_ids(tok)
    audio_id = resolve_audio_token_id(hf.config, tok)

    for wav in args.wavs:
        messages = [{"role": "user", "content": [{"type": "audio", "audio": str(wav)}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        positions = (inputs["input_ids"][0] == audio_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise RuntimeError(f"{wav}: no audio soft tokens in the prefill")

        with torch.no_grad(), ActivationRecorder(model.layers, at=[READ_LAYER]) as rec:
            hf(**inputs, use_cache=False)
        residual = rec.activations[READ_LAYER][0][positions].float()

        lens_logits = model.unembed(lens.transport(residual, READ_LAYER)).float().cpu()

        print(f"== {clip_label(wav)}  ({positions.numel()} audio tokens, "
              f"seq {inputs['input_ids'].shape[1]}) ==")

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
