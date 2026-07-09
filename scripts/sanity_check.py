"""Text-side sanity for the fitted gemma-4-E2B-it lens.

1. Top-k readout on probe prompts (should show the forward-looking
   workspace behavior, e.g. currency tokens before "is").
2. Correlation against Neuronpedia's base-model gemma-4-e2b lens on the
   same prompts (instruct-vs-base lenses should agree strongly).
3. Mood-anchor vetting on the Gemma tokenizer (single-token coverage).

Run after downloading the fitted lens:
    modal volume get audiolens-vol lenses/gemma-4-E2B-it_jacobian_lens.pt lenses/
    uv run python scripts/sanity_check.py
"""

from __future__ import annotations


import torch
import transformers

import jlens

MODEL_ID = "google/gemma-4-E2B-it"
OUR_LENS = "lenses/gemma-4-E2B-it_jacobian_lens.pt"
REF_REPO = "neuronpedia/jacobian-lens"
REF_FILE = "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt"

PROBES = [
    "Fact: The currency used in the country shaped like a boot is",
    "The fridge had been unplugged for three weeks, and the smell hit us the moment the door opened.",
]

# Mood anchors from jlens-mood (vendored list to avoid a cross-project dep in
# a sanity script; sync manually if the clusters change).
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


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16  # matches the fit dtype; fp32 E2B is ~20 GB
    ).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)

    ours = jlens.JacobianLens.from_pretrained(OUR_LENS)
    print(f"our lens: {ours}")
    ref = jlens.JacobianLens.from_pretrained(REF_REPO, filename=REF_FILE)
    print(f"ref lens: {ref}")

    shared = sorted(set(ours.source_layers) & set(ref.source_layers))
    # Readout resolves late on gemma-4-E2B: L29/L33 show the currency cluster,
    # mid layers are filler for both lenses (layer sweep, 2026-07-08).
    late = shared[-5]

    print(f"\n== top-k readout (our lens, L{late}) ==")
    for probe in PROBES:
        lens_logits, _, _ = ours.apply(model, probe, layers=[late])
        top = lens_logits[late][-1].topk(6)
        print(f"  ...{probe[-45:]!r}")
        print("   -> " + ", ".join(tok.decode([i]) for i in top.indices.tolist()))

    print("\n== ours vs neuronpedia base lens (per-position lens-logit correlation) ==")
    for probe in PROBES:
        ours_logits, _, _ = ours.apply(model, probe, layers=[late])
        ref_logits, _, _ = ref.apply(model, probe, layers=[late])
        a, b = ours_logits[late].flatten(), ref_logits[late].flatten()
        r = torch.corrcoef(torch.stack([a, b]))[0, 1]
        # top-10 overlap at the final position
        ours_top = set(ours_logits[late][-1].topk(10).indices.tolist())
        ref_top = set(ref_logits[late][-1].topk(10).indices.tolist())
        print(f"  r={r:.3f}  top10 overlap {len(ours_top & ref_top)}/10  ...{probe[-40:]!r}")

    print("\n== mood-anchor vetting on the Gemma tokenizer ==")
    for emotion, words in EMOTION_ANCHORS.items():
        dropped = []
        for w in words:
            ok = any(
                len(tok.encode(v, add_special_tokens=False)) == 1
                for v in (f" {w}", f" {w.capitalize()}")
            )
            if not ok:
                dropped.append(w)
        status = f"{len(words) - len(dropped)}/{len(words)}"
        print(f"  {emotion:<9} {status}" + (f"  dropped: {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
