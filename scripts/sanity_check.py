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

import argparse

import torch
import transformers

import jlens

from audiolens import EMOTION_ANCHORS, single_token_id
from audiolens.fitting import MODEL_REVISION

MODEL_ID = "google/gemma-4-E2B-it"
OUR_LENS = "lenses/gemma-4-E2B-it_jacobian_lens.pt"
REF_REPO = "neuronpedia/jacobian-lens"
REF_FILE = "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt"

PROBES = [
    "Fact: The currency used in the country shaped like a boot is",
    "The fridge had been unplugged for three weeks, and the smell hit us the moment the door opened.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", default=OUR_LENS, help="candidate lens path")
    parser.add_argument(
        "--baseline", default=OUR_LENS, help="text-only baseline lens path"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,  # matches the fit dtype; fp32 E2B is ~20 GB
        attn_implementation="eager",
    ).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = jlens.from_hf(hf, tok)

    ours = jlens.JacobianLens.from_pretrained(args.lens)
    baseline = jlens.JacobianLens.from_pretrained(args.baseline)
    print(f"candidate lens: {ours}")
    print(f"baseline lens:  {baseline}")
    ref = jlens.JacobianLens.from_pretrained(REF_REPO, filename=REF_FILE)
    print(f"ref lens: {ref}")

    shared = sorted(
        set(ours.source_layers) & set(baseline.source_layers) & set(ref.source_layers)
    )
    # Readout resolves late on gemma-4-E2B: L29/L33 show the currency cluster,
    # mid layers are filler for both lenses (layer sweep, 2026-07-08).
    late = shared[-5]

    print(f"\n== top-k readout (our lens, L{late}) ==")
    for probe in PROBES:
        lens_logits, _, _ = ours.apply(model, probe, layers=[late])
        top = lens_logits[late][-1].topk(6)
        print(f"  ...{probe[-45:]!r}")
        print("   -> " + ", ".join(tok.decode([i]) for i in top.indices.tolist()))

    print("\n== candidate vs baseline / neuronpedia (lens-logit correlation) ==")
    for probe in PROBES:
        ours_logits, _, _ = ours.apply(model, probe, layers=[late])
        baseline_logits, _, _ = baseline.apply(model, probe, layers=[late])
        ref_logits, _, _ = ref.apply(model, probe, layers=[late])
        a = ours_logits[late].flatten()
        b = baseline_logits[late].flatten()
        c = ref_logits[late].flatten()
        baseline_r = torch.corrcoef(torch.stack([a, b]))[0, 1]
        ref_r = torch.corrcoef(torch.stack([a, c]))[0, 1]
        # top-10 overlap at the final position
        ours_top = set(ours_logits[late][-1].topk(10).indices.tolist())
        ref_top = set(ref_logits[late][-1].topk(10).indices.tolist())
        print(
            f"  baseline r={baseline_r:.3f}  ref r={ref_r:.3f}  "
            f"ref top10 overlap {len(ours_top & ref_top)}/10  ...{probe[-40:]!r}"
        )

    print("\n== candidate vs baseline Jacobian change by layer ==")
    for layer in shared:
        candidate_j = ours.jacobians[layer].flatten()
        baseline_j = baseline.jacobians[layer].flatten()
        cosine = torch.nn.functional.cosine_similarity(candidate_j, baseline_j, dim=0)
        relative_l2 = (candidate_j - baseline_j).norm() / baseline_j.norm()
        print(f"  L{layer:02d} cosine={cosine:.6f}  relative_l2={relative_l2:.6f}")

    print("\n== mood-anchor vetting on the Gemma tokenizer ==")
    for emotion, words in EMOTION_ANCHORS.items():
        dropped = [w for w in words if single_token_id(tok, w) is None]
        status = f"{len(words) - len(dropped)}/{len(words)}"
        print(f"  {emotion:<9} {status}" + (f"  dropped: {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
