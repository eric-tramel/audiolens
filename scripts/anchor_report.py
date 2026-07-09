"""Audit an anchor vocabulary against the model's tokenizer.

For each cluster: how many token ids it resolves to (both ' word' / ' Word'
case variants count) and which words contribute nothing. The readout is only
as robust as this instrument — run it before adopting a new anchors YAML.

    uv run python scripts/anchor_report.py                     # built-in
    uv run python scripts/anchor_report.py anchors/extended.yaml
"""

from __future__ import annotations

import sys

import transformers

from audiolens import MODEL_ID, anchor_fingerprint, load_anchors, variant_token_ids


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    anchors, colors = load_anchors(path)
    tok = transformers.AutoProcessor.from_pretrained(MODEL_ID).tokenizer

    print(f"anchors: {path or 'builtin'}  fingerprint: {anchor_fingerprint(anchors)}")
    total_ids = 0
    for emotion, words in anchors.items():
        per_word = {w: variant_token_ids(tok, w) for w in words}
        ids = set(t for ts in per_word.values() for t in ts)  # dedupe, like anchor_token_ids
        dropped = [w for w, ts in per_word.items() if not ts]
        total_ids += len(ids)
        color = f"  color={colors[emotion]}" if emotion in colors else ""
        print(f"{emotion:<10} {len(words):>3} words -> {len(ids):>3} token ids{color}")
        if dropped:
            print(f"           dropped (no single-token variant): {', '.join(dropped)}")
    print(f"total: {total_ids} token ids across {len(anchors)} clusters")


if __name__ == "__main__":
    main()
