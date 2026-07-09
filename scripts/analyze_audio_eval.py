"""Aggregate the Modal RAVDESS eval: prosody vs mood readout, all actors.

For each (actor, statement) group, the baseline is the mean anchor mass over
that group's *neutral* clips. Every other clip's masses become lifts
(mass / baseline), so per-voice and per-sentence biases divide out — the
smoke-test finding was that raw masses share a large speech-generic component
(curiosity dominance).

Reports, per layer: the acted-emotion x anchor-cluster mean-lift matrix, the
rank of each acted emotion's own cluster, argmax-lift accuracy, and a
normal-vs-strong intensity split.

    uv run python scripts/analyze_audio_eval.py eval/ravdess_gemma-4-E2B-it.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

from audiolens import ACTED_TO_CLUSTER, EMOTION_ANCHORS

CLUSTERS = list(EMOTION_ANCHORS)


def load_records(path: pathlib.Path) -> list[dict]:
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"warning: skipped {skipped} unparseable line(s)")
    return [r for r in records if r.get("meta")]


def lifts_by_layer(records: list[dict], layer: str) -> list[dict]:
    """Per-clip anchor-mass lifts vs the (actor, statement) neutral baseline."""
    baseline: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r["meta"]["emotion"] == "neutral":
            key = (r["meta"]["actor"], r["meta"]["statement"])
            for e, m in r["layers"][layer]["anchor_mass"].items():
                baseline[key][e].append(m)

    out = []
    for r in records:
        meta = r["meta"]
        if meta["emotion"] == "neutral":
            continue
        key = (meta["actor"], meta["statement"])
        if key not in baseline:
            continue
        base = {e: sum(v) / len(v) for e, v in baseline[key].items()}
        mass = r["layers"][layer]["anchor_mass"]
        out.append(
            {
                "acted": meta["emotion"],
                "intensity": meta["intensity"],
                "lift": {e: mass[e] / base[e] for e in CLUSTERS},
            }
        )
    return out


def matrix(lifts: list[dict]) -> dict[str, dict[str, float]]:
    """Mean lift per (acted emotion, anchor cluster)."""
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in lifts:
        counts[row["acted"]] += 1
        for e, v in row["lift"].items():
            sums[row["acted"]][e] += v
    return {
        acted: {e: sums[acted][e] / counts[acted] for e in CLUSTERS}
        for acted in sorted(sums)
    }


def report(lifts: list[dict], label: str) -> None:
    if not lifts:
        print(f"-- {label}: no clips --")
        return
    mat = matrix(lifts)
    counts = defaultdict(int)
    for row in lifts:
        counts[row["acted"]] += 1
    print(f"-- {label} ({len(lifts)} clips) --")
    print(f"{'acted':<10}{'n':>5}" + "".join(f"{e[:7]:>9}" for e in CLUSTERS))
    hits = 0
    ranked_own = []
    for acted, row in mat.items():
        own = ACTED_TO_CLUSTER.get(acted)
        cells = []
        for e in CLUSTERS:
            mark = "*" if e == own else " "
            cells.append(f"{row[e]:>8.2f}{mark}")
        print(f"{acted:<10}{counts[acted]:>5}" + "".join(cells))
        if own:
            rank = sorted(row, key=row.get, reverse=True).index(own) + 1
            ranked_own.append((acted, rank))
            hits += row[own] == max(row.values())
    n = len(ranked_own)
    print(f"argmax-lift accuracy: {hits}/{n}   "
          + "own-cluster rank: "
          + "  ".join(f"{a}={r}" for a, r in ranked_own))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--layer", default=None, help="single layer to report")
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"no usable records in {args.results}")
    n_non_neutral = sum(1 for r in records if r["meta"]["emotion"] != "neutral")
    print(f"{len(records)} records from {args.results}\n")
    layers = [args.layer] if args.layer else sorted(records[0]["layers"], key=int)

    for layer in layers:
        print(f"===== layer L{layer} =====")
        lifts = lifts_by_layer(records, layer)
        if len(lifts) < n_non_neutral:
            print(f"warning: {n_non_neutral - len(lifts)} clip(s) dropped "
                  "(no neutral baseline for their actor+statement group)")
        report(lifts, "all intensities")
        for intensity in ("normal", "strong"):
            report([row for row in lifts if row["intensity"] == intensity], intensity)


if __name__ == "__main__":
    main()
