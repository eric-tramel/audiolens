"""Aggregate the Modal RAVDESS eval: prosody vs mood readout, all actors.

For each (actor, statement) group, the baseline is the mean anchor mass over
that group's *neutral* clips. Every other clip's masses become lifts
(mass / baseline), so per-voice and per-sentence biases divide out — the
smoke-test finding was that raw masses share a large speech-generic component
(curiosity dominance).

Reports, per layer: the acted-emotion x anchor-cluster mean-lift matrix, the
rank of each acted emotion's own cluster, argmax-lift accuracy, and a
normal-vs-strong intensity split.

    uv run python scripts/analyze_audio_eval.py eval/ravdess-paired-<digest>.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

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


def lifts_by_layer(
    records: list[dict], layer: str, clusters: list[str] = CLUSTERS
) -> list[dict]:
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
                "lift": {e: mass[e] / base[e] for e in clusters},
            }
        )
    return out


def matrix(
    lifts: list[dict], clusters: list[str] = CLUSTERS
) -> dict[str, dict[str, float]]:
    """Mean lift per (acted emotion, anchor cluster)."""
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in lifts:
        counts[row["acted"]] += 1
        for e, v in row["lift"].items():
            sums[row["acted"]][e] += v
    return {
        acted: {e: sums[acted][e] / counts[acted] for e in clusters}
        for acted in sorted(sums)
    }


def report(
    lifts: list[dict],
    label: str,
    clusters: list[str] = CLUSTERS,
    acted_to_cluster: dict[str, str] = ACTED_TO_CLUSTER,
) -> None:
    if not lifts:
        print(f"-- {label}: no clips --")
        return
    mat = matrix(lifts, clusters)
    counts = defaultdict(int)
    for row in lifts:
        counts[row["acted"]] += 1
    print(f"-- {label} ({len(lifts)} clips) --")
    print(f"{'acted':<10}{'n':>5}" + "".join(f"{e[:7]:>9}" for e in clusters))
    hits = 0
    ranked_own = []
    for acted, row in mat.items():
        own = acted_to_cluster.get(acted)
        cells = []
        for e in clusters:
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


def load_paired_metadata(results_path: pathlib.Path) -> dict:
    from audiolens.fitting import config_digest

    metadata_path = results_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"paired results need metadata sidecar {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if pathlib.Path(metadata.get("results", "")).name != results_path.name:
        raise ValueError("metadata results path does not match JSONL")
    config = metadata.get("config")
    if not isinstance(config, dict) or metadata.get("config_sha256") != config_digest(config):
        raise ValueError("paired metadata config digest mismatch")
    return metadata


def validate_paired_records(records: list[dict], metadata: dict) -> None:
    from audiolens.fitting import AudioFitContractError, validate_paired_result_row

    config = metadata["config"]
    expected_count = config.get("n_clips")
    if (
        metadata.get("completed") is not True
        or not isinstance(expected_count, int)
        or metadata.get("n_records") != expected_count
        or len(records) != expected_count
    ):
        raise ValueError(
            f"paired evaluation is incomplete: records={len(records)}, "
            f"expected={expected_count}, metadata={metadata.get('n_records')}"
        )
    clips: set[str] = set()
    for row in records:
        clip = row["clip"]
        if clip in clips:
            raise ValueError(f"duplicate clip {clip}")
        clips.add(clip)
        try:
            validate_paired_result_row(row, config)
        except AudioFitContractError as exc:
            raise ValueError(str(exc)) from exc
    if expected_count == 1_440:
        actor_counts = Counter(row["meta"]["actor"] for row in records)
        expected_actors = {f"{actor:02d}" for actor in range(1, 25)}
        if set(actor_counts) != expected_actors or any(
            count != 60 for count in actor_counts.values()
        ):
            raise ValueError(
                "paired evaluation does not contain 60 clips for all 24 actors: "
                f"{actor_counts}"
            )


def paired_estimands(
    records: list[dict],
    *,
    layer: str,
    clusters: list[str],
    acted_to_cluster: dict[str, str],
    actor_draw: list[str] | None = None,
) -> dict[str, float]:
    """Candidate-minus-baseline log-lift metrics for an actor-block draw.

    Neutral baselines are rebuilt from the records on every invocation, so a
    bootstrap replicate cannot accidentally reuse a point-estimate baseline.
    Duplicate actor IDs in ``actor_draw`` give that actor repeated block weight.
    """
    by_actor: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_actor[row["meta"]["actor"]].append(row)
    if actor_draw is None:
        actor_draw = sorted(by_actor)

    own_deltas: list[float] = []
    intensity_deltas: list[float] = []
    for actor in actor_draw:
        actor_records = by_actor[actor]
        neutral: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in actor_records:
            meta = row["meta"]
            if meta["emotion"] != "neutral":
                continue
            for lens in ("text400", "mixed528"):
                mass = row["readouts"][lens]["layers"][layer]["anchor_mass"]
                for cluster in clusters:
                    neutral[(meta["statement"], lens, cluster)].append(mass[cluster])
        bases = {key: sum(values) / len(values) for key, values in neutral.items()}

        contrasts: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        for row in actor_records:
            meta = row["meta"]
            cluster = acted_to_cluster.get(meta["emotion"])
            if cluster is None:
                continue
            log_lifts = {}
            for lens in ("text400", "mixed528"):
                mass = row["readouts"][lens]["layers"][layer]["anchor_mass"][cluster]
                base = bases[(meta["statement"], lens, cluster)]
                log_lifts[lens] = math.log(mass / base)
                contrasts[(meta["statement"], meta["emotion"], meta["intensity"], lens)].append(
                    log_lifts[lens]
                )
            own_deltas.append(log_lifts["mixed528"] - log_lifts["text400"])

        groups = {(statement, emotion) for statement, emotion, _intensity, _lens in contrasts}
        for statement, emotion in groups:
            lens_contrasts = {}
            for lens in ("text400", "mixed528"):
                normal = contrasts[(statement, emotion, "normal", lens)]
                strong = contrasts[(statement, emotion, "strong", lens)]
                if not normal or not strong:
                    continue
                lens_contrasts[lens] = sum(strong) / len(strong) - sum(normal) / len(normal)
            if len(lens_contrasts) == 2:
                intensity_deltas.append(
                    lens_contrasts["mixed528"] - lens_contrasts["text400"]
                )

    if not own_deltas or not intensity_deltas:
        raise ValueError("paired records do not contain complete estimands")
    return {
        "own_cluster_log_lift_delta": sum(own_deltas) / len(own_deltas),
        "strong_minus_normal_log_lift_delta": sum(intensity_deltas) / len(intensity_deltas),
    }


def actor_block_bootstrap(
    records: list[dict],
    *,
    layer: str,
    clusters: list[str],
    acted_to_cluster: dict[str, str],
    seed: int = 20260709,
    n_replicates: int = 10_000,
) -> dict[str, dict[str, float]]:
    actors = sorted({row["meta"]["actor"] for row in records})
    point = paired_estimands(
        records, layer=layer, clusters=clusters, acted_to_cluster=acted_to_cluster
    )
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_replicates) for name in point}
    for index in range(n_replicates):
        draw = rng.choice(actors, size=len(actors), replace=True).tolist()
        estimate = paired_estimands(
            records,
            layer=layer,
            clusters=clusters,
            acted_to_cluster=acted_to_cluster,
            actor_draw=draw,
        )
        for name, value in estimate.items():
            samples[name][index] = value
    return {
        name: {
            "estimate": value,
            "ci_low": float(np.quantile(samples[name], 0.025)),
            "ci_high": float(np.quantile(samples[name], 0.975)),
        }
        for name, value in point.items()
    }


def report_paired(records: list[dict], metadata: dict, layer: str) -> None:
    anchors = metadata["config"]["anchors"]
    summary = actor_block_bootstrap(
        records,
        layer=layer,
        clusters=anchors["clusters"],
        acted_to_cluster=anchors["acted_to_cluster"],
    )
    print(f"{len(records)} paired records · layer L{layer} · 10,000 actor-block replicates")
    for name, values in summary.items():
        print(
            f"{name}: {values['estimate']:+.4f} "
            f"(95% CI {values['ci_low']:+.4f}, {values['ci_high']:+.4f})"
        )
    for lens in ("text400", "mixed528"):
        lens_records = [
            {
                "meta": row["meta"],
                "layers": row["readouts"][lens]["layers"],
            }
            for row in records
        ]
        report(
            lifts_by_layer(lens_records, layer, anchors["clusters"]),
            f"{lens} descriptive own-cluster rank/argmax",
            anchors["clusters"],
            anchors["acted_to_cluster"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--layer", default=None, help="single layer to report")
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"no usable records in {args.results}")
    if "readouts" in records[0]:
        metadata = load_paired_metadata(args.results)
        validate_paired_records(records, metadata)
        layers = [args.layer] if args.layer else [str(layer) for layer in metadata["config"]["read_layers"]]
        for layer in layers:
            report_paired(records, metadata, layer)
        return
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
