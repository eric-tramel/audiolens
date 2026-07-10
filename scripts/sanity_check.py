"""Inspect a completed, content-addressed workspace-evaluation report.

This command is deliberately a report reader, not a second evaluator.  It
prints summaries already present in the validated report and never loads a
model, lens tensor, benchmark fixture, or inferred sidecar.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any, Mapping

os.environ["AUDIOLENS_REPORT_INSPECTOR_ONLY"] = "1"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from modal_workspace_eval import load_completed_workspace_report  # noqa: E402


def render_workspace_report(report: Mapping[str, Any]) -> str:
    """Render only validated, precomputed report fields."""
    fit = report["fit_identity"]
    eligibility = report["eligibility"]["distributions"]
    summaries = report["summaries"]
    aggregate = summaries["equal_distribution_aggregate"]["layer_sets"]
    adjudication = report["adjudication"]
    lines = [
        "AudioLens canonical J-lens workspace evaluation",
        f"status: {adjudication['status']}",
        f"evaluation config: {report['evaluation_config_sha256']}",
        f"report content: {report['workspace_report_sha256']}",
        f"fit config: {fit['fit_config_sha256']}",
        (
            "lens: "
            f"{fit['lens_relative_path']} sha256={fit['lens_sha256']} "
            f"bytes={fit['lens_bytes']} dtype={fit['lens_dtype']} "
            f"prompts={fit['lens_n_prompts']}"
        ),
        "fixtures / tokenizer eligibility:",
    ]
    for fixture in report["fixtures"]:
        coverage = eligibility[fixture["slug"]]
        lines.append(
            "  "
            f"{fixture['slug']}: source={fixture['publication_count']} "
            f"eligible_items={coverage['eligible_items']} "
            f"eligible_concepts={coverage['eligible_concepts']} "
            f"fixture_sha256={fixture['sha256']} "
            f"names_sha256={fixture['selected_name_sha256']}"
        )
    lines.append("all-distribution summaries:")
    for slug in fixture_slugs(report):
        layer_sets = summaries["distributions"][slug]["layer_sets"]
        all_layers = layer_sets["all_l0_l33"]["variants"]
        band = layer_sets["candidate_l13_l31"]["variants"]
        lines.append(f"  {slug}:")
        for label, values in (("all L0-L33", all_layers), ("band L13-L31", band)):
            candidate = values["candidate"]
            lines.append(
                f"    {label}: candidate_auc={candidate['log_k_auc']:.6f} "
                f"logit_auc={values['logit']['log_k_auc']:.6f} "
                f"transposed_auc={values['transposed']['log_k_auc']:.6f} "
                f"permuted_auc={values['permuted']['log_k_auc']:.6f} "
                f"candidate_pass@k={candidate['pass_at_k']}"
            )
    lines.append("fixed equal-distribution region curves:")
    for region in ("early_l0_l12", "candidate_l13_l31", "motor_l32_l33"):
        region_summary = aggregate[region]
        candidate = region_summary["variants"]["candidate"]
        motor = region_summary["motor_metrics"]
        lines.append(
            f"  {region}: layers={region_summary['layers']} "
            f"intermediate_auc={candidate['log_k_auc']:.6f} "
            f"candidate_pass@k={candidate['pass_at_k']} "
            "next_token_agreement@k="
            f"{motor['next_token_agreement_at_k']['candidate']} "
            f"j_logit_top1_agreement={motor['candidate_logit_top1_agreement']:.6f} "
            f"j_logit_js_nats={motor['candidate_logit_js_nats']:.6f}"
        )
    failed = adjudication["failed_criteria"]
    lines.append("failed criteria: " + (", ".join(failed) if failed else "none"))
    lines.append(
        "claim: "
        + (
            "fixed L13-L31 workspace-like transfer validated"
            if adjudication["status"] == "validated"
            else "no workspace-like band validated; canonical J-lens remains valid"
        )
    )
    return "\n".join(lines)


def fixture_slugs(report: Mapping[str, Any]) -> list[str]:
    return [fixture["slug"] for fixture in report["fixtures"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        required=True,
        help="explicit completed workspace-evaluation JSON (bare .pt is rejected)",
    )
    arguments = parser.parse_args()
    report = load_completed_workspace_report(arguments.report)
    print(render_workspace_report(report))


if __name__ == "__main__":
    main()
