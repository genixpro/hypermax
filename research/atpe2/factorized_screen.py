"""Protocol-locked aggregation for ATPE2 factorized behavioral screens."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


SMOKE_SEED = 7
THREE_QUICK_SEEDS = (7, 19, 31)
FIVE_QUICK_SEEDS = (7, 19, 23, 31, 47)
TARGETED_SEEDS = (19, 23, 31)
PERFORMANCE_FAMILIES = ("smooth", "rugged", "interaction", "hierarchical")


def _paired_rows(report: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    output: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in report["rows"]:
        key = (
            row["task"],
            row["trajectory_lineage"],
            row["checkpoint"],
            row["seed"],
        )
        output[key][row["optimizer"]] = row
    return output


def _family_deltas(
    reports: list[dict[str, Any]], comparator: str, field: str
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for paired in _paired_rows(report).values():
            if "atpe2" not in paired or comparator not in paired:
                continue
            candidate = paired["atpe2"]
            if candidate["family"] not in PERFORMANCE_FAMILIES:
                continue
            values[candidate["family"]].append(
                float(candidate[field]) - float(paired[comparator][field])
            )
    return values


def assess_five_quick_screens(
    candidate_reports: list[dict[str, Any]],
    incumbent_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    seeds = tuple(sorted(int(report["seed"]) for report in candidate_reports))
    if seeds != FIVE_QUICK_SEEDS:
        raise ValueError(f"five-quick assessment requires seeds {FIVE_QUICK_SEEDS}")
    incumbent_by_seed = {int(report["seed"]): report for report in incumbent_reports}
    if set(incumbent_by_seed) != set(FIVE_QUICK_SEEDS):
        raise ValueError("incumbent reports must use the same five quick seeds")
    regret = _family_deltas(candidate_reports, "hyperopt_tpe", "normalized_regret")
    improvement_vs_classic = _family_deltas(
        candidate_reports, "legacy_atpe", "scale_normalized_improvement_auc"
    )
    incumbent_vs_classic = _family_deltas(
        [incumbent_by_seed[seed] for seed in FIVE_QUICK_SEEDS],
        "legacy_atpe",
        "scale_normalized_improvement_auc",
    )
    candidate_mean_gap = float(
        np.mean([value for values in improvement_vs_classic.values() for value in values])
    )
    incumbent_mean_gap = float(
        np.mean([value for values in incumbent_vs_classic.values() for value in values])
    )
    incumbent_deficit = max(0.0, -incumbent_mean_gap)
    candidate_deficit = max(0.0, -candidate_mean_gap)
    gap_closure = (
        1.0 - candidate_deficit / incumbent_deficit
        if incumbent_deficit > 0
        else float(candidate_mean_gap >= incumbent_mean_gap)
    )
    candidate_overheads = [
        float(report["summary"]["optimizer_overhead"]["atpe2"]["median_block_seconds"])
        for report in candidate_reports
    ]
    incumbent_overheads = [
        float(incumbent_by_seed[seed]["summary"]["optimizer_overhead"]["atpe2"]["median_block_seconds"])
        for seed in FIVE_QUICK_SEEDS
    ]
    overhead_ratio = float(np.median(candidate_overheads) / np.median(incumbent_overheads))
    family_mean_regret = {
        family: float(np.mean(regret.get(family, [np.inf])))
        for family in PERFORMANCE_FAMILIES
    }
    seed_family_regret: dict[str, float] = {}
    for report in candidate_reports:
        per_report = _family_deltas([report], "hyperopt_tpe", "normalized_regret")
        for family in PERFORMANCE_FAMILIES:
            seed_family_regret[f"{report['seed']}:{family}"] = float(
                np.mean(per_report.get(family, [np.inf]))
            )
    checks = {
        "all_five_quick_screens_pass": all(
            report["summary"]["screen_passed"] for report in candidate_reports
        ),
        "every_family_mean_regret_non_positive": all(
            value <= 0.0 for value in family_mean_regret.values()
        ),
        "no_seed_family_regret_above_0_10": all(
            value <= 0.10 for value in seed_family_regret.values()
        ),
        "classic_deficit_closed_at_least_50_percent": gap_closure >= 0.50,
        "controller_overhead_at_most_1_25x": overhead_ratio <= 1.25,
        "all_decisions_adaptive_without_fallback": all(
            report["summary"]["checks"]["all_atpe2_blocks_have_adaptive_decisions"]
            for report in candidate_reports
        ),
    }
    return {
        "schema_version": "atpe2-factorized-five-quick-assessment-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seeds": list(seeds),
        "family_mean_paired_regret_vs_tpe": family_mean_regret,
        "seed_family_mean_paired_regret_vs_tpe": seed_family_regret,
        "classic_gap": {
            "incumbent_mean_atpe2_minus_classic": incumbent_mean_gap,
            "candidate_mean_atpe2_minus_classic": candidate_mean_gap,
            "deficit_closure_fraction": gap_closure,
        },
        "controller_overhead_ratio": overhead_ratio,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=Path, nargs=5, required=True)
    parser.add_argument("--incumbent", type=Path, nargs=5, required=True)
    args = parser.parse_args(argv)
    report = assess_five_quick_screens(
        [json.loads(path.read_text()) for path in args.candidate],
        [json.loads(path.read_text()) for path in args.incumbent],
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
