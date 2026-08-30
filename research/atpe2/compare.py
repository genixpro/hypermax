"""Paired comparison of two directional ATPE2 iteration reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evidence import paired_bootstrap_interval
from .iterate import PERFORMANCE_FAMILIES


COMPARISON_SCHEMA_VERSION = "atpe2-iteration-comparison-v1"


def _atpe2_rows(report: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in report.get("rows", []):
        if row.get("optimizer") != "atpe2":
            continue
        key = (
            row["task"],
            row["trajectory_lineage"],
            int(row["checkpoint"]),
            int(row["seed"]),
        )
        if key in rows:
            raise ValueError(f"Duplicate ATPE2 comparison block: {key}")
        rows[key] = row
    if not rows:
        raise ValueError("Iteration report contains no ATPE2 rows")
    return rows


def _direction(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _summary(deltas: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(deltas), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Comparison requires finite paired deltas")
    directions = [_direction(float(value)) for value in values]
    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "wins": directions.count(1),
        "ties": directions.count(0),
        "losses": directions.count(-1),
        "blocks": len(values),
    }


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    if baseline.get("seed") != candidate.get("seed"):
        raise ValueError("Iteration reports must use the same root seed")
    baseline_rows = _atpe2_rows(baseline)
    candidate_rows = _atpe2_rows(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Iteration reports do not contain identical paired blocks")

    deltas: dict[tuple[Any, ...], float] = {}
    exact_curve_matches = 0
    task_deltas: dict[str, list[float]] = defaultdict(list)
    for key, baseline_row in baseline_rows.items():
        candidate_row = candidate_rows[key]
        if (
            baseline_row["family"] != candidate_row["family"]
            or baseline_row["trajectory_method"]
            != candidate_row["trajectory_method"]
            or baseline_row["initial_incumbent"]
            != candidate_row["initial_incumbent"]
        ):
            raise ValueError(f"Iteration block provenance mismatch: {key}")
        delta = float(
            candidate_row["scale_normalized_improvement_auc"]
            - baseline_row["scale_normalized_improvement_auc"]
        )
        deltas[key] = delta
        if candidate_row["incumbent_curve"] == baseline_row["incumbent_curve"]:
            exact_curve_matches += 1
        if candidate_row["family"] in PERFORMANCE_FAMILIES:
            task_deltas[candidate_row["task"]].append(delta)

    task_medians = [float(np.median(values)) for values in task_deltas.values()]
    performance_keys = [
        key
        for key, row in candidate_rows.items()
        if row["family"] in PERFORMANCE_FAMILIES
    ]
    by_family = {
        family: _summary(
            deltas[key]
            for key in performance_keys
            if candidate_rows[key]["family"] == family
        )
        for family in sorted(PERFORMANCE_FAMILIES)
    }
    by_checkpoint = {
        str(checkpoint): _summary(
            deltas[key]
            for key in performance_keys
            if int(candidate_rows[key]["checkpoint"]) == checkpoint
        )
        for checkpoint in sorted(
            {int(candidate_rows[key]["checkpoint"]) for key in performance_keys}
        )
    }
    baseline_overhead = float(
        baseline["summary"]["optimizer_overhead"]["atpe2"]["median_block_seconds"]
    )
    candidate_overhead = float(
        candidate["summary"]["optimizer_overhead"]["atpe2"]["median_block_seconds"]
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "evidence_eligible": False,
        "seed": seed,
        "baseline_model_manifest_sha256": baseline.get("model_manifest_sha256"),
        "candidate_model_manifest_sha256": candidate.get("model_manifest_sha256"),
        "performance_family_candidate_minus_baseline": _summary(
            deltas[key] for key in performance_keys
        ),
        "task_median_95_percent_interval": paired_bootstrap_interval(
            task_medians, seed
        ),
        "by_family": by_family,
        "by_checkpoint": by_checkpoint,
        "exact_incumbent_curve_matches": exact_curve_matches,
        "paired_blocks": len(deltas),
        "median_optimizer_overhead_ratio": (
            candidate_overhead / baseline_overhead
            if baseline_overhead > 0
            else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    comparison = compare_reports(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
