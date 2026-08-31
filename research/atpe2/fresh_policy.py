"""Small held-out multi-step policy selection before behavioral screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble

from .iterate import _add_improvement_auc, _history_loss_scale
from .synthetic import synthetic_tasks
from .validate import _make_optimizer, _run_five_steps, build_initial_history


FRESH_POLICY_SCHEMA_VERSION = "atpe2-fresh-policy-quality-v1"
FRESH_POLICY_CHECKPOINT = 25
FRESH_POLICY_STEPS = 5
REJECTED_POLICY_QUALITY = -1.0e9


def _task_quality(
    task: Any, task_index: int, seed: int, manifest_path: Path
) -> dict[str, Any]:
    ensemble = ATPE2ModelEnsemble.load(manifest_path)
    if not ensemble.research_ready or not ensemble.is_factorized:
        raise ValueError("Fresh policy quality requires a factorized candidate")
    history_seed = seed + task_index * 1_000_000
    rollout_seed = seed + task_index * 100_000_000 + 17
    history = build_initial_history(task, history_seed, FRESH_POLICY_CHECKPOINT, "tpe")
    optimizer = _make_optimizer("atpe2", rollout_seed, {"atpe2": ensemble})
    result = _run_five_steps(
        task, history, optimizer, rollout_seed, steps=FRESH_POLICY_STEPS
    )
    _add_improvement_auc(result, _history_loss_scale(history))
    decisions = list(optimizer.controller.records)
    adaptive = bool(decisions) and all(
        decision.get("decision_mode") == "adaptive_model"
        and decision.get("fallback_reason") is None
        for decision in decisions
    )
    return {
        "task": task.task_id,
        "family": task.family,
        "scale_normalized_improvement_auc": result[
            "scale_normalized_improvement_auc"
        ],
        "adaptive_without_fallback": adaptive,
        "controller_decisions": decisions,
    }


def summarize_fresh_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Fresh policy quality requires held-out rows")
    family_means = {
        family: float(
            np.mean(
                [
                    row["scale_normalized_improvement_auc"]
                    for row in rows
                    if row["family"] == family
                ]
            )
        )
        for family in sorted({row["family"] for row in rows})
    }
    values = np.asarray(list(family_means.values()), dtype=float)
    adaptive = all(row["adaptive_without_fallback"] for row in rows)
    quality = float(np.mean(values) - 0.10 * np.std(values))
    return {
        "held_out_stable_policy_quality": quality if adaptive else REJECTED_POLICY_QUALITY,
        "family_mean_improvement_auc": family_means,
        "all_decisions_adaptive_without_fallback": adaptive,
    }


def evaluate_manifest(
    manifest_path: Path, seed: int, workers: int
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    tasks = synthetic_tasks(seed + 80_000_000, instances_per_family=1)
    jobs = [
        (task, task_index, seed, manifest_path)
        for task_index, task in enumerate(tasks)
    ]
    if workers == 1:
        rows = [_task_quality(*job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            rows = pool.starmap(_task_quality, jobs)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_version": manifest["model_version"],
        "model_artifact_sha256": manifest["model_artifact_sha256"],
        "summary": summarize_fresh_policy(rows),
        "rows": rows,
    }


def run_fresh_policy_selection(
    manifests: list[Path], output: Path, seed: int, workers: int = 4
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("At least one selected manifest is required")
    if len(manifests) > 12:
        raise ValueError("Fresh policy selection is capped at twelve manifests")
    results = [evaluate_manifest(path, seed, workers) for path in manifests]
    mapping = {
        result["model_artifact_sha256"]: result["summary"][
            "held_out_stable_policy_quality"
        ]
        for result in results
    }
    report = {
        "schema_version": FRESH_POLICY_SCHEMA_VERSION,
        "evidence_eligible": False,
        "seed": seed,
        "protocol": {
            "instances_per_family": 1,
            "checkpoint": FRESH_POLICY_CHECKPOINT,
            "steps": FRESH_POLICY_STEPS,
            "trajectory_method": "tpe",
            "candidate_cap": 12,
        },
        "quality_by_model_artifact_sha256": mapping,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = run_fresh_policy_selection(
        args.manifests, args.output, args.seed, args.workers
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "quality_by_model_artifact_sha256": report[
                    "quality_by_model_artifact_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
