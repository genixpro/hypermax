"""Fast, directional ATPE2 development benchmark.

This suite is intentionally not an evidence gate.  Its fixed synthetic
development tasks may be used repeatedly while iterating; Gate 2 and external
Gate 3 remain the independent checks after a controller is frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble

from .database import content_hash
from .evidence import paired_bootstrap_interval
from .synthetic import synthetic_tasks
from .validate import (
    _make_optimizer,
    _rank_utilities,
    _run_five_steps,
    build_initial_history,
)


ITERATION_SCHEMA_VERSION = "atpe2-iteration-screen-v3"
ITERATION_OPTIMIZERS = ("random", "hyperopt_tpe", "legacy_atpe", "atpe2")
ITERATION_TRAJECTORIES = ("random", "tpe", "space_filled_atpe2")
ITERATION_CHECKPOINTS = (10, 25, 50)
ITERATION_ROLLOUT_SEEDS = 2
ITERATION_INSTANCES_PER_FAMILY = 2
ITERATION_STEPS = 10
PERFORMANCE_FAMILIES = {"smooth", "rugged", "interaction", "hierarchical"}
ITERATION_FAMILIES = (
    "constant_zero",
    "hierarchical",
    "interaction",
    "plateau",
    "rugged",
    "smooth",
)
COMPARATOR_OPTIMIZERS = ("random", "hyperopt_tpe", "legacy_atpe")
COMPARATOR_CACHE_SCHEMA_VERSION = "atpe2-iteration-comparators-v1"


def environment_identity() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
    }


def validated_comparator_rows(
    cache: dict[str, Any], seed: int, task_families: tuple[str, ...] = ITERATION_FAMILIES
) -> list[dict[str, Any]]:
    expected = {
        "schema_version": COMPARATOR_CACHE_SCHEMA_VERSION,
        "iteration_schema_version": ITERATION_SCHEMA_VERSION,
        "seed": seed,
        "optimizers": list(COMPARATOR_OPTIMIZERS),
        "environment": environment_identity(),
        "task_families": list(task_families),
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f"Comparator cache mismatch for {key}")
    rows = cache.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Comparator cache rows must be a list")
    return rows


def _history_loss_scale(history: list[dict[str, Any]]) -> float:
    losses = np.asarray(
        [
            float(row["loss"])
            for row in history
            if row.get("status", "ok") == "ok"
            and row.get("loss") is not None
            and np.isfinite(row["loss"])
        ],
        dtype=float,
    )
    if len(losses) < 2:
        return 1.0
    scale = float(np.quantile(losses, 0.75) - np.quantile(losses, 0.25))
    if scale <= np.finfo(float).eps:
        scale = float(np.max(losses) - np.min(losses))
    return scale if scale > np.finfo(float).eps else 1.0


def _add_improvement_auc(result: dict[str, Any], scale: float) -> None:
    initial = result["initial_incumbent"]
    if initial is None or not np.isfinite(initial):
        result["scale_normalized_improvement_auc"] = -1.0
        return
    improvements = [
        (float(initial) - float(value)) / scale
        if value is not None and np.isfinite(value)
        else -1.0
        for value in result["incumbent_curve"]
    ]
    result["scale_normalized_improvement_auc"] = float(np.mean(improvements))


def _task_rows(
    task,
    task_index: int,
    seed: int,
    manifest_path: Path | None,
    optimizer_names: tuple[str, ...] = ITERATION_OPTIMIZERS,
) -> list[dict[str, Any]]:
    ensembles = {}
    if "atpe2" in optimizer_names:
        if manifest_path is None:
            raise ValueError("Candidate iteration rows require an ATPE2 manifest")
        ensemble = ATPE2ModelEnsemble.load(manifest_path)
        if not ensemble.research_ready:
            raise ValueError("Iteration benchmark requires a fitted ATPE2 ensemble")
        ensembles["atpe2"] = ensemble
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(ITERATION_TRAJECTORIES):
        history_seed = seed + task_index * 1_000_000 + method_index * 10_000
        for checkpoint in ITERATION_CHECKPOINTS:
            history = build_initial_history(task, history_seed, checkpoint, method)
            loss_scale = _history_loss_scale(history)
            lineage = content_hash(
                [task.specification(), ITERATION_SCHEMA_VERSION, method, history_seed]
            )
            for rollout_index in range(ITERATION_ROLLOUT_SEEDS):
                rollout_seed = (
                    seed
                    + task_index * 100_000_000
                    + method_index * 1_000_000
                    + checkpoint * 1_000
                    + rollout_index * 101
                )
                paired = []
                for name in optimizer_names:
                    optimizer = _make_optimizer(name, rollout_seed, ensembles)
                    result = _run_five_steps(
                        task,
                        history,
                        optimizer,
                        rollout_seed,
                        steps=ITERATION_STEPS,
                    )
                    _add_improvement_auc(result, loss_scale)
                    decisions = (
                        list(optimizer.controller.records)
                        if name == "atpe2"
                        else []
                    )
                    result.update(
                        {
                            "task": task.task_id,
                            "family": task.family,
                            "trajectory_method": method,
                            "trajectory_lineage": lineage,
                            "checkpoint": checkpoint,
                            "seed": rollout_seed,
                            "optimizer": name,
                            "controller_decisions": decisions,
                        }
                    )
                    paired.append(result)
                _rank_utilities(paired)
                for result in paired:
                    result["rank_utility_auc"] = result["five_step_utility"]
                rows.extend(paired)
    return rows


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def summarize_iteration(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["task"],
            row["trajectory_lineage"],
            row["checkpoint"],
            row["seed"],
        )
        paired[key][row["optimizer"]] = row

    task_improvement_deltas: dict[str, list[float]] = defaultdict(list)
    family_regret: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    family_improvement_deltas: dict[str, list[float]] = defaultdict(list)
    family_regret_deltas: dict[str, list[float]] = defaultdict(list)
    wins = ties = losses = 0
    for values in paired.values():
        atpe2 = values["atpe2"]
        tpe = values["hyperopt_tpe"]
        rank_delta = float(atpe2["rank_utility_auc"] - tpe["rank_utility_auc"])
        improvement_delta = float(
            atpe2["scale_normalized_improvement_auc"]
            - tpe["scale_normalized_improvement_auc"]
        )
        if atpe2["family"] in PERFORMANCE_FAMILIES:
            task_improvement_deltas[atpe2["task"]].append(improvement_delta)
            family_improvement_deltas[atpe2["family"]].append(improvement_delta)
            family_regret_deltas[atpe2["family"]].append(
                float(atpe2["normalized_regret"] - tpe["normalized_regret"])
            )
        if rank_delta > 1e-12:
            wins += 1
        elif rank_delta < -1e-12:
            losses += 1
        else:
            ties += 1
        for optimizer in ("atpe2", "hyperopt_tpe"):
            row = values[optimizer]
            family_regret[row["family"]][optimizer].append(
                float(row["normalized_regret"])
            )

    task_improvement_medians = [
        float(np.median(deltas))
        for deltas in task_improvement_deltas.values()
    ]
    interval = paired_bootstrap_interval(task_improvement_medians, seed)
    family_degradation = {
        family: float(
            np.median(values["atpe2"])
            - np.median(values["hyperopt_tpe"])
        )
        for family, values in family_regret.items()
    }
    family_improvement = {
        family: float(np.median(values))
        for family, values in family_improvement_deltas.items()
    }
    family_paired_regret = {
        family: {
            "mean_delta": float(np.mean(values)),
            "median_delta": float(np.median(values)),
        }
        for family, values in family_regret_deltas.items()
    }
    overhead = {}
    for optimizer in ITERATION_OPTIMIZERS:
        values = [
            float(row["optimizer_overhead_seconds"])
            for row in rows
            if row["optimizer"] == optimizer
        ]
        overhead[optimizer] = {
            "median_block_seconds": float(np.median(values)),
            "p95_block_seconds": _percentile(values, 0.95),
            "total_seconds": float(np.sum(values)),
        }

    decisions = [
        decision
        for row in rows
        if row["optimizer"] == "atpe2"
        for decision in row["controller_decisions"]
    ]
    configurations = {
        json.dumps(decision["configuration"], sort_keys=True)
        for decision in decisions
    }
    all_adaptive = bool(decisions) and all(
        decision.get("decision_mode") == "adaptive_model"
        and decision.get("fallback_reason") is None
        for decision in decisions
    )
    effective_decisions = [
        decision
        for decision in decisions
        if "successful_history_count" in decision
        and "filtered_history_count" in decision
    ]
    locked_parameter_counts = Counter(
        name
        for decision in decisions
        for name in decision.get("locked_parameters", [])
    )
    retained_history_ratios = [
        decision["filtered_history_count"] / decision["successful_history_count"]
        for decision in effective_decisions
        if decision["successful_history_count"] > 0
    ]
    worst_paired_regret = max(
        (
            values["mean_delta"]
            for values in family_paired_regret.values()
        ),
        default=0.0,
    )
    worst_improvement = min(family_improvement.values(), default=0.0)
    checks = {
        "median_task_improvement_auc_delta_at_least_minus_0_02": (
            float(np.median(task_improvement_medians)) >= -0.02
        ),
        # Two development tasks per family are intentionally too small to
        # enforce Gate 2's 5% bound.  Use paired block effects here; subtracting
        # separate family medians can produce a large, misleading difference.
        "no_performance_family_mean_paired_regret_delta_above_0_10": (
            worst_paired_regret <= 0.10
        ),
        "no_performance_family_improvement_auc_delta_below_minus_0_25": (
            worst_improvement >= -0.25
        ),
        "all_atpe2_blocks_have_adaptive_decisions": all_adaptive,
        "adaptive_action_diversity": len(configurations) > 1,
    }
    return {
        "screen_passed": all(checks.values()),
        "checks": checks,
        "atpe2_minus_tpe": {
            "median_task_scale_normalized_improvement_auc_delta": float(
                np.median(task_improvement_medians)
            ),
            "directional_95_percent_improvement_auc_interval": interval,
            "rank_utility_block_wins": wins,
            "rank_utility_block_ties": ties,
            "rank_utility_block_losses": losses,
        },
        "family_marginal_median_normalized_regret_degradation": family_degradation,
        "performance_family_paired_normalized_regret_delta": family_paired_regret,
        "performance_family_median_improvement_auc_delta": family_improvement,
        "optimizer_overhead": overhead,
        "adaptive_telemetry": {
            "decision_count": len(decisions),
            "distinct_configurations": len(configurations),
            "outside_training_envelope_fraction": float(
                np.mean(
                    [
                        bool(decision.get("outside_training_envelope"))
                        for decision in decisions
                    ]
                )
            ),
            "decisions_with_effective_locks": sum(
                bool(decision.get("locked_parameters")) for decision in decisions
            ),
            "locked_parameter_counts": dict(sorted(locked_parameter_counts.items())),
            "history_filtering_decisions": sum(
                decision["filtered_history_count"]
                < decision["successful_history_count"]
                for decision in effective_decisions
            ),
            "median_retained_history_fraction": (
                float(np.median(retained_history_ratios))
                if retained_history_ratios
                else None
            ),
        },
    }


def run_iteration(
    manifest_path: Path,
    output: Path,
    seed: int,
    workers: int = 6,
    comparator_cache: Path | None = None,
    task_families: tuple[str, ...] = ITERATION_FAMILIES,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    tasks = synthetic_tasks(
        seed + 70_000_000,
        instances_per_family=ITERATION_INSTANCES_PER_FAMILY,
    )
    unknown = set(task_families) - set(ITERATION_FAMILIES)
    if unknown or not task_families:
        raise ValueError(f"Unknown or empty iteration families: {sorted(unknown)}")
    tasks = [task for task in tasks if task.family in task_families]
    optimizer_names = ("atpe2",) if comparator_cache else ITERATION_OPTIMIZERS
    jobs = [
        (task, task_index, seed, manifest_path, optimizer_names)
        for task_index, task in enumerate(tasks)
    ]
    if workers == 1:
        blocks = [_task_rows(*job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            blocks = pool.starmap(_task_rows, jobs)
    rows = [row for block in blocks for row in block]
    comparator_identity = None
    if comparator_cache:
        cache = json.loads(comparator_cache.read_text())
        rows.extend(validated_comparator_rows(cache, seed, task_families))
        paired: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                row["task"],
                row["trajectory_lineage"],
                row["checkpoint"],
                row["seed"],
            )
            paired[key].append(row)
        if not paired or any(
            {row["optimizer"] for row in values} != set(ITERATION_OPTIMIZERS)
            for values in paired.values()
        ):
            raise ValueError("Comparator cache does not pair every candidate block")
        for values in paired.values():
            _rank_utilities(values)
            for row in values:
                row["rank_utility_auc"] = row["five_step_utility"]
        comparator_identity = hashlib.sha256(comparator_cache.read_bytes()).hexdigest()
    report = {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "evidence_eligible": False,
        "warning": (
            "Directional development screen only. Repeated use can overfit these "
            "tasks; this report cannot satisfy Gate 2 or Gate 3."
        ),
        "seed": seed,
        "model_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "comparator_cache_sha256": comparator_identity,
        "protocol": {
            "task_families": sorted({task.family for task in tasks}),
            "instances_per_family": ITERATION_INSTANCES_PER_FAMILY,
            "trajectory_methods": list(ITERATION_TRAJECTORIES),
            "checkpoints": list(ITERATION_CHECKPOINTS),
            "rollout_seeds": ITERATION_ROLLOUT_SEEDS,
            "optimizers": list(ITERATION_OPTIMIZERS),
            "paired_blocks": len(rows) // len(ITERATION_OPTIMIZERS),
            "rollout_steps": ITERATION_STEPS,
            "objective_evaluations": len(rows) * ITERATION_STEPS,
            "workers": min(workers, len(jobs)),
        },
        "summary": summarize_iteration(rows, seed),
        "wall_clock_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--comparator-cache", type=Path)
    parser.add_argument(
        "--families", nargs="+", choices=ITERATION_FAMILIES, default=ITERATION_FAMILIES
    )
    args = parser.parse_args(argv)
    report = run_iteration(
        args.manifest,
        args.output,
        args.seed,
        args.workers,
        args.comparator_cache,
        tuple(args.families),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "evidence_eligible": report["evidence_eligible"],
                "protocol": report["protocol"],
                "summary": report["summary"],
                "wall_clock_seconds": report["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
