"""Held-out synthetic controller validation and executable ablations."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from hypermax.algorithms.atpe2.controller import ATPE2Action
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.algorithms.random_search_optimizer import RandomSearchOptimizer
from hypermax.hyperparameter import Hyperparameter

from .database import content_hash
from .gates import gate2_report
from .generate import TRAJECTORY_METHODS, build_trajectory
from .synthetic import SyntheticTask, synthetic_tasks


_LEGACY_OPTIMIZER = None
_VARIANT_CACHE: dict[tuple[str, str | None, str | None], dict[str, ATPE2ModelEnsemble]] = {}


def _fixed_tpe(seed: int) -> ATPE2Optimizer:
    optimizer = ATPE2Optimizer(seed=seed)
    anchor = ATPE2Action.anchor()
    optimizer.controller.action_for = lambda space, results, current: anchor
    return optimizer


def _clone_ensemble(
    ensemble: ATPE2ModelEnsemble,
    constants: dict[str, Any] | None = None,
    disable_uncertainty_penalty: bool = False,
) -> ATPE2ModelEnsemble:
    manifest = copy.deepcopy(ensemble.manifest)
    if constants:
        manifest["ablation_action_overrides"] = {
            **manifest.get("ablation_action_overrides", {}),
            **constants,
        }
    if disable_uncertainty_penalty:
        manifest.setdefault("calibration", {})["uncertainty_penalty"] = 0.0
    return ATPE2ModelEnsemble(manifest, ensemble.models)


def controller_variants(
    pruned_manifest: Path,
    full_manifest: Path | None = None,
    legacy_feature_manifest: Path | None = None,
) -> dict[str, ATPE2ModelEnsemble]:
    pruned = ATPE2ModelEnsemble.load(pruned_manifest)
    if not pruned.research_ready:
        raise ValueError("Pruned/final ATPE2 ensemble is not research-ready")
    variants = {
        "atpe2": pruned,
        "ablation_filtering": _clone_ensemble(
            pruned,
            {
                "history_keep_fraction": 1.0,
                "history_recency_bias": 0.0,
                "history_quality_bias": 0.0,
            },
        ),
        "ablation_locking": _clone_ensemble(
            pruned,
            {"secondary_fraction": 0.0, "secondary_lock_probability": 0.0},
        ),
        "ablation_adaptive_gamma_candidates": _clone_ensemble(
            pruned, {"gamma": 0.25, "n_ei_candidates": 24}
        ),
        "ablation_uncertainty_penalty": _clone_ensemble(
            pruned, disable_uncertainty_penalty=True
        ),
        "ablation_pruned_schema": pruned,
    }
    if full_manifest:
        full = ATPE2ModelEnsemble.load(full_manifest)
        if not full.research_ready:
            raise ValueError("Full-schema ensemble is not research-ready")
        variants["ablation_full_schema"] = full
    if legacy_feature_manifest:
        legacy = ATPE2ModelEnsemble.load(legacy_feature_manifest)
        if not legacy.research_ready:
            raise ValueError("Legacy-feature ensemble is not research-ready")
        variants["ablation_legacy_features"] = legacy
    return variants


def _cached_variants(
    pruned_manifest: Path,
    full_manifest: Path | None,
    legacy_feature_manifest: Path | None,
) -> dict[str, ATPE2ModelEnsemble]:
    key = (
        str(pruned_manifest.resolve()),
        str(full_manifest.resolve()) if full_manifest else None,
        str(legacy_feature_manifest.resolve()) if legacy_feature_manifest else None,
    )
    if key not in _VARIANT_CACHE:
        _VARIANT_CACHE[key] = controller_variants(
            pruned_manifest, full_manifest, legacy_feature_manifest
        )
    return _VARIANT_CACHE[key]


def build_initial_history(
    task: SyntheticTask, seed: int, checkpoint: int, method: str = "tpe"
) -> list[dict[str, Any]]:
    return build_trajectory(task, method, seed, checkpoint)


def _run_five_steps(
    task: SyntheticTask,
    history: list[dict[str, Any]],
    optimizer: Any,
    rollout_seed: int,
    steps: int = 5,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    rows = [dict(row) for row in history]
    parameter = Hyperparameter(task.space)
    finite = [
        float(row["loss"])
        for row in rows
        if row.get("status", "ok") == "ok"
        and row.get("loss") is not None
        and np.isfinite(row["loss"])
    ]
    initial_incumbent = min(finite, default=None)
    incumbent = initial_incumbent
    curve = []
    failures = []
    overhead = 0.0
    for step in range(steps):
        started = time.perf_counter()
        try:
            params = optimizer.recommendNextParameters(task.space, rows, [])
            overhead += time.perf_counter() - started
            outcome = task.evaluate(params, rollout_seed, step)
            row = parameter.convertToFlatValues(params)
        except Exception as exc:  # Research evidence retains optimizer failures.
            overhead += time.perf_counter() - started
            outcome = {"status": "fail", "loss": None, "error": repr(exc)}
            row = {descriptor.name: "" for descriptor in parameter.getFlatParameters()}
        row.update(
            {
                "trial": len(rows),
                "status": outcome.get("status", "fail"),
                "loss": outcome.get("loss"),
            }
        )
        rows.append(row)
        loss = outcome.get("loss")
        if outcome.get("status") == "ok" and loss is not None and np.isfinite(loss):
            incumbent = float(loss) if incumbent is None else min(incumbent, float(loss))
        else:
            failures.append({"step": step, "error": outcome.get("error", "failure")})
        curve.append(incumbent)
    return {
        "initial_incumbent": initial_incumbent,
        "incumbent_curve": curve,
        "failures": failures,
        "optimizer_overhead_seconds": overhead,
    }


def _make_optimizer(
    name: str,
    seed: int,
    ensembles: dict[str, ATPE2ModelEnsemble],
):
    if name == "random":
        # Pure random search should not condition on prior trials.
        class FreshRandom:
            def __init__(self, inner):
                self.inner = inner

            def recommendNextParameters(self, space, results, current):
                return self.inner.recommendNextParameters(space, [], current)

        return FreshRandom(RandomSearchOptimizer(seed=seed))
    if name == "hyperopt_tpe":
        return _fixed_tpe(seed)
    if name == "legacy_atpe":
        global _LEGACY_OPTIMIZER
        random.seed(seed)
        np.random.seed(seed % (2**32))
        if _LEGACY_OPTIMIZER is None:
            from hypermax.algorithms.atpe_optimizer import ATPEOptimizer

            _LEGACY_OPTIMIZER = ATPEOptimizer()
        return _LEGACY_OPTIMIZER
    return ATPE2Optimizer(seed=seed, ensemble=ensembles[name])


def _rank_utilities(records: list[dict[str, Any]]) -> None:
    curve_lengths = {len(record["incumbent_curve"]) for record in records}
    if len(curve_lengths) != 1:
        raise ValueError("Paired incumbent curves must have equal lengths")
    for step in range(curve_lengths.pop()):
        values = np.asarray(
            [
                float(record["incumbent_curve"][step])
                if record["incumbent_curve"][step] is not None
                else np.inf
                for record in records
            ]
        )
        finite = np.isfinite(values)
        ranks = np.full(len(values), len(values), dtype=float)
        if finite.any():
            from scipy.stats import rankdata

            ranks[finite] = rankdata(values[finite], method="average")
        utilities = 1.0 - (ranks - 1.0) / max(1, len(values) - 1)
        for record, utility in zip(records, utilities):
            record.setdefault("step_utilities", []).append(float(utility))
    best = min(
        (
            float(value)
            for record in records
            for value in record["incumbent_curve"]
            if value is not None and np.isfinite(value)
        ),
        default=None,
    )
    for record in records:
        record["five_step_utility"] = float(np.mean(record.pop("step_utilities")))
        final = record["incumbent_curve"][-1]
        initial = record["initial_incumbent"]
        if best is None or final is None or not np.isfinite(final):
            record["normalized_regret"] = 1.0
        elif initial is None or initial <= best:
            record["normalized_regret"] = 0.0 if final <= best else 1.0
        else:
            record["normalized_regret"] = float(
                np.clip((float(final) - best) / (float(initial) - best), 0.0, 1.0)
            )


def _validate_task(
    task: SyntheticTask,
    task_index: int,
    seed: int,
    rollout_seeds: int,
    checkpoints: tuple[int, ...],
    trajectory_methods: tuple[str, ...],
    pruned_manifest: Path,
    full_manifest: Path | None,
    legacy_feature_manifest: Path | None,
) -> list[dict[str, Any]]:
    ensembles = _cached_variants(
        pruned_manifest, full_manifest, legacy_feature_manifest
    )
    names = ["random", "hyperopt_tpe", "legacy_atpe", *ensembles]
    task_rows = []
    for method_index, method in enumerate(trajectory_methods):
        history_seed = seed + task_index * 1_000_000 + method_index * 10_000
        for checkpoint in checkpoints:
            history = build_initial_history(task, history_seed, checkpoint, method)
            lineage = content_hash([task.task_id, "validation", method, history_seed])
            for rollout_index in range(rollout_seeds):
                rollout_seed = (
                    seed
                    + task_index * 100_000_000
                    + method_index * 1_000_000
                    + checkpoint * 1_000
                    + rollout_index * 101
                )
                paired = []
                for name in names:
                    result = _run_five_steps(
                        task,
                        history,
                        _make_optimizer(name, rollout_seed, ensembles),
                        rollout_seed,
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
                        }
                    )
                    paired.append(result)
                _rank_utilities(paired)
                task_rows.extend(paired)
    return task_rows


def validate_synthetic(
    pruned_manifest: Path,
    output: Path,
    seed: int,
    pruning_report: Path,
    full_manifest: Path | None = None,
    legacy_feature_manifest: Path | None = None,
    instances_per_family: int = 4,
    rollout_seeds: int = 5,
    checkpoints: tuple[int, ...] = (12, 20, 40),
    trajectory_methods: tuple[str, ...] = TRAJECTORY_METHODS,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    ensembles = _cached_variants(
        pruned_manifest, full_manifest, legacy_feature_manifest
    )
    names = ["random", "hyperopt_tpe", "legacy_atpe", *ensembles]
    tasks = synthetic_tasks(seed + 50_000_000, instances_per_family)
    jobs = [
        (
            task,
            task_index,
            seed,
            rollout_seeds,
            checkpoints,
            trajectory_methods,
            pruned_manifest,
            full_manifest,
            legacy_feature_manifest,
        )
        for task_index, task in enumerate(tasks)
    ]
    if workers == 1:
        blocks = [_validate_task(*job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            blocks = pool.starmap(_validate_task, jobs)
    rows = [row for block in blocks for row in block]

    pruning = json.loads(pruning_report.read_text())
    manifest = json.loads(pruned_manifest.read_text())
    completed_ablations = []
    if {"ablation_filtering"}.issubset(names):
        completed_ablations.append("filtering")
    if {"ablation_locking"}.issubset(names):
        completed_ablations.append("locking")
    if {"ablation_adaptive_gamma_candidates"}.issubset(names):
        completed_ablations.append("adaptive_gamma_candidates")
    if {"ablation_uncertainty_penalty"}.issubset(names):
        completed_ablations.append("uncertainty_penalty")
    if {"ablation_full_schema", "ablation_pruned_schema"}.issubset(names):
        completed_ablations.append("full_vs_pruned_schema")
    if "ablation_legacy_features" in names:
        completed_ablations.append("legacy_vs_robust_features")
    gate = gate2_report(
        rows,
        pruning,
        manifest.get("dataset_aggregation_rounds", []),
        completed_ablations,
        seed,
    )
    ablations = {
        name: {
            "median_five_step_utility": float(
                np.median([row["five_step_utility"] for row in rows if row["optimizer"] == name])
            ),
            "median_normalized_regret": float(
                np.median([row["normalized_regret"] for row in rows if row["optimizer"] == name])
            ),
            "failures": sum(
                len(row["failures"]) for row in rows if row["optimizer"] == name
            ),
        }
        for name in names
    }
    report = {
        "schema_version": "atpe2-synthetic-validation-v1",
        "seed": seed,
        "artifact_hashes": {
            "pruned_manifest_sha256": hashlib.sha256(
                pruned_manifest.read_bytes()
            ).hexdigest(),
            "pruning_report_sha256": hashlib.sha256(
                pruning_report.read_bytes()
            ).hexdigest(),
            "full_manifest_sha256": (
                hashlib.sha256(full_manifest.read_bytes()).hexdigest()
                if full_manifest
                else None
            ),
            "legacy_feature_manifest_sha256": (
                hashlib.sha256(legacy_feature_manifest.read_bytes()).hexdigest()
                if legacy_feature_manifest
                else None
            ),
        },
        "rows": rows,
        "ablations": ablations,
        "completed_ablations": completed_ablations,
        "gate2": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("pruned_manifest", type=Path)
    parser.add_argument("pruning_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--full-manifest", type=Path)
    parser.add_argument("--legacy-feature-manifest", type=Path)
    parser.add_argument("--instances-per-family", type=int, default=4)
    parser.add_argument("--rollout-seeds", type=int, default=5)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[12, 20, 40])
    parser.add_argument(
        "--trajectory-methods", nargs="+", choices=TRAJECTORY_METHODS, default=list(TRAJECTORY_METHODS)
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    report = validate_synthetic(
        args.pruned_manifest,
        args.output,
        args.seed,
        args.pruning_report,
        args.full_manifest,
        args.legacy_feature_manifest,
        args.instances_per_family,
        args.rollout_seeds,
        tuple(args.checkpoints),
        tuple(args.trajectory_methods),
        args.workers,
    )
    print(json.dumps({"gate2": report["gate2"], "ablations": report["ablations"]}, indent=2))


if __name__ == "__main__":
    main()
