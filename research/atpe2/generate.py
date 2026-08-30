"""Resumable paired-rollout data collection for ATPE2.

The generator retains every action, uses common objective-noise seeds, and labels
each state only against other actions evaluated from that exact state and seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import time
from typing import Any

import hyperopt
import numpy as np
from scipy.stats import qmc

from hypermax.algorithms.atpe2.controller import ATPE2Action, ATPE2Controller
from hypermax.algorithms.atpe2.features import (
    ATPE2ContextV4FeatureExtractor,
    ATPE2FeatureExtractor,
)
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.random_search_optimizer import RandomSearchOptimizer
from hypermax.algorithms.tpe_optimizer import TPEOptimizer
from hypermax.hyperparameter import Hyperparameter

from .database import ResearchStore, content_hash
from .labels import label_state
from .synthetic import (
    GENERATOR_VERSION,
    SyntheticTask,
    factorized_training_tasks,
    synthetic_tasks,
)
from .acquisition import (
    ACQUISITION_IMPLEMENTATION_VERSION,
    AcquisitionCandidate,
    acquisition_lineage,
    structured_action_candidates,
)


TRAJECTORY_METHODS = ("random", "tpe", "legacy_atpe", "space_filled_atpe2")


@lru_cache(maxsize=1)
def dependency_versions() -> dict[str, str]:
    packages = ("hyperopt", "lightgbm", "numpy", "scipy")
    versions = {package: importlib.metadata.version(package) for package in packages}
    versions["python"] = platform.python_version()
    return versions


def action_candidates(
    seed: int,
    acquisition_round: int = 0,
    fixed_constants: dict[str, Any] | None = None,
    action_ranges: dict[str, dict[str, Any]] | None = None,
) -> list[ATPE2Action]:
    """Return 64 Sobol configurations plus the neutral TPE anchor."""
    controller = ATPE2Controller(seed)
    # Offline acquisition has its own fixed 64-action protocol.  It must not
    # inherit a pruned runtime policy's candidate count, fixed constants, or
    # ranges when the packaged model is replaced by a faster ensemble.
    controller.ensemble.manifest["runtime_candidate_count"] = 64
    controller.ensemble.manifest["validation_selected_constants"] = {}
    controller.ensemble.manifest.pop("ablation_action_overrides", None)
    controller.ensemble.manifest["action_ranges"] = action_ranges or {}
    candidates = controller._candidate_actions(acquisition_round)[:64]
    actions = candidates + [ATPE2Action.anchor()]
    if fixed_constants:
        actions = [ATPE2Action(**{**asdict(action), **fixed_constants}) for action in actions]
    return actions


def _optimizer_for_trajectory(
    method: str,
    seed: int,
    model_manifest: Path | None = None,
    fixed_constants: dict[str, Any] | None = None,
    action_ranges: dict[str, dict[str, Any]] | None = None,
):
    if method == "random":
        return RandomSearchOptimizer(seed=seed)
    if method == "tpe":
        return TPEOptimizer(seed=seed)
    if method == "legacy_atpe":
        from hypermax.algorithms.atpe_optimizer import ATPEOptimizer

        return ATPEOptimizer(seed=seed)
    if method == "space_filled_atpe2":
        optimizer = ATPE2Optimizer(seed=seed)
        actions = action_candidates(
            seed, fixed_constants=fixed_constants, action_ranges=action_ranges
        )
        action = actions[seed % 64]
        optimizer.controller.action_for = lambda space, results, current: action
        return optimizer
    if method == "deployed_atpe2":
        if model_manifest is None:
            raise ValueError("Dataset aggregation requires --model-manifest")
        ensemble = ATPE2ModelEnsemble.load(model_manifest)
        if not ensemble.research_ready:
            raise ValueError("Dataset aggregation requires a fitted controller candidate")
        return ATPE2Optimizer(seed=seed, ensemble=ensemble)
    raise ValueError(f"Unknown trajectory method: {method}")


def build_trajectory(
    task: SyntheticTask,
    method: str,
    seed: int,
    length: int,
    model_manifest: Path | None = None,
    fixed_constants: dict[str, Any] | None = None,
    action_ranges: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    optimizer = _optimizer_for_trajectory(
        method, seed, model_manifest, fixed_constants, action_ranges
    )
    history = []
    parameter = Hyperparameter(task.space)
    for trial in range(length):
        params = optimizer.recommendNextParameters(task.space, history, [])
        outcome = task.evaluate(params, seed, trial)
        row = parameter.convertToFlatValues(params)
        row.update({"trial": trial, "status": outcome["status"], "loss": outcome.get("loss")})
        history.append(row)
    return history


def rollout_action(
    task: SyntheticTask,
    history: list[dict[str, Any]],
    action: ATPE2Action,
    rollout_seed: int,
) -> tuple[list[float | None], list[dict], dict]:
    optimizer = ATPE2Optimizer(seed=rollout_seed)
    optimizer.controller.action_for = lambda space, results, current: action
    rollout_history = [dict(row) for row in history]
    finite_history = [
        float(row["loss"])
        for row in history
        if row.get("status", "ok") == "ok"
        and row.get("loss") is not None
        and np.isfinite(row["loss"])
    ]
    incumbent = min(finite_history, default=None)
    curve: list[float | None] = []
    failures = []
    started = time.perf_counter()
    parameter = Hyperparameter(task.space)
    for step in range(5):
        try:
            params = optimizer.recommendNextParameters(task.space, rollout_history, [])
            outcome = task.evaluate(params, rollout_seed, step)
            row = parameter.convertToFlatValues(params)
        except Exception as exc:  # Research collection must retain optimizer failures.
            outcome = {
                "status": "fail",
                "loss": None,
                "error": f"optimizer_exception:{type(exc).__name__}:{exc}",
            }
            row = {
                descriptor.name: ""
                for descriptor in parameter.getFlatParameters()
            }
        row.update({"trial": len(rollout_history), "status": outcome["status"], "loss": outcome.get("loss")})
        rollout_history.append(row)
        loss = outcome.get("loss")
        if outcome["status"] == "ok" and loss is not None and np.isfinite(loss):
            incumbent = float(loss) if incumbent is None else min(incumbent, float(loss))
        else:
            failures.append({"step": step, "status": outcome["status"], "error": outcome.get("error")})
        curve.append(incumbent)
    return curve, failures, {"seconds": time.perf_counter() - started}


def _add_rollout(
    store: ResearchStore,
    task: SyntheticTask,
    state_id: str,
    lineage: str,
    history: list[dict],
    action: ATPE2Action,
    rollout_seed: int,
    repeat_index: int,
    acquisition_role: str = "legacy_joint",
    varied_groups: tuple[str, ...] = (),
) -> None:
    configuration = asdict(action)
    if store.has_evaluation(state_id, configuration, rollout_seed, repeat_index):
        return
    curve, failures, timing = rollout_action(task, history, action, rollout_seed)
    store.add_evaluation(
        {
            "state_id": state_id,
            "task_id": task.task_id,
            "generator_version": GENERATOR_VERSION,
            "history_lineage": lineage,
            "configuration": configuration,
            "rollout_seed": rollout_seed,
            "repeat_index": repeat_index,
            "incumbent_curve": curve,
            "failures": failures,
            "timing": timing,
            "dependency_versions": dependency_versions(),
            "acquisition_role": acquisition_role,
            "varied_groups": list(varied_groups),
        }
    )


def _third_repeat_candidates(records: list[dict], fraction: float = 0.20) -> set[str]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        curve = record["incumbent_curve"]
        final = curve[-1]
        grouped.setdefault(record["configuration_hash"], []).append(float(final) if final is not None else np.inf)
    count = max(1, int(math.ceil(fraction * len(grouped))))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    uncertainty = {key: float(np.std(values)) if np.all(np.isfinite(values)) else np.inf for key, values in grouped.items()}
    promising = sorted(grouped, key=lambda key: (means[key], key))[:count]
    uncertain = sorted(grouped, key=lambda key: (-uncertainty[key], key))[:count]
    return set(promising) | set(uncertain)


def _write_labels(store: ResearchStore, state_id: str) -> None:
    records = store.state_evaluations(state_id)
    labels = label_state(records)
    with store.transaction() as connection:
        for record in records:
            key = (record["configuration_hash"], int(record["rollout_seed"]), int(record["repeat_index"]))
            connection.execute(
                "UPDATE evaluations SET utility=? WHERE evaluation_id=?",
                (labels[key], record["evaluation_id"]),
            )


def generate_state(
    store: ResearchStore,
    task: SyntheticTask,
    method: str,
    trajectory_seed: int,
    checkpoint: int,
    acquisition_round: int,
    model_manifest: Path | None = None,
    fixed_constants: dict[str, Any] | None = None,
    action_ranges: dict[str, dict[str, Any]] | None = None,
) -> str:
    if acquisition_round not in {0, 1}:
        raise ValueError("Factorized collection retains exactly acquisition rounds {0,1}")
    base_lineage = content_hash(
        [task.specification(), GENERATOR_VERSION, method, trajectory_seed]
    )
    history = build_trajectory(
        task,
        method,
        trajectory_seed,
        checkpoint,
        model_manifest,
        fixed_constants,
        action_ranges,
    )
    extractor = (
        ATPE2ContextV4FeatureExtractor()
        if acquisition_round == 1
        else ATPE2FeatureExtractor()
    )
    context = {
        "__feature_schema_version": extractor.version,
        **extractor.extract(task.space, history, []),
    }
    acquisition_metadata: dict[str, Any] = {}
    if acquisition_round == 1:
        if model_manifest is None:
            raise ValueError("Structured round one requires a provisional manifest")
        if fixed_constants:
            raise ValueError("Factorized structured acquisition requires all eight public knobs")
        ensemble = ATPE2ModelEnsemble.load(model_manifest)
        if not ensemble.research_ready:
            raise ValueError("Structured acquisition requires a fitted provisional policy")
        provisional_controller = ATPE2Controller(trajectory_seed, ensemble=ensemble)
        provisional_action = provisional_controller.action_for(task.space, history, [])
        policy_context = provisional_controller.extractor.vector(task.space, history, [])
        records = structured_action_candidates(
            trajectory_seed,
            provisional_action,
            ensemble,
            policy_context,
            action_ranges,
        )
        provisional_sha256 = hashlib.sha256(model_manifest.read_bytes()).hexdigest()
        acquisition_metadata = acquisition_lineage(records, provisional_sha256)
        lineage = content_hash([base_lineage, acquisition_metadata])
    else:
        lineage = base_lineage
        actions = action_candidates(
            trajectory_seed, acquisition_round, fixed_constants, action_ranges
        )
        records = [
            AcquisitionCandidate(
                action,
                "neutral_anchor" if action == ATPE2Action.anchor() else "round_zero_joint_sobol",
                () if action == ATPE2Action.anchor() else ("search", "history", "locking"),
            )
            for action in actions
        ]
    state_id = store.upsert_state(
        task.task_id,
        method,
        lineage,
        checkpoint,
        history,
        context,
        acquisition_round,
        acquisition_metadata,
    )
    if len(fixed_constants or {}) == len(ATPE2Action.NAMES):
        # Empirical pruning is allowed to remove every knob. In that degenerate
        # but valid outcome the controller is exactly the fixed TPE anchor, so
        # repeated Sobol coordinates do not represent distinct actions.
        records = [records[-1]]
    paired_seeds = [trajectory_seed * 1000 + 101, trajectory_seed * 1000 + 202]
    for record in records:
        for repeat_index, rollout_seed in enumerate(paired_seeds):
            _add_rollout(
                store,
                task,
                state_id,
                lineage,
                history,
                record.action,
                rollout_seed,
                repeat_index,
                record.role,
                record.varied_groups,
            )
    # Derive the adaptive third-repeat set only from the two mandatory paired
    # rollouts. Otherwise resuming a completed state could let already-added
    # third repeats change the uncertainty ordering and slowly expand the set.
    initial = [
        record
        for record in store.state_evaluations(state_id)
        if int(record["repeat_index"]) < 2
    ]
    third_hashes = _third_repeat_candidates(initial)
    by_hash = {content_hash(asdict(record.action)): record for record in records}
    for configuration_hash in third_hashes:
        record = by_hash[configuration_hash]
        _add_rollout(
            store,
            task,
            state_id,
            lineage,
            history,
            record.action,
            trajectory_seed * 1000 + 303,
            2,
            record.role,
            record.varied_groups,
        )
    _write_labels(store, state_id)
    store.mark_state_complete(state_id)
    return state_id


def canonical_action(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def _generate_job(
    database: str,
    task: SyntheticTask,
    method: str,
    trajectory_seed: int,
    checkpoint: int,
    acquisition_round: int,
    model_manifest: str | None,
    fixed_constants: dict[str, Any],
    action_ranges: dict[str, dict[str, Any]],
) -> str:
    with ResearchStore(database) as store:
        return generate_state(
            store,
            task,
            method,
            trajectory_seed,
            checkpoint,
            acquisition_round,
            Path(model_manifest) if model_manifest else None,
            fixed_constants,
            action_ranges,
        )


def generate(
    database: Path,
    seed: int,
    checkpoints: list[int],
    max_tasks: int | None = None,
    acquisition_round: int = 0,
    model_manifest: Path | None = None,
    action_schema: dict[str, Any] | None = None,
    instances_per_family: int = 4,
    workers: int = 1,
    task_count: int | None = 192,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    if acquisition_round > 0:
        if model_manifest is None:
            raise ValueError("Dataset aggregation requires a fitted model manifest")
        manifest = json.loads(model_manifest.read_text())
        training_task_ids = set(manifest.get("task_splits", {}).get("train", []))
        with ResearchStore(database) as store:
            tasks = [
                SyntheticTask.from_specification(specification)
                for specification in store.task_specifications(training_task_ids)
            ]
        if not tasks or {task.task_id for task in tasks} != training_task_ids:
            raise ValueError("Dataset aggregation could not resolve every training task")
    else:
        tasks = (
            factorized_training_tasks(seed, task_count)
            if task_count is not None
            else synthetic_tasks(seed, instances_per_family)
        )
    tasks = tasks[:max_tasks]
    with ResearchStore(database) as store:
        schema = (action_schema or {}).get("backward_knockout", action_schema or {})
        fixed_constants = schema.get("fixed_constants", {})
        action_ranges = schema.get("ranges", {})
        store.set_metadata("generator_version", GENERATOR_VERSION)
        store.set_metadata("root_seed", seed)
        store.set_metadata(
            f"collection_round_{acquisition_round}",
            {
                "seed": seed,
                "checkpoints": list(checkpoints),
                "instances_per_family": instances_per_family,
                "task_count": len(tasks),
                "task_family_counts": {
                    family: sum(task.family == family for task in tasks)
                    for family in sorted({task.family for task in tasks})
                },
                "acquisition_mix": (
                    {"fixed_anchor": 1}
                    if len(fixed_constants) == len(ATPE2Action.NAMES)
                    else (
                        {
                            "neutral_anchor": 1,
                            "provisional_policy": 1,
                            "group_isolated": 32,
                            "two_group": 18,
                            "full_action_regime": 8,
                            "highest_disagreement": 5,
                        }
                        if acquisition_round == 1
                        else {"space_filling": 64, "anchor": 1}
                    )
                ),
                "acquisition_implementation_version": (
                    ACQUISITION_IMPLEMENTATION_VERSION
                    if acquisition_round == 1
                    else "atpe2-round-zero-joint-sobol-v1"
                ),
                "model_manifest_sha256": (
                    content_hash(json.loads(model_manifest.read_text()))
                    if model_manifest
                    else None
                ),
                "action_schema_sha256": (
                    content_hash(action_schema) if action_schema else None
                ),
            },
        )
        if action_schema:
            store.set_metadata("action_schema", action_schema)
        for task in tasks:
            store.upsert_task(task.task_id, task.family, task.specification(), GENERATOR_VERSION)
    methods = ("deployed_atpe2",) if acquisition_round > 0 else TRAJECTORY_METHODS
    jobs = [
        (
            str(database),
            task,
            method,
            seed + method_index,
            checkpoint,
            acquisition_round,
            str(model_manifest) if model_manifest else None,
            fixed_constants,
            action_ranges,
        )
        for task in tasks
        for method_index, method in enumerate(methods)
        for checkpoint in checkpoints
    ]
    if workers == 1:
        for job in jobs:
            _generate_job(*job)
    else:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            pool.starmap(_generate_job, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--acquisition-round", type=int, default=0)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--action-schema", type=Path)
    parser.add_argument("--instances-per-family", type=int, default=4)
    parser.add_argument(
        "--task-count",
        type=int,
        default=192,
        help="factorized protocol task count; use 0 for legacy equal-per-family generation",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.acquisition_round > 0 and args.model_manifest is None:
        parser.error("--model-manifest is required for dataset aggregation rounds")
    action_schema = json.loads(args.action_schema.read_text()) if args.action_schema else None
    generate(
        args.database,
        args.seed,
        args.checkpoints,
        args.max_tasks,
        args.acquisition_round,
        args.model_manifest,
        action_schema,
        args.instances_per_family,
        args.workers,
        None if args.task_count == 0 else args.task_count,
    )


if __name__ == "__main__":
    main()
