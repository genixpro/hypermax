"""Training-only five-plus-ten-step continuation acquisition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from hypermax.algorithms.atpe2.controller import ATPE2Action, ATPE2Controller
from hypermax.algorithms.atpe2.features import ATPE2ContextV4FeatureExtractor
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.hyperparameter import Hyperparameter

from .acquisition import acquisition_lineage, structured_action_candidates
from .database import ResearchStore, content_hash
from .generate import build_trajectory
from .labels import label_curves
from .synthetic import SyntheticTask


CONTINUATION_VERSION = "atpe2-continuation-v1"
CONTINUATION_CHECKPOINTS = (25, 50)
CONTINUATION_TASKS = 24
CONTINUATION_LONG_WEIGHTS = (0.0, 0.25, 0.5)


def _selected_tasks(
    store: ResearchStore, manifest: dict[str, Any], seed: int
) -> list[SyntheticTask]:
    allowed = set(manifest.get("task_splits", {}).get("train", []))
    tasks = [
        SyntheticTask.from_specification(specification)
        for specification in store.task_specifications(allowed)
    ]
    if {task.task_id for task in tasks} != allowed:
        raise ValueError("Continuation collection could not resolve training-only tasks")
    by_family: dict[str, list[SyntheticTask]] = {}
    for task in tasks:
        by_family.setdefault(task.family, []).append(task)
    rng = np.random.default_rng(seed)
    for values in by_family.values():
        values.sort(key=lambda task: task.task_id)
        rng.shuffle(values)
    selected = []
    depth = 0
    while len(selected) < min(CONTINUATION_TASKS, len(tasks)):
        progressed = False
        for family in sorted(by_family):
            if depth < len(by_family[family]):
                selected.append(by_family[family][depth])
                progressed = True
                if len(selected) == CONTINUATION_TASKS:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != CONTINUATION_TASKS:
        raise ValueError("Continuation protocol requires 24 training-only tasks")
    return selected


def _run_steps(
    task: SyntheticTask,
    history: list[dict[str, Any]],
    optimizer: Any,
    rollout_seed: int,
    steps: range,
) -> tuple[list[dict[str, Any]], list[float | None], list[dict[str, Any]]]:
    rollout_history = [dict(row) for row in history]
    finite = [
        float(row["loss"])
        for row in rollout_history
        if row.get("status", "ok") == "ok"
        and row.get("loss") is not None
        and np.isfinite(row["loss"])
    ]
    incumbent = min(finite, default=None)
    curve: list[float | None] = []
    failures = []
    parameter = Hyperparameter(task.space)
    for step in steps:
        try:
            params = optimizer.recommendNextParameters(task.space, rollout_history, [])
            outcome = task.evaluate(params, rollout_seed, step)
            row = parameter.convertToFlatValues(params)
        except Exception as exc:
            outcome = {
                "status": "fail",
                "loss": None,
                "error": f"optimizer_exception:{type(exc).__name__}:{exc}",
            }
            row = {descriptor.name: "" for descriptor in parameter.getFlatParameters()}
        row.update(
            {
                "trial": len(rollout_history),
                "status": outcome["status"],
                "loss": outcome.get("loss"),
            }
        )
        rollout_history.append(row)
        loss = outcome.get("loss")
        if outcome["status"] == "ok" and loss is not None and np.isfinite(loss):
            incumbent = float(loss) if incumbent is None else min(incumbent, float(loss))
        else:
            failures.append(
                {"step": step, "status": outcome["status"], "error": outcome.get("error")}
            )
        curve.append(incumbent)
    return rollout_history, curve, failures


def _collect_state(
    database: str,
    task: SyntheticTask,
    checkpoint: int,
    seed: int,
    manifest_path: str,
) -> str:
    manifest_path_object = Path(manifest_path)
    ensemble = ATPE2ModelEnsemble.load(manifest_path_object)
    history_seed = seed + int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16)
    history = build_trajectory(task, "tpe", history_seed, checkpoint)
    controller = ATPE2Controller(history_seed, ensemble=ensemble)
    provisional = controller.action_for(task.space, history, [])
    policy_context = controller.extractor.vector(task.space, history, [])
    candidates = structured_action_candidates(
        history_seed, provisional, ensemble, policy_context
    )
    manifest_sha = hashlib.sha256(manifest_path_object.read_bytes()).hexdigest()
    metadata = {
        **acquisition_lineage(candidates, manifest_sha),
        "continuation_version": CONTINUATION_VERSION,
        "training_only": True,
    }
    lineage = content_hash(
        [task.specification(), CONTINUATION_VERSION, history_seed, checkpoint, metadata]
    )
    context_extractor = ATPE2ContextV4FeatureExtractor()
    context = {
        "__feature_schema_version": context_extractor.version,
        **context_extractor.extract(task.space, history, []),
    }
    with ResearchStore(database) as store:
        state_id = store.upsert_state(
            task.task_id,
            "tpe_continuation",
            lineage,
            checkpoint,
            history,
            context,
            1,
            metadata,
        )
        for candidate in candidates:
            for repeat_index, rollout_seed in enumerate(
                (history_seed * 1000 + 404, history_seed * 1000 + 505)
            ):
                if store.has_continuation(
                    state_id, asdict(candidate.action), rollout_seed, repeat_index
                ):
                    continue
                forced = ATPE2Optimizer(seed=rollout_seed, ensemble=ensemble)
                forced.controller.action_for = (
                    lambda space, results, current, action=candidate.action: action
                )
                forced_history, forced_curve, forced_failures = _run_steps(
                    task, history, forced, rollout_seed, range(5)
                )
                released = ATPE2Optimizer(seed=rollout_seed + 1, ensemble=ensemble)
                _, continuation_curve, continuation_failures = _run_steps(
                    task, forced_history, released, rollout_seed, range(5, 15)
                )
                store.add_continuation(
                    {
                        "state_id": state_id,
                        "configuration": asdict(candidate.action),
                        "acquisition_role": candidate.role,
                        "varied_groups": candidate.varied_groups,
                        "rollout_seed": rollout_seed,
                        "repeat_index": repeat_index,
                        "forced_curve": forced_curve,
                        "continuation_curve": continuation_curve,
                        "failures": forced_failures + continuation_failures,
                        "provisional_manifest_sha256": manifest_sha,
                    }
                )
        records = store.continuation_rows()
        records = [record for record in records if record["state_id"] == state_id]
        five_records = [
            {**record, "configuration_hash": record["configuration_hash"]}
            for record in records
        ]
        fifteen_records = [
            {
                **record,
                "configuration_hash": record["configuration_hash"],
                "full_curve": record["forced_curve"] + record["continuation_curve"],
            }
            for record in records
        ]
        utility_5 = label_curves(five_records, "forced_curve", 5)
        utility_15 = label_curves(fifteen_records, "full_curve", 15)
        for record in records:
            key = (
                record["configuration_hash"],
                int(record["rollout_seed"]),
                int(record["repeat_index"]),
            )
            store.set_continuation_utilities(
                record["continuation_id"], utility_5[key], utility_15[key]
            )
        store.mark_state_complete(state_id)
    return state_id


def collect_continuations(
    database: Path,
    manifest_path: Path,
    seed: int,
    workers: int = 4,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("component_gates", {}).get("passed", False):
        raise ValueError("Multi-horizon work begins only after factorized component gates pass")
    with ResearchStore(database) as store:
        tasks = _selected_tasks(store, manifest, seed)
    jobs = [
        (str(database), task, checkpoint, seed, str(manifest_path))
        for task in tasks
        for checkpoint in CONTINUATION_CHECKPOINTS
    ]
    if workers == 1:
        for job in jobs:
            _collect_state(*job)
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            pool.starmap(_collect_state, jobs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    collect_continuations(args.database, args.manifest, args.seed, args.workers)


if __name__ == "__main__":
    main()
