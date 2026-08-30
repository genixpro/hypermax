"""Training-only classic-ATPE mode labels for factorized heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hypermax.algorithms.atpe2.controller import ATPE2Action
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.algorithms.atpe_optimizer import ATPEOptimizer

from .continuation import _run_steps
from .database import ResearchStore
from .labels import label_curves
from .synthetic import SyntheticTask


TEACHER_VERSION = "atpe2-classic-teacher-v1"


def _search_regime(parameters: dict[str, Any]) -> str:
    gamma = float(parameters["gamma"])
    candidates = int(parameters["nEICandidates"])
    if gamma <= 0.5 and candidates >= 24:
        return "exploit"
    if gamma >= 1.0 or candidates <= 16:
        return "explore"
    return "balanced"


def _paired_utility(records: list[dict[str, Any]]) -> dict[str, float]:
    labeled = label_curves(records, "curve", 5)
    values: dict[str, list[float]] = {"classic": [], "anchor": []}
    for record in records:
        key = (
            record["configuration_hash"],
            record["rollout_seed"],
            record["repeat_index"],
        )
        values[record["name"]].append(labeled[key])
    return {name: float(np.mean(scores)) for name, scores in values.items()}


def collect_teacher_labels(
    database: Path, manifest_path: Path, seed: int
) -> int:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("component_gates", {}).get("passed", False):
        raise ValueError("Classic teacher work begins only after factorization is validated")
    training_tasks = set(manifest.get("task_splits", {}).get("train", []))
    ensemble = ATPE2ModelEnsemble.load(manifest_path)
    with ResearchStore(database) as store:
        existing = {row["state_id"] for row in store.teacher_rows()}
        states = [
            state
            for state in store.state_records(training_tasks, acquisition_round=1)
            if state["trajectory_method"] != "tpe_continuation"
            and state["state_id"] not in existing
        ]
        for state in states:
            task = SyntheticTask.from_specification(state["task_specification"])
            records = []
            recorded_modes = None
            state_seed = seed + int(state["state_id"][:8], 16)
            for repeat_index, rollout_seed in enumerate(
                (state_seed * 1000 + 606, state_seed * 1000 + 707)
            ):
                classic = ATPEOptimizer(seed=rollout_seed)
                classic_history, first_curve, first_failures = _run_steps(
                    task, state["history"], classic, rollout_seed, range(1)
                )
                if recorded_modes is None:
                    parameters = dict(classic.lastATPEParameters)
                    recorded_modes = {
                        "search_regime": _search_regime(parameters),
                        "filtering_mode": parameters["resultFilteringMode"],
                        "locking_mode": (
                            f"{parameters['secondaryProbabilityMode']}:"
                            f"{parameters['secondaryLockingMode']}"
                        ),
                        "classic_parameters": parameters,
                    }
                _, remaining_curve, remaining_failures = _run_steps(
                    task, classic_history, classic, rollout_seed, range(1, 5)
                )
                anchor = ATPE2Optimizer(seed=rollout_seed, ensemble=ensemble)
                anchor.controller.action_for = (
                    lambda space, results, current: ATPE2Action.anchor()
                )
                _, anchor_curve, anchor_failures = _run_steps(
                    task, state["history"], anchor, rollout_seed, range(5)
                )
                records.extend(
                    [
                        {
                            "name": "classic",
                            "configuration_hash": "classic",
                            "rollout_seed": rollout_seed,
                            "repeat_index": repeat_index,
                            "curve": first_curve + remaining_curve,
                            "failures": first_failures + remaining_failures,
                        },
                        {
                            "name": "anchor",
                            "configuration_hash": "anchor",
                            "rollout_seed": rollout_seed,
                            "repeat_index": repeat_index,
                            "curve": anchor_curve,
                            "failures": anchor_failures,
                        },
                    ]
                )
            utility = _paired_utility(records)
            store.upsert_teacher_label(
                {
                    "state_id": state["state_id"],
                    **recorded_modes,
                    "classic_utility": utility["classic"],
                    "anchor_utility": utility["anchor"],
                    "eligible": utility["classic"] >= utility["anchor"],
                    "metadata": {
                        "version": TEACHER_VERSION,
                        "seed": state_seed,
                        "runtime_executes_classic_atpe": False,
                        "manifest_sha256": hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                    },
                }
            )
    return len(states)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {"teacher_states": collect_teacher_labels(args.database, args.manifest, args.seed)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
