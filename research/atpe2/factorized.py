"""Derived targets and bounded experiment definitions for factorized ATPE2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from hypermax.algorithms.atpe2.controller import ACTION_GROUPS, ATPE2Action


TARGET_DEFINITIONS_VERSION = "atpe2-factorized-targets-v1"
FACTORIZED_MANIFEST_VERSION = "atpe2-factorized-manifest-v1"

GROUP_WEIGHT_GRID = {
    "equal": {"search": 1.0, "history": 1.0, "locking": 1.0},
    "search_heavy": {"search": 1.5, "history": 1.0, "locking": 1.0},
    "history_heavy": {"search": 1.0, "history": 1.5, "locking": 1.0},
    "locking_heavy": {"search": 1.0, "history": 1.0, "locking": 1.5},
}
RESIDUAL_WEIGHT_GRID = (0.0, 0.25, 0.5)
DOWNSIDE_WEIGHT_GRID = (0.0, 0.25, 0.5)
SUPPORT_WEIGHT_GRID = (0.0, 0.10, 0.25)
TREE_GRID = (8, 12, 16)
LEAF_GRID = tuple(
    {"num_leaves": leaves, "min_child_samples": children}
    for leaves in (7, 15, 31)
    for children in (5, 10, 20)
)

ABLATION_BRANCHES = {
    "structured_joint": {
        "policy": "joint",
        "residual": False,
        "downside": False,
        "support": False,
        "context_v4": False,
    },
    "factorized_no_residual": {
        "policy": "factorized",
        "residual": False,
        "downside": False,
        "support": False,
        "context_v4": False,
    },
    "factorized_residual": {
        "policy": "factorized",
        "residual": True,
        "downside": False,
        "support": False,
        "context_v4": False,
    },
    "residual_downside": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": False,
        "context_v4": False,
    },
    "downside_support": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": True,
        "context_v4": False,
    },
    "context_v4": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": True,
        "context_v4": True,
    },
    "multi_horizon": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": True,
        "context_v4": True,
        "multi_horizon": True,
    },
    "teacher": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": True,
        "context_v4": True,
        "teacher": True,
    },
    "multi_horizon_teacher": {
        "policy": "factorized",
        "residual": True,
        "downside": True,
        "support": True,
        "context_v4": True,
        "multi_horizon": True,
        "teacher": True,
    },
}


def action_key(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def group_key(configuration: dict[str, Any], group: str) -> tuple[float, ...]:
    return tuple(float(configuration[name]) for name in ACTION_GROUPS[group])


def _rollout_downside(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = float(np.mean(values))
    negative = np.maximum(0.0, center - np.asarray(values, dtype=float))
    return float(np.sqrt(np.mean(negative * negative)))


def derive_factorized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mandatory repeats and attach group/additive/downside targets."""
    mandatory = [row for row in rows if int(row.get("repeat_index", 0)) < 2]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mandatory:
        grouped[(row["state_id"], action_key(row["configuration"]))].append(row)

    states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (state_id, _), repeats in grouped.items():
        first = repeats[0]
        utilities = [float(row["utility"]) for row in repeats]
        failures = sum(len(row.get("failures", [])) for row in repeats)
        rollout_steps = sum(int(row.get("rollout_steps", 5)) for row in repeats)
        states[state_id].append(
            {
                **{
                    key: first[key]
                    for key in (
                        "state_id",
                        "task_id",
                        "family",
                        "trajectory_lineage",
                        "checkpoint",
                        "acquisition_round",
                        "context",
                        "history",
                        "space",
                        "configuration",
                        "acquisition_role",
                        "varied_groups",
                    )
                    if key in first
                },
                "utility": float(np.mean(utilities)),
                "rollout_variability": _rollout_downside(utilities),
                "failed_step_fraction": failures / float(max(1, rollout_steps)),
                "mandatory_repeat_count": len(repeats),
            }
        )

    output: list[dict[str, Any]] = []
    anchor_configuration = asdict(ATPE2Action.anchor())
    for state_id, state_rows in states.items():
        anchor = next(
            (
                row
                for row in state_rows
                if row["configuration"] == anchor_configuration
                or row.get("acquisition_role") == "neutral_anchor"
            ),
            None,
        )
        if anchor is None:
            raise ValueError(f"State {state_id} has no neutral anchor")
        anchor_utility = float(anchor["utility"])
        isolated: dict[str, dict[tuple[float, ...], float]] = {
            group: {group_key(anchor_configuration, group): 0.0}
            for group in ACTION_GROUPS
        }
        for row in state_rows:
            varied = tuple(row.get("varied_groups", ()))
            if len(varied) == 1 and varied[0] in ACTION_GROUPS:
                group = varied[0]
                isolated[group][group_key(row["configuration"], group)] = (
                    float(row["utility"]) - anchor_utility
                )

        for row in state_rows:
            advantages: dict[str, float | None] = {}
            for group in ACTION_GROUPS:
                advantages[group] = isolated[group].get(
                    group_key(row["configuration"], group)
                )
            matched = all(value is not None for value in advantages.values())
            additive = (
                anchor_utility
                + sum(float(value) for value in advantages.values() if value is not None)
                if matched
                else None
            )
            varied_count = len(tuple(row.get("varied_groups", ())))
            residual = (
                float(row["utility"]) - float(additive)
                if additive is not None and varied_count >= 2
                else None
            )
            harm = max(0.0, anchor_utility - float(row["utility"]))
            downside = float(
                np.clip(
                    harm
                    + 0.5 * float(row["rollout_variability"])
                    + float(row["failed_step_fraction"]),
                    0.0,
                    1.0,
                )
            )
            output.append(
                {
                    **row,
                    "anchor_utility": anchor_utility,
                    "group_advantages": advantages,
                    "additive_prediction": additive,
                    "residual_target": residual,
                    "harm_target": harm,
                    "downside_target": downside,
                    "catastrophic_label": int(
                        float(row["utility"]) <= anchor_utility - 0.15
                        or float(row["failed_step_fraction"]) > 0.0
                    ),
                }
            )

    round_counts = {
        round_index: sum(row["acquisition_round"] == round_index for row in output)
        for round_index in {row["acquisition_round"] for row in output}
    }
    for row in output:
        round_mass = 0.10 if row["acquisition_round"] == 1 else 0.90
        row["round_weight"] = round_mass / max(1, round_counts[row["acquisition_round"]])
    return output


def robust_support_artifact(
    matrix: np.ndarray,
    task_ids: Iterable[str],
    seed: int,
    size: int = 512,
    k: int = 5,
) -> dict[str, Any]:
    """Build deterministic task-stratified robustly normalized representatives."""
    matrix = np.asarray(matrix, dtype=float)
    task_ids = np.asarray(list(task_ids), dtype=object)
    if matrix.ndim != 2 or len(matrix) != len(task_ids) or len(matrix) == 0:
        raise ValueError("support artifact requires a non-empty aligned matrix")
    center = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    scale = q75 - q25
    mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    standard = np.std(matrix, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, mad)
    scale = np.where(scale > np.finfo(float).eps, scale, standard)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    normalized = (matrix - center) / scale
    by_task = {
        task: sorted(
            np.flatnonzero(task_ids == task).tolist(),
            key=lambda index: hashlib.sha256(
                normalized[index].tobytes() + str(seed).encode()
            ).hexdigest(),
        )
        for task in sorted(set(task_ids.tolist()))
    }
    selected: list[int] = []
    depth = 0
    target = min(size, len(matrix))
    while len(selected) < target:
        progressed = False
        for task in sorted(by_task):
            if depth < len(by_task[task]):
                selected.append(by_task[task][depth])
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        depth += 1
    representatives = normalized[np.asarray(selected, dtype=int)]
    # Calibrate raw distance into a monotone [0, 1] support cost using the
    # empirical representative distribution itself.
    pairwise = np.sqrt(
        np.mean(
            (representatives[:, None, :] - representatives[None, :, :]) ** 2,
            axis=2,
        )
    )
    local_k = min(k + 1, len(representatives))
    local = np.partition(pairwise, local_k - 1, axis=1)[:, 1:local_k]
    reference_distances = np.mean(local, axis=1) if local.size else np.zeros(1)
    quantiles = np.linspace(0.0, 1.0, 21)
    distance_knots = np.maximum.accumulate(
        np.quantile(reference_distances, quantiles)
    )
    return {
        "schema_version": "atpe2-support-v1",
        "requested_size": size,
        "representative_count": len(representatives),
        "k": k,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "representatives": representatives.tolist(),
        "calibration": {
            "distance": distance_knots.tolist(),
            "cost": quantiles.tolist(),
        },
    }


def normalize_component(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -5.0, 5.0)
