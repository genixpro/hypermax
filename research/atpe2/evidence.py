"""Statistical gates, ablation declarations, and evidence-based knob pruning."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from hypermax.algorithms.atpe2.controller import ATPE2Action
from hypermax.algorithms.atpe2.controller_schema import ACTION_DOMAINS
from hypermax.algorithms.atpe2.features import ATPE2FeatureExtractor
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from .database import ResearchStore
from .train import row_vector


ABLATIONS = {
    "filtering": ("history_keep_fraction", "history_recency_bias", "history_quality_bias"),
    "locking": ("secondary_fraction", "secondary_lock_probability", "secondary_top_fraction"),
    "adaptive_gamma_candidates": ("gamma", "n_ei_candidates"),
    "uncertainty_penalty": ("uncertainty_penalty",),
    "full_vs_pruned_schema": tuple(ATPE2Action.NAMES),
    "legacy_vs_robust_features": ("feature_schema",),
}


ACTION_RANGES = {
    name: {
        "minimum": domain["minimum"],
        "maximum": domain["maximum"],
        "type": domain["type"],
        "scale": domain["scale"],
    }
    for name, domain in ACTION_DOMAINS.items()
}


def structurally_inactive_locking_knobs(removed: set[str]) -> list[str]:
    """Return controls made behaviorally inert by a neutral locking gate."""
    if "secondary_fraction" in removed:
        candidates = ("secondary_lock_probability", "secondary_top_fraction")
    elif "secondary_lock_probability" in removed:
        candidates = ("secondary_fraction", "secondary_top_fraction")
    else:
        candidates = ()
    return [name for name in candidates if name not in removed]


def boundary_diagnostics(
    rows: list[dict[str, Any]], band_fraction: float = 0.05
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Detect task-consistent accumulation of top actions at initial bounds."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["task_id"],
                row["state_id"],
                json.dumps(row["configuration"], sort_keys=True, separators=(",", ":")),
            )
        ].append(row)
    aggregated = [
        {
            "task_id": key[0],
            "configuration": group[0]["configuration"],
            "utility": float(np.mean([row["utility"] for row in group])),
        }
        for key, group in grouped.items()
    ]
    tasks = sorted({row["task_id"] for row in aggregated})
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        rows_by_task[row["task_id"]].append(row)
    diagnostics = {}
    expanded = {name: dict(domain) for name, domain in ACTION_RANGES.items()}
    for name, domain in ACTION_RANGES.items():
        minimum, maximum = float(domain["minimum"]), float(domain["maximum"])
        span = maximum - minimum
        task_sides: dict[str, tuple[float, float]] = {}
        for task in tasks:
            task_rows = rows_by_task[task]
            utilities = np.asarray([row["utility"] for row in task_rows], dtype=float)
            threshold = float(np.quantile(utilities, 0.80))
            top = [row for row in task_rows if row["utility"] >= threshold]
            normalized = np.asarray(
                [
                    (float(row["configuration"][name]) - minimum) / span
                    for row in top
                ]
            )
            task_sides[task] = (
                float(np.mean(normalized <= band_fraction)),
                float(np.mean(normalized >= 1.0 - band_fraction)),
            )
        lower_consistency = float(
            np.mean([sides[0] >= 0.25 for sides in task_sides.values()])
        )
        upper_consistency = float(
            np.mean([sides[1] >= 0.25 for sides in task_sides.values()])
        )
        expand_lower = lower_consistency >= 0.60
        expand_upper = upper_consistency >= 0.60
        if expand_lower:
            if name == "n_ei_candidates":
                expanded[name]["minimum"] = max(4, int(minimum // 2))
            elif name in {"gamma", "secondary_top_fraction"}:
                expanded[name]["minimum"] = max(0.005, minimum / 2.0)
            elif name == "history_keep_fraction":
                expanded[name]["minimum"] = max(0.05, minimum - 0.15)
            elif name in {"history_recency_bias", "history_quality_bias"}:
                expanded[name]["minimum"] = minimum - 1.0
        if expand_upper:
            if name == "n_ei_candidates":
                expanded[name]["maximum"] = min(256, int(maximum * 2))
            elif name == "gamma":
                expanded[name]["maximum"] = maximum * 1.5
            elif name in {"history_recency_bias", "history_quality_bias"}:
                expanded[name]["maximum"] = maximum + 1.0
            elif name == "secondary_fraction":
                expanded[name]["maximum"] = min(1.0, maximum + 0.15)
        diagnostics[name] = {
            "band_fraction": band_fraction,
            "task_fraction_with_lower_accumulation": lower_consistency,
            "task_fraction_with_upper_accumulation": upper_consistency,
            "expanded_lower": expanded[name]["minimum"] != domain["minimum"],
            "expanded_upper": expanded[name]["maximum"] != domain["maximum"],
            "resulting_range": expanded[name],
        }
    return diagnostics, expanded


def paired_bootstrap_interval(
    differences: Iterable[float], seed: int, confidence: float = 0.95, repeats: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap needs finite paired differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    samples = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def external_study_statistics(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Analyze frozen benchmark rows without removing tasks or failures.

    Each row needs optimizer, task, family, seed, budget and normalized_regret.
    """
    required = {"optimizer", "task", "family", "seed", "budget", "normalized_regret"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"Every external row needs {sorted(required)}")
    optimizers = sorted({row["optimizer"] for row in rows})
    seed_blocks: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = float(row["normalized_regret"])
        if not math.isfinite(value):
            value = 1.0
        seed_blocks[(row["task"], row["seed"], row["budget"])][row["optimizer"]] = value
    complete_seed_blocks = [
        (key, block)
        for key, block in seed_blocks.items()
        if set(block) == set(optimizers)
    ]
    if not complete_seed_blocks:
        raise ValueError("No complete task/seed/budget blocks")
    # Seeds are repeated measurements of a task. Aggregate them before the
    # omnibus and paired tests so a benchmark with more seeds cannot create
    # pseudo-replication.
    task_values: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (task, _seed_name, budget), block in complete_seed_blocks:
        for name, value in block.items():
            task_values[(task, budget)][name].append(value)
    complete = [
        {name: float(np.median(values[name])) for name in optimizers}
        for values in task_values.values()
        if set(values) == set(optimizers)
    ]
    if len(complete) < 2:
        raise ValueError("External statistical tests need at least two complete tasks")
    samples = [np.asarray([block[name] for block in complete]) for name in optimizers]
    if all(np.allclose(samples[0], sample) for sample in samples[1:]):
        friedman_statistic, friedman_p = 0.0, 1.0
    else:
        friedman = stats.friedmanchisquare(*samples)
        friedman_statistic, friedman_p = float(friedman.statistic), float(friedman.pvalue)
    best_name = min(
        optimizers, key=lambda name: np.median(samples[optimizers.index(name)])
    )
    raw_p = {}
    for name in optimizers:
        if name == best_name:
            continue
        left = samples[optimizers.index(best_name)]
        right = samples[optimizers.index(name)]
        raw_p[name] = (
            1.0
            if np.allclose(left, right)
            else float(stats.wilcoxon(left, right).pvalue)
        )
    return {
        "optimizers": {
            name: {
                "median": float(np.median(samples[optimizers.index(name)])),
                "interquartile_mean": float(
                    stats.trim_mean(samples[optimizers.index(name)], 0.25)
                ),
            }
            for name in optimizers
        },
        "statistical_unit": "task (median across the five prescribed seeds)",
        "complete_tasks": len(complete),
        "friedman": {"statistic": friedman_statistic, "p_value": friedman_p},
        "paired_holm_against_best": {"best": best_name, "adjusted_p_values": holm_correction(raw_p)},
    }


def _predict_ensemble(ensemble: ATPE2ModelEnsemble, matrix: np.ndarray) -> np.ndarray:
    means, uncertainty = ensemble.predict(matrix)
    return means - ensemble.uncertainty_penalty * uncertainty


def grouped_permutation_importance(
    ensemble: ATPE2ModelEnsemble, matrix: np.ndarray, labels: np.ndarray, groups: list[str], seed: int
) -> dict[str, float]:
    baseline = np.mean((_predict_ensemble(ensemble, matrix) - labels) ** 2)
    rng = np.random.default_rng(seed)
    names = list(ATPE2FeatureExtractor.FEATURE_NAMES) + list(ATPE2Action.NAMES)
    output = {}
    groups_array = np.asarray(groups)
    grouped_positions: dict[str, list[int]] = defaultdict(list)
    for position, group in enumerate(groups_array):
        grouped_positions[str(group)].append(position)
    positions_by_group = {
        group: np.asarray(positions, dtype=int)
        for group, positions in grouped_positions.items()
    }
    for column, name in enumerate(names):
        permuted = matrix.copy()
        # Shuffle between whole task groups by mapping each group's column values
        # to another group, preserving within-task dependence and repeated states.
        unique = np.asarray(sorted(positions_by_group))
        mapped = rng.permutation(unique)
        for source, destination in zip(unique, mapped):
            source_indices = positions_by_group[source]
            destination_values = matrix[positions_by_group[destination], column]
            permuted[source_indices, column] = np.resize(destination_values, len(source_indices))
        output[name] = float(np.mean((_predict_ensemble(ensemble, permuted) - labels) ** 2) - baseline)
    return output


def partial_dependence_and_interactions(
    ensemble: ATPE2ModelEnsemble, matrix: np.ndarray
) -> tuple[dict[str, list], dict[str, float]]:
    # Partial-dependence estimates do not improve materially by repeating every
    # rollout repeat.  A deterministic, evenly spaced cap keeps the analysis
    # tractable while retaining coverage across the held-out rows.
    if len(matrix) > 512:
        matrix = matrix[np.linspace(0, len(matrix) - 1, 512, dtype=int)]
    offset = len(ATPE2FeatureExtractor.FEATURE_NAMES)
    dependence = {}
    centered = {}
    grids = {}
    for action_index, name in enumerate(ATPE2Action.NAMES):
        column = offset + action_index
        grid = np.unique(np.quantile(matrix[:, column], np.linspace(0.05, 0.95, 7)))
        batches = []
        for point in grid:
            modified = matrix.copy()
            modified[:, column] = point
            batches.append(modified)
        predictions = _predict_ensemble(ensemble, np.vstack(batches)).reshape(
            len(grid), len(matrix)
        )
        values = np.mean(predictions, axis=1).tolist()
        grids[name] = grid
        centered[name] = np.asarray(values) - np.mean(values)
        dependence[name] = {"grid": grid.tolist(), "mean_prediction": values}
    interactions = {}
    for left in range(len(ATPE2Action.NAMES)):
        for right in range(left + 1, len(ATPE2Action.NAMES)):
            left_name, right_name = ATPE2Action.NAMES[left], ATPE2Action.NAMES[right]
            batches = []
            for left_index, left_value in enumerate(grids[left_name]):
                for right_index, right_value in enumerate(grids[right_name]):
                    modified = matrix.copy()
                    modified[:, offset + left] = left_value
                    modified[:, offset + right] = right_value
                    batches.append(modified)
            joint_predictions = _predict_ensemble(
                ensemble, np.vstack(batches)
            ).reshape(len(batches), len(matrix))
            joint_means = np.mean(joint_predictions, axis=1)
            residuals = []
            pair_index = 0
            for left_index in range(len(grids[left_name])):
                for right_index in range(len(grids[right_name])):
                    residuals.append(
                        joint_means[pair_index]
                        - centered[left_name][left_index]
                        - centered[right_name][right_index]
                    )
                    pair_index += 1
            interactions[f"{left_name}:{right_name}"] = float(np.std(residuals))
    return dependence, interactions


def _controller_regrets(
    ensemble: ATPE2ModelEnsemble,
    rows: list[dict[str, Any]],
    removed: set[str],
    base_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    by_state: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        configuration_key = json.dumps(
            row["configuration"], sort_keys=True, separators=(",", ":")
        )
        by_state[row["state_id"]][configuration_key].append(index)
    if base_matrix is None:
        matrix = np.asarray([row_vector(row) for row in rows], dtype=float)
    else:
        matrix = np.array(base_matrix, copy=True)
    offset = len(ATPE2FeatureExtractor.FEATURE_NAMES)
    anchor = ATPE2Action.anchor()
    for name in removed:
        matrix[:, offset + ATPE2Action.NAMES.index(name)] = float(getattr(anchor, name))
    predictions = _predict_ensemble(ensemble, matrix)
    regrets, tasks = [], []
    for configurations in by_state.values():
        groups = list(configurations.values())
        predicted = [float(np.mean(predictions[indices])) for indices in groups]
        utilities = [
            float(np.mean([rows[index]["utility"] for index in indices]))
            for indices in groups
        ]
        selected_group = int(np.argmax(predicted))
        regrets.append(float(max(utilities) - utilities[selected_group]))
        tasks.append(rows[groups[selected_group][0]]["task_id"])
    return np.asarray(regrets), tasks


def backward_knockout(
    ensemble: ATPE2ModelEnsemble, rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    removed: set[str] = set()
    evidence = []
    base_matrix = np.asarray([row_vector(row) for row in rows], dtype=float)
    while True:
        full_regret, tasks = _controller_regrets(
            ensemble, rows, removed, base_matrix
        )
        candidates = []
        for knob in ATPE2Action.NAMES:
            if knob in removed:
                continue
            reduced, _ = _controller_regrets(
                ensemble, rows, removed | {knob}, base_matrix
            )
            delta = reduced - full_regret
            rng = np.random.default_rng(seed + len(removed))
            task_positions: dict[str, list[int]] = defaultdict(list)
            for position, task in enumerate(tasks):
                task_positions[task].append(position)
            task_names = np.asarray(sorted(task_positions))
            task_sums = np.asarray(
                [
                    float(np.sum(delta[np.asarray(task_positions[task], dtype=int)]))
                    for task in task_names
                ]
            )
            task_counts = np.asarray(
                [len(task_positions[task]) for task in task_names]
            )
            sampled = rng.integers(
                0, len(task_names), size=(2000, len(task_names))
            )
            boot = np.sum(task_sums[sampled], axis=1) / np.sum(
                task_counts[sampled], axis=1
            )
            standard_error = float(np.std(boot, ddof=1))
            median_change = float(np.median(reduced) - np.median(full_regret))
            worst_change = float(np.quantile(reduced, 0.75) - np.quantile(full_regret, 0.75))
            within_one_se = float(np.mean(delta)) <= standard_error
            removable = within_one_se and median_change < 0.01 and worst_change < 0.02
            candidates.append(
                {
                    "knob": knob,
                    "mean_delta": float(np.mean(delta)),
                    "standard_error": standard_error,
                    "median_regret_change": median_change,
                    "worst_quartile_regret_change": worst_change,
                    "within_one_standard_error": within_one_se,
                    "removable": removable,
                }
            )
        removable = [candidate for candidate in candidates if candidate["removable"]]
        evidence.append(candidates)
        if not removable:
            break
        chosen = min(removable, key=lambda candidate: (candidate["mean_delta"], candidate["knob"]))
        removed.add(chosen["knob"])
    structurally_inactive = structurally_inactive_locking_knobs(removed)
    if structurally_inactive:
        evidence.append(
            [
                {
                    "knob": name,
                    "mean_delta": 0.0,
                    "standard_error": 0.0,
                    "median_regret_change": 0.0,
                    "worst_quartile_regret_change": 0.0,
                    "within_one_standard_error": True,
                    "removable": True,
                    "basis": "exact structural equivalence: neutral locking gate disables this control",
                }
                for name in structurally_inactive
            ]
        )
        removed.update(structurally_inactive)
    anchor = ATPE2Action.anchor()
    return {
        "retained_knobs": [name for name in ATPE2Action.NAMES if name not in removed],
        "fixed_constants": {name: getattr(anchor, name) for name in sorted(removed)},
        "ranges": {name: ACTION_RANGES[name] for name in ATPE2Action.NAMES if name not in removed},
        "iterations": evidence,
    }


def pruning_report(database: Path, manifest: Path, output: Path, seed: int) -> dict[str, Any]:
    ensemble = ATPE2ModelEnsemble.load(manifest)
    if not ensemble.research_ready:
        raise ValueError("Knob evidence requires a fitted, schema-compatible candidate")
    if tuple(ensemble.action_names) != tuple(ATPE2Action.NAMES):
        raise ValueError("Knob pruning must start from the complete eight-knob pilot schema")
    held_out_tasks = set(ensemble.manifest.get("task_splits", {}).get("validation", []))
    with ResearchStore(database) as store:
        rows = store.training_rows(held_out_tasks)
    if not rows:
        raise ValueError("No grouped validation rows match the model manifest")
    matrix = np.asarray([row_vector(row) for row in rows], dtype=float)
    labels = np.asarray([row["utility"] for row in rows])
    permutation = grouped_permutation_importance(ensemble, matrix, labels, [row["task_id"] for row in rows], seed)
    dependence, interactions = partial_dependence_and_interactions(ensemble, matrix)
    knockout = backward_knockout(ensemble, rows, seed)
    boundaries, expanded_ranges = boundary_diagnostics(rows)
    knockout["ranges"] = {
        name: expanded_ranges[name] for name in knockout["retained_knobs"]
    }
    report = {
        "seed": seed,
        "removal_thresholds": {
            "within_one_standard_error": True,
            "maximum_median_regret_change": 0.01,
            "maximum_worst_quartile_regret_change": 0.02,
        },
        "grouped_permutation_importance": permutation,
        "partial_dependence": dependence,
        "pairwise_interactions": interactions,
        "boundary_diagnostics": boundaries,
        "backward_knockout": knockout,
        "required_ablations": ABLATIONS,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(pruning_report(args.database, args.manifest, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
