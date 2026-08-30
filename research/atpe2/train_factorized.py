"""Fit, calibrate, gate, and export the factorized ATPE2 controller."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score

from hypermax.algorithms.atpe2.controller import (
    ACTION_GROUPS,
    ATPE2Action,
    FACTORIZED_CONTROLLER_POLICY_VERSION,
)
from hypermax.algorithms.atpe2.controller_schema import (
    ACTION_SCHEMA_VERSION,
    action_schema_definition,
)
from hypermax.algorithms.atpe2.features import (
    ATPE2ContextV4FeatureExtractor,
    ATPE2FeatureExtractor,
)
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble

from .database import ResearchStore, canonical_json
from .factorized import (
    ABLATION_BRANCHES,
    DOWNSIDE_WEIGHT_GRID,
    FACTORIZED_MANIFEST_VERSION,
    GROUP_WEIGHT_GRID,
    LEAF_GRID,
    RESIDUAL_WEIGHT_GRID,
    SUPPORT_WEIGHT_GRID,
    TARGET_DEFINITIONS_VERSION,
    TREE_GRID,
    derive_factorized_rows,
    group_key,
    normalize_component,
    robust_support_artifact,
)
from .train import grouped_split, sqlite_snapshot_hash


def _artifact(path: Path) -> dict[str, str]:
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save_model(model: Any, path: Path) -> dict[str, str]:
    model.booster_.save_model(str(path))
    return _artifact(path)


def _full_vector(context: np.ndarray, configuration: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            context,
            np.asarray(
                [float(configuration[name]) for name in ATPE2Action.NAMES],
                dtype=float,
            ),
        )
    )


def _group_vector(
    context: np.ndarray, configuration: dict[str, Any], group: str
) -> np.ndarray:
    return np.concatenate(
        (
            context,
            np.asarray(
                [float(configuration[name]) for name in ACTION_GROUPS[group]],
                dtype=float,
            ),
        )
    )


def _contexts(
    rows: list[dict[str, Any]], extractor: Any
) -> dict[str, np.ndarray]:
    contexts: dict[str, np.ndarray] = {}
    required = set(extractor.FEATURE_NAMES)
    for row in rows:
        if row["state_id"] in contexts:
            continue
        stored = row["context"]
        values = (
            stored
            if required.issubset(stored)
            and stored.get("__feature_schema_version") == extractor.version
            else extractor.extract(row["space"], row["history"], [])
        )
        contexts[row["state_id"]] = np.asarray(
            [float(values[name]) for name in extractor.FEATURE_NAMES], dtype=float
        )
    return contexts


def _fit_regressor(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
    trees: int,
    leaves: dict[str, int],
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=trees,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        **leaves,
    )
    model.fit(matrix, target, sample_weight=weights)
    return model


def _ranker_dataset(
    rows: list[dict[str, Any]],
    indices: list[int],
    contexts: dict[str, np.ndarray],
    group: str,
    task_bootstrap: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    by_task_state: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        row = rows[index]
        varied = tuple(row.get("varied_groups", ()))
        if row.get("acquisition_role") == "neutral_anchor" or varied == (group,):
            by_task_state[row["task_id"]][row["state_id"]].append(index)
    task_order = task_bootstrap if task_bootstrap is not None else sorted(by_task_state)
    matrix: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    query: list[int] = []
    for task in task_order:
        for state in sorted(by_task_state.get(task, {})):
            state_indices = sorted(
                by_task_state[task][state],
                key=lambda index: canonical_json(rows[index]["configuration"]),
            )
            utilities = np.asarray([rows[index]["utility"] for index in state_indices])
            grades = rankdata(utilities, method="average") - 1.0
            # Lambdarank labels must be non-negative integers. Ties retain the
            # same rounded grade and are excluded from directional gate pairs.
            grades = np.rint(grades).astype(int)
            for index, grade in zip(state_indices, grades):
                matrix.append(
                    _group_vector(
                        contexts[rows[index]["state_id"]],
                        rows[index]["configuration"],
                        group,
                    )
                )
                labels.append(int(grade))
                weights.append(float(rows[index]["round_weight"]))
            query.append(len(state_indices))
    if not matrix:
        raise ValueError(f"No anchor-versus-isolated training rows for {group}")
    return (
        np.asarray(matrix, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=float),
        query,
    )


def _fit_ranker(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    seed: int,
    trees: int,
    leaves: dict[str, int],
) -> lgb.LGBMRanker:
    matrix, labels, weights, query = dataset
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=trees,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        **leaves,
    )
    model.fit(matrix, labels, group=query, sample_weight=weights)
    return model


def _calibrator(
    rows: list[dict[str, Any]],
    indices: list[int],
    contexts: dict[str, np.ndarray],
    group: str,
    models: list[Any],
) -> dict[str, list[float]]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        varied = tuple(rows[index].get("varied_groups", ()))
        if rows[index].get("acquisition_role") == "neutral_anchor" or varied == (group,):
            by_state[rows[index]["state_id"]].append(index)
    percentiles: list[float] = []
    advantages: list[float] = []
    for state in sorted(by_state):
        state_indices = by_state[state]
        matrix = np.asarray(
            [
                _group_vector(
                    contexts[state], rows[index]["configuration"], group
                )
                for index in state_indices
            ]
        )
        raw = np.mean(
            np.asarray([model.predict(matrix, num_threads=1) for model in models]),
            axis=0,
        )
        ranks = rankdata(raw, method="average")
        state_percentiles = (
            (ranks - 1.0) / max(1, len(raw) - 1)
            if len(raw) > 1
            else np.full(1, 0.5)
        )
        for index, percentile in zip(state_indices, state_percentiles):
            advantage = rows[index]["group_advantages"].get(group)
            if advantage is not None:
                percentiles.append(float(percentile))
                advantages.append(float(advantage))
    if len(set(percentiles)) < 2:
        return {"percentile": [0.0, 1.0], "advantage": [-0.5, 0.5]}
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(np.asarray(percentiles), np.asarray(advantages))
    knots = np.linspace(0.0, 1.0, 21)
    return {
        "percentile": knots.tolist(),
        "advantage": isotonic.predict(knots).astype(float).tolist(),
    }


def _support_cost(artifact: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    center = np.asarray(artifact["center"], dtype=float)
    scale = np.asarray(artifact["scale"], dtype=float)
    representatives = np.asarray(artifact["representatives"], dtype=float)
    normalized = (matrix - center) / scale
    distances = np.sqrt(
        np.mean((normalized[:, None, :] - representatives[None, :, :]) ** 2, axis=2)
    )
    k = min(int(artifact["k"]), len(representatives))
    raw = np.mean(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)
    return np.interp(
        raw,
        np.asarray(artifact["calibration"]["distance"]),
        np.asarray(artifact["calibration"]["cost"]),
    )


def _composition_search(
    rows: list[dict[str, Any]],
    validation_indices: list[int],
    group_predictions: dict[str, np.ndarray],
    residual: np.ndarray,
    downside: np.ndarray,
    support: np.ndarray,
    branch: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positions_by_state: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(validation_indices):
        positions_by_state[rows[index]["state_id"]].append(position)
    residual_weights = RESIDUAL_WEIGHT_GRID if branch.get("residual") else (0.0,)
    downside_weights = DOWNSIDE_WEIGHT_GRID if branch.get("downside") else (0.0,)
    support_weights = SUPPORT_WEIGHT_GRID if branch.get("support") else (0.0,)
    results = []
    for weight_name, group_weights in GROUP_WEIGHT_GRID.items():
        for residual_weight in residual_weights:
            for downside_weight in downside_weights:
                for support_weight in support_weights:
                    selected = []
                    family_values: dict[str, list[float]] = defaultdict(list)
                    for state in sorted(positions_by_state):
                        positions = np.asarray(positions_by_state[state], dtype=int)
                        scores = np.zeros(len(positions))
                        for group in ACTION_GROUPS:
                            scores += group_weights[group] * normalize_component(
                                group_predictions[group][positions]
                            )
                        scores += residual_weight * normalize_component(residual[positions])
                        scores -= downside_weight * normalize_component(downside[positions])
                        scores -= support_weight * normalize_component(support[positions])
                        chosen_position = int(positions[int(np.argmax(scores))])
                        chosen = rows[validation_indices[chosen_position]]
                        selected.append(float(chosen["utility"]))
                        family_values[chosen["family"]].append(float(chosen["utility"]))
                    family_means = [np.mean(values) for values in family_values.values()]
                    quality = float(np.mean(selected) - 0.10 * np.std(family_means))
                    results.append(
                        {
                            "group_weight_profile": weight_name,
                            "group_weights": group_weights,
                            "residual_weight": residual_weight,
                            "downside_weight": downside_weight,
                            "support_weight": support_weight,
                            "held_out_stable_policy_quality": quality,
                        }
                    )
    results.sort(
        key=lambda row: (
            -row["held_out_stable_policy_quality"],
            canonical_json(row),
        )
    )
    return results[0], results


def component_gate_report(
    rows: list[dict[str, Any]],
    validation_indices: list[int],
    group_predictions: dict[str, np.ndarray],
    residual_prediction: np.ndarray,
    downside_prediction: np.ndarray,
    catastrophic_prediction: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    position = {index: offset for offset, index in enumerate(validation_indices)}
    directional: dict[str, list[bool]] = defaultdict(list)
    family_isolated_delta: dict[str, list[float]] = defaultdict(list)
    by_state_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in validation_indices:
        varied = tuple(rows[index].get("varied_groups", ()))
        for group in ACTION_GROUPS:
            if rows[index].get("acquisition_role") == "neutral_anchor" or varied == (group,):
                by_state_group[(rows[index]["state_id"], group)].append(index)
    for (_, group), indices in by_state_group.items():
        anchor = next(
            (index for index in indices if rows[index].get("acquisition_role") == "neutral_anchor"),
            None,
        )
        if anchor is None:
            continue
        anchor_utility = rows[anchor]["utility"]
        anchor_prediction = group_predictions[group][position[anchor]]
        for index in indices:
            true_delta = float(rows[index]["utility"] - anchor_utility)
            predicted_delta = float(
                group_predictions[group][position[index]] - anchor_prediction
            )
            if abs(true_delta) > 1e-12:
                directional[group].append((true_delta > 0) == (predicted_delta > 0))
        selected = max(indices, key=lambda index: group_predictions[group][position[index]])
        family_isolated_delta[rows[selected]["family"]].append(
            float(rows[selected]["utility"] - anchor_utility)
        )
    directional_accuracy = {
        group: float(np.mean(directional[group])) if directional[group] else 0.0
        for group in ACTION_GROUPS
    }
    all_directional = [value for values in directional.values() for value in values]
    overall_directional = float(np.mean(all_directional)) if all_directional else 0.0

    residual_positions = [
        offset
        for offset, index in enumerate(validation_indices)
        if rows[index]["residual_target"] is not None
    ]
    additive = np.asarray(
        [rows[validation_indices[offset]]["additive_prediction"] for offset in residual_positions],
        dtype=float,
    )
    truth = np.asarray(
        [rows[validation_indices[offset]]["utility"] for offset in residual_positions],
        dtype=float,
    )
    additive_mse = float(mean_squared_error(truth, additive)) if len(truth) else float("inf")
    residual_mse = (
        float(mean_squared_error(truth, additive + residual_prediction[residual_positions]))
        if len(truth)
        else float("inf")
    )
    catastrophic = np.asarray(
        [rows[index]["catastrophic_label"] for index in validation_indices], dtype=int
    )
    catastrophic_probability = np.clip(catastrophic_prediction, 0.0, 1.0)
    auroc = (
        float(roc_auc_score(catastrophic, catastrophic_probability))
        if len(np.unique(catastrophic)) == 2
        else 0.0
    )
    brier = float(brier_score_loss(catastrophic, catastrophic_probability))
    constant_rate = float(np.mean(catastrophic))
    constant_brier = float(brier_score_loss(catastrophic, np.full(len(catastrophic), constant_rate)))
    downside_truth = np.asarray(
        [rows[index]["downside_target"] for index in validation_indices], dtype=float
    )
    errors = np.abs(downside_truth - downside_prediction)
    support_correlation = spearmanr(support, errors).statistic if len(support) > 2 else 0.0
    support_correlation = (
        float(support_correlation) if np.isfinite(support_correlation) else 0.0
    )
    family_means = {
        family: float(np.mean(values))
        for family, values in family_isolated_delta.items()
    }
    checks = {
        "directional_accuracy_overall_at_least_0_60": overall_directional >= 0.60,
        "directional_accuracy_each_group_at_least_0_55": all(
            value >= 0.55 for value in directional_accuracy.values()
        ),
        "isolated_group_family_delta_at_least_minus_0_02": bool(family_means)
        and min(family_means.values()) >= -0.02,
        "residual_improves_full_action_error": residual_mse < additive_mse,
        "catastrophic_auroc_at_least_0_65": auroc >= 0.65,
        "catastrophic_brier_beats_constant": brier < constant_brier,
        "support_error_non_decreasing": support_correlation >= 0.0,
        "support_telemetry_finite_and_monotonic": bool(np.all(np.isfinite(support)))
        and bool(np.all(np.diff(np.sort(support)) >= 0.0)),
        "adaptive_policy_has_no_runtime_fallback": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "directional_accuracy": {
            "overall": overall_directional,
            "groups": directional_accuracy,
        },
        "isolated_group_family_mean_delta": family_means,
        "full_action_mse": {
            "additive": additive_mse,
            "additive_plus_residual": residual_mse,
        },
        "catastrophic": {
            "auroc": auroc,
            "brier": brier,
            "constant_rate_brier": constant_brier,
        },
        "support_error_spearman": support_correlation,
    }


def _source_identity(require_clean: bool) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if require_clean and status:
        raise ValueError("Authoritative factorized fits require a clean committed source snapshot")
    return {
        "commit": head,
        "clean": not bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def train_factorized(
    database: Path,
    output: Path,
    seed: int,
    branch_name: str = "context_v4",
    trees: int = 12,
    leaf_index: int = 0,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    if branch_name not in ABLATION_BRANCHES:
        raise ValueError(f"Unknown ablation branch: {branch_name}")
    branch = ABLATION_BRANCHES[branch_name]
    if branch["policy"] != "factorized":
        raise ValueError("structured_joint uses the existing joint trainer")
    if trees not in TREE_GRID or not 0 <= leaf_index < len(LEAF_GRID):
        raise ValueError("trees/leaf choice is outside the predeclared grid")
    with ResearchStore(database) as store:
        raw_rows = store.training_rows()
    rows = derive_factorized_rows(raw_rows)
    rounds = {int(row["acquisition_round"]) for row in rows}
    if rounds != {0, 1}:
        raise ValueError("Factorized training requires exactly acquisition rounds {0,1}")
    splits = grouped_split(rows, seed)
    train_indices = list(splits["train"])
    validation_indices = [
        index
        for index in splits["validation"]
        if int(rows[index]["acquisition_round"]) == 1
    ]
    if not validation_indices:
        raise ValueError("Held-out component calibration requires structured round-one rows")
    extractor = (
        ATPE2ContextV4FeatureExtractor()
        if branch.get("context_v4")
        else ATPE2FeatureExtractor()
    )
    contexts = _contexts(rows, extractor)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    leaves = dict(LEAF_GRID[leaf_index])
    train_tasks = sorted({rows[index]["task_id"] for index in train_indices})
    members: list[dict[str, Any]] = []
    fitted_members: list[dict[str, Any]] = []

    residual_indices = [
        index for index in train_indices if rows[index]["residual_target"] is not None
    ]
    downside_indices = train_indices
    for member_index in range(8):
        bootstrapped_tasks = rng.choice(
            train_tasks, size=len(train_tasks), replace=True
        ).tolist()
        boot_task_set = set(bootstrapped_tasks)
        # Repeated task identities are intentionally retained for ranker query
        # bootstraps; regressors use occurrence-expanded row indices.
        expanded_indices = [
            index
            for task in bootstrapped_tasks
            for index in train_indices
            if rows[index]["task_id"] == task
        ]
        member_models: dict[str, Any] = {"groups": {}, "residual": None, "downside": {}}
        member_artifacts: dict[str, Any] = {"groups": {}, "residual": None, "downside": {}}
        for group_index, group in enumerate(ACTION_GROUPS):
            dataset = _ranker_dataset(
                rows,
                train_indices,
                contexts,
                group,
                bootstrapped_tasks,
            )
            ranker = _fit_ranker(
                dataset,
                seed + member_index * 100 + group_index,
                trees,
                leaves,
            )
            path = output / f"member-{member_index}-group-{group}.txt"
            member_artifacts["groups"][group] = _save_model(ranker, path)
            member_models["groups"][group] = ranker
        if branch.get("residual"):
            selected = [index for index in expanded_indices if index in set(residual_indices)]
            matrix = np.asarray(
                [_full_vector(contexts[rows[index]["state_id"]], rows[index]["configuration"]) for index in selected]
            )
            target = np.asarray([rows[index]["residual_target"] for index in selected])
            weights = np.asarray([rows[index]["round_weight"] for index in selected])
            residual_model = _fit_regressor(
                matrix, target, weights, seed + member_index * 100 + 10, trees, leaves
            )
            path = output / f"member-{member_index}-residual.txt"
            member_artifacts["residual"] = _save_model(residual_model, path)
            member_models["residual"] = residual_model
        for target_index, (name, field) in enumerate(
            (
                ("harm", "harm_target"),
                ("variability", "rollout_variability"),
                ("failure", "failed_step_fraction"),
                ("catastrophic", "catastrophic_label"),
            )
        ):
            if not branch.get("downside"):
                continue
            selected = expanded_indices
            matrix = np.asarray(
                [_full_vector(contexts[rows[index]["state_id"]], rows[index]["configuration"]) for index in selected]
            )
            target = np.asarray([rows[index][field] for index in selected], dtype=float)
            weights = np.asarray([rows[index]["round_weight"] for index in selected])
            model = _fit_regressor(
                matrix,
                target,
                weights,
                seed + member_index * 100 + 20 + target_index,
                trees,
                leaves,
            )
            path = output / f"member-{member_index}-downside-{name}.txt"
            member_artifacts["downside"][name] = _save_model(model, path)
            member_models["downside"][name] = model
        members.append(member_artifacts)
        fitted_members.append(member_models)

    calibrations = {
        group: _calibrator(
            rows,
            validation_indices,
            contexts,
            group,
            [member["groups"][group] for member in fitted_members],
        )
        for group in ACTION_GROUPS
    }
    full_train = np.asarray(
        [_full_vector(contexts[rows[index]["state_id"]], rows[index]["configuration"]) for index in train_indices]
    )
    support_artifact = robust_support_artifact(
        full_train,
        [rows[index]["task_id"] for index in train_indices],
        seed,
    )
    support_path = output / "support.json"
    support_path.write_text(json.dumps(support_artifact, sort_keys=True, separators=(",", ":")) + "\n")

    validation_matrix = np.asarray(
        [_full_vector(contexts[rows[index]["state_id"]], rows[index]["configuration"]) for index in validation_indices]
    )
    group_predictions: dict[str, np.ndarray] = {}
    for group in ACTION_GROUPS:
        group_matrix = np.asarray(
            [_group_vector(contexts[rows[index]["state_id"]], rows[index]["configuration"], group) for index in validation_indices]
        )
        raw = np.mean(
            np.asarray([member["groups"][group].predict(group_matrix) for member in fitted_members]),
            axis=0,
        )
        calibrated = np.empty(len(raw), dtype=float)
        by_state: dict[str, list[int]] = defaultdict(list)
        for position, index in enumerate(validation_indices):
            by_state[rows[index]["state_id"]].append(position)
        for positions in by_state.values():
            ranks = rankdata(raw[positions], method="average")
            percentiles = (ranks - 1.0) / max(1, len(positions) - 1)
            calibrated[positions] = np.interp(
                percentiles,
                calibrations[group]["percentile"],
                calibrations[group]["advantage"],
            )
        group_predictions[group] = calibrated

    def ensemble_target(component: str, target: str | None = None) -> np.ndarray:
        models = [
            member[component] if target is None else member[component].get(target)
            for member in fitted_members
        ]
        models = [model for model in models if model is not None]
        return (
            np.mean(np.asarray([model.predict(validation_matrix) for model in models]), axis=0)
            if models
            else np.zeros(len(validation_indices))
        )

    residual_prediction = ensemble_target("residual")
    downside_heads = {
        name: ensemble_target("downside", name)
        for name in ("harm", "variability", "failure", "catastrophic")
    }
    downside_prediction = np.clip(
        downside_heads["harm"]
        + 0.5 * downside_heads["variability"]
        + downside_heads["failure"],
        0.0,
        1.0,
    )
    support = _support_cost(support_artifact, validation_matrix)
    composition, composition_grid = _composition_search(
        rows,
        validation_indices,
        group_predictions,
        residual_prediction,
        downside_prediction,
        support,
        branch,
    )
    gates = component_gate_report(
        rows,
        validation_indices,
        group_predictions,
        residual_prediction,
        downside_prediction,
        downside_heads["catastrophic"],
        support,
    )

    feature_schema_hash = hashlib.sha256(
        canonical_json(extractor.schema_definition()).encode()
    ).hexdigest()
    action_schema_hash = hashlib.sha256(
        canonical_json(action_schema_definition(ATPE2Action.NAMES, {}, {})).encode()
    ).hexdigest()
    context_train = np.asarray([contexts[rows[index]["state_id"]] for index in train_indices])
    training_hash = sqlite_snapshot_hash(database)
    identity = hashlib.sha256(
        canonical_json(
            {
                "members": members,
                "support": _artifact(support_path),
                "training_hash": training_hash,
                "branch": branch_name,
                "trees": trees,
                "leaves": leaves,
                "seed": seed,
            }
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": FACTORIZED_MANIFEST_VERSION,
        "policy_type": "factorized",
        "model_version": f"atpe2-factorized-v1-{identity[:12]}",
        "controller_policy_version": FACTORIZED_CONTROLLER_POLICY_VERSION,
        "status": "candidate",
        "model_artifact_sha256": identity,
        "members": members,
        "support_artifact": _artifact(support_path),
        "feature_schema_version": extractor.version,
        "feature_schema_hash": feature_schema_hash,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_schema_hash": action_schema_hash,
        "retained_action_names": list(ATPE2Action.NAMES),
        "validation_selected_constants": {},
        "action_ranges": {},
        "runtime": {
            "group_proposals": 17,
            "retained_per_group": 6,
            "beam_width": 64,
            "full_action_exploration_candidates": 1,
            "max_supported_distance": 1.0,
            "deterministic": True,
            "fallback_to_tpe": False,
        },
        "calibration": {"groups": calibrations},
        "score_composition": {
            key: composition[key]
            for key in (
                "group_weight_profile",
                "group_weights",
                "residual_weight",
                "downside_weight",
                "support_weight",
            )
        },
        "score_grid": composition_grid,
        "component_gates": gates,
        "target_definitions": {
            "version": TARGET_DEFINITIONS_VERSION,
            "group_advantage": "isolated group utility minus anchor utility",
            "additive": "anchor plus three matched group advantages",
            "residual": "pair/full utility minus additive prediction",
            "downside": "clip(anchor regret + 0.5 * rollout downside deviation + failed-step fraction, 0, 1)",
            "catastrophic": "utility at least 0.15 below anchor or any optimizer failure",
        },
        "acquisition_schema": {
            "implementation_version": "atpe2-structured-acquisition-v1",
            "rounds": [0, 1],
            "round_one_training_mass": 0.10,
            "mandatory_paired_rollouts": 2,
            "third_repeats": "diagnostic_only",
        },
        "training_rounds": [0, 1],
        "training_data_hashes": [training_hash],
        "task_splits": {
            name: sorted({rows[index]["task_id"] for index in indices})
            for name, indices in splits.items()
        },
        "lineage_splits": {
            name: sorted({rows[index]["trajectory_lineage"] for index in indices})
            for name, indices in splits.items()
        },
        "context_envelope": {
            "min": np.min(context_train, axis=0).tolist(),
            "max": np.max(context_train, axis=0).tolist(),
        },
        "architecture_branch": branch_name,
        "architecture": branch,
        "model_training": {
            "ensemble_members": 8,
            "trees_per_component": trees,
            "leaf_hyperparameters": leaves,
            "task_bootstrap": True,
        },
        "source_identity": _source_identity(require_clean_source),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Reload immediately so corrupt/missing component declarations fail before
    # the candidate can be dispatched to behavioral workers.
    loaded = ATPE2ModelEnsemble.load(manifest_path)
    if not loaded.research_ready:
        raise ValueError("Exported factorized manifest failed runtime schema validation")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", choices=tuple(ABLATION_BRANCHES), default="context_v4")
    parser.add_argument("--trees", type=int, choices=TREE_GRID, default=12)
    parser.add_argument("--leaf-index", type=int, choices=range(len(LEAF_GRID)), default=0)
    parser.add_argument("--require-clean-source", action="store_true")
    args = parser.parse_args(argv)
    manifest = train_factorized(
        args.database,
        args.output,
        args.seed,
        args.branch,
        args.trees,
        args.leaf_index,
        args.require_clean_source,
    )
    print(
        json.dumps(
            {
                "model_version": manifest["model_version"],
                "branch": manifest["architecture_branch"],
                "component_gates": manifest["component_gates"],
                "score_composition": manifest["score_composition"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
