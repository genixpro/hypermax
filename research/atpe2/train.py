"""Grouped ATPE2 ensemble training, calibration, and frozen manifest export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np

from hypermax.algorithms.atpe2.controller import (
    ATPE2Action,
    CONTROLLER_POLICY_VERSION,
)
from hypermax.algorithms.atpe2.controller_schema import (
    ACTION_SCHEMA_VERSION,
    action_schema_definition,
)
from hypermax.algorithms.atpe2.features import ATPE2FeatureExtractor
from hypermax.algorithms.atpe2.legacy_features import LegacyATPEFeatureExtractor
from .database import ResearchStore, canonical_json


def sqlite_snapshot_hash(path: Path) -> str:
    """Checkpoint WAL contents, then hash the complete SQLite snapshot."""
    connection = sqlite3.connect(path, timeout=120)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_split(rows: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    """Assign whole tasks and lineages to deterministic, family-stratified splits."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["task_id"], []).append(index)
    rng = np.random.default_rng(seed)
    if len(groups) < 3:
        raise ValueError("Grouped split requires at least three distinct tasks")

    task_family = {
        task: str(rows[indices[0]].get("family", "__all__"))
        for task, indices in groups.items()
    }
    family_tasks: dict[str, list[str]] = {}
    for task, family in task_family.items():
        family_tasks.setdefault(family, []).append(task)
    stratified = all(len(tasks) >= 3 for tasks in family_tasks.values())
    strata = (
        [family_tasks[family] for family in sorted(family_tasks)]
        if stratified
        else [list(groups)]
    )
    assignments: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for tasks in strata:
        tasks = sorted(tasks)
        rng.shuffle(tasks)
        train_count = max(1, min(len(tasks) - 2, int(round(0.70 * len(tasks)))))
        validation_count = max(
            1,
            min(len(tasks) - train_count - 1, int(round(0.15 * len(tasks)))),
        )
        assignments["train"].extend(tasks[:train_count])
        assignments["validation"].extend(
            tasks[train_count : train_count + validation_count]
        )
        assignments["test"].extend(tasks[train_count + validation_count :])
    return {
        name: [index for task in tasks for index in groups[task]]
        for name, tasks in assignments.items()
    }


def row_vector(
    row: dict[str, Any],
    action_names: list[str] | tuple[str, ...] = ATPE2Action.NAMES,
    context: dict[str, float] | None = None,
    feature_names: tuple[str, ...] = ATPE2FeatureExtractor.FEATURE_NAMES,
) -> list[float]:
    context_values = context if context is not None else row["context"]
    context_vector = [float(context_values[name]) for name in feature_names]
    action = [float(row["configuration"][name]) for name in action_names]
    return context_vector + action


def train_ensemble(
    database: Path,
    output: Path,
    seed: int,
    action_schema: dict[str, Any] | None = None,
    feature_schema: str = "robust",
    n_estimators: int = 300,
    runtime_candidate_count: int = 1024,
) -> dict[str, Any]:
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if runtime_candidate_count < 2 or runtime_candidate_count & (runtime_candidate_count - 1):
        raise ValueError("runtime_candidate_count must be a power of two")
    with ResearchStore(database) as store:
        rows = store.training_rows()
    if len(rows) < 30:
        raise ValueError("At least 30 labeled rollout rows are required")
    splits = grouped_split(rows, seed)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Grouped split requires multiple task/trajectory lineages")
    schema = (action_schema or {}).get("backward_knockout", action_schema or {})
    action_names = list(schema.get("retained_knobs", ATPE2Action.NAMES))
    fixed_constants = dict(schema.get("fixed_constants", {}))
    if not set(action_names).issubset(ATPE2Action.NAMES):
        raise ValueError("Action schema contains unknown retained knobs")
    if feature_schema == "robust":
        extractor = ATPE2FeatureExtractor()
        contexts = {}
        required_features = set(extractor.FEATURE_NAMES)
        for row in rows:
            if row["state_id"] in contexts:
                continue
            stored = row["context"]
            contexts[row["state_id"]] = (
                stored
                if required_features.issubset(stored)
                and stored.get("__feature_schema_version") == extractor.version
                else extractor.extract(row["space"], row["history"], [])
            )
    elif feature_schema == "legacy":
        extractor = LegacyATPEFeatureExtractor()
        contexts = {}
        for row in rows:
            if row["state_id"] not in contexts:
                contexts[row["state_id"]] = extractor.extract(
                    row["space"], row["history"], []
                )
    else:
        raise ValueError("feature_schema must be 'robust' or 'legacy'")
    matrix = np.empty(
        (len(rows), len(extractor.FEATURE_NAMES) + len(action_names)), dtype=float
    )
    labels = np.empty(len(rows), dtype=float)
    for index, row in enumerate(rows):
        matrix[index] = row_vector(
            row,
            action_names,
            contexts[row["state_id"]],
            extractor.FEATURE_NAMES,
        )
        labels[index] = row["utility"]
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(labels)):
        raise ValueError("Training data contains non-finite values")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train_indices = np.asarray(splits["train"])
    validation_indices = np.asarray(splits["validation"])
    # Tree complexity is selected on whole held-out tasks before task-bootstrap
    # members are fit. Trees need no fitted scaling or imputation preprocessing.
    candidates = [
        {"num_leaves": leaves, "min_child_samples": children}
        for leaves in (7, 15, 31)
        for children in (5, 10, 20)
    ]
    validation_task_positions: dict[str, list[int]] = {}
    for position, row_index in enumerate(validation_indices):
        validation_task_positions.setdefault(rows[row_index]["task_id"], []).append(
            position
        )
    candidate_scores = []
    for candidate_index, candidate in enumerate(candidates):
        probe = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.04,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed + 10_000 + candidate_index,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **candidate,
        )
        probe.fit(matrix[train_indices], labels[train_indices])
        prediction = probe.predict(matrix[validation_indices])
        task_errors = []
        validation_labels = labels[validation_indices]
        for task in sorted(validation_task_positions):
            positions = np.asarray(validation_task_positions[task])
            task_errors.append(
                float(
                    np.mean(
                        (prediction[positions] - validation_labels[positions]) ** 2
                    )
                )
            )
        candidate_scores.append(float(np.mean(task_errors)))
    selected_hyperparameters = candidates[int(np.argmin(candidate_scores))]
    models = []
    artifacts = []
    validation_predictions = []
    train_indices_by_task: dict[str, list[int]] = {}
    for row_index in train_indices:
        train_indices_by_task.setdefault(rows[row_index]["task_id"], []).append(
            int(row_index)
        )
    train_tasks = sorted(train_indices_by_task)
    for model_index in range(8):
        bootstrapped_tasks = rng.choice(train_tasks, size=len(train_tasks), replace=True)
        bootstrapped = np.concatenate(
            [
                np.asarray(train_indices_by_task[task], dtype=int)
                for task in bootstrapped_tasks
            ]
        )
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.035,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed + model_index,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **selected_hyperparameters,
        )
        model.fit(matrix[bootstrapped], labels[bootstrapped])
        path = output / f"model-{model_index}.txt"
        model.booster_.save_model(str(path))
        payload = path.read_bytes()
        artifacts.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest()})
        models.append(model)
        validation_predictions.append(model.predict(matrix[validation_indices]))
    predictions = np.asarray(validation_predictions)
    means = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0, ddof=1)
    errors = np.abs(labels[validation_indices] - means)
    penalties = np.linspace(0, 2, 21)
    validation_states: dict[str, dict[str, list[int]]] = {}
    for position, row_index in enumerate(validation_indices):
        configuration_key = canonical_json(rows[row_index]["configuration"])
        validation_states.setdefault(rows[row_index]["state_id"], {}).setdefault(
            configuration_key, []
        ).append(position)
    penalty_scores = []
    for candidate in penalties:
        risk_adjusted = means - candidate * uncertainty
        selected_utilities = []
        for configurations in validation_states.values():
            groups = list(configurations.values())
            selected_group = int(
                np.argmax(
                    [float(np.mean(risk_adjusted[positions])) for positions in groups]
                )
            )
            selected_utilities.append(
                float(
                    np.mean(
                        labels[validation_indices[np.asarray(groups[selected_group])]]
                    )
                )
            )
        penalty_scores.append(float(np.mean(selected_utilities)))
    penalty = float(penalties[int(np.argmax(penalty_scores))])
    uncertainty_p95 = float(np.quantile(uncertainty, 0.95))
    context_width = len(extractor.FEATURE_NAMES)
    context_train = matrix[train_indices, :context_width]
    try:
        commit_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        repository_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        )
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], text=False)
        untracked_names = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            text=False,
        ).split(b"\0")
        untracked = {
            name.decode("utf-8"): hashlib.sha256(
                (repository_root / name.decode("utf-8")).read_bytes()
            ).hexdigest()
            for name in untracked_names
            if name and (repository_root / name.decode("utf-8")).is_file()
        }
        identity_payload = {
            "head": commit_head,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files": untracked,
        }
        commit = {
            "head": commit_head,
            "dirty": bool(diff or untracked),
            "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files": untracked,
            "working_tree_identity_sha256": hashlib.sha256(
                canonical_json(identity_payload).encode()
            ).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        commit = None
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("hyperopt", "lightgbm", "numpy", "scipy", "scikit-learn")
    }
    dependencies["python"] = platform.python_version()
    dependencies["platform"] = platform.platform()
    feature_schema_hash = hashlib.sha256(
        canonical_json(extractor.schema_definition()).encode()
    ).hexdigest()
    action_schema_hash = hashlib.sha256(
        canonical_json(
            action_schema_definition(
                action_names, fixed_constants, schema.get("ranges", {})
            )
        ).encode()
    ).hexdigest()
    training_data_hash = sqlite_snapshot_hash(database)
    artifact_identity = hashlib.sha256(
        canonical_json(
            {
                "models": artifacts,
                "training_data_hash": training_data_hash,
                "feature_schema_hash": feature_schema_hash,
                "action_schema_hash": action_schema_hash,
                "root_seed": seed,
                "n_estimators": n_estimators,
                "runtime_candidate_count": runtime_candidate_count,
            }
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "atpe2-manifest-v1",
        "model_version": f"atpe2-ensemble-v1-{artifact_identity[:12]}",
        "controller_policy_version": CONTROLLER_POLICY_VERSION,
        "runtime_candidate_count": runtime_candidate_count,
        "model_artifact_sha256": artifact_identity,
        "status": "candidate",
        "models": artifacts,
        "feature_schema_version": extractor.version,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash,
        "action_schema_hash": action_schema_hash,
        "retained_action_names": action_names,
        "training_data_hashes": [training_data_hash],
        "task_splits": {name: sorted(set(rows[index]["task_id"] for index in indices)) for name, indices in splits.items()},
        "lineage_splits": {name: sorted(set(rows[index]["trajectory_lineage"] for index in indices)) for name, indices in splits.items()},
        "family_splits": {
            name: sorted(set(rows[index]["family"] for index in indices))
            for name, indices in splits.items()
        },
        "dependency_versions": dependencies,
        "seeds": [seed + index for index in range(8)],
        "commit_identity": commit,
        "validation_selected_constants": fixed_constants,
        "action_ranges": schema.get("ranges", {}),
        "knob_pruning_evidence": {
            "mode": "full_pilot_schema" if action_schema is None else "backward_knockout",
            "source_sha256": (
                hashlib.sha256(canonical_json(action_schema).encode()).hexdigest()
                if action_schema is not None
                else None
            ),
            "iterations": schema.get("iterations", []),
        },
        "calibration": {
            "uncertainty_penalty": penalty,
            "validation_uncertainty_p95": uncertainty_p95,
        },
        "context_envelope": {
            "min": np.min(context_train, axis=0).tolist(),
            "max": np.max(context_train, axis=0).tolist(),
        },
        "validation": {"mae": float(np.mean(errors)), "rows": len(splits["validation"])},
        "selected_tree_hyperparameters": selected_hyperparameters,
        "model_training": {
            "ensemble_members": 8,
            "n_estimators": n_estimators,
            "learning_rate": 0.035,
        },
        "preprocessing": "none (finite versioned runtime features; tree model)",
        "dataset_aggregation_rounds": sorted(set(row["acquisition_round"] for row in rows)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_manifest = {
        "schema_version": "atpe2-grouped-splits-v1",
        "root_seed": seed,
        "tasks": manifest["task_splits"],
        "trajectory_lineages": manifest["lineage_splits"],
        "families": manifest["family_splits"],
    }
    (output / "splits.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_card = f"""# ATPE2 model card

Status: research candidate. Runtime publication requires a passing Gate 2 report
and is performed by `python -m research.atpe2.freeze`.

## Model

- Ensemble: eight task-bootstrap LightGBM regressors
- Trees per ensemble member: {n_estimators}
- Runtime Sobol candidates per decision: {runtime_candidate_count} plus the TPE anchor
- Artifact identity: `{artifact_identity}`
- Feature schema: `{manifest['feature_schema_version']}`
- Action schema: `{manifest['action_schema_version']}`
- Retained actions: {', '.join(action_names)}
- Fixed constants: `{json.dumps(fixed_constants, sort_keys=True)}`
- Validation MAE: {manifest['validation']['mae']:.8f} over {manifest['validation']['rows']} rows
- Uncertainty penalty: {penalty:.4f}
- Validation uncertainty p95 (diagnostic only): {uncertainty_p95:.8f}

## Data and evaluation boundaries

- Training data SHA-256: `{manifest['training_data_hashes'][0]}`
- Grouping unit: whole task and trajectory lineage
- Train/validation/test task counts: {len(manifest['task_splits']['train'])}/{len(manifest['task_splits']['validation'])}/{len(manifest['task_splits']['test'])}
- Dataset-aggregation rounds present: {manifest['dataset_aggregation_rounds']}
- External HPO-B/YAHPO data was not used for model fitting or calibration.

## Intended use and limitations

This candidate predicts scale-free five-step controller utility for ATPE2. It is
not a multi-fidelity controller. `abh2` is a compatibility path through the
existing scheduler and carries no performance claim. Publication is guarded by
`research.atpe2.freeze`, which requires exact matching pruning and passing Gate 2
evidence.
"""
    (output / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action-schema", type=Path)
    parser.add_argument("--feature-schema", choices=("robust", "legacy"), default="robust")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--runtime-candidate-count", type=int, default=1024)
    args = parser.parse_args(argv)
    action_schema = json.loads(args.action_schema.read_text()) if args.action_schema else None
    print(
        json.dumps(
            train_ensemble(
                args.database,
                args.output,
                args.seed,
                action_schema,
                args.feature_schema,
                args.n_estimators,
                args.runtime_candidate_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
