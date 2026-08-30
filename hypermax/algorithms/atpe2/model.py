"""Loading and scoring of a frozen ATPE2 LightGBM ensemble."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np

from .controller_schema import action_schema_definition
from .features import (
    ATPE2ContextV4FeatureExtractor,
    ATPE2FeatureExtractor,
    FEATURE_EXTRACTORS,
)
from .legacy_features import LegacyATPEFeatureExtractor


class ATPE2ModelEnsemble:
    """Eight task-bootstrap regressors plus validation calibration metadata."""

    def __init__(
        self,
        manifest: dict[str, Any],
        models: list[Any],
        support: dict[str, Any] | None = None,
        teacher_models: dict[str, Any] | None = None,
    ):
        self.manifest = manifest
        self.models = models
        self.support = support
        self.teacher_models = teacher_models or {}

    @staticmethod
    def _checked_payload(directory: Any, artifact: dict[str, Any]) -> bytes:
        payload = directory.joinpath(artifact["file"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"ATPE2 model checksum mismatch for {artifact['file']}")
        return payload

    @classmethod
    def _load_components(
        cls, directory: Any, manifest: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any] | None, dict[str, Any]]:
        if manifest.get("policy_type") != "factorized":
            models = [
                lightgbm.Booster(
                    model_str=cls._checked_payload(directory, artifact).decode("utf-8")
                )
                for artifact in manifest.get("models", [])
            ]
            return models, None, {}

        members = []
        for member in manifest.get("members", []):
            groups = {
                name: lightgbm.Booster(
                    model_str=cls._checked_payload(directory, artifact).decode("utf-8")
                )
                for name, artifact in member.get("groups", {}).items()
            }
            residual_artifact = member.get("residual")
            residual = (
                lightgbm.Booster(
                    model_str=cls._checked_payload(directory, residual_artifact).decode("utf-8")
                )
                if residual_artifact
                else None
            )
            downside = {
                name: lightgbm.Booster(
                    model_str=cls._checked_payload(directory, artifact).decode("utf-8")
                )
                for name, artifact in member.get("downside", {}).items()
            }
            members.append({"groups": groups, "residual": residual, "downside": downside})
        support_artifact = manifest.get("support_artifact")
        support = None
        if support_artifact:
            support = json.loads(cls._checked_payload(directory, support_artifact).decode("utf-8"))
        teacher_models = {
            name: lightgbm.Booster(
                model_str=cls._checked_payload(directory, artifact).decode("utf-8")
            )
            for name, artifact in manifest.get("teacher_artifacts", {}).items()
        }
        return members, support, teacher_models

    @classmethod
    def load_default(cls) -> "ATPE2ModelEnsemble":
        directory = files("hypermax").joinpath("atpe2_models")
        manifest = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
        models, support, teacher_models = cls._load_components(directory, manifest)
        return cls(manifest, models, support, teacher_models)

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ATPE2ModelEnsemble":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models, support, teacher_models = cls._load_components(
            manifest_path.parent, manifest
        )
        return cls(manifest, models, support, teacher_models)

    @property
    def is_factorized(self) -> bool:
        return self.manifest.get("policy_type") == "factorized"

    @property
    def ready(self) -> bool:
        return len(self.models) == 8 and self.manifest.get("status") == "frozen" and self.schema_compatible

    @property
    def research_ready(self) -> bool:
        return (
            len(self.models) == 8
            and self.manifest.get("status") in {"candidate", "frozen"}
            and self.schema_compatible
        )

    @property
    def action_names(self) -> tuple[str, ...]:
        names = self.manifest.get(
            "retained_action_names",
            [
                "gamma",
                "n_ei_candidates",
                "history_keep_fraction",
                "history_recency_bias",
                "history_quality_bias",
                "secondary_fraction",
                "secondary_lock_probability",
                "secondary_top_fraction",
            ],
        )
        return tuple(names)

    @property
    def runtime_candidate_count(self) -> int:
        """Return the versioned Sobol action count for each decision boundary."""
        count = int(self.manifest.get("runtime_candidate_count", 1024))
        if count < 2 or count & (count - 1):
            raise ValueError("ATPE2 runtime_candidate_count must be a power of two")
        return count

    @property
    def feature_extractor_class(self):
        version = self.manifest.get("feature_schema_version")
        if version == LegacyATPEFeatureExtractor.version:
            return LegacyATPEFeatureExtractor
        return FEATURE_EXTRACTORS.get(version)

    @property
    def schema_compatible(self) -> bool:
        valid_actions = {
            "gamma",
            "n_ei_candidates",
            "history_keep_fraction",
            "history_recency_bias",
            "history_quality_bias",
            "secondary_fraction",
            "secondary_lock_probability",
            "secondary_top_fraction",
        }
        feature_schemas = {
            ATPE2FeatureExtractor.version: ATPE2FeatureExtractor.schema_definition(),
            ATPE2ContextV4FeatureExtractor.version: ATPE2ContextV4FeatureExtractor.schema_definition(),
            LegacyATPEFeatureExtractor.version: LegacyATPEFeatureExtractor.schema_definition(),
        }
        feature_schema = feature_schemas.get(self.manifest.get("feature_schema_version"))
        if feature_schema is None:
            return False
        action_schema = action_schema_definition(
            self.action_names,
            self.manifest.get("validation_selected_constants", {}),
            self.manifest.get("action_ranges", {}),
        )
        constants = set(self.manifest.get("validation_selected_constants", {}))
        retained = set(self.action_names)
        complete_action_partition = (
            len(self.action_names) == len(retained)
            and retained.isdisjoint(constants)
            and retained | constants == valid_actions
        )
        expected_feature_hash = hashlib.sha256(
            json.dumps(feature_schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected_action_hash = hashlib.sha256(
            json.dumps(action_schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not self.is_factorized:
            try:
                self.runtime_candidate_count
            except (TypeError, ValueError):
                return False
        factorized_components_valid = True
        if self.is_factorized:
            expected_groups = {"search", "history", "locking"}
            factorized_components_valid = (
                self.manifest.get("schema_version") == "atpe2-factorized-manifest-v1"
                and len(self.models) == 8
                and all(
                    set(member.get("groups", {})) == expected_groups
                    for member in self.models
                )
                and self.support is not None
                and (
                    not self.manifest.get("architecture", {}).get("teacher")
                    or set(self.teacher_models) == {"search", "filtering", "locking"}
                )
                and int(self.manifest.get("runtime", {}).get("group_proposals", 0)) == 17
                and int(self.manifest.get("runtime", {}).get("beam_width", 0)) <= 64
            )
        return (
            self.manifest.get("action_schema_version") == "atpe2-action-v1"
            and complete_action_partition
            and self.manifest.get("feature_schema_hash") == expected_feature_hash
            and self.manifest.get("action_schema_hash") == expected_action_hash
            and factorized_components_valid
        )

    def in_envelope(self, context: np.ndarray) -> bool:
        lower = np.asarray(self.manifest.get("context_envelope", {}).get("min", []), dtype=float)
        upper = np.asarray(self.manifest.get("context_envelope", {}).get("max", []), dtype=float)
        return lower.shape == context.shape and upper.shape == context.shape and bool(
            np.all(context >= lower) and np.all(context <= upper)
        )

    def augment_context(self, context: np.ndarray) -> np.ndarray:
        """Append classic-mode probabilities without executing classic ATPE."""
        if not self.teacher_models:
            return np.asarray(context, dtype=float)
        features = []
        classes = self.manifest.get("teacher_classes", {})
        for name in ("search", "filtering", "locking"):
            model = self.teacher_models[name]
            prediction = np.asarray(
                model.predict(np.asarray(context, dtype=float)[None, :], num_threads=1)
            )
            if prediction.ndim == 1:
                prediction = prediction[None, :]
            values = prediction[0]
            expected = len(classes.get(name, []))
            if expected == 2 and len(values) == 1:
                probability = float(values[0])
                values = np.asarray([1.0 - probability, probability])
            if len(values) != expected:
                raise ValueError(f"ATPE2 teacher {name} probability width mismatch")
            features.extend(float(value) for value in values)
        output = np.concatenate((np.asarray(context, dtype=float), np.asarray(features)))
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("ATPE2 teacher features are non-finite")
        return output

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.research_ready:
            raise RuntimeError("the ATPE2 ensemble is not schema-compatible or fitted")
        # Runtime scoring evaluates only 1,025 rows at a time.  Letting every
        # Booster create its own OpenMP team badly oversubscribes multi-worker
        # optimization runs; process-level parallelism is the intended layer.
        predictions = np.asarray(
            [model.predict(matrix, num_threads=1) for model in self.models],
            dtype=float,
        )
        means = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0, ddof=1)
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(uncertainty)):
            raise FloatingPointError("ATPE2 ensemble emitted a non-finite prediction")
        return means, uncertainty

    @staticmethod
    def _component_prediction(models: list[Any], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not models:
            zeros = np.zeros(len(matrix), dtype=float)
            return zeros, zeros
        predictions = np.asarray(
            [model.predict(matrix, num_threads=1) for model in models], dtype=float
        )
        means = np.mean(predictions, axis=0)
        uncertainty = (
            np.std(predictions, axis=0, ddof=1)
            if len(predictions) > 1
            else np.zeros(predictions.shape[1], dtype=float)
        )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(uncertainty)):
            raise FloatingPointError("ATPE2 factorized component emitted non-finite predictions")
        return means, uncertainty

    def predict_group(self, group: str, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_factorized or not self.research_ready:
            raise RuntimeError("factorized group scoring requires a fitted factorized manifest")
        means, uncertainty = self._component_prediction(
            [member["groups"][group] for member in self.models], matrix
        )
        if len(means) <= 1:
            percentiles = np.full(len(means), 0.5, dtype=float)
        else:
            order = np.argsort(np.argsort(means, kind="stable"), kind="stable")
            percentiles = order.astype(float) / float(len(means) - 1)
        calibration = self.manifest.get("calibration", {}).get("groups", {}).get(group, {})
        x = np.asarray(calibration.get("percentile", [0.0, 1.0]), dtype=float)
        y = np.asarray(calibration.get("advantage", [-0.5, 0.5]), dtype=float)
        calibrated = np.interp(percentiles, x, y)
        return calibrated, uncertainty

    def predict_residual(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._component_prediction(
            [member.get("residual") for member in self.models if member.get("residual") is not None],
            matrix,
        )

    def predict_downside(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for target in ("harm", "variability", "failure", "catastrophic"):
            models = [
                member.get("downside", {}).get(target)
                for member in self.models
                if member.get("downside", {}).get(target) is not None
            ]
            output[target] = self._component_prediction(models, matrix)[0]
        return output

    def support_cost(self, matrix: np.ndarray) -> np.ndarray:
        """Calibrated k-NN distance to deterministic training representatives."""
        if self.support is None:
            return np.zeros(len(matrix), dtype=float)
        center = np.asarray(self.support["center"], dtype=float)
        scale = np.asarray(self.support["scale"], dtype=float)
        representatives = np.asarray(self.support["representatives"], dtype=float)
        if matrix.shape[1] != len(center) or representatives.shape[1] != len(center):
            raise ValueError("ATPE2 support artifact width does not match model input")
        normalized = (np.asarray(matrix, dtype=float) - center) / scale
        distances = np.sqrt(
            np.mean((normalized[:, None, :] - representatives[None, :, :]) ** 2, axis=2)
        )
        k = min(int(self.support.get("k", 5)), distances.shape[1])
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        raw = np.mean(nearest, axis=1)
        calibration = self.support.get("calibration", {})
        x = np.asarray(calibration.get("distance", [0.0, 1.0]), dtype=float)
        y = np.asarray(calibration.get("cost", [0.0, 1.0]), dtype=float)
        costs = np.interp(raw, x, y, left=float(y[0]), right=float(y[-1]))
        if not np.all(np.isfinite(costs)):
            raise FloatingPointError("ATPE2 support scoring emitted non-finite costs")
        return costs

    @property
    def uncertainty_penalty(self) -> float:
        return float(self.manifest.get("calibration", {}).get("uncertainty_penalty", 1.0))
