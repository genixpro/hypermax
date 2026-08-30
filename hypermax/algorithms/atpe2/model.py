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
from .features import ATPE2FeatureExtractor
from .legacy_features import LegacyATPEFeatureExtractor


class ATPE2ModelEnsemble:
    """Eight task-bootstrap regressors plus validation calibration metadata."""

    def __init__(self, manifest: dict[str, Any], models: list[Any]):
        self.manifest = manifest
        self.models = models

    @classmethod
    def load_default(cls) -> "ATPE2ModelEnsemble":
        directory = files("hypermax").joinpath("atpe2_models")
        manifest = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
        models = []
        for artifact in manifest.get("models", []):
            payload = directory.joinpath(artifact["file"]).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError(f"ATPE2 model checksum mismatch for {artifact['file']}")
            # Booster accepts a model string and avoids temporary files for packaged resources.
            models.append(lightgbm.Booster(model_str=payload.decode("utf-8")))
        return cls(manifest, models)

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ATPE2ModelEnsemble":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = []
        for artifact in manifest.get("models", []):
            path = manifest_path.parent / artifact["file"]
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
                raise ValueError(f"ATPE2 model checksum mismatch for {path.name}")
            models.append(lightgbm.Booster(model_file=str(path)))
        return cls(manifest, models)

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
        try:
            self.runtime_candidate_count
        except (TypeError, ValueError):
            return False
        return (
            self.manifest.get("action_schema_version") == "atpe2-action-v1"
            and complete_action_partition
            and self.manifest.get("feature_schema_hash") == expected_feature_hash
            and self.manifest.get("action_schema_hash") == expected_action_hash
        )

    def in_envelope(self, context: np.ndarray) -> bool:
        lower = np.asarray(self.manifest.get("context_envelope", {}).get("min", []), dtype=float)
        upper = np.asarray(self.manifest.get("context_envelope", {}).get("max", []), dtype=float)
        return lower.shape == context.shape and upper.shape == context.shape and bool(
            np.all(context >= lower) and np.all(context <= upper)
        )

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

    @property
    def uncertainty_penalty(self) -> float:
        return float(self.manifest.get("calibration", {}).get("uncertainty_penalty", 1.0))
