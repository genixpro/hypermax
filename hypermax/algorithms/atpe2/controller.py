"""Joint ATPE2 controller action selection and history transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.stats import qmc, rankdata

from .controller_schema import ACTION_DOMAINS, ACTION_SCHEMA_VERSION
from .features import ATPE2FeatureExtractor, ParameterDescriptor, describe_space, parameter_relevance, successful_results
from .legacy_features import LegacyATPEFeatureExtractor
from .model import ATPE2ModelEnsemble


MODEL_SCHEMA_VERSION = "atpe2-ensemble-v1"
CONTROLLER_POLICY_VERSION = "atpe2-adaptive-only-v2"


@dataclass(frozen=True)
class ATPE2Action:
    gamma: float = 0.25
    n_ei_candidates: int = 24
    history_keep_fraction: float = 1.0
    history_recency_bias: float = 0.0
    history_quality_bias: float = 0.0
    secondary_fraction: float = 0.0
    secondary_lock_probability: float = 0.0
    secondary_top_fraction: float = 1.0

    NAMES = (
        "gamma",
        "n_ei_candidates",
        "history_keep_fraction",
        "history_recency_bias",
        "history_quality_bias",
        "secondary_fraction",
        "secondary_lock_probability",
        "secondary_top_fraction",
    )

    @classmethod
    def anchor(cls) -> "ATPE2Action":
        return cls()

    def vector(self) -> np.ndarray:
        return np.asarray([float(getattr(self, name)) for name in self.NAMES], dtype=float)

    def vector_for(self, names: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray([float(getattr(self, name)) for name in names], dtype=float)


class ATPE2Controller:
    """Select and cache one complete action for each five-completion block."""

    def __init__(self, seed: int, ensemble: ATPE2ModelEnsemble | None = None):
        self.seed = int(seed)
        self.ensemble = ensemble if ensemble is not None else ATPE2ModelEnsemble.load_default()
        self.extractor = (
            LegacyATPEFeatureExtractor()
            if self.ensemble.manifest.get("feature_schema_version")
            == LegacyATPEFeatureExtractor.version
            else ATPE2FeatureExtractor()
        )
        self._cached_boundary: int | None = None
        self._cached_action = ATPE2Action.anchor()
        self.last_record: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def action_for(
        self,
        space: dict[str, Any],
        results: list[dict[str, Any]],
        current_trials: list[dict[str, Any]],
    ) -> ATPE2Action:
        completed = len(results)
        boundary = completed // 5
        if self._cached_boundary == boundary:
            return self._cached_action

        context = self.extractor.vector(space, results, current_trials)
        if not self.ensemble.research_ready:
            raise RuntimeError(
                "ATPE2 requires a fitted, schema-compatible eight-model ensemble"
            )
        candidates = self._candidate_actions(boundary)
        matrix = np.column_stack(
            (
                np.repeat(context[None, :], len(candidates), axis=0),
                np.asarray(
                    [
                        candidate.vector_for(self.ensemble.action_names)
                        for candidate in candidates
                    ]
                ),
            )
        )
        means, uncertainties = self.ensemble.predict(matrix)
        utility = means - self.ensemble.uncertainty_penalty * uncertainties
        if not np.all(np.isfinite(utility)):
            raise FloatingPointError("ATPE2 produced non-finite risk-adjusted utility")
        index = int(np.argmax(utility))
        selected = candidates[index]
        expected_utility = float(means[index])
        risk_adjusted_expected_utility = float(utility[index])
        uncertainty = float(uncertainties[index])
        outside_training_envelope = not self.ensemble.in_envelope(context)

        self._cached_boundary = boundary
        self._cached_action = selected
        self.last_record = {
            "completed_trials": completed,
            "boundary": boundary,
            "configuration": asdict(selected),
            "expected_utility": expected_utility,
            "risk_adjusted_expected_utility": risk_adjusted_expected_utility,
            "uncertainty": uncertainty,
            "fallback_reason": None,
            "decision_mode": "adaptive_model",
            "outside_training_envelope": outside_training_envelope,
            "seed": self.seed,
            "model_version": self.ensemble.manifest.get("model_version", MODEL_SCHEMA_VERSION),
            "controller_policy_version": CONTROLLER_POLICY_VERSION,
            "feature_schema_version": self.extractor.version,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "feature_schema_hash": self.ensemble.manifest.get("feature_schema_hash"),
            "action_schema_hash": self.ensemble.manifest.get("action_schema_hash"),
        }
        self.records.append(self.last_record)
        return selected

    def _candidate_actions(self, boundary: int) -> list[ATPE2Action]:
        sampler = qmc.Sobol(d=8, scramble=True, seed=self._derived_seed("sobol", boundary))
        points = sampler.random_base2(
            m=int(math.log2(self.ensemble.runtime_candidate_count))
        )
        ranges = {
            name: {
                **ACTION_DOMAINS[name],
                **self.ensemble.manifest.get("action_ranges", {}).get(name, {}),
            }
            for name in ATPE2Action.NAMES
        }

        def sample(name: str, coordinate: float) -> float | int:
            domain = ranges[name]
            minimum, maximum = float(domain["minimum"]), float(domain["maximum"])
            if domain.get("scale") == "log":
                value = math.exp(
                    math.log(minimum)
                    + coordinate * (math.log(maximum) - math.log(minimum))
                )
            else:
                value = minimum + coordinate * (maximum - minimum)
            if domain.get("type") == "integer":
                return int(np.clip(round(value), int(minimum), int(maximum)))
            return float(value)

        actions = []
        for point in points:
            actions.append(
                ATPE2Action(
                    **{
                        name: sample(name, float(point[index]))
                        for index, name in enumerate(ATPE2Action.NAMES)
                    }
                )
            )
        actions.append(ATPE2Action.anchor())
        constants = self.ensemble.manifest.get("validation_selected_constants", {})
        constants = {
            **constants,
            **self.ensemble.manifest.get("ablation_action_overrides", {}),
        }
        if constants and self.ensemble.research_ready:
            actions = [ATPE2Action(**{**asdict(action), **constants}) for action in actions]
        return actions

    def filter_history(
        self, results: list[dict[str, Any]], action: ATPE2Action, boundary: int
    ) -> list[dict[str, Any]]:
        successes = successful_results(results)
        if not successes:
            return []
        keep = min(len(successes), max(1, int(math.ceil(action.history_keep_fraction * len(successes)))))
        if keep == len(successes):
            return list(successes)
        age = np.linspace(0.0, 1.0, len(successes))
        losses = np.asarray([float(result["loss"]) for result in successes])
        quality = 1.0 - (rankdata(losses, method="average") - 1.0) / max(1, len(losses) - 1)
        log_weights = action.history_recency_bias * age + action.history_quality_bias * quality
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        rng = np.random.default_rng(self._derived_seed("history", boundary))
        indices = rng.choice(len(successes), size=keep, replace=False, p=weights)
        # Preserve chronology after weighted selection because Hyperopt trial order is meaningful.
        return [successes[index] for index in sorted(indices.tolist())]

    def locking_values(
        self,
        space: dict[str, Any],
        results: list[dict[str, Any]],
        action: ATPE2Action,
        boundary: int,
    ) -> dict[str, Any]:
        successes = successful_results(results)
        descriptors = describe_space(space)
        if not successes or action.secondary_fraction <= 0 or action.secondary_lock_probability <= 0:
            return {}
        relevance = parameter_relevance(descriptors, successes)
        lockable = [descriptor for descriptor in descriptors if relevance[descriptor.name] < 1.0]
        eligible_count = int(math.ceil(action.secondary_fraction * len(descriptors)))
        eligible_names = {
            descriptor.name
            for descriptor in sorted(lockable, key=lambda descriptor: (relevance[descriptor.name], descriptor.name))[:eligible_count]
        }
        ordered = sorted(descriptors, key=lambda descriptor: (not descriptor.is_selector, descriptor.depth, descriptor.name))
        rng = np.random.default_rng(self._derived_seed("locking", boundary))
        locked: dict[str, Any] = {}

        for descriptor in ordered:
            if descriptor.name not in eligible_names or rng.random() > action.secondary_lock_probability:
                continue
            if not self._conditions_compatible(descriptor, locked):
                continue
            compatible = [
                result
                for result in successes
                if result.get(descriptor.name) not in {None, ""}
                and all(result.get(selector) == branch for selector, branch in descriptor.conditions)
                and self._is_complete_configuration(result, descriptors)
                and all(result.get(name) == value for name, value in locked.items())
            ]
            if not compatible:
                continue
            compatible.sort(key=lambda result: float(result["loss"]))
            top_count = max(1, int(math.ceil(action.secondary_top_fraction * len(compatible))))
            source = compatible[int(rng.integers(0, top_count))]
            locked[descriptor.name] = source[descriptor.name]
        return locked

    @staticmethod
    def _conditions_compatible(descriptor: ParameterDescriptor, locked: dict[str, Any]) -> bool:
        return all(selector not in locked or locked[selector] == branch for selector, branch in descriptor.conditions)

    @staticmethod
    def _is_complete_configuration(
        result: dict[str, Any], descriptors: list[ParameterDescriptor]
    ) -> bool:
        for candidate in descriptors:
            active_branch = all(result.get(selector) == branch for selector, branch in candidate.conditions)
            if active_branch and result.get(candidate.name) in {None, ""}:
                return False
        return True

    def _derived_seed(self, purpose: str, boundary: int) -> int:
        payload = json.dumps([self.seed, purpose, boundary], separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
