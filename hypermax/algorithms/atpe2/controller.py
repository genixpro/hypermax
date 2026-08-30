"""Joint ATPE2 controller action selection and history transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
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
FACTORIZED_CONTROLLER_POLICY_VERSION = "atpe2-factorized-v1"
ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "search": ("gamma", "n_ei_candidates"),
    "history": (
        "history_keep_fraction",
        "history_recency_bias",
        "history_quality_bias",
    ),
    "locking": (
        "secondary_fraction",
        "secondary_lock_probability",
        "secondary_top_fraction",
    ),
}


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
        extractor_class = getattr(self.ensemble, "feature_extractor_class", None)
        self.extractor = (
            extractor_class()
            if extractor_class is not None
            else (
                LegacyATPEFeatureExtractor()
                if self.ensemble.manifest.get("feature_schema_version")
                == LegacyATPEFeatureExtractor.version
                else ATPE2FeatureExtractor()
            )
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
        factorized = bool(getattr(self.ensemble, "is_factorized", False))
        if factorized:
            selected, decision = self._factorized_action(context, boundary)
            expected_utility = float(decision["expected_utility"])
            risk_adjusted_expected_utility = float(decision["score"])
            uncertainty = float(decision["ensemble_disagreement"])
        else:
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
            decision = {}
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
            "controller_policy_version": (
                FACTORIZED_CONTROLLER_POLICY_VERSION
                if factorized
                else CONTROLLER_POLICY_VERSION
            ),
            "feature_schema_version": self.extractor.version,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "feature_schema_hash": self.ensemble.manifest.get("feature_schema_hash"),
            "action_schema_hash": self.ensemble.manifest.get("action_schema_hash"),
            **decision,
        }
        self.records.append(self.last_record)
        return selected

    def _factorized_action(
        self, context: np.ndarray, boundary: int
    ) -> tuple[ATPE2Action, dict[str, Any]]:
        runtime = self.ensemble.manifest.get("runtime", {})
        retained = max(1, min(6, int(runtime.get("retained_per_group", 6))))
        max_support = float(runtime.get("max_supported_distance", math.inf))
        group_weights = {
            **{name: 1.0 for name in ACTION_GROUPS},
            **self.ensemble.manifest.get("score_composition", {}).get(
                "group_weights", {}
            ),
        }
        group_records: dict[str, dict[str, Any]] = {}
        retained_actions: dict[str, list[tuple[ATPE2Action, float, float, float]]] = {}
        anchor = ATPE2Action.anchor()
        exploration = self._seeded_full_action(boundary)
        exploration_advantages: dict[str, float] = {}
        exploration_disagreements: dict[str, float] = {}

        for group, names in ACTION_GROUPS.items():
            proposals = self._group_proposals(group, boundary)
            scoring_proposals = proposals + [exploration]
            group_matrix = np.column_stack(
                (
                    np.repeat(context[None, :], len(scoring_proposals), axis=0),
                    np.asarray(
                        [proposal.vector_for(names) for proposal in scoring_proposals]
                    ),
                )
            )
            advantages, disagreements = self.ensemble.predict_group(group, group_matrix)
            exploration_advantages[group] = float(advantages[-1])
            exploration_disagreements[group] = float(disagreements[-1])
            advantages = advantages[:-1]
            disagreements = disagreements[:-1]
            support_matrix = self._full_matrix(context, proposals)
            support_costs = self.ensemble.support_cost(support_matrix)
            ranked = sorted(
                range(len(proposals)),
                key=lambda index: (
                    -float(advantages[index]),
                    self._action_key(proposals[index]),
                ),
            )
            supported = [index for index in ranked if support_costs[index] <= max_support]
            if not supported:
                supported = ranked[:1]
            selected_indices = supported[:retained]
            retained_actions[group] = [
                (
                    proposals[index],
                    float(advantages[index]),
                    float(disagreements[index]),
                    float(support_costs[index]),
                )
                for index in selected_indices
            ]
            winner = ranked[0]
            group_records[group] = {
                "inactive": proposals[winner].vector_for(names).tolist()
                == anchor.vector_for(names).tolist(),
                "winning_advantage": float(advantages[winner]),
                "winning_disagreement": float(disagreements[winner]),
                "winning_support_cost": float(support_costs[winner]),
                "retained_proposals": len(selected_indices),
            }

        combinations: dict[str, tuple[ATPE2Action, dict[str, float], dict[str, float]]] = {}
        for entries in product(*(retained_actions[name] for name in ACTION_GROUPS)):
            values = asdict(anchor)
            advantages: dict[str, float] = {}
            disagreements: dict[str, float] = {}
            for group, entry in zip(ACTION_GROUPS, entries):
                proposal, advantage, disagreement, _ = entry
                for name in ACTION_GROUPS[group]:
                    values[name] = getattr(proposal, name)
                advantages[group] = advantage
                disagreements[group] = disagreement
            action = ATPE2Action(**values)
            combinations.setdefault(
                self._action_key(action), (action, advantages, disagreements)
            )
        ordered = sorted(
            combinations.values(),
            key=lambda entry: (
                -sum(
                    group_weights[group] * entry[1][group]
                    for group in ACTION_GROUPS
                ),
                self._action_key(entry[0]),
            ),
        )[: min(64, int(runtime.get("beam_width", 64)))]

        exploration_key = self._action_key(exploration)
        if exploration_key not in {self._action_key(entry[0]) for entry in ordered}:
            ordered.append((exploration, exploration_advantages, exploration_disagreements))
        candidates = [entry[0] for entry in ordered]
        full_matrix = self._full_matrix(context, candidates)
        residual, residual_disagreement = self.ensemble.predict_residual(full_matrix)
        downside_heads = self.ensemble.predict_downside(full_matrix)
        downside = np.clip(
            downside_heads["harm"]
            + 0.5 * downside_heads["variability"]
            + downside_heads["failure"],
            0.0,
            1.0,
        )
        support = self.ensemble.support_cost(full_matrix)
        group_components = {
            group: np.asarray([entry[1][group] for entry in ordered], dtype=float)
            for group in ACTION_GROUPS
        }
        composition = self.ensemble.manifest.get("score_composition", {})
        score = np.zeros(len(candidates), dtype=float)
        for group in ACTION_GROUPS:
            score += group_weights[group] * self._normalize_component(
                group_components[group]
            )
        score += float(composition.get("residual_weight", 0.0)) * self._normalize_component(residual)
        score -= float(composition.get("downside_weight", 0.0)) * self._normalize_component(downside)
        score -= float(composition.get("support_weight", 0.0)) * self._normalize_component(support)
        if not np.all(np.isfinite(score)):
            raise FloatingPointError("ATPE2 factorized policy produced non-finite scores")
        index = int(np.argmax(score))
        selected = candidates[index]
        group_sum = sum(
            group_weights[group] * group_components[group][index]
            for group in ACTION_GROUPS
        )
        expected_utility = float(group_sum + float(composition.get("residual_weight", 0.0)) * residual[index])
        ensemble_disagreement = float(
            np.mean(
                [
                    *[ordered[index][2][group] for group in ACTION_GROUPS],
                    residual_disagreement[index],
                ]
            )
        )
        return selected, {
            "score": float(score[index]),
            "expected_utility": expected_utility,
            "ensemble_disagreement": ensemble_disagreement,
            "support_cost": float(support[index]),
            "catastrophic_probability": float(
                np.clip(downside_heads["catastrophic"][index], 0.0, 1.0)
            ),
            "downside_prediction": float(downside[index]),
            "candidate_count": len(candidates),
            "group_decisions": group_records,
        }

    def _group_proposals(self, group: str, boundary: int) -> list[ATPE2Action]:
        names = ACTION_GROUPS[group]
        sampler = qmc.Sobol(
            d=len(names), scramble=True, seed=self._derived_seed(f"group-{group}", boundary)
        )
        points = sampler.random_base2(m=4)
        actions = [self._action_from_coordinates(names, point) for point in points]
        actions.append(ATPE2Action.anchor())
        unique: dict[str, ATPE2Action] = {}
        for action in actions:
            unique.setdefault(self._action_key(action), action)
        refill = 0
        while len(unique) < 17:
            fallback = qmc.Sobol(
                d=len(names),
                scramble=True,
                seed=self._derived_seed(f"group-refill-{group}-{refill}", boundary),
            ).random(1)[0]
            action = self._action_from_coordinates(names, fallback)
            unique.setdefault(self._action_key(action), action)
            refill += 1
            if refill > 4096:
                raise RuntimeError(f"Could not construct 17 unique {group} proposals")
        return list(unique.values())[:17]

    def _seeded_full_action(self, boundary: int) -> ATPE2Action:
        point = qmc.Sobol(
            d=len(ATPE2Action.NAMES),
            scramble=True,
            seed=self._derived_seed("full-exploration", boundary),
        ).random(1)[0]
        return self._action_from_coordinates(ATPE2Action.NAMES, point)

    def _action_from_coordinates(
        self, names: tuple[str, ...] | list[str], coordinates: np.ndarray
    ) -> ATPE2Action:
        values = asdict(ATPE2Action.anchor())
        ranges = {
            name: {
                **ACTION_DOMAINS[name],
                **self.ensemble.manifest.get("action_ranges", {}).get(name, {}),
            }
            for name in names
        }
        for name, coordinate in zip(names, coordinates):
            domain = ranges[name]
            minimum, maximum = float(domain["minimum"]), float(domain["maximum"])
            if domain.get("scale") == "log":
                value = math.exp(
                    math.log(minimum)
                    + float(coordinate) * (math.log(maximum) - math.log(minimum))
                )
            else:
                value = minimum + float(coordinate) * (maximum - minimum)
            values[name] = (
                int(np.clip(round(value), int(minimum), int(maximum)))
                if domain.get("type") == "integer"
                else float(value)
            )
        return ATPE2Action(**values)

    @staticmethod
    def _normalize_component(values: np.ndarray) -> np.ndarray:
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

    def _full_matrix(
        self, context: np.ndarray, actions: list[ATPE2Action]
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.repeat(context[None, :], len(actions), axis=0),
                np.asarray([action.vector() for action in actions]),
            )
        )

    @staticmethod
    def _action_key(action: ATPE2Action) -> str:
        return json.dumps(asdict(action), sort_keys=True, separators=(",", ":"))

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
