"""Finite, loss-sign-agnostic context features and parameter relevance.

The old ATPE feature set contains ratios whose denominators can be zero and whose
meaning changes when a constant is added to every loss.  This module deliberately
uses ranks and robust standardized differences instead.  The public feature vector
has a fixed width so a frozen controller can be packaged independently of a search
space's number of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np
from scipy import stats


FEATURE_SCHEMA_VERSION = "atpe2-context-v2"
FEATURE_SCHEMA_V4_VERSION = "atpe2-context-v4"


@dataclass(frozen=True)
class ParameterDescriptor:
    name: str
    kind: str
    conditions: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    cardinality: float = 20.0
    depth: int = 0

    @property
    def is_selector(self) -> bool:
        return self.kind == "selector"


def describe_space(space: dict[str, Any]) -> list[ParameterDescriptor]:
    """Flatten a Hypermax schema, retaining conditional branch requirements."""

    descriptors: list[ParameterDescriptor] = []

    def walk(
        node: dict[str, Any],
        path: str,
        conditions: tuple[tuple[str, int], ...],
        depth: int,
    ) -> None:
        alternatives = node.get("anyOf", node.get("oneOf"))
        if alternatives is not None:
            if path:
                descriptors.append(
                    ParameterDescriptor(path, "selector", conditions, len(alternatives), depth)
                )
            for index, child in enumerate(alternatives):
                child_path = f"{path}.{index}" if path else str(index)
                child_conditions = conditions + (((path, index),) if path else ())
                walk(child, child_path, child_conditions, depth + 1)
            return

        if "enum" in node:
            descriptors.append(
                ParameterDescriptor(path, "categorical", conditions, len(node["enum"]), depth)
            )
            return
        if "constant" in node:
            return
        if node.get("type") == "object":
            for key, child in node.get("properties", {}).items():
                walk(child, f"{path}.{key}" if path else key, conditions, depth)
            return
        if node.get("type") == "number":
            integer = node.get("mode") == "randint"
            rounding = node.get("rounding")
            if rounding is not None and float(rounding).is_integer():
                integer = True
            cardinality = 20.0
            if node.get("mode") == "randint":
                cardinality = max(1.0, float(node.get("max", 1)))
            elif rounding and "min" in node and "max" in node:
                cardinality = max(1.0, (float(node["max"]) - float(node["min"])) / float(rounding) + 1)
            descriptors.append(
                ParameterDescriptor(path, "integer" if integer else "numeric", conditions, cardinality, depth)
            )

    walk(space, "", (), 0)
    return descriptors


def successful_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for result in results:
        status = str(result.get("status", "ok")).lower()
        loss = result.get("loss")
        if status in {"ok", "success", "completed"} and _finite_number(loss):
            successful.append(result)
    return successful


def _finite_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_active(result: dict[str, Any], descriptor: ParameterDescriptor) -> bool:
    value = result.get(descriptor.name)
    if value is None or value == "":
        return False
    return all(result.get(selector) == branch for selector, branch in descriptor.conditions)


def _correlation_ratio(categories: list[Any], values: np.ndarray) -> float:
    if len(categories) < 2 or len(set(map(repr, categories))) < 2:
        return 0.0
    overall = float(np.mean(values))
    denominator = float(np.sum((values - overall) ** 2))
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    category_array = np.asarray([repr(value) for value in categories], dtype=object)
    for category in np.unique(category_array):
        group = values[category_array == category]
        numerator += len(group) * float(np.mean(group) - overall) ** 2
    return float(np.clip(math.sqrt(numerator / denominator), 0.0, 1.0))


def parameter_relevance(
    descriptors: list[ParameterDescriptor], results: Iterable[dict[str, Any]]
) -> dict[str, float]:
    """Return conservative relevance values in ``[0, 1]``.

    A value of one means "do not lock".  Conditional parameters combine active
    value relevance with the relevance of being active at all.
    """

    successes = successful_results(results)
    if not successes:
        return {descriptor.name: 1.0 for descriptor in descriptors}
    losses = np.asarray([float(result["loss"]) for result in successes], dtype=float)
    loss_ranks = stats.rankdata(losses, method="average")
    relevance: dict[str, float] = {}

    for descriptor in descriptors:
        active_mask = np.asarray([_is_active(result, descriptor) for result in successes])
        active_indices = np.flatnonzero(active_mask)
        active_values = [successes[index].get(descriptor.name) for index in active_indices]
        distinct = len(set(map(repr, active_values)))
        if len(active_indices) < 5 or distinct < 2:
            relevance[descriptor.name] = 1.0
            continue

        active_ranks = loss_ranks[active_indices]
        if descriptor.kind in {"categorical", "selector"}:
            value_relevance = _correlation_ratio(active_values, active_ranks)
        else:
            try:
                numeric_values = np.asarray(active_values, dtype=float)
                if len(np.unique(numeric_values)) < 2 or len(np.unique(active_ranks)) < 2:
                    value_relevance = 0.0
                else:
                    coefficient = stats.spearmanr(numeric_values, active_ranks).statistic
                    value_relevance = abs(float(coefficient)) if math.isfinite(float(coefficient)) else 0.0
            except (TypeError, ValueError):
                value_relevance = 0.0

        activation_relevance = 0.0
        if descriptor.conditions and active_mask.any() and (~active_mask).any():
            activation_relevance = _correlation_ratio(active_mask.tolist(), loss_ranks)
        relevance[descriptor.name] = float(np.clip(max(value_relevance, activation_relevance), 0.0, 1.0))
    return relevance


class ATPE2FeatureExtractor:
    """Create the fixed-width ATPE2 context vector."""

    FEATURE_NAMES = (
        "numeric_count",
        "integer_count",
        "categorical_count",
        "selector_count",
        "effective_active_dimension",
        "max_branch_depth",
        "log1p_cardinality",
        "conditional_fraction",
        "successful_count",
        "failed_count",
        "pending_count",
        "failure_rate",
        "duplication_rate",
        "active_coverage_mean",
        "active_coverage_min",
        "loss_iqr_scaled",
        "loss_mad_scaled",
        "loss_recent_rank_mean_5",
        "loss_recent_rank_mean_10",
        "loss_recent_robust_z_5",
        "loss_recent_robust_z_10",
        "improvement_slope_5",
        "improvement_slope_10",
        "stagnation_fraction",
        "relevance_mean",
        "relevance_q25",
        "relevance_q75",
        "activity_mean",
        "activity_min",
        "missing_loss_indicator",
        "constant_loss_indicator",
        "small_history_indicator",
    )

    version = FEATURE_SCHEMA_VERSION

    @classmethod
    def schema_definition(cls) -> dict[str, Any]:
        return {"version": cls.version, "names": list(cls.FEATURE_NAMES), "dtype": "float64"}

    def extract(
        self,
        space: dict[str, Any],
        results: Iterable[dict[str, Any]],
        current_trials: Iterable[dict[str, Any]] = (),
    ) -> dict[str, float]:
        results = list(results)
        descriptors = describe_space(space)
        successes = successful_results(results)
        success_ids = {id(result) for result in successes}
        failures = [result for result in results if id(result) not in success_ids]
        losses = np.asarray([float(result["loss"]) for result in successes], dtype=float)
        count = len(results)

        activities = np.asarray(
            [
                np.mean([_is_active(result, descriptor) for result in successes])
                if successes
                else 0.0
                for descriptor in descriptors
            ],
            dtype=float,
        )
        active_dimensions = [
            sum(_is_active(result, descriptor) and not descriptor.is_selector for descriptor in descriptors)
            for result in successes
        ]
        effective_dimension = float(np.mean(active_dimensions)) if active_dimensions else float(
            sum(not descriptor.conditions and not descriptor.is_selector for descriptor in descriptors)
        )
        cardinality_log = sum(math.log1p(max(1.0, descriptor.cardinality)) for descriptor in descriptors)

        unique_losses = len(set(losses.tolist())) if losses.size else 0
        duplicate_rate = 1.0 - unique_losses / len(losses) if losses.size else 0.0
        loss_iqr_scaled, loss_mad_scaled, constant = self._dispersion(losses)
        recent_rank_5 = self._recent_rank_mean(losses, 5)
        recent_rank_10 = self._recent_rank_mean(losses, 10)
        recent_robust_z_5 = self._recent_robust_difference(losses, 5)
        recent_robust_z_10 = self._recent_robust_difference(losses, 10)
        slope_5 = self._improvement_slope(losses, 5)
        slope_10 = self._improvement_slope(losses, 10)
        stagnation = self._stagnation(losses)
        relevances = np.asarray(list(parameter_relevance(descriptors, successes).values()), dtype=float)

        values = {
            "numeric_count": sum(descriptor.kind == "numeric" for descriptor in descriptors),
            "integer_count": sum(descriptor.kind == "integer" for descriptor in descriptors),
            "categorical_count": sum(descriptor.kind == "categorical" for descriptor in descriptors),
            "selector_count": sum(descriptor.kind == "selector" for descriptor in descriptors),
            "effective_active_dimension": effective_dimension,
            "max_branch_depth": max((descriptor.depth for descriptor in descriptors), default=0),
            "log1p_cardinality": cardinality_log,
            "conditional_fraction": np.mean([bool(descriptor.conditions) for descriptor in descriptors]) if descriptors else 0,
            "successful_count": len(successes),
            "failed_count": len(failures),
            "pending_count": len(list(current_trials)),
            "failure_rate": len(failures) / count if count else 0,
            "duplication_rate": duplicate_rate,
            "active_coverage_mean": float(np.mean(activities)) if activities.size else 0,
            "active_coverage_min": float(np.min(activities)) if activities.size else 0,
            "loss_iqr_scaled": loss_iqr_scaled,
            "loss_mad_scaled": loss_mad_scaled,
            "loss_recent_rank_mean_5": recent_rank_5,
            "loss_recent_rank_mean_10": recent_rank_10,
            "loss_recent_robust_z_5": recent_robust_z_5,
            "loss_recent_robust_z_10": recent_robust_z_10,
            "improvement_slope_5": slope_5,
            "improvement_slope_10": slope_10,
            "stagnation_fraction": stagnation,
            "relevance_mean": float(np.mean(relevances)) if relevances.size else 1,
            "relevance_q25": float(np.quantile(relevances, 0.25)) if relevances.size else 1,
            "relevance_q75": float(np.quantile(relevances, 0.75)) if relevances.size else 1,
            "activity_mean": float(np.mean(activities)) if activities.size else 0,
            "activity_min": float(np.min(activities)) if activities.size else 0,
            "missing_loss_indicator": float(bool(failures)),
            "constant_loss_indicator": float(constant),
            "small_history_indicator": float(len(successes) < 12),
        }
        return {
            name: self._finite(values[name])
            for name in ATPE2FeatureExtractor.FEATURE_NAMES
        }

    def vector(self, *args: Any, **kwargs: Any) -> np.ndarray:
        features = self.extract(*args, **kwargs)
        return np.asarray([features[name] for name in self.FEATURE_NAMES], dtype=float)

    @staticmethod
    def _finite(value: Any) -> float:
        try:
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _dispersion(losses: np.ndarray) -> tuple[float, float, bool]:
        if losses.size < 2:
            return 0.0, 0.0, True
        # Exact equality is the only scale-independent constant-history test.
        # Normalize before subtraction so +/-1e308 histories cannot overflow;
        # ratios of centered spreads remain translation- and scale-invariant.
        constant = bool(np.all(losses == losses[0]))
        if constant:
            return 0.0, 0.0, True
        magnitude = float(np.max(np.abs(losses)))
        normalized = losses / magnitude if magnitude > 0 else losses
        median = float(np.median(normalized))
        absolute = np.abs(normalized - median)
        mad = float(np.median(absolute))
        q25, q75 = np.quantile(normalized, [0.25, 0.75])
        iqr = float(q75 - q25)
        spread = float(np.max(normalized) - np.min(normalized))
        if not math.isfinite(spread) or spread <= 0:
            return 0.0, 0.0, True
        return iqr / spread, mad / spread, False

    @staticmethod
    def _recent_rank_mean(losses: np.ndarray, window: int) -> float:
        if losses.size == 0:
            return 0.5
        ranks = stats.rankdata(losses, method="average") / len(losses)
        return float(np.mean(ranks[-min(window, len(ranks)) :]))

    @staticmethod
    def _recent_robust_difference(losses: np.ndarray, window: int) -> float:
        """Robustly standardize a recent median without assuming positive loss."""
        if losses.size < 2 or np.all(losses == losses[0]):
            return 0.0
        magnitude = float(np.max(np.abs(losses)))
        normalized = losses / magnitude if magnitude > 0 else losses
        center = float(np.median(normalized))
        deviations = np.abs(normalized - center)
        mad_scale = 1.4826 * float(np.median(deviations))
        q25, q75 = np.quantile(normalized, [0.25, 0.75])
        scale = max(float(q75 - q25), mad_scale)
        if not math.isfinite(scale) or scale <= 0:
            return 0.0
        recent = float(np.median(normalized[-min(window, len(normalized)) :]))
        value = (recent - center) / scale
        return float(np.clip(value, -20.0, 20.0)) if math.isfinite(value) else 0.0

    @staticmethod
    def _improvement_slope(losses: np.ndarray, window: int) -> float:
        if losses.size < 2:
            return 0.0
        incumbents = np.minimum.accumulate(losses)
        ranks = stats.rankdata(losses, method="average") / len(losses)
        incumbent_ranks = np.minimum.accumulate(ranks)
        values = incumbent_ranks[-min(window, len(losses)) :]
        if len(values) < 2:
            return 0.0
        return float(np.polyfit(np.linspace(0, 1, len(values)), values, 1)[0])

    @staticmethod
    def _stagnation(losses: np.ndarray) -> float:
        if losses.size < 2:
            return 0.0
        incumbents = np.minimum.accumulate(losses)
        changes = np.count_nonzero(np.diff(incumbents) < 0)
        return float(1.0 - changes / (len(losses) - 1))


class ATPE2ContextV4FeatureExtractor(ATPE2FeatureExtractor):
    """The v2 context plus local relevance, progress, and support summaries.

    V2 remains a separate class because its frozen manifests are part of the
    runtime compatibility contract.  V4 deliberately appends features so an
    older model can never silently consume a different column layout.
    """

    V4_FEATURE_NAMES = (
        "recent_10_relevance_mean",
        "recent_25pct_relevance_mean",
        "top_quartile_relevance_mean",
        "global_recent_relevance_stability",
        "recent_10_improvement_density",
        "recent_25pct_improvement_density",
        "incumbent_age_fraction",
        "active_support_min",
        "active_support_q25",
    )
    FEATURE_NAMES = ATPE2FeatureExtractor.FEATURE_NAMES + V4_FEATURE_NAMES
    version = FEATURE_SCHEMA_V4_VERSION

    def extract(
        self,
        space: dict[str, Any],
        results: Iterable[dict[str, Any]],
        current_trials: Iterable[dict[str, Any]] = (),
    ) -> dict[str, float]:
        results = list(results)
        current_trials = list(current_trials)
        base = super().extract(space, results, current_trials)
        descriptors = describe_space(space)
        successes = successful_results(results)
        recent_10 = successes[-min(10, len(successes)) :]
        recent_fraction_count = max(1, int(math.ceil(0.25 * len(successes)))) if successes else 0
        recent_fraction = successes[-recent_fraction_count:] if recent_fraction_count else []
        top_count = max(1, int(math.ceil(0.25 * len(successes)))) if successes else 0
        top_quartile = (
            sorted(successes, key=lambda row: float(row["loss"]))[:top_count]
            if top_count
            else []
        )

        global_relevance = parameter_relevance(descriptors, successes)
        recent_10_relevance = parameter_relevance(descriptors, recent_10)
        recent_fraction_relevance = parameter_relevance(descriptors, recent_fraction)
        top_relevance = parameter_relevance(descriptors, top_quartile)

        def relevance_mean(values: dict[str, float]) -> float:
            return float(np.mean(list(values.values()))) if values else 1.0

        common = sorted(set(global_relevance) & set(recent_10_relevance))
        stability = (
            1.0
            - float(
                np.mean(
                    [
                        abs(global_relevance[name] - recent_10_relevance[name])
                        for name in common
                    ]
                )
            )
            if common
            else 1.0
        )
        losses = np.asarray([float(row["loss"]) for row in successes], dtype=float)
        activities = np.asarray(
            [
                np.mean([_is_active(row, descriptor) for row in successes])
                if successes
                else 0.0
                for descriptor in descriptors
            ],
            dtype=float,
        )
        additions = {
            "recent_10_relevance_mean": relevance_mean(recent_10_relevance),
            "recent_25pct_relevance_mean": relevance_mean(recent_fraction_relevance),
            "top_quartile_relevance_mean": relevance_mean(top_relevance),
            "global_recent_relevance_stability": float(np.clip(stability, 0.0, 1.0)),
            "recent_10_improvement_density": self._improvement_density(losses, 10),
            "recent_25pct_improvement_density": self._improvement_density(
                losses, recent_fraction_count
            ),
            "incumbent_age_fraction": self._incumbent_age(losses),
            "active_support_min": float(np.min(activities)) if activities.size else 0.0,
            "active_support_q25": (
                float(np.quantile(activities, 0.25)) if activities.size else 0.0
            ),
        }
        return {
            name: self._finite({**base, **additions}[name])
            for name in self.FEATURE_NAMES
        }

    @staticmethod
    def _improvement_density(losses: np.ndarray, window: int) -> float:
        if losses.size < 2 or window < 2:
            return 0.0
        start = max(0, len(losses) - int(window))
        incumbents = np.minimum.accumulate(losses)
        changes = np.diff(incumbents)
        relevant = changes[max(0, start - 1) :]
        return float(np.mean(relevant < 0)) if relevant.size else 0.0

    @staticmethod
    def _incumbent_age(losses: np.ndarray) -> float:
        if losses.size < 2:
            return 0.0
        best = float(np.min(losses))
        latest_best = int(np.flatnonzero(losses == best)[-1])
        return float((len(losses) - 1 - latest_best) / (len(losses) - 1))


FEATURE_EXTRACTORS = {
    ATPE2FeatureExtractor.version: ATPE2FeatureExtractor,
    ATPE2ContextV4FeatureExtractor.version: ATPE2ContextV4FeatureExtractor,
}
