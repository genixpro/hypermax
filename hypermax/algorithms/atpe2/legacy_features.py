"""Finite reproduction of the legacy 113-feature ATPE context for ablation only."""

from __future__ import annotations

import math
from typing import Any, Iterable
import warnings

import numpy as np
from scipy import stats

from hypermax.hyperparameter import Hyperparameter
from .features import successful_results


LEGACY_FEATURE_SCHEMA_VERSION = "atpe2-legacy-context-v1"
_PREFIXES = (
    "all",
    "recent_10",
    "recent_15%",
    "recent_25",
    "top_10%",
    "top_20%",
    "top_30%",
)
_METRICS = (
    "correlation_best_percentile25_ratio",
    "correlation_best_percentile50_ratio",
    "correlation_best_percentile75_ratio",
    "correlation_kurtosis",
    "correlation_percentile5_percentile25_ratio",
    "correlation_skew",
    "correlation_stddev_best_ratio",
    "correlation_stddev_median_ratio",
    "loss_best_percentile25_ratio",
    "loss_best_percentile50_ratio",
    "loss_best_percentile75_ratio",
    "loss_kurtosis",
    "loss_percentile5_percentile25_ratio",
    "loss_skew",
    "loss_stddev_best_ratio",
    "loss_stddev_median_ratio",
)


class LegacyATPEFeatureExtractor:
    """The old ratio-heavy layout, made finite so an ablation can be trained."""

    FEATURE_NAMES = tuple(
        sorted(
            [f"{prefix}_{metric}" for prefix in _PREFIXES for metric in _METRICS]
            + ["log10_cardinality"]
        )
    )
    version = LEGACY_FEATURE_SCHEMA_VERSION

    @classmethod
    def schema_definition(cls) -> dict[str, Any]:
        return {"version": cls.version, "names": list(cls.FEATURE_NAMES), "dtype": "float64"}

    def extract(
        self,
        space: dict[str, Any],
        results: Iterable[dict[str, Any]],
        current_trials: Iterable[dict[str, Any]] = (),
    ) -> dict[str, float]:
        del current_trials
        successes = successful_results(results)
        losses = np.asarray([float(row["loss"]) for row in successes], dtype=float)
        subsets: dict[str, list[dict[str, Any]]] = {"all": successes}
        for percentile in (10, 20, 30):
            if losses.size:
                threshold = float(np.percentile(losses, percentile))
                subsets[f"top_{percentile}%"] = [
                    row for row in successes if float(row["loss"]) <= threshold
                ]
            else:
                subsets[f"top_{percentile}%"] = []
        subsets["recent_10"] = successes[-min(10, len(successes)) :]
        subsets["recent_25"] = successes[-min(25, len(successes)) :]
        recent_fraction = min(len(successes), max(5, math.ceil(len(successes) * 0.15)))
        subsets["recent_15%"] = successes[-recent_fraction:] if recent_fraction else []

        output = {"log10_cardinality": self._finite(Hyperparameter(space).getLog10Cardinality())}
        for prefix in _PREFIXES:
            partial = self._partial(space, subsets[prefix])
            for metric, value in partial.items():
                output[f"{prefix}_{metric}"] = self._finite(value)
        return {name: output.get(name, 0.0) for name in self.FEATURE_NAMES}

    def vector(self, *args: Any, **kwargs: Any) -> np.ndarray:
        features = self.extract(*args, **kwargs)
        return np.asarray([features[name] for name in self.FEATURE_NAMES], dtype=float)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    @classmethod
    def _distribution(cls, values: np.ndarray, best_high: bool) -> dict[str, float]:
        if values.size == 0 or len(np.unique(values)) < 2:
            return {
                "skew": 0.0,
                "kurtosis": 0.0,
                "stddev_median_ratio": 0.0,
                "stddev_best_ratio": 0.0,
                "best_percentile25_ratio": 0.0,
                "best_percentile50_ratio": 0.0,
                "best_percentile75_ratio": 0.0,
                "percentile5_percentile25_ratio": 0.0,
            }
        if best_high:
            best, p5, p25, p50, p75 = np.percentile(values, [100, 95, 75, 50, 25])
        else:
            best, p5, p25, p50, p75 = np.percentile(values, [0, 5, 25, 50, 75])
        stddev = float(np.std(values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))
        return {
            "skew": skew,
            "kurtosis": kurtosis,
            "stddev_median_ratio": cls._ratio(stddev, float(p50)),
            "stddev_best_ratio": cls._ratio(stddev, float(best)),
            "best_percentile25_ratio": cls._ratio(float(best), float(p25)),
            "best_percentile50_ratio": cls._ratio(float(best), float(p50)),
            "best_percentile75_ratio": cls._ratio(float(best), float(p75)),
            "percentile5_percentile25_ratio": cls._ratio(float(p5), float(p25)),
        }

    @classmethod
    def _partial(cls, space: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
        losses = np.asarray([float(row["loss"]) for row in rows], dtype=float)
        loss_stats = cls._distribution(losses, best_high=False)
        correlations = []
        for parameter in Hyperparameter(space).getFlatParameters():
            if parameter.config.get("type") != "number":
                continue
            pairs = [
                (float(row[parameter.name]), float(row["loss"]))
                for row in rows
                if row.get(parameter.name) not in {None, ""}
            ]
            if len(pairs) < 2 or len({value for value, _ in pairs}) < 2:
                correlations.append(0.0)
                continue
            coefficient = stats.spearmanr(
                [value for value, _ in pairs], [loss for _, loss in pairs]
            ).statistic
            correlations.append(
                abs(float(coefficient)) if math.isfinite(float(coefficient)) else 0.0
            )
        correlation_stats = cls._distribution(np.asarray(correlations), best_high=True)
        return {
            **{f"loss_{name}": value for name, value in loss_stats.items()},
            **{f"correlation_{name}": value for name, value in correlation_stats.items()},
        }

    @staticmethod
    def _finite(value: Any) -> float:
        try:
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0
