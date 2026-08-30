import random

import numpy as np
import pytest

from hypermax.algorithms.atpe_optimizer import ATPEOptimizer
from hypermax.hyperparameter import Hyperparameter
from hypermax.algorithms.random_search_optimizer import RandomSearchOptimizer
from hypermax.algorithms.tpe_optimizer import TPEOptimizer


SPACE = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "min": -2, "max": 2},
        "kind": {"enum": ["a", "b"]},
    },
}


@pytest.mark.parametrize("optimizer", [RandomSearchOptimizer(), TPEOptimizer()])
def test_standard_optimizers_sample_with_current_hyperopt(optimizer):
    sample = optimizer.recommendNextParameters(SPACE, [], [])

    assert -2 <= sample["x"] <= 2
    assert sample["kind"] in {"a", "b"}


def test_atpe_models_load_from_package_resources():
    optimizer = ATPEOptimizer()

    assert len(optimizer.parameterModels) == len(optimizer.atpeParameters)
    sample = optimizer.recommendNextParameters(SPACE, [], [])
    assert -2 <= sample["x"] <= 2


def test_legacy_atpe_remains_compatible_with_numpy2_and_categorical_history():
    rows = [
        {
            "trial": index,
            "status": "ok",
            "loss": float((index - 7) ** 2 - 10),
            "x": float((index % 10) / 3 - 1.5),
            "kind": ["a", "b"][index % 2],
        }
        for index in range(20)
    ]
    random.seed(1)
    np.random.seed(1)
    sample = ATPEOptimizer().recommendNextParameters(SPACE, rows, [])

    assert -2 <= sample["x"] <= 2
    assert sample["kind"] in {"a", "b"}


def test_legacy_atpe_reconstructs_conditional_selectors_from_history():
    space = {
        "type": "object",
        "properties": {
            "branch": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"left": {"type": "number", "min": -1, "max": 1}},
                    },
                    {
                        "type": "object",
                        "properties": {"right": {"type": "number", "min": 1, "max": 3}},
                    },
                ]
            }
        },
    }
    rows = [
        {
            "trial": index,
            "status": "ok",
            "loss": float((index % 5) - 2),
            "branch": index % 2,
            "branch.0.left": float(index % 3) / 2 if index % 2 == 0 else "",
            "branch.1.right": float(index % 3 + 1) if index % 2 else "",
        }
        for index in range(20)
    ]
    random.seed(3)
    np.random.seed(3)
    sample = ATPEOptimizer().recommendNextParameters(space, rows, [])
    assert sample["branch"]["$index"] in {0, 1}


def test_legacy_statistics_are_finite_for_zero_and_constant_correlations():
    optimizer = ATPEOptimizer()
    rows = [
        {"trial": index, "status": "ok", "loss": 0.0, "x": 1.0, "kind": "a"}
        for index in range(12)
    ]
    statistics = optimizer.computePartialResultStatistics(SPACE, rows)
    assert statistics
    assert all(np.isfinite(float(value)) for value in statistics.values())


def test_legacy_random_locking_reads_normal_and_randint_parameter_configs():
    optimizer = ATPEOptimizer()
    normal = Hyperparameter(
        {"type": "number", "mode": "normal", "mean": 0.0, "stddev": 1.0}
    )
    integer = Hyperparameter({"type": "number", "mode": "randint", "max": 4})
    assert np.isfinite(optimizer.chooseRandomValueForParameter(normal))
    assert optimizer.chooseRandomValueForParameter(integer) in range(4)
