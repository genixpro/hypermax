import pytest

from hypermax.algorithms.atpe_optimizer import ATPEOptimizer
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
