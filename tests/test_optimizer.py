import json

import pytest

from hypermax.optimizer import Optimizer


def test_optimizer_runs_exact_iteration_count_and_writes_restart_files(tmp_path):
    config = {
        "hyperparameters": {
            "type": "object",
            "properties": {"x": {"type": "number", "min": -2, "max": 2}},
        },
        "function": {
            "type": "python_function",
            "module": "unused",
            "name": "unused",
            "func": lambda params: params["x"] ** 2,
        },
        "search": {"method": "random", "iterations": 3},
        "results": {"directory": str(tmp_path / "results"), "graphs": False},
        "ui": {"enabled": False},
    }
    optimizer = Optimizer(config)

    optimizer.runOptimizationThread()
    optimizer.thread.join(timeout=20)

    assert not optimizer.thread.is_alive()
    assert optimizer.completed() == 3
    assert sorted(result["trial"] for result in optimizer.results) == [0, 1, 2]

    output_directory = tmp_path / "results_0"
    assert (output_directory / "results.csv").is_file()
    assert (output_directory / "best.json").is_file()
    written_config = json.loads((output_directory / "search.json").read_text())
    assert "func" not in written_config["function"]


def test_optimizer_rejects_invalid_search_configuration(tmp_path):
    config = {
        "hyperparameters": {"type": "object", "properties": {}},
        "function": {"type": "python_function", "func": lambda params: 0},
        "search": {"method": "unknown", "iterations": 0},
        "results": {"directory": str(tmp_path / "results")},
    }

    with pytest.raises(ValueError, match="search.method"):
        Optimizer(config)
