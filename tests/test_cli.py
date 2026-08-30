import json
import os
import subprocess
import sys


def test_cli_runs_search_to_completion(tmp_path):
    (tmp_path / "cli_model.py").write_text(
        "def evaluate(params):\n"
        "    return {'loss': params['x'] ** 2}\n",
        encoding="utf-8",
    )
    config = {
        "hyperparameters": {
            "type": "object",
            "properties": {"x": {"type": "number", "min": -1, "max": 1}},
        },
        "function": {
            "type": "python_function",
            "module": "cli_model",
            "name": "evaluate",
            "parallel": 1,
        },
        "search": {"method": "random", "iterations": 2},
        "results": {"directory": "results", "graphs": False},
        "ui": {"enabled": False},
    }
    config_path = tmp_path / "search.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / ".matplotlib")

    completed = subprocess.run(
        [sys.executable, "-m", "hypermax.cli", str(config_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    results_file = tmp_path / "results_0" / "results.csv"
    assert results_file.is_file()
    assert len(results_file.read_text(encoding="utf-8").splitlines()) == 3
