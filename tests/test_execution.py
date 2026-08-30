import time

from hypermax.execution import Execution


def test_in_process_function_normalizes_a_scalar_result():
    execution = Execution(
        {"type": "python_function", "func": lambda params: params["x"] ** 2},
        {"x": 3},
    )

    assert execution.run() == {
        "loss": 9,
        "status": "ok",
        "log": "",
        "error": None,
    }


def test_python_subprocess_uses_active_interpreter_and_preserves_json_values(tmp_path, monkeypatch):
    (tmp_path / "sample_model.py").write_text(
        "def evaluate(params):\n"
        "    assert params == {'enabled': True, 'value': None}\n"
        "    return {'loss': 0.25, 'message': \"it's valid JSON\"}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    execution = Execution(
        {"type": "python_function", "module": "sample_model", "name": "evaluate"},
        {"enabled": True, "value": None},
    )
    result = execution.run()

    assert result["status"] == "ok"
    assert result["loss"] == 0.25
    assert result["message"] == "it's valid JSON"
    assert execution.process is None


def test_auto_kill_time_limit_is_enforced_while_subprocess_runs(tmp_path, monkeypatch):
    (tmp_path / "slow_model.py").write_text(
        "import time\n"
        "def evaluate(params):\n"
        "    time.sleep(5)\n"
        "    return 0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    execution = Execution(
        {
            "type": "python_function",
            "module": "slow_model",
            "name": "evaluate",
            "auto_kill_max_time": 0.1,
            "auto_kill_loss": 99,
        },
        {},
    )

    started = time.monotonic()
    result = execution.run()

    assert time.monotonic() - started < 2
    assert result["status"] == "fail"
    assert result["loss"] == 99
