"""Resumable HPO-B v3 and fixed-max-fidelity YAHPO external study.

Third-party data is read from user-supplied directories and is never copied into
the package.  Every optimizer sees the same five initial configurations for a
task/seed block.  Failed trials are retained and charged the task's normalization
worst value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import time
from typing import Any, Callable
import warnings

import numpy as np
import psutil

from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.atpe2.controller import ATPE2Action
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.algorithms.random_search_optimizer import RandomSearchOptimizer
from hypermax.algorithms.atpe_optimizer import ATPEOptimizer
from hypermax.hyperparameter import Hyperparameter

from .configspace_adapter import ConfigSpaceAdapter
from .database import canonical_json, content_hash


OPTIMIZERS = (
    "random",
    "hyperopt_tpe",
    "legacy_atpe",
    "atpe2",
    "optuna_tpe",
    "smac",
)
TEST_SEEDS = tuple(f"test{index}" for index in range(5))
YAHPO_METRICS = {
    "lcbench": ("val_accuracy", "maximize"),
    "rbv2_xgboost": ("logloss", "minimize"),
    "rbv2_super": ("logloss", "minimize"),
    "nb301": ("val_accuracy", "maximize"),
}
EXPECTED_YAHPO_TASKS = {
    "lcbench": 34,
    "rbv2_xgboost": 119,
    "rbv2_super": 103,
    "nb301": 1,
}
EXTERNAL_SCHEMA_VERSION = "atpe2-external-traces-v1"
_ENSEMBLES: dict[str, ATPE2ModelEnsemble] = {}
_HPOB_DATA: dict[str, Any] = {}
_HPOB_INITIALIZATIONS: dict[str, Any] = {}
_YAHPO_BENCHMARKS: dict[tuple[str, str], Any] = {}
_YAHPO_STATISTICS: dict[str, dict[tuple[str, str, str, str], float]] = {}


def _ensemble(path: Path) -> ATPE2ModelEnsemble:
    key = str(path.resolve())
    if key not in _ENSEMBLES:
        ensemble = ATPE2ModelEnsemble.load(path)
        ensemble.manifest["_source_manifest_sha256"] = _sha256_file(path)
        _ENSEMBLES[key] = ensemble
    return _ENSEMBLES[key]


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json(list(parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class ExternalStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=120)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY, value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs(
                run_id TEXT PRIMARY KEY,
                benchmark TEXT NOT NULL,
                scenario TEXT NOT NULL,
                family TEXT NOT NULL,
                task TEXT NOT NULL,
                optimizer TEXT NOT NULL,
                seed TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                UNIQUE(benchmark, scenario, task, optimizer, seed)
            );
            CREATE TABLE IF NOT EXISTS traces(
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                trial INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY(run_id, trial)
            );
            """
        )
        self.set_metadata("schema_version", EXTERNAL_SCHEMA_VERSION)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def set_metadata(self, key: str, value: Any) -> None:
        encoded = canonical_json(_jsonable(value))
        existing = self.connection.execute(
            "SELECT value_json FROM metadata WHERE key=?", (key,)
        ).fetchone()
        if existing is not None and existing["value_json"] != encoded:
            raise ValueError(
                f"External database metadata conflict for {key}; use a separate study database"
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata VALUES(?,?)",
            (key, encoded),
        )
        self.connection.commit()

    def begin(self, identity: dict[str, Any]) -> str:
        run_id = content_hash([identity[key] for key in ("benchmark", "scenario", "task", "optimizer", "seed")])
        self.connection.execute(
            "INSERT OR IGNORE INTO runs(run_id,benchmark,scenario,family,task,optimizer,seed) VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                identity["benchmark"],
                identity["scenario"],
                identity["family"],
                identity["task"],
                identity["optimizer"],
                str(identity["seed"]),
            ),
        )
        self.connection.commit()
        return run_id

    def rows(self, run_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(row["row_json"])
            for row in self.connection.execute(
                "SELECT row_json FROM traces WHERE run_id=? ORDER BY trial", (run_id,)
            )
        ]

    def complete(self, run_id: str, budget: int) -> bool:
        row = self.connection.execute(
            "SELECT status,(SELECT COUNT(*) FROM traces WHERE run_id=?) AS n FROM runs WHERE run_id=?",
            (run_id, run_id),
        ).fetchone()
        return row is not None and row["status"] == "complete" and row["n"] == budget

    def add(self, run_id: str, trial: int, row: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO traces VALUES(?,?,?)",
            (run_id, trial, canonical_json(_jsonable(row))),
        )
        self.connection.commit()

    def mark_complete(self, run_id: str) -> None:
        self.connection.execute("UPDATE runs SET status='complete' WHERE run_id=?", (run_id,))
        self.connection.commit()

    def export(self, path: Path) -> int:
        """Stream a deterministic JSON array without retaining every trial row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            handle.write("[\n")
            for row in self.connection.execute(
                """SELECT t.row_json FROM traces t JOIN runs r USING(run_id)
                   ORDER BY r.benchmark,r.scenario,r.task,r.optimizer,r.seed,t.trial"""
            ):
                if count:
                    handle.write(",\n")
                handle.write("  ")
                handle.write(row[0])
                count += 1
            handle.write("\n]\n")
        return count


class HypermaxEngine:
    def __init__(
        self,
        name: str,
        adapter: ConfigSpaceAdapter,
        initial: list[dict[str, Any]],
        seed: int,
        ensemble: ATPE2ModelEnsemble,
    ):
        self.name = name
        self.adapter = adapter
        self.initial = initial
        self.history: list[dict[str, Any]] = []
        self.parameter = Hyperparameter(adapter.hypermax_space())
        if name == "random":
            self.optimizer = RandomSearchOptimizer(seed=seed)
        elif name == "hyperopt_tpe":
            self.optimizer = ATPE2Optimizer(seed=seed)
            anchor = ATPE2Action.anchor()
            self.optimizer.controller.action_for = lambda space, results, current: anchor
        elif name == "legacy_atpe":
            random.seed(seed)
            np.random.seed(seed % (2**32))
            self.optimizer = ATPEOptimizer()
        elif name == "atpe2":
            self.optimizer = ATPE2Optimizer(seed=seed, ensemble=ensemble)
        else:
            raise ValueError(name)

    def ask(self) -> tuple[dict[str, Any], Any]:
        if len(self.history) < len(self.initial):
            flat = self.initial[len(self.history)]
            return flat, self.adapter.to_structured(flat)
        history = [] if self.name == "random" else self.history
        structured = self.optimizer.recommendNextParameters(
            self.adapter.hypermax_space(), history, []
        )
        return self.adapter.to_flat(structured), structured

    def tell(self, flat: dict[str, Any], structured: Any, loss: float | None, status: str) -> None:
        if structured is None:
            structured = self.adapter.to_structured(flat)
        row = self.parameter.convertToFlatValues(structured)
        row.update(
            {
                "trial": len(self.history),
                "loss": loss if status == "ok" else None,
                "status": status,
            }
        )
        self.history.append(row)


class OptunaEngine:
    def __init__(self, adapter: ConfigSpaceAdapter, initial: list[dict], seed: int):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.adapter = adapter
        self.study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        constants = set(adapter.constant_values)
        for config in initial:
            self.study.enqueue_trial({key: value for key, value in config.items() if key not in constants})
        self.trial = None

    def ask(self):
        self.trial = self.study.ask()
        return self.adapter.suggest_optuna(self.trial), None

    def tell(self, flat, structured, loss, status):
        import optuna

        value = (
            float(loss)
            if status == "ok" and loss is not None and np.isfinite(loss)
            else None
        )
        state = optuna.trial.TrialState.COMPLETE if value is not None else optuna.trial.TrialState.FAIL
        self.study.tell(self.trial, value, state=state)


class SmacEngine:
    def __init__(
        self,
        adapter: ConfigSpaceAdapter,
        initial: list[dict],
        seed: int,
        run_id: str,
        budget: int,
    ):
        import ConfigSpace
        from smac import HyperparameterOptimizationFacade, Scenario

        self.adapter = adapter
        configs = [ConfigSpace.Configuration(adapter.config_space, values=row) for row in initial]
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"hypermax-atpe2-smac-{run_id[:8]}-"
        )
        scenario = Scenario(
            adapter.config_space,
            deterministic=True,
            n_trials=budget,
            seed=seed,
            use_default_config=False,
            output_directory=Path(self._temporary_directory.name),
        )
        design = HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=0, additional_configs=configs
        )
        self.facade = HyperparameterOptimizationFacade(
            scenario,
            target_function=None,
            initial_design=design,
            logging_level=False,
            overwrite=True,
        )
        self.info = None

    def __del__(self):
        temporary = getattr(self, "_temporary_directory", None)
        if temporary is not None:
            temporary.cleanup()

    def ask(self):
        self.info = self.facade.ask()
        return dict(self.info.config), None

    def tell(self, flat, structured, loss, status):
        from smac.runhistory.dataclasses import TrialValue
        from smac.runhistory.enumerations import StatusType

        value = float(loss) if loss is not None and np.isfinite(loss) else float("inf")
        trial_status = (
            StatusType.SUCCESS
            if status == "ok" and np.isfinite(value)
            else StatusType.CRASHED
        )
        self.facade.tell(
            self.info, TrialValue(cost=value, status=trial_status), save=False
        )


def make_engine(
    name: str,
    adapter: ConfigSpaceAdapter,
    initial: list[dict],
    seed: int,
    ensemble: ATPE2ModelEnsemble,
    run_id: str,
    budget: int,
):
    if name in {"random", "hyperopt_tpe", "legacy_atpe", "atpe2"}:
        return HypermaxEngine(name, adapter, initial, seed, ensemble)
    if name == "optuna_tpe":
        return OptunaEngine(adapter, initial, seed)
    if name == "smac":
        return SmacEngine(adapter, initial, seed, run_id, budget)
    raise ValueError(name)


def run_trace(
    store: ExternalStore,
    identity: dict[str, Any],
    adapter: ConfigSpaceAdapter,
    initial: list[dict[str, Any]],
    evaluate: Callable[[dict[str, Any], int], tuple[float, dict[str, Any]]],
    optimum: float,
    worst: float,
    ensemble: ATPE2ModelEnsemble,
    study_seed: int,
    budget: int = 100,
) -> None:
    run_id = store.begin(identity)
    if store.complete(run_id, budget):
        return
    # Every optimizer in a paired block receives the same numeric seed. Each
    # library consumes it differently, but the experimental seed itself is not
    # optimizer-dependent.
    seed = _seed(
        study_seed,
        identity["benchmark"],
        identity["scenario"],
        identity["task"],
        identity["seed"],
    )
    process = psutil.Process()
    baseline_memory = process.memory_info().rss
    engine = make_engine(
        identity["optimizer"], adapter, initial, seed, ensemble, run_id, budget
    )
    existing = store.rows(run_id)
    peak_memory = process.memory_info().rss
    for trial in range(budget):
        started = time.perf_counter()
        ask_error = None
        try:
            proposed, structured = engine.ask()
        except Exception as exc:
            proposed, structured = {}, None
            ask_error = repr(exc)
        overhead = time.perf_counter() - started
        if trial < len(existing):
            row = existing[trial]
            if ask_error is not None:
                if row.get("status") != "fail" or row.get("configuration") != {}:
                    raise RuntimeError("External resume diverged at optimizer ask failure")
                continue
            if str(row.get("error", "")).startswith("optimizer_ask:"):
                raise RuntimeError("External resume no longer reproduces an optimizer ask failure")
            # HPO-B's discrete wrapper has a pending-pool side effect.  Replay
            # only its cheap nearest-neighbor selection so a resumed trace has
            # exactly the same remaining candidate set.
            if "pending_index" in row.get("raw_objective", {}):
                replay_loss, replay_raw = evaluate(row["configuration"], trial)
                if int(replay_raw["pending_index"]) != int(row["raw_objective"]["pending_index"]):
                    raise RuntimeError("HPO-B resume replay diverged from stored pending index")
            try:
                engine.tell(
                    row["configuration"],
                    None,
                    row.get("loss"),
                    row.get("status", "ok"),
                )
            except Exception as exc:
                if "optimizer_tell:" not in str(row.get("error", "")):
                    raise RuntimeError(
                        "External resume diverged at optimizer tell"
                    ) from exc
            continue
        if ask_error is not None:
            raw = {}
            status = "fail"
            error = f"optimizer_ask: {ask_error}"
            loss = None
            tell_loss = float(worst)
        else:
            try:
                loss, raw = evaluate(proposed, trial)
                if not np.isfinite(loss):
                    raise ValueError(f"non-finite objective {loss}")
                status = "ok"
                error = None
                tell_loss = float(loss)
            except Exception as exc:
                raw = {}
                status = "fail"
                error = f"objective: {repr(exc)}"
                loss = None
                tell_loss = float(worst)
            tell_started = time.perf_counter()
            try:
                engine.tell(proposed, structured, tell_loss, status)
            except Exception as exc:
                status = "fail"
                loss = None
                error = (
                    f"{error}; optimizer_tell: {repr(exc)}"
                    if error
                    else f"optimizer_tell: {repr(exc)}"
                )
            overhead += time.perf_counter() - tell_started
        peak_memory = max(peak_memory, process.memory_info().rss)
        row = {
            **identity,
            "split": "test" if identity["benchmark"] == "hpob-v3" else "external-shift",
            "trial": trial,
            "configuration": _jsonable(proposed),
            "loss": loss,
            "status": status,
            "error": error,
            "raw_objective": _jsonable(raw),
            "optimum": float(optimum),
            "normalization_worst": float(worst),
            "optimizer_overhead_seconds": overhead,
            "baseline_memory_bytes": baseline_memory,
            "peak_memory_bytes": peak_memory,
            "optimizer_memory_delta_bytes": max(0, peak_memory - baseline_memory),
            "model_schema_version": ensemble.manifest.get("schema_version"),
            "model_version": ensemble.manifest.get("model_version"),
            "controller_policy_version": ensemble.manifest.get(
                "controller_policy_version"
            ),
            "model_manifest_sha256": ensemble.manifest.get(
                "_source_manifest_sha256"
            ),
            "controller_decision": (
                _jsonable(engine.optimizer.controller.last_record)
                if identity["optimizer"] == "atpe2"
                and trial >= len(initial)
                else None
            ),
            "study_seed": study_seed,
            "optimizer_seed": seed,
        }
        store.add(run_id, trial, row)
    store.mark_complete(run_id)


def _yahpo_stats(path: Path) -> dict[tuple[str, str, str, str], float]:
    import csv

    key = str(path.resolve())
    if key in _YAHPO_STATISTICS:
        return _YAHPO_STATISTICS[key]
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[(row["scenario"], row["instance"], row["metric"], row["statistic"])] = float(row["value"])
    _YAHPO_STATISTICS[key] = output
    return output


def run_yahpo_task_seed(
    database: Path,
    data_path: Path,
    manifest_path: Path,
    scenario: str,
    instance: str,
    seed_name: str,
    study_seed: int,
    budget: int,
) -> None:
    warnings.filterwarnings(
        "ignore", message="Loky-backed parallel loops cannot be called in a multiprocessing"
    )
    config_path = database.parent / "yahpo-local.yml"
    os.environ["YAHPO_LOCAL_CONFIG"] = str(config_path)
    from yahpo_gym.local_config import local_config

    local_config.init_config(str(data_path))
    from yahpo_gym import BenchmarkSet

    benchmark_key = (str(data_path.resolve()), scenario)
    if benchmark_key not in _YAHPO_BENCHMARKS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _YAHPO_BENCHMARKS[benchmark_key] = BenchmarkSet(
                scenario, active_session=True, multithread=False, noisy=False
            )
    benchmark = _YAHPO_BENCHMARKS[benchmark_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # YAHPO 1.0.2 called this private ConfigSpace 0.6 helper.  Modern
        # ConfigSpace preserves the deserialized order already, so the old
        # in-place sorter is safely represented by a no-op at this boundary.
        if not hasattr(benchmark.config_space, "_sort_hyperparameters"):
            benchmark.config_space._sort_hyperparameters = lambda: None
        benchmark.set_instance(instance)
        for fidelity in benchmark.config.fidelity_params:
            hp = benchmark.config_space[fidelity]
            benchmark.set_constant(fidelity, hp.upper)
        config_space = benchmark.get_opt_space(
            drop_fidelity_params=True,
            seed=_seed(study_seed, scenario, instance, seed_name),
        )
    adapter = ConfigSpaceAdapter(config_space)
    sampled = config_space.sample_configuration(size=5)
    if not isinstance(sampled, list):
        sampled = [sampled]
    initial = [dict(configuration) for configuration in sampled]
    metric, direction = YAHPO_METRICS[scenario]
    statistics = _yahpo_stats(data_path / "global_statistics" / "instance_target_statistics.csv")
    stats_instance = "CIFAR10" if scenario == "nb301" else str(instance)
    minimum = statistics[(scenario, stats_instance, metric, "min")]
    maximum = statistics[(scenario, stats_instance, metric, "max")]
    optimum, worst = (minimum, maximum) if direction == "minimize" else (-maximum, -minimum)

    def evaluate(configuration: dict[str, Any], trial: int):
        complete_configuration = {**configuration, **benchmark.constants}
        result = benchmark.objective_function(
            complete_configuration,
            seed=_seed(study_seed, seed_name, trial),
            multithread=False,
        )[0]
        value = float(result[metric])
        return (value if direction == "minimize" else -value), result

    ensemble = _ensemble(manifest_path)
    with ExternalStore(database) as store:
        for optimizer in OPTIMIZERS:
            identity = {
                "benchmark": "yahpo",
                "scenario": scenario,
                "family": scenario,
                "task": f"{scenario}/{instance}",
                "optimizer": optimizer,
                "seed": seed_name,
            }
            run_trace(
                store,
                identity,
                adapter,
                initial,
                evaluate,
                optimum,
                worst,
                ensemble,
                study_seed,
                budget,
            )


def run_yahpo(
    database: Path,
    data_path: Path,
    manifest: Path,
    workers: int,
    seed: int,
    budget: int,
    max_tasks: int | None,
) -> None:
    model = _ensemble(manifest)
    if not model.research_ready:
        raise ValueError("External study requires a fitted, schema-compatible ATPE2 model")
    if max_tasks is None and budget == 100 and (
        not model.ready
        or not model.manifest.get("validation_evidence", {}).get("gate2_passed")
    ):
        raise ValueError("Full YAHPO study is locked until Gate 2 evidence is frozen")
    config_path = database.parent / "yahpo-discovery.yml"
    os.environ["YAHPO_LOCAL_CONFIG"] = str(config_path)
    from yahpo_gym.local_config import local_config

    local_config.init_config(str(data_path))
    from yahpo_gym import BenchmarkSet

    tasks = []
    for scenario in YAHPO_METRICS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            benchmark = BenchmarkSet(scenario, active_session=False)
        tasks.extend((scenario, str(instance)) for instance in benchmark.instances)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    jobs = [
        (database, data_path, manifest, scenario, instance, seed_name, seed, budget)
        for scenario, instance in tasks
        for seed_name in TEST_SEEDS
    ]
    with ExternalStore(database) as store:
        store.set_metadata("yahpo_expected_tasks", EXPECTED_YAHPO_TASKS)
        store.set_metadata("yahpo_data_version", (data_path / "VERSION").read_text().strip())
        store.set_metadata(
            "yahpo_global_statistics_sha256",
            _sha256_file(data_path / "global_statistics" / "instance_target_statistics.csv"),
        )
        store.set_metadata("atpe2_model_manifest_sha256", _sha256_file(manifest))
        store.set_metadata("study_budget", budget)
        store.set_metadata("root_seed", seed)
        store.set_metadata("smoke_subset", max_tasks is not None or budget != 100)
    if workers == 1:
        for job in jobs:
            run_yahpo_task_seed(*job)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.starmap(run_yahpo_task_seed, jobs)


def _numeric_adapter(dimension: int, seed: int) -> ConfigSpaceAdapter:
    import ConfigSpace
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    space = ConfigSpace.ConfigurationSpace(seed=seed)
    space.add(
        [UniformFloatHyperparameter(f"x{index}", 0.0, 1.0) for index in range(dimension)]
    )
    return ConfigSpaceAdapter(space)


def run_hpob_task_seed(
    database: Path,
    data_path: Path,
    manifest_path: Path,
    space_id: str,
    dataset_id: str,
    seed_name: str,
    study_seed: int,
    budget: int,
) -> None:
    warnings.filterwarnings(
        "ignore", message="Loky-backed parallel loops cannot be called in a multiprocessing"
    )
    import ujson

    data_key = str(data_path.resolve())
    if data_key not in _HPOB_DATA:
        _HPOB_DATA[data_key] = ujson.load(
            (data_path / "meta-test-dataset.json").open()
        )
        _HPOB_INITIALIZATIONS[data_key] = ujson.load(
            (data_path / "bo-initializations.json").open()
        )
    values = _HPOB_DATA[data_key][space_id][dataset_id]
    initial_indices = _HPOB_INITIALIZATIONS[data_key][space_id][dataset_id][seed_name]
    matrix = np.asarray(values["X"], dtype=float)
    responses = np.asarray(values["y"], dtype=float).reshape(-1)
    adapter = _numeric_adapter(
        matrix.shape[1], _seed(study_seed, space_id, dataset_id, seed_name)
    )

    def row(index: int) -> dict[str, float]:
        return {f"x{column}": float(value) for column, value in enumerate(matrix[index])}

    initial = [row(index) for index in initial_indices]
    optimum, worst = -float(np.max(responses)), -float(np.min(responses))
    ensemble = _ensemble(manifest_path)
    with ExternalStore(database) as store:
        for optimizer in OPTIMIZERS:
            used: set[int] = set()

            def evaluate(configuration: dict[str, Any], trial: int):
                if trial < len(initial_indices):
                    # Preserve the official initialization index even when the
                    # tabular benchmark contains duplicate feature vectors.
                    selected = int(initial_indices[trial])
                else:
                    vector = np.asarray(
                        [configuration[f"x{index}"] for index in range(matrix.shape[1])]
                    )
                    pending = np.asarray(
                        [index for index in range(len(matrix)) if index not in used]
                    )
                    if not len(pending):
                        raise RuntimeError("HPO-B pending configuration pool exhausted")
                    distances = np.sum((matrix[pending] - vector) ** 2, axis=1)
                    selected = int(pending[int(np.argmin(distances))])
                if selected in used:
                    raise RuntimeError("HPO-B official initialization contains a duplicate index")
                used.add(selected)
                accuracy = float(responses[selected])
                return -accuracy, {"accuracy": accuracy, "pending_index": selected}

            identity = {
                "benchmark": "hpob-v3",
                "scenario": str(space_id),
                "family": f"hpob-space-{space_id}",
                "task": f"{space_id}/{dataset_id}",
                "optimizer": optimizer,
                "seed": seed_name,
            }
            run_trace(
                store,
                identity,
                adapter,
                initial,
                evaluate,
                optimum,
                worst,
                ensemble,
                study_seed,
                budget,
            )


def run_hpob(
    database: Path,
    data_path: Path,
    task_metadata: Path,
    manifest: Path,
    workers: int,
    seed: int,
    budget: int,
    max_tasks: int | None,
) -> None:
    model = _ensemble(manifest)
    if not model.research_ready:
        raise ValueError("External study requires a fitted, schema-compatible ATPE2 model")
    if max_tasks is None and budget == 100 and (
        not model.ready
        or not model.manifest.get("validation_evidence", {}).get("gate2_passed")
    ):
        raise ValueError("HPO-B v3 test study is locked until Gate 2 evidence is frozen")
    official = json.loads(task_metadata.read_text())
    tasks = [
        (str(space), str(dataset))
        for space, datasets in official.items()
        for dataset in datasets
    ]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    jobs = [
        (
            database,
            data_path,
            manifest,
            space,
            dataset,
            seed_name,
            seed,
            budget,
        )
        for space, dataset in tasks
        for seed_name in TEST_SEEDS
    ]
    with ExternalStore(database) as store:
        store.set_metadata("hpob_expected_tasks", len(tasks) if max_tasks is not None else 86)
        store.set_metadata("hpob_mode", "v3-test")
        store.set_metadata(
            "hpob_meta_test_dataset_sha256",
            _sha256_file(data_path / "meta-test-dataset.json"),
        )
        store.set_metadata(
            "hpob_initializations_sha256",
            _sha256_file(data_path / "bo-initializations.json"),
        )
        store.set_metadata("hpob_task_split_sha256", _sha256_file(task_metadata))
        store.set_metadata("atpe2_model_manifest_sha256", _sha256_file(manifest))
        store.set_metadata("study_budget", budget)
        store.set_metadata("root_seed", seed)
        store.set_metadata("smoke_subset", max_tasks is not None or budget != 100)
    if workers == 1:
        for job in jobs:
            run_hpob_task_seed(*job)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.starmap(run_hpob_task_seed, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    yahpo = subparsers.add_parser("yahpo")
    yahpo.add_argument("database", type=Path)
    yahpo.add_argument("data", type=Path)
    yahpo.add_argument("manifest", type=Path)
    yahpo.add_argument("--workers", type=int, default=1)
    yahpo.add_argument("--seed", type=int, required=True)
    yahpo.add_argument("--budget", type=int, default=100)
    yahpo.add_argument("--max-tasks", type=int)
    hpob = subparsers.add_parser("hpob")
    hpob.add_argument("database", type=Path)
    hpob.add_argument("data", type=Path)
    hpob.add_argument("task_metadata", type=Path)
    hpob.add_argument("manifest", type=Path)
    hpob.add_argument("--workers", type=int, default=1)
    hpob.add_argument("--seed", type=int, required=True)
    hpob.add_argument("--budget", type=int, default=100)
    hpob.add_argument("--max-tasks", type=int)
    export = subparsers.add_parser("export")
    export.add_argument("database", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "yahpo":
        run_yahpo(
            args.database,
            args.data,
            args.manifest,
            args.workers,
            args.seed,
            args.budget,
            args.max_tasks,
        )
    elif args.command == "hpob":
        run_hpob(
            args.database,
            args.data,
            args.task_metadata,
            args.manifest,
            args.workers,
            args.seed,
            args.budget,
            args.max_tasks,
        )
    else:
        with ExternalStore(args.database) as store:
            store.set_metadata("export_seed", args.seed)
            print(json.dumps({"rows": store.export(args.output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
