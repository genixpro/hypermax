"""Machine-checkable Gate 2 and Gate 3 acceptance reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from scipy import stats

from .evidence import ABLATIONS, external_study_statistics, paired_bootstrap_interval


EXTERNAL_OPTIMIZERS = {
    "random",
    "hyperopt_tpe",
    "legacy_atpe",
    "atpe2",
    "optuna_tpe",
    "smac",
}
EXTERNAL_BUDGETS = {25, 50, 100}
YAHPO_SCENARIOS = {"lcbench", "rbv2_xgboost", "rbv2_super", "nb301"}
SYNTHETIC_FAMILIES = {
    "smooth",
    "rugged",
    "interaction",
    "hierarchical",
    "plateau",
    "constant_zero",
}
SYNTHETIC_TRAJECTORIES = {
    "random",
    "tpe",
    "legacy_atpe",
    "space_filled_atpe2",
}
SYNTHETIC_CHECKPOINTS = {12, 20, 40}
SYNTHETIC_INSTANCES_PER_FAMILY = 4
SYNTHETIC_ROLLOUT_SEEDS = 5


def configurations_equivalent(first: Any, second: Any, tolerance: float = 1e-12) -> bool:
    """Compare configurations while tolerating library serialization rounding.

    ConfigSpace/SMAC canonicalizes floating-point values to roughly 13 decimal
    places.  The represented configurations are unchanged, but requiring
    byte-identical JSON would incorrectly reject common initial designs.  Keep
    categorical and structural comparisons exact and permit only sub-picounit
    numeric drift.
    """
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            configurations_equivalent(first[key], second[key], tolerance)
            for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            configurations_equivalent(left, right, tolerance)
            for left, right in zip(first, second)
        )
    if (
        isinstance(first, (int, float, np.number))
        and not isinstance(first, (bool, np.bool_))
        and isinstance(second, (int, float, np.number))
        and not isinstance(second, (bool, np.bool_))
    ):
        return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))
    return first == second


def _initial_designs_equivalent(designs: dict[str, list[Any]]) -> bool:
    if len(designs) != len(EXTERNAL_OPTIMIZERS):
        return False
    reference = next(iter(designs.values()))
    return all(
        len(candidate) == len(reference)
        and all(
            configurations_equivalent(left, right)
            for left, right in zip(reference, candidate)
        )
        for candidate in designs.values()
    )


def gate2_report(
    rows: list[dict[str, Any]],
    pruning: dict[str, Any],
    aggregation_rounds: list[int],
    completed_ablations: list[str],
    seed: int,
) -> dict[str, Any]:
    """Evaluate synthetic utility, per-family regret, pruning, DAgger, and ablations."""
    pairs: dict[tuple, dict[str, float]] = defaultdict(dict)
    family_regret: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["task"], row["trajectory_lineage"], row["checkpoint"], row["seed"])
        pairs[key][row["optimizer"]] = float(row["five_step_utility"])
        family_regret[row["family"]][row["optimizer"]].append(float(row["normalized_regret"]))
    differences_by_task: dict[str, list[float]] = defaultdict(list)
    for key, values in pairs.items():
        if {"atpe2", "hyperopt_tpe"}.issubset(values):
            differences_by_task[str(key[0])].append(
                values["atpe2"] - values["hyperopt_tpe"]
            )
    differences = [
        float(np.median(values)) for values in differences_by_task.values()
    ]
    interval = paired_bootstrap_interval(differences, seed)
    family_degradation = {}
    for family, optimizers in family_regret.items():
        if not {"atpe2", "hyperopt_tpe"}.issubset(optimizers):
            family_degradation[family] = None
        else:
            family_degradation[family] = float(
                np.median(optimizers["atpe2"]) - np.median(optimizers["hyperopt_tpe"])
            )
    tasks_by_family = {
        family: {row["task"] for row in rows if row["family"] == family}
        for family in SYNTHETIC_FAMILIES
    }
    protocol_blocks: dict[tuple, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        block = (
            row["task"],
            row.get("trajectory_method"),
            int(row["checkpoint"]),
        )
        protocol_blocks[block][int(row["seed"])].append(row["optimizer"])
    expected_optimizers = {
        "random",
        "hyperopt_tpe",
        "legacy_atpe",
        "atpe2",
        "ablation_filtering",
        "ablation_locking",
        "ablation_adaptive_gamma_candidates",
        "ablation_uncertainty_penalty",
        "ablation_pruned_schema",
        "ablation_full_schema",
        "ablation_legacy_features",
    }
    expected_block_count = (
        len(SYNTHETIC_FAMILIES)
        * SYNTHETIC_INSTANCES_PER_FAMILY
        * len(SYNTHETIC_TRAJECTORIES)
        * len(SYNTHETIC_CHECKPOINTS)
    )
    tasks = {row["task"] for row in rows}
    expected_blocks = {
        (task, method, checkpoint)
        for task in tasks
        for method in SYNTHETIC_TRAJECTORIES
        for checkpoint in SYNTHETIC_CHECKPOINTS
    }
    protocol_complete = (
        {row["family"] for row in rows} == SYNTHETIC_FAMILIES
        and all(
            len(family_tasks) == SYNTHETIC_INSTANCES_PER_FAMILY
            for family_tasks in tasks_by_family.values()
        )
        and len(tasks) == len(SYNTHETIC_FAMILIES) * SYNTHETIC_INSTANCES_PER_FAMILY
        and {row.get("trajectory_method") for row in rows}
        == SYNTHETIC_TRAJECTORIES
        and {int(row["checkpoint"]) for row in rows} == SYNTHETIC_CHECKPOINTS
        and set(protocol_blocks) == expected_blocks
        and len(protocol_blocks) == expected_block_count
        and all(
            len(seed_optimizers) == SYNTHETIC_ROLLOUT_SEEDS
            and all(
                len(optimizers) == len(expected_optimizers)
                and set(optimizers) == expected_optimizers
                for optimizers in seed_optimizers.values()
            )
            for seed_optimizers in protocol_blocks.values()
        )
    )
    checks = {
        "full_held_out_protocol_complete": protocol_complete,
        "paired_ci_non_negative": interval[0] >= 0,
        "family_degradation_at_most_5_percent": bool(family_degradation) and all(
            value is not None and value <= 0.05 for value in family_degradation.values()
        ),
        "knob_pruning_complete": "retained_knobs" in pruning.get("backward_knockout", {}),
        "exactly_one_dataset_aggregation_round": set(aggregation_rounds) == {0, 1},
        "ablations_complete": set(ABLATIONS).issubset(completed_ablations),
    }
    return {
        "gate": 2,
        "passed": all(checks.values()),
        "checks": checks,
        "atpe2_minus_tpe_95_percent_paired_bootstrap_ci": interval,
        "paired_bootstrap_unit": "task (median across trajectory/checkpoint/seed blocks)",
        "family_median_normalized_regret_degradation": family_degradation,
        "protocol": {
            "families": sorted({row["family"] for row in rows}),
            "tasks_per_family": {
                family: len(tasks) for family, tasks in sorted(tasks_by_family.items())
            },
            "trajectory_methods": sorted(
                {str(row.get("trajectory_method")) for row in rows}
            ),
            "checkpoints": sorted({int(row["checkpoint"]) for row in rows}),
            "paired_blocks": sum(len(seeds) for seeds in protocol_blocks.values()),
            "expected_paired_blocks": expected_block_count * SYNTHETIC_ROLLOUT_SEEDS,
        },
    }


def incumbent_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create fixed-budget regrets and AUC from complete optimizer traces."""
    traces: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        traces[(row["benchmark"], row["scenario"], row["family"], row["task"], row["optimizer"], row["seed"])].append(row)
    metrics = []
    for key, trace in traces.items():
        trace.sort(key=lambda row: int(row["trial"]))
        incumbent = np.minimum.accumulate(
            [float(row["loss"]) if row.get("status", "ok") == "ok" and row.get("loss") is not None else np.inf for row in trace]
        )
        optimum = float(trace[0]["optimum"])
        worst = float(trace[0]["normalization_worst"])
        incumbent = np.where(np.isfinite(incumbent), incumbent, worst)
        denominator = max(np.finfo(float).eps, worst - optimum)
        regret = np.clip((incumbent - optimum) / denominator, 0, 1)
        by_budget = {}
        for budget in EXTERNAL_BUDGETS:
            if len(regret) < budget:
                raise ValueError(f"Incomplete trace for {key}: needs {budget} trials")
            by_budget[str(budget)] = float(regret[budget - 1])
        auc = float(np.trapezoid(regret[:100], dx=1) / 99)
        metrics.append(
            {
                "benchmark": key[0],
                "scenario": key[1],
                "family": key[2],
                "task": key[3],
                "optimizer": key[4],
                "seed": key[5],
                "regret": by_budget,
                "auc_incumbent_regret": auc,
                "optimizer_overhead_seconds": float(sum(row.get("optimizer_overhead_seconds", 0) for row in trace[:100])),
                "peak_memory_bytes": max(int(row.get("peak_memory_bytes", 0)) for row in trace[:100]),
                "optimizer_memory_delta_bytes": max(
                    int(row.get("optimizer_memory_delta_bytes", 0))
                    for row in trace[:100]
                ),
                "failures": sum(row.get("status", "ok") != "ok" for row in trace[:100]),
            }
        )
    return metrics


def _endpoint_value(row: dict[str, Any], endpoint: str) -> float:
    return (
        float(row["auc_incumbent_regret"])
        if endpoint == "auc_incumbent_regret"
        else float(row["regret"][endpoint])
    )


def _task_stratified_summary(
    rows: list[dict[str, Any]], endpoint: str, seed: int, repeats: int = 10_000
) -> dict[str, Any]:
    """Summarize an endpoint while resampling whole tasks with all their seeds."""
    rng = np.random.default_rng(seed)
    output = {}
    for optimizer in sorted(EXTERNAL_OPTIMIZERS):
        selected = [row for row in rows if row["optimizer"] == optimizer]
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            by_task[f'{row["benchmark"]}:{row["scenario"]}:{row["task"]}'].append(
                _endpoint_value(row, endpoint)
            )
        tasks = sorted(by_task)
        task_medians = np.asarray(
            [np.median(by_task[task]) for task in tasks], dtype=float
        )
        sampled = rng.integers(0, len(tasks), size=(repeats, len(tasks)))
        bootstrap_medians = np.median(task_medians[sampled], axis=1)
        output[optimizer] = {
            "median": float(np.median(task_medians)),
            "interquartile_mean": float(stats.trim_mean(task_medians, 0.25)),
            "task_stratified_median_95_percent_ci": [
                float(np.quantile(bootstrap_medians, 0.025)),
                float(np.quantile(bootstrap_medians, 0.975)),
            ],
            "tasks": len(tasks),
            "runs": len(selected),
        }
    return output


def external_summaries(metrics: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    endpoints = {
        endpoint: _task_stratified_summary(metrics, endpoint, seed + index)
        for index, endpoint in enumerate(("25", "50", "100", "auc_incumbent_regret"))
    }
    per_family = {}
    for family in sorted({row["family"] for row in metrics}):
        family_rows = [row for row in metrics if row["family"] == family]
        family_endpoints = {}
        for endpoint in ("25", "50", "100", "auc_incumbent_regret"):
            optimizer_summary = {}
            for optimizer in sorted(EXTERNAL_OPTIMIZERS):
                by_task: dict[str, list[float]] = defaultdict(list)
                for row in family_rows:
                    if row["optimizer"] == optimizer:
                        by_task[row["task"]].append(_endpoint_value(row, endpoint))
                if by_task:
                    values = [float(np.median(rows)) for rows in by_task.values()]
                    optimizer_summary[optimizer] = {
                        "median": float(np.median(values)),
                        "interquartile_mean": float(stats.trim_mean(values, 0.25)),
                        "tasks": len(values),
                    }
            family_endpoints[endpoint] = optimizer_summary
        per_family[family] = family_endpoints
    resources = {}
    for optimizer in sorted(EXTERNAL_OPTIMIZERS):
        selected = [row for row in metrics if row["optimizer"] == optimizer]
        overhead = [row["optimizer_overhead_seconds"] for row in selected]
        memory = [row["peak_memory_bytes"] for row in selected]
        memory_delta = [row["optimizer_memory_delta_bytes"] for row in selected]
        resources[optimizer] = {
            "median_optimizer_overhead_seconds_per_100_trials": float(
                np.median(overhead)
            ),
            "interquartile_mean_optimizer_overhead_seconds_per_100_trials": float(
                stats.trim_mean(overhead, 0.25)
            ),
            "median_peak_memory_bytes": float(np.median(memory)),
            "maximum_peak_memory_bytes": int(max(memory)),
            "median_optimizer_memory_delta_bytes": float(np.median(memory_delta)),
            "maximum_optimizer_memory_delta_bytes": int(max(memory_delta)),
            "failed_trials": int(sum(row["failures"] for row in selected)),
        }
    return {"endpoints": endpoints, "per_family": per_family, "resources": resources}


def validate_external_protocol(
    rows: list[dict[str, Any]], frozen_model: bool, model_manifest_sha256: str | None = None
) -> None:
    if not frozen_model:
        raise ValueError("External test data may only be analyzed after the model is frozen")
    if {row["benchmark"] for row in rows} != {"hpob-v3", "yahpo"}:
        raise ValueError("External study must contain both HPO-B v3 and YAHPO")
    if model_manifest_sha256 is not None and {
        row.get("model_manifest_sha256") for row in rows
    } != {model_manifest_sha256}:
        raise ValueError("External rows were not all produced by the supplied frozen model")
    if len({row.get("study_seed") for row in rows}) != 1:
        raise ValueError("External rows must share one explicit study seed")
    if any(
        row["optimizer"] == "atpe2"
        and int(row["trial"]) >= 5
        and (
            not isinstance(row.get("controller_decision"), dict)
            or row["controller_decision"].get("decision_mode")
            != "adaptive_model"
            or row["controller_decision"].get("fallback_reason") is not None
        )
        for row in rows
    ):
        raise ValueError("ATPE2 external rows must prove adaptive-only decisions")
    optimizers = {row["optimizer"] for row in rows}
    if optimizers != EXTERNAL_OPTIMIZERS:
        raise ValueError(f"Expected exactly these optimizers: {sorted(EXTERNAL_OPTIMIZERS)}")
    if any(row["benchmark"] == "hpob-v3" and row.get("split") != "test" for row in rows):
        raise ValueError("HPO-B v3 external study rows must use its official test split")
    yahpo = {row["scenario"] for row in rows if row["benchmark"] == "yahpo"}
    if yahpo != YAHPO_SCENARIOS:
        raise ValueError(f"YAHPO study must contain exactly {sorted(YAHPO_SCENARIOS)}")
    expected_yahpo_counts = {
        "lcbench": 34,
        "rbv2_xgboost": 119,
        "rbv2_super": 103,
        "nb301": 1,
    }
    for scenario, expected in expected_yahpo_counts.items():
        tasks = {
            row["task"]
            for row in rows
            if row["benchmark"] == "yahpo" and row["scenario"] == scenario
        }
        if len(tasks) != expected:
            raise ValueError(f"YAHPO {scenario} needs all {expected} official instances")
    hpob_tasks = {row["task"] for row in rows if row["benchmark"] == "hpob-v3"}
    if len(hpob_tasks) != 86:
        raise ValueError("HPO-B v3 test study needs all 86 official test tasks")

    seeds_by_task: dict[tuple, set] = defaultdict(set)
    numeric_seeds_by_block: dict[tuple, set] = defaultdict(set)
    traces: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seeds_by_task[(row["benchmark"], row["scenario"], row["task"], row["optimizer"])].add(row["seed"])
        numeric_seeds_by_block[
            (row["benchmark"], row["scenario"], row["task"], row["seed"])
        ].add(row.get("optimizer_seed"))
        traces[(row["benchmark"], row["scenario"], row["task"], row["optimizer"], row["seed"])].append(row)
    counts = {len(seeds) for seeds in seeds_by_task.values()}
    if counts != {5}:
        raise ValueError("Every external task/optimizer must contain all five prescribed seeds")
    if any(seeds != {"test0", "test1", "test2", "test3", "test4"} for seeds in seeds_by_task.values()):
        raise ValueError("Every external task must use the five prescribed test seed identifiers")
    if any(len(values) != 1 or None in values for values in numeric_seeds_by_block.values()):
        raise ValueError("Optimizers in each paired block must receive the same numeric seed")
    for key, trace in traces.items():
        trials = sorted(int(row["trial"]) for row in trace)
        if trials != list(range(100)):
            raise ValueError(f"External trace {key} is incomplete or duplicated")

    initial: dict[tuple, dict[str, list[str]]] = defaultdict(dict)
    for key, trace in traces.items():
        block = (key[0], key[1], key[2], key[4])
        ordered = sorted(trace, key=lambda row: int(row["trial"]))[:5]
        initial[block][key[3]] = [row["configuration"] for row in ordered]
    for block, designs in initial.items():
        if not _initial_designs_equivalent(designs):
            raise ValueError(f"Optimizers did not share an identical initial design for {block}")


def validate_external_database(
    path: Path, frozen_model: bool, model_manifest_sha256: str | None = None
) -> None:
    """Validate the complete protocol without materializing one million rows."""
    if not frozen_model:
        raise ValueError("External test data may only be analyzed after the model is frozen")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        if model_manifest_sha256 is not None:
            stored = connection.execute(
                "SELECT value_json FROM metadata WHERE key='atpe2_model_manifest_sha256'"
            ).fetchone()
            if stored is None or json.loads(stored[0]) != model_manifest_sha256:
                raise ValueError(
                    "External database was not produced by the supplied frozen model"
                )
        root_seed_row = connection.execute(
            "SELECT value_json FROM metadata WHERE key='root_seed'"
        ).fetchone()
        trace_seeds = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT json_extract(row_json,'$.study_seed') FROM traces"
            )
        }
        if root_seed_row is None or trace_seeds != {json.loads(root_seed_row[0])}:
            raise ValueError("External rows must share the database's explicit study seed")
        benchmarks = {
            row[0] for row in connection.execute("SELECT DISTINCT benchmark FROM runs")
        }
        if benchmarks != {"hpob-v3", "yahpo"}:
            raise ValueError("External study must contain both HPO-B v3 and YAHPO")
        optimizers = {
            row[0] for row in connection.execute("SELECT DISTINCT optimizer FROM runs")
        }
        if optimizers != EXTERNAL_OPTIMIZERS:
            raise ValueError(
                f"Expected exactly these optimizers: {sorted(EXTERNAL_OPTIMIZERS)}"
            )
        invalid_split = connection.execute(
            """SELECT COUNT(*) FROM traces t JOIN runs r USING(run_id)
               WHERE r.benchmark='hpob-v3'
                 AND json_extract(t.row_json,'$.split') != 'test'"""
        ).fetchone()[0]
        if invalid_split:
            raise ValueError("HPO-B v3 external study rows must use its official test split")
        scenarios = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT scenario FROM runs WHERE benchmark='yahpo'"
            )
        }
        if scenarios != YAHPO_SCENARIOS:
            raise ValueError(f"YAHPO study must contain exactly {sorted(YAHPO_SCENARIOS)}")
        expected_yahpo_counts = {
            "lcbench": 34,
            "rbv2_xgboost": 119,
            "rbv2_super": 103,
            "nb301": 1,
        }
        for scenario, expected in expected_yahpo_counts.items():
            actual = connection.execute(
                "SELECT COUNT(DISTINCT task) FROM runs WHERE benchmark='yahpo' AND scenario=?",
                (scenario,),
            ).fetchone()[0]
            if actual != expected:
                raise ValueError(f"YAHPO {scenario} needs all {expected} official instances")
        hpob_tasks = connection.execute(
            "SELECT COUNT(DISTINCT task) FROM runs WHERE benchmark='hpob-v3'"
        ).fetchone()[0]
        if hpob_tasks != 86:
            raise ValueError("HPO-B v3 test study needs all 86 official test tasks")
        seeds = {row[0] for row in connection.execute("SELECT DISTINCT seed FROM runs")}
        if seeds != {"test0", "test1", "test2", "test3", "test4"}:
            raise ValueError("Every external task must use the five prescribed test seed identifiers")
        incomplete_seed_groups = connection.execute(
            """SELECT COUNT(*) FROM (
                 SELECT benchmark,scenario,task,optimizer,COUNT(DISTINCT seed) AS n
                 FROM runs GROUP BY benchmark,scenario,task,optimizer HAVING n != 5
               )"""
        ).fetchone()[0]
        if incomplete_seed_groups:
            raise ValueError("Every external task/optimizer must contain all five prescribed seeds")
        incomplete_optimizer_groups = connection.execute(
            """SELECT COUNT(*) FROM (
                 SELECT benchmark,scenario,task,seed,COUNT(DISTINCT optimizer) AS n
                 FROM runs GROUP BY benchmark,scenario,task,seed HAVING n != 6
               )"""
        ).fetchone()[0]
        if incomplete_optimizer_groups:
            raise ValueError("Every external task/seed must contain all six optimizers")
        mismatched_numeric_seeds = connection.execute(
            """SELECT COUNT(*) FROM (
                 SELECT r.benchmark,r.scenario,r.task,r.seed,
                        COUNT(DISTINCT json_extract(t.row_json,'$.optimizer_seed')) AS n
                 FROM runs r JOIN traces t USING(run_id)
                 GROUP BY r.benchmark,r.scenario,r.task,r.seed HAVING n != 1
               )"""
        ).fetchone()[0]
        if mismatched_numeric_seeds:
            raise ValueError("Optimizers in a paired block did not receive the same numeric seed")
        invalid_adaptive_rows = connection.execute(
            """SELECT COUNT(*) FROM traces t JOIN runs r USING(run_id)
               WHERE r.optimizer='atpe2' AND t.trial>=5 AND (
                 COALESCE(json_type(t.row_json,'$.controller_decision'),'missing')!='object' OR
                 COALESCE(json_extract(t.row_json,'$.controller_decision.decision_mode'),'')!='adaptive_model' OR
                 COALESCE(json_type(t.row_json,'$.controller_decision.fallback_reason'),'null')!='null'
               )"""
        ).fetchone()[0]
        if invalid_adaptive_rows:
            raise ValueError("ATPE2 external rows must prove adaptive-only decisions")
        incomplete_traces = connection.execute(
            """SELECT COUNT(*) FROM (
                 SELECT r.run_id,COUNT(t.trial) AS n,COUNT(DISTINCT t.trial) AS distinct_n,
                        MIN(t.trial) AS first_trial,MAX(t.trial) AS last_trial,r.status
                 FROM runs r LEFT JOIN traces t USING(run_id) GROUP BY r.run_id
                 HAVING n != 100 OR distinct_n != 100 OR first_trial != 0 OR
                        last_trial != 99 OR status != 'complete'
               )"""
        ).fetchone()[0]
        if incomplete_traces:
            raise ValueError("External study contains incomplete or duplicated traces")

        initial: dict[tuple, dict[str, list[str]]] = defaultdict(dict)
        cursor = connection.execute(
            """SELECT r.benchmark,r.scenario,r.task,r.seed,r.optimizer,t.trial,t.row_json
               FROM runs r JOIN traces t USING(run_id) WHERE t.trial < 5
               ORDER BY r.benchmark,r.scenario,r.task,r.seed,r.optimizer,t.trial"""
        )
        for row in cursor:
            block = (row["benchmark"], row["scenario"], row["task"], row["seed"])
            configuration = json.loads(row["row_json"])["configuration"]
            initial[block].setdefault(row["optimizer"], []).append(configuration)
        for block, designs in initial.items():
            if not _initial_designs_equivalent(designs):
                raise ValueError(
                    f"Optimizers did not share an identical initial design for {block}"
                )
    finally:
        connection.close()


def incumbent_metrics_from_database(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    metrics: list[dict[str, Any]] = []
    current_run = None
    trace: list[dict[str, Any]] = []
    try:
        cursor = connection.execute(
            """SELECT r.run_id,t.row_json FROM runs r JOIN traces t USING(run_id)
               ORDER BY r.run_id,t.trial"""
        )
        for row in cursor:
            if current_run is not None and row["run_id"] != current_run:
                metrics.extend(incumbent_metrics(trace))
                trace = []
            current_run = row["run_id"]
            trace.append(json.loads(row["row_json"]))
        if trace:
            metrics.extend(incumbent_metrics(trace))
    finally:
        connection.close()
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=Path, help="External trace SQLite database or exported JSON array"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble

    ensemble = ATPE2ModelEnsemble.load(args.model_manifest)
    validation_evidence = ensemble.manifest.get("validation_evidence", {})
    frozen = ensemble.ready and validation_evidence.get("gate2_passed") is True
    model_manifest_sha256 = hashlib.sha256(
        args.model_manifest.read_bytes()
    ).hexdigest()
    artifact_size = sum(
        path.stat().st_size
        for path in args.model_manifest.parent.iterdir()
        if path.is_file()
    )
    if args.input.suffix in {".sqlite", ".db", ".sqlite3"}:
        validate_external_database(args.input, frozen, model_manifest_sha256)
        metrics = incumbent_metrics_from_database(args.input)
    else:
        rows = json.loads(args.input.read_text())
        validate_external_protocol(rows, frozen, model_manifest_sha256)
        metrics = incumbent_metrics(rows)
    study_rows = [
        {
            "optimizer": row["optimizer"],
            "task": f'{row["benchmark"]}:{row["scenario"]}:{row["task"]}',
            "family": row["family"],
            "seed": row["seed"],
            "budget": 100,
            "normalized_regret": row["auc_incumbent_regret"],
        }
        for row in metrics
    ]
    statistics = external_study_statistics(study_rows, args.seed)
    paired: dict[tuple, dict[str, float]] = defaultdict(dict)
    per_family: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in metrics:
        paired[(row["benchmark"], row["scenario"], row["task"], row["seed"])][row["optimizer"]] = row["auc_incumbent_regret"]
        per_family[row["family"]][row["optimizer"]][row["task"]].append(
            row["auc_incumbent_regret"]
        )
    improvements_by_task: dict[str, list[float]] = defaultdict(list)
    for (benchmark, scenario, task, _seed_name), values in paired.items():
        if {"atpe2", "hyperopt_tpe"}.issubset(values):
            improvements_by_task[f"{benchmark}:{scenario}:{task}"].append(
                values["hyperopt_tpe"] - values["atpe2"]
            )
    improvements = [
        float(np.median(values)) for values in improvements_by_task.values()
    ]
    confidence_interval = paired_bootstrap_interval(improvements, args.seed)
    adjusted = statistics["paired_holm_against_best"]
    corrected_significance = (
        statistics["friedman"]["p_value"] < 0.05
        and adjusted["best"] == "atpe2"
        and adjusted["adjusted_p_values"].get("hyperopt_tpe", 1.0) < 0.05
    )
    family_degradation = {
        family: float(
            np.median(
                [np.median(seed_values) for seed_values in values["atpe2"].values()]
            )
            - np.median(
                [
                    np.median(seed_values)
                    for seed_values in values["hyperopt_tpe"].values()
                ]
            )
        )
        for family, values in per_family.items()
    }
    claim_checks = {
        "friedman_omnibus_significant": statistics["friedman"]["p_value"] < 0.05,
        "holm_corrected_significance_vs_tpe": corrected_significance,
        "confidence_interval_excludes_zero": confidence_interval[0] > 0,
        "no_family_over_5_percent_median_degradation": all(
            degradation <= 0.05 for degradation in family_degradation.values()
        ),
    }
    report = {
        "gate": 3,
        "protocol_complete": True,
        "frozen_model_manifest_sha256": model_manifest_sha256,
        "frozen_model_artifact_size_bytes": artifact_size,
        "metrics": metrics,
        "summaries": external_summaries(metrics, args.seed),
        "statistics": statistics,
        "publication_claim": {
            "allowed": all(claim_checks.values()),
            "checks": claim_checks,
            "tpe_minus_atpe2_auc_task_stratified_95_percent_ci": confidence_interval,
            "family_median_auc_degradation": family_degradation,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
