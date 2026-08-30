"""Three-hour local throughput gate using the real resumable generator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import time

import psutil

from .database import ResearchStore, content_hash
from .generate import GENERATOR_VERSION, TRAJECTORY_METHODS, generate_state
from .synthetic import pilot_tasks
from .train import train_ensemble


MIB = 1024**2
GIB = 1024**3


def database_footprint(path: Path) -> int:
    """Include SQLite's main database, WAL, and shared-memory files."""
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def _generation_loop(
    database: str,
    worker_index: int,
    phase_seed: int,
    stop: mp.Event,
    state_counter: mp.Value,
    configuration_counter: mp.Value,
    evaluation_counter: mp.Value,
) -> None:
    """Generate complete labeled production-format states until stopped."""
    templates = pilot_tasks(phase_seed + worker_index * 10_000)
    state_index = 0
    with ResearchStore(database) as store:
        while not stop.is_set():
            template = templates[state_index % len(templates)]
            task_seed = phase_seed * 1_000_000 + worker_index * 10_000 + state_index
            task = replace(
                template,
                task_id=f"pilot-{phase_seed}-{worker_index}-{state_index}-{template.family}",
                seed=task_seed,
            )
            store.upsert_task(task.task_id, task.family, task.specification(), GENERATOR_VERSION)
            method = TRAJECTORY_METHODS[state_index % len(TRAJECTORY_METHODS)]
            trajectory_seed = task_seed + 17
            lineage = content_hash(
                [task.specification(), GENERATOR_VERSION, method, trajectory_seed]
            )
            state_id = content_hash(
                [task.task_id, lineage, (12, 20, 40)[state_index % 3], 0]
            )
            before = len(store.state_evaluations(state_id))
            state_id = generate_state(
                store,
                task,
                method,
                trajectory_seed,
                checkpoint=(12, 20, 40)[state_index % 3],
                acquisition_round=0,
            )
            new_configurations = len(store.state_evaluations(state_id)) - before
            if new_configurations:
                with state_counter.get_lock():
                    state_counter.value += 1
            with configuration_counter.get_lock():
                configuration_counter.value += new_configurations
            with evaluation_counter.get_lock():
                evaluation_counter.value += new_configurations * 5
            state_index += 1


def benchmark_workers(
    database: Path,
    worker_count: int,
    seconds: int,
    phase_seed: int,
) -> dict:
    stop = mp.Event()
    state_counter = mp.Value("Q", 0)
    configuration_counter = mp.Value("Q", 0)
    evaluation_counter = mp.Value("Q", 0)
    processes = [
        mp.Process(
            target=_generation_loop,
            args=(
                str(database),
                worker_index,
                phase_seed,
                stop,
                state_counter,
                configuration_counter,
                evaluation_counter,
            ),
        )
        for worker_index in range(worker_count)
    ]
    started = time.monotonic()
    for process in processes:
        process.start()
    observed_processes = [psutil.Process(process.pid) for process in processes]
    for observed in observed_processes:
        observed.cpu_percent(interval=None)
    peak_rss = 0
    cpu_samples = []
    while time.monotonic() - started < seconds:
        rss = 0
        cpu = 0.0
        for observed in observed_processes:
            try:
                rss += observed.memory_info().rss
                cpu += observed.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_rss = max(peak_rss, rss)
        cpu_samples.append(cpu)
        time.sleep(0.25)
    stop.set()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"Pilot workers failed with exit codes {failures}")
    elapsed = time.monotonic() - started
    configurations = int(configuration_counter.value)
    states = int(state_counter.value)
    evaluations = int(evaluation_counter.value)
    aggregate_cpu = sum(cpu_samples) / max(1, len(cpu_samples))
    return {
        "workers": worker_count,
        "seconds": elapsed,
        "state_blocks": states,
        "state_blocks_per_minute": states * 60 / elapsed,
        "configuration_blocks": configurations,
        "configuration_blocks_per_minute": configurations * 60 / elapsed,
        "objective_evaluations": evaluations,
        "objective_evaluations_per_second": evaluations / elapsed,
        "peak_resident_memory_bytes": peak_rss,
        "peak_resident_memory_gib": peak_rss / GIB,
        "mean_aggregate_cpu_percent": aggregate_cpu,
        "mean_machine_cpu_percent": aggregate_cpu / max(1, psutil.cpu_count()),
    }


def run_pilot(
    database: Path,
    output: Path,
    seed: int,
    benchmark_seconds: int,
    total_seconds: int,
    projected_main_states: int = 10_000,
) -> dict:
    if benchmark_seconds < 1 or total_seconds < benchmark_seconds * 5:
        raise ValueError("total_seconds must cover all five worker benchmarks")
    started = time.monotonic()
    initial_database_size = database_footprint(database)
    benchmarks = [
        benchmark_workers(database, workers, benchmark_seconds, seed + phase * 100_000)
        for phase, workers in enumerate((1, 2, 4, 6, 8))
    ]
    eligible = [row for row in benchmarks if row["peak_resident_memory_bytes"] < 18 * GIB]
    if not eligible:
        raise RuntimeError("No worker count stayed below the 18 GB limit")
    chosen = max(eligible, key=lambda row: row["state_blocks_per_minute"])
    remaining = max(0, int(math.ceil(total_seconds - (time.monotonic() - started))))
    generation = (
        benchmark_workers(database, chosen["workers"], remaining, seed + 9_000_000)
        if remaining > 0
        else None
    )
    collection_wall_seconds = time.monotonic() - started

    model_output = output.parent / f"{output.stem}-models"
    with ResearchStore(database) as store:
        labeled_task_count = len(
            {row["task_id"] for row in store.training_rows()}
        )
    manifest = None
    model_fit_time = None
    if labeled_task_count >= 3:
        fit_started = time.perf_counter()
        manifest = train_ensemble(database, model_output, seed)
        model_fit_time = time.perf_counter() - fit_started

    database_size = database_footprint(database)
    all_phases = benchmarks + ([generation] if generation else [])
    states = sum(row["state_blocks"] for row in all_phases)
    configurations = sum(row["configuration_blocks"] for row in all_phases)
    evaluations = sum(row["objective_evaluations"] for row in all_phases)
    generation_seconds = sum(row["seconds"] for row in all_phases)
    state_rate = states / generation_seconds if generation_seconds else 0.0
    configuration_rate = configurations / generation_seconds if generation_seconds else 0.0
    bytes_per_state = max(0, database_size - initial_database_size) / max(1, states)
    projected_seconds = projected_main_states / max(state_rate, 1e-12)
    report = {
        "gate": 1,
        "passed": (
            collection_wall_seconds >= total_seconds
            and chosen["peak_resident_memory_bytes"] < 18 * GIB
            and manifest is not None
        ),
        "platform": platform.platform(),
        "seed": seed,
        "required_total_seconds": total_seconds,
        "benchmark_seconds_per_worker_count": benchmark_seconds,
        "benchmarks": benchmarks,
        "selected_workers": chosen["workers"],
        "generation": generation,
        "totals": {
            "collection_wall_seconds": collection_wall_seconds,
            "generation_seconds": generation_seconds,
            "state_blocks": states,
            "state_blocks_per_minute": state_rate * 60,
            "configuration_blocks": configurations,
            "configuration_blocks_per_minute": configuration_rate * 60,
            "objective_evaluations": evaluations,
            "objective_evaluations_per_second": evaluations / max(generation_seconds, 1e-12),
        },
        "database_size_bytes": database_size,
        "database_size_mib": database_size / MIB,
        "model_fit_time_seconds": model_fit_time,
        "model_manifest": str(model_output / "manifest.json") if manifest else None,
        "model_validation": manifest["validation"] if manifest else None,
        "labeled_task_count": labeled_task_count,
        "projected_main_dataset": {
            "assumed_state_blocks": projected_main_states,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600,
            "projected_storage_bytes": bytes_per_state * projected_main_states,
            "estimated_bytes_per_state_block": bytes_per_state,
        },
        "wall_clock_seconds_including_model_fit": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--benchmark-seconds", type=int, default=180)
    parser.add_argument("--total-seconds", type=int, default=3 * 60 * 60)
    parser.add_argument("--projected-main-states", type=int, default=10_000)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_pilot(
                args.database,
                args.output,
                args.seed,
                args.benchmark_seconds,
                args.total_seconds,
                args.projected_main_states,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
