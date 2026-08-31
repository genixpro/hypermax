"""Generate reusable same-host comparators for factorized behavior screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

from .iterate import (
    COMPARATOR_CACHE_SCHEMA_VERSION,
    COMPARATOR_OPTIMIZERS,
    ITERATION_CHECKPOINTS,
    ITERATION_INSTANCES_PER_FAMILY,
    ITERATION_FAMILIES,
    ITERATION_ROLLOUT_SEEDS,
    ITERATION_SCHEMA_VERSION,
    ITERATION_STEPS,
    ITERATION_TRAJECTORIES,
    _task_rows,
    environment_identity,
)
from .synthetic import synthetic_tasks


def run_comparator_cache(
    output: Path,
    seed: int,
    workers: int = 6,
    task_families: tuple[str, ...] = ITERATION_FAMILIES,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    tasks = synthetic_tasks(
        seed + 70_000_000,
        instances_per_family=ITERATION_INSTANCES_PER_FAMILY,
    )
    unknown = set(task_families) - set(ITERATION_FAMILIES)
    if unknown or not task_families:
        raise ValueError(f"Unknown or empty comparator families: {sorted(unknown)}")
    tasks = [task for task in tasks if task.family in task_families]
    jobs = [
        (task, task_index, seed, None, COMPARATOR_OPTIMIZERS)
        for task_index, task in enumerate(tasks)
    ]
    if workers == 1:
        blocks = [_task_rows(*job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            blocks = pool.starmap(_task_rows, jobs)
    rows = [row for block in blocks for row in block]
    report = {
        "schema_version": COMPARATOR_CACHE_SCHEMA_VERSION,
        "iteration_schema_version": ITERATION_SCHEMA_VERSION,
        "seed": seed,
        "optimizers": list(COMPARATOR_OPTIMIZERS),
        "environment": environment_identity(),
        "task_families": list(task_families),
        "protocol": {
            "trajectory_methods": list(ITERATION_TRAJECTORIES),
            "checkpoints": list(ITERATION_CHECKPOINTS),
            "rollout_seeds": ITERATION_ROLLOUT_SEEDS,
            "steps": ITERATION_STEPS,
            "instances_per_family": ITERATION_INSTANCES_PER_FAMILY,
        },
        "task_specification_sha256": hashlib.sha256(
            json.dumps(
                [task.specification() for task in tasks],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--families", nargs="+", choices=ITERATION_FAMILIES, default=ITERATION_FAMILIES
    )
    args = parser.parse_args(argv)
    report = run_comparator_cache(
        args.output, args.seed, args.workers, tuple(args.families)
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema_version",
                    "seed",
                    "optimizers",
                    "task_specification_sha256",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
