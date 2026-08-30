"""Resumable, shardable execution of the declared ATPE2 model grid."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any

from .continuation import CONTINUATION_LONG_WEIGHTS
from .factorized import ABLATION_BRANCHES, LEAF_GRID, TREE_GRID
from .train import train_ensemble
from .train_factorized import train_factorized


BASE_BRANCHES = (
    "structured_joint",
    "factorized_no_residual",
    "factorized_residual",
    "residual_downside",
    "downside_support",
    "context_v4",
)


def declared_jobs(branches: tuple[str, ...]) -> list[dict[str, Any]]:
    unknown = set(branches) - set(ABLATION_BRANCHES)
    if unknown:
        raise ValueError(f"Unknown branches: {sorted(unknown)}")
    jobs = []
    for branch in branches:
        weights = (
            CONTINUATION_LONG_WEIGHTS
            if ABLATION_BRANCHES[branch].get("multi_horizon")
            else (0.0,)
        )
        leaf_indices = (0,) if branch == "structured_joint" else range(len(LEAF_GRID))
        for trees, leaf_index, long_weight in product(
            TREE_GRID, leaf_indices, weights
        ):
            jobs.append(
                {
                    "branch": branch,
                    "trees": trees,
                    "leaf_index": leaf_index,
                    "long_horizon_weight": long_weight,
                }
            )
    return jobs


def run_grid_shard(
    database: Path,
    output: Path,
    seed: int,
    branches: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("grid shard must satisfy 0 <= index < count")
    jobs = [
        job
        for index, job in enumerate(declared_jobs(branches))
        if index % shard_count == shard_index
    ]
    results = []
    for job in jobs:
        suffix = (
            f"{job['branch']}-t{job['trees']}-l{job['leaf_index']}"
            f"-h{str(job['long_horizon_weight']).replace('.', '_')}"
        )
        directory = output / suffix
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        elif job["branch"] == "structured_joint":
            manifest = train_ensemble(
                database,
                directory,
                seed,
                n_estimators=job["trees"],
                runtime_candidate_count=64,
            )
            manifest["architecture_branch"] = "structured_joint"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
        else:
            manifest = train_factorized(
                database,
                directory,
                seed,
                job["branch"],
                job["trees"],
                job["leaf_index"],
                True,
                job["long_horizon_weight"],
            )
        results.append(
            {
                **job,
                "manifest": str(manifest_path),
                "model_version": manifest["model_version"],
                "component_gates_passed": manifest.get("component_gates", {}).get(
                    "passed"
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / f"grid-shard-{shard_index}-of-{shard_count}.json"
    index_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branches", nargs="+", choices=tuple(ABLATION_BRANCHES), default=BASE_BRANCHES)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    results = run_grid_shard(
        args.database,
        args.output,
        args.seed,
        tuple(args.branches),
        args.shard_index,
        args.shard_count,
    )
    print(json.dumps({"fits": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
