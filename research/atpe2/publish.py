"""Publish compact, checksummed ATPE2 evidence without vendoring raw benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any


SCHEMA_VERSION = "atpe2-published-evidence-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _external_counts(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """SELECT benchmark,COUNT(*),SUM(status='complete'),SUM(status='failed'),
                      COUNT(DISTINCT task)
               FROM runs GROUP BY benchmark ORDER BY benchmark"""
        ).fetchall()
        return {
            "by_benchmark": {
                benchmark: {
                    "runs": int(runs),
                    "complete_runs": int(complete),
                    "failed_runs": int(failed),
                    "tasks": int(tasks),
                }
                for benchmark, runs, complete, failed, tasks in rows
            },
            "trace_rows": int(
                connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            ),
            "failed_trials": int(
                connection.execute(
                    "SELECT COUNT(*) FROM traces WHERE json_extract(row_json,'$.status')!='ok'"
                ).fetchone()[0]
            ),
        }
    finally:
        connection.close()


def publish(
    output: Path,
    gate1_path: Path,
    pruning_path: Path,
    gate2_path: Path,
    gate3_path: Path,
    split_path: Path,
    model_manifest: Path,
    external_database: Path,
    raw_export: Path,
    seed: int,
) -> dict[str, Any]:
    gate1 = json.loads(gate1_path.read_text())
    pruning = json.loads(pruning_path.read_text())
    gate2 = json.loads(gate2_path.read_text())
    gate3 = json.loads(gate3_path.read_text())
    if not gate1.get("passed"):
        raise ValueError("Published evidence requires a passing Gate 1 report")
    if not gate2.get("gate2", {}).get("passed"):
        raise ValueError("Published evidence requires a passing Gate 2 report")
    if not gate3.get("protocol_complete"):
        raise ValueError("Published evidence requires a complete Gate 3 protocol")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_path, output / "splits.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "publication_seed": int(seed),
        "gate1": gate1,
        "pruning": pruning,
        "gate2": {key: value for key, value in gate2.items() if key != "rows"},
        "gate3": {key: value for key, value in gate3.items() if key != "metrics"},
        "external_counts": _external_counts(external_database),
        "artifacts": {
            "gate1_report": _artifact(gate1_path),
            "pruning_report": _artifact(pruning_path),
            "gate2_report": _artifact(gate2_path),
            "gate3_report": _artifact(gate3_path),
            "grouped_splits": _artifact(output / "splits.json"),
            "frozen_model_manifest": _artifact(model_manifest),
            "external_database": _artifact(external_database),
            "raw_external_trace_export": _artifact(raw_export),
        },
        "third_party_data_redistributed": False,
    }
    (output / "evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("gate1", type=Path)
    parser.add_argument("pruning", type=Path)
    parser.add_argument("gate2", type=Path)
    parser.add_argument("gate3", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("model_manifest", type=Path)
    parser.add_argument("external_database", type=Path)
    parser.add_argument("raw_export", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    result = publish(
        args.output,
        args.gate1,
        args.pruning,
        args.gate2,
        args.gate3,
        args.splits,
        args.model_manifest,
        args.external_database,
        args.raw_export,
        args.seed,
    )
    print(json.dumps({"output": str(args.output), "schema_version": result["schema_version"]}))


if __name__ == "__main__":
    main()
