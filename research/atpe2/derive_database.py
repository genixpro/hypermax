"""Create the factorized training database from a pristine round-zero snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from .database import ResearchStore


def _json(row: sqlite3.Row, name: str) -> Any:
    return json.loads(row[name])


def create_factorized_database(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(
            "Derived factorized database must be new; refusing to mix prior acquisition rounds"
        )
    digest = hashlib.sha256()
    for path in (source, Path(f"{source}-wal")):
        if path.exists():
            payload = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    source_hash = digest.hexdigest()
    source_connection = sqlite3.connect(source)
    source_connection.row_factory = sqlite3.Row
    try:
        rounds = {
            int(row[0])
            for row in source_connection.execute(
                "SELECT DISTINCT acquisition_round FROM states"
            )
        }
        if rounds - {0}:
            raise ValueError(
                "Source database is not pristine round zero; previous joint-acquisition rounds are prohibited"
            )
        with ResearchStore(destination) as destination_store:
            tasks = source_connection.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
            for row in tasks:
                destination_store.upsert_task(
                    row["task_id"],
                    row["family"],
                    _json(row, "specification_json"),
                    row["generator_version"],
                )
            states = source_connection.execute(
                "SELECT * FROM states WHERE acquisition_round=0 ORDER BY state_id"
            ).fetchall()
            state_ids: dict[str, str] = {}
            state_columns = {column[1] for column in source_connection.execute("PRAGMA table_info(states)")}
            for row in states:
                state_id = destination_store.upsert_state(
                    row["task_id"],
                    row["trajectory_method"],
                    row["trajectory_lineage"],
                    int(row["checkpoint"]),
                    _json(row, "history_json"),
                    _json(row, "context_json"),
                    0,
                    (
                        _json(row, "acquisition_metadata_json")
                        if "acquisition_metadata_json" in state_columns
                        else {}
                    ),
                )
                state_ids[row["state_id"]] = state_id
                if row["status"] == "complete":
                    destination_store.mark_state_complete(state_id)
            evaluations = source_connection.execute(
                """SELECT e.* FROM evaluations e
                   JOIN states s ON s.state_id=e.state_id
                   WHERE s.acquisition_round=0
                   ORDER BY e.evaluation_id"""
            ).fetchall()
            evaluation_columns = {
                column[1]
                for column in source_connection.execute("PRAGMA table_info(evaluations)")
            }
            for row in evaluations:
                destination_store.add_evaluation(
                    {
                        "state_id": state_ids[row["state_id"]],
                        "task_id": row["task_id"],
                        "generator_version": row["generator_version"],
                        "history_lineage": row["history_lineage"],
                        "configuration": _json(row, "configuration_json"),
                        "rollout_seed": int(row["rollout_seed"]),
                        "repeat_index": int(row["repeat_index"]),
                        "incumbent_curve": _json(row, "incumbent_curve_json"),
                        "failures": _json(row, "failures_json"),
                        "timing": _json(row, "timing_json"),
                        "dependency_versions": _json(row, "dependency_versions_json"),
                        "utility": row["utility"],
                        "acquisition_role": (
                            row["acquisition_role"]
                            if "acquisition_role" in evaluation_columns
                            else "round_zero_joint_sobol"
                        ),
                        "varied_groups": (
                            _json(row, "varied_groups_json")
                            if "varied_groups_json" in evaluation_columns
                            else ["search", "history", "locking"]
                        ),
                    }
                )
            destination_store.set_metadata(
                "factorized_derivation",
                {
                    "schema_version": "atpe2-factorized-derived-db-v1",
                    "source_database_sha256": source_hash,
                    "source_rounds": [0],
                    "copied_states": len(states),
                    "copied_evaluations": len(evaluations),
                    "prohibits_previous_joint_round_one": True,
                },
            )
    finally:
        source_connection.close()
    return {
        "source_database_sha256": source_hash,
        "destination": str(destination),
        "states": len(states),
        "evaluations": len(evaluations),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(create_factorized_database(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
