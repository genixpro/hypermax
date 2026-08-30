"""Merge disjoint ATPE2 task shards into the authoritative research store."""

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


def merge_database_shards(destination: Path, sources: list[Path]) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one source shard is required")
    source_records = []
    copied_states = copied_evaluations = 0
    with ResearchStore(destination) as output:
        for source in sources:
            connection = sqlite3.connect(source)
            connection.row_factory = sqlite3.Row
            try:
                state_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(states)")
                }
                evaluation_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(evaluations)")
                }
                for row in connection.execute("SELECT * FROM tasks ORDER BY task_id"):
                    output.upsert_task(
                        row["task_id"],
                        row["family"],
                        _json(row, "specification_json"),
                        row["generator_version"],
                    )
                state_ids: dict[str, str] = {}
                for row in connection.execute("SELECT * FROM states ORDER BY state_id"):
                    state_id = output.upsert_state(
                        row["task_id"],
                        row["trajectory_method"],
                        row["trajectory_lineage"],
                        int(row["checkpoint"]),
                        _json(row, "history_json"),
                        _json(row, "context_json"),
                        int(row["acquisition_round"]),
                        (
                            _json(row, "acquisition_metadata_json")
                            if "acquisition_metadata_json" in state_columns
                            else {}
                        ),
                    )
                    state_ids[row["state_id"]] = state_id
                    if row["status"] == "complete":
                        output.mark_state_complete(state_id)
                    copied_states += 1
                for row in connection.execute(
                    "SELECT * FROM evaluations ORDER BY evaluation_id"
                ):
                    inserted = output.add_evaluation(
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
                            "dependency_versions": _json(
                                row, "dependency_versions_json"
                            ),
                            "utility": row["utility"],
                            "acquisition_role": (
                                row["acquisition_role"]
                                if "acquisition_role" in evaluation_columns
                                else "legacy_joint"
                            ),
                            "varied_groups": (
                                _json(row, "varied_groups_json")
                                if "varied_groups_json" in evaluation_columns
                                else []
                            ),
                        }
                    )
                    copied_evaluations += int(inserted)
                payload = source.read_bytes()
                source_records.append(
                    {
                        "name": source.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                )
            finally:
                connection.close()
        prior_row = output.connection.execute(
            "SELECT value_json FROM metadata WHERE key='merged_shards'"
        ).fetchone()
        prior = json.loads(prior_row[0]) if prior_row else []
        output.set_metadata("merged_shards", prior + source_records)
    return {
        "destination": str(destination),
        "sources": source_records,
        "states_seen": copied_states,
        "evaluations_inserted": copied_evaluations,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args(argv)
    print(json.dumps(merge_database_shards(args.destination, args.sources), indent=2))


if __name__ == "__main__":
    main()
