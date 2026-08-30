"""SQLite persistence for resumable ATPE2 state/action rollouts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterator


SCHEMA_VERSION = "atpe2-research-sqlite-v3"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class ResearchStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ResearchStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                specification_json TEXT NOT NULL,
                generator_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS states (
                state_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                trajectory_method TEXT NOT NULL,
                trajectory_lineage TEXT NOT NULL,
                checkpoint INTEGER NOT NULL,
                history_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                acquisition_metadata_json TEXT NOT NULL DEFAULT '{}',
                acquisition_round INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                UNIQUE(task_id, trajectory_lineage, checkpoint, acquisition_round)
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                state_id TEXT NOT NULL REFERENCES states(state_id),
                task_id TEXT NOT NULL,
                generator_version TEXT NOT NULL,
                history_lineage TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                rollout_seed INTEGER NOT NULL,
                repeat_index INTEGER NOT NULL,
                incumbent_curve_json TEXT NOT NULL,
                failures_json TEXT NOT NULL,
                timing_json TEXT NOT NULL,
                dependency_versions_json TEXT NOT NULL,
                acquisition_role TEXT NOT NULL DEFAULT 'legacy_joint',
                varied_groups_json TEXT NOT NULL DEFAULT '[]',
                utility REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(state_id, configuration_hash, rollout_seed, repeat_index)
            );
            CREATE INDEX IF NOT EXISTS evaluations_state ON evaluations(state_id);
            CREATE INDEX IF NOT EXISTS states_lineage ON states(task_id, trajectory_lineage);
            CREATE TABLE IF NOT EXISTS continuation_evaluations (
                continuation_id TEXT PRIMARY KEY,
                state_id TEXT NOT NULL REFERENCES states(state_id),
                configuration_json TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                acquisition_role TEXT NOT NULL,
                varied_groups_json TEXT NOT NULL,
                rollout_seed INTEGER NOT NULL,
                repeat_index INTEGER NOT NULL,
                forced_curve_json TEXT NOT NULL,
                continuation_curve_json TEXT NOT NULL,
                failures_json TEXT NOT NULL,
                provisional_manifest_sha256 TEXT NOT NULL,
                utility_5 REAL,
                utility_15 REAL,
                UNIQUE(state_id, configuration_hash, rollout_seed, repeat_index)
            );
            CREATE INDEX IF NOT EXISTS continuations_state
                ON continuation_evaluations(state_id);
            CREATE TABLE IF NOT EXISTS classic_teacher_labels (
                state_id TEXT PRIMARY KEY REFERENCES states(state_id),
                search_regime TEXT NOT NULL,
                filtering_mode TEXT NOT NULL,
                locking_mode TEXT NOT NULL,
                classic_utility REAL NOT NULL,
                anchor_utility REAL NOT NULL,
                eligible INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        # SQLite has no portable ``ADD COLUMN IF NOT EXISTS``.  These guarded
        # migrations keep pristine v1 research databases readable and resumable.
        state_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(states)")
        }
        if "acquisition_metadata_json" not in state_columns:
            self.connection.execute(
                "ALTER TABLE states ADD COLUMN acquisition_metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        evaluation_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(evaluations)")
        }
        if "acquisition_role" not in evaluation_columns:
            self.connection.execute(
                "ALTER TABLE evaluations ADD COLUMN acquisition_role TEXT NOT NULL DEFAULT 'legacy_joint'"
            )
        if "varied_groups_json" not in evaluation_columns:
            self.connection.execute(
                "ALTER TABLE evaluations ADD COLUMN varied_groups_json TEXT NOT NULL DEFAULT '[]'"
            )
        self.set_metadata("schema_version", SCHEMA_VERSION)

    def set_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, canonical_json(value)),
        )
        self.connection.commit()

    def upsert_task(self, task_id: str, family: str, specification: dict, generator_version: str) -> None:
        self.connection.execute(
            "INSERT INTO tasks(task_id, family, specification_json, generator_version) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(task_id) DO NOTHING",
            (task_id, family, canonical_json(specification), generator_version),
        )
        self.connection.commit()

    def task_specifications(self, task_ids: set[str] | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT task_id, specification_json FROM tasks ORDER BY task_id"
        ).fetchall()
        return [
            json.loads(row["specification_json"])
            for row in rows
            if task_ids is None or row["task_id"] in task_ids
        ]

    def upsert_state(
        self,
        task_id: str,
        method: str,
        lineage: str,
        checkpoint: int,
        history: list[dict],
        context: dict,
        acquisition_round: int = 0,
        acquisition_metadata: dict[str, Any] | None = None,
    ) -> str:
        identity = [task_id, lineage, checkpoint, acquisition_round]
        state_id = content_hash(identity)
        self.connection.execute(
            """INSERT INTO states(
                state_id, task_id, trajectory_method, trajectory_lineage, checkpoint,
                history_json, context_json, acquisition_metadata_json,
                acquisition_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                history_json=excluded.history_json,
                context_json=excluded.context_json,
                acquisition_metadata_json=excluded.acquisition_metadata_json
            """,
            (
                state_id,
                task_id,
                method,
                lineage,
                checkpoint,
                canonical_json(history),
                canonical_json(context),
                canonical_json(acquisition_metadata or {}),
                acquisition_round,
            ),
        )
        self.connection.commit()
        return state_id

    def add_evaluation(self, record: dict[str, Any]) -> bool:
        configuration_hash = content_hash(record["configuration"])
        evaluation_id = content_hash(
            [record["state_id"], configuration_hash, record["rollout_seed"], record["repeat_index"]]
        )
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO evaluations(
                evaluation_id, state_id, task_id, generator_version, history_lineage,
                configuration_json, configuration_hash, rollout_seed, repeat_index,
                incumbent_curve_json, failures_json, timing_json,
                dependency_versions_json, acquisition_role, varied_groups_json,
                utility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluation_id,
                record["state_id"],
                record["task_id"],
                record["generator_version"],
                record["history_lineage"],
                canonical_json(record["configuration"]),
                configuration_hash,
                int(record["rollout_seed"]),
                int(record["repeat_index"]),
                canonical_json(record["incumbent_curve"]),
                canonical_json(record.get("failures", [])),
                canonical_json(record.get("timing", {})),
                canonical_json(record.get("dependency_versions", {})),
                str(record.get("acquisition_role", "legacy_joint")),
                canonical_json(record.get("varied_groups", [])),
                record.get("utility"),
            ),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def has_evaluation(
        self, state_id: str, configuration: dict[str, Any], rollout_seed: int, repeat_index: int
    ) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM evaluations WHERE state_id=? AND configuration_hash=? AND rollout_seed=? AND repeat_index=?",
            (state_id, content_hash(configuration), int(rollout_seed), int(repeat_index)),
        ).fetchone()
        return row is not None

    def task_action_summaries(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT configuration_json, AVG(utility) AS mean_utility,
                      AVG(utility * utility) - AVG(utility) * AVG(utility) AS variance
               FROM evaluations WHERE task_id=? AND utility IS NOT NULL
               GROUP BY configuration_hash, configuration_json""",
            (task_id,),
        ).fetchall()
        return [
            {
                "configuration": json.loads(row["configuration_json"]),
                "mean_utility": float(row["mean_utility"]),
                "uncertainty": math.sqrt(max(0.0, float(row["variance"] or 0.0))),
            }
            for row in rows
        ]

    def state_evaluations(self, state_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM evaluations WHERE state_id=? ORDER BY configuration_hash, rollout_seed, repeat_index",
            (state_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "configuration": json.loads(row["configuration_json"]),
                "incumbent_curve": json.loads(row["incumbent_curve_json"]),
                "failures": json.loads(row["failures_json"]),
                "timing": json.loads(row["timing_json"]),
                "dependency_versions": json.loads(row["dependency_versions_json"]),
                "acquisition_role": row["acquisition_role"],
                "varied_groups": json.loads(row["varied_groups_json"]),
            }
            for row in rows
        ]

    def training_rows(
        self, task_ids: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Load labeled rows without multiplying large state payloads in memory.

        A state history is shared by roughly 140 configuration rollouts.  Joining
        and parsing it on every evaluation row made a long pilot require tens of
        gigabytes at fit time.  Parse each state once, retain shared object
        references, and stream the small evaluation projection instead.
        """
        selected_tasks = set(task_ids) if task_ids is not None else None
        states: dict[str, dict[str, Any]] = {}
        state_cursor = self.connection.execute(
            """SELECT s.state_id,s.task_id,s.trajectory_lineage,s.checkpoint,
                      s.acquisition_round,s.context_json,s.history_json,
                      s.acquisition_metadata_json,
                      t.family,t.specification_json
               FROM states s JOIN tasks t ON t.task_id=s.task_id
               ORDER BY s.state_id"""
        )
        for row in state_cursor:
            if selected_tasks is not None and row["task_id"] not in selected_tasks:
                continue
            states[row["state_id"]] = {
                "state_id": row["state_id"],
                "task_id": row["task_id"],
                "family": row["family"],
                "trajectory_lineage": row["trajectory_lineage"],
                "checkpoint": row["checkpoint"],
                "acquisition_round": row["acquisition_round"],
                "acquisition_metadata": json.loads(row["acquisition_metadata_json"]),
                "context": json.loads(row["context_json"]),
                "history": json.loads(row["history_json"]),
                "space": json.loads(row["specification_json"]).get("space", {}),
            }
        output = []
        evaluation_cursor = self.connection.execute(
            """SELECT state_id,configuration_json,utility,rollout_seed,
                      repeat_index,incumbent_curve_json,failures_json,
                      acquisition_role,varied_groups_json
               FROM evaluations WHERE utility IS NOT NULL
               ORDER BY state_id,configuration_hash,rollout_seed,repeat_index"""
        )
        for row in evaluation_cursor:
            state = states.get(row["state_id"])
            if state is None:
                continue
            output.append(
                {
                    **state,
                    "configuration": json.loads(row["configuration_json"]),
                    "utility": float(row["utility"]),
                    "rollout_seed": int(row["rollout_seed"]),
                    "repeat_index": int(row["repeat_index"]),
                    "incumbent_curve": json.loads(row["incumbent_curve_json"]),
                    "failures": json.loads(row["failures_json"]),
                    "acquisition_role": row["acquisition_role"],
                    "varied_groups": json.loads(row["varied_groups_json"]),
                }
            )
        return output

    def mark_state_complete(self, state_id: str) -> None:
        self.connection.execute("UPDATE states SET status='complete' WHERE state_id=?", (state_id,))
        self.connection.commit()

    def add_continuation(self, record: dict[str, Any]) -> bool:
        configuration_hash = content_hash(record["configuration"])
        continuation_id = content_hash(
            [
                record["state_id"],
                configuration_hash,
                record["rollout_seed"],
                record["repeat_index"],
            ]
        )
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO continuation_evaluations(
                   continuation_id,state_id,configuration_json,configuration_hash,
                   acquisition_role,varied_groups_json,rollout_seed,repeat_index,
                   forced_curve_json,continuation_curve_json,failures_json,
                   provisional_manifest_sha256,utility_5,utility_15
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                continuation_id,
                record["state_id"],
                canonical_json(record["configuration"]),
                configuration_hash,
                record["acquisition_role"],
                canonical_json(record.get("varied_groups", [])),
                int(record["rollout_seed"]),
                int(record["repeat_index"]),
                canonical_json(record["forced_curve"]),
                canonical_json(record["continuation_curve"]),
                canonical_json(record.get("failures", [])),
                record["provisional_manifest_sha256"],
                record.get("utility_5"),
                record.get("utility_15"),
            ),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def has_continuation(
        self,
        state_id: str,
        configuration: dict[str, Any],
        rollout_seed: int,
        repeat_index: int,
    ) -> bool:
        row = self.connection.execute(
            """SELECT 1 FROM continuation_evaluations
               WHERE state_id=? AND configuration_hash=?
                     AND rollout_seed=? AND repeat_index=?""",
            (
                state_id,
                content_hash(configuration),
                int(rollout_seed),
                int(repeat_index),
            ),
        ).fetchone()
        return row is not None

    def continuation_rows(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT c.*,s.task_id,s.trajectory_lineage,s.checkpoint,
                      s.context_json,s.history_json,t.family,t.specification_json
               FROM continuation_evaluations c
               JOIN states s ON s.state_id=c.state_id
               JOIN tasks t ON t.task_id=s.task_id
               ORDER BY c.state_id,c.configuration_hash,c.rollout_seed,c.repeat_index"""
        ).fetchall()
        return [
            {
                **dict(row),
                "configuration": json.loads(row["configuration_json"]),
                "varied_groups": json.loads(row["varied_groups_json"]),
                "forced_curve": json.loads(row["forced_curve_json"]),
                "continuation_curve": json.loads(row["continuation_curve_json"]),
                "failures": json.loads(row["failures_json"]),
                "context": json.loads(row["context_json"]),
                "history": json.loads(row["history_json"]),
                "space": json.loads(row["specification_json"]).get("space", {}),
            }
            for row in rows
        ]

    def set_continuation_utilities(
        self, continuation_id: str, utility_5: float, utility_15: float
    ) -> None:
        self.connection.execute(
            "UPDATE continuation_evaluations SET utility_5=?,utility_15=? WHERE continuation_id=?",
            (float(utility_5), float(utility_15), continuation_id),
        )
        self.connection.commit()

    def upsert_teacher_label(self, record: dict[str, Any]) -> None:
        self.connection.execute(
            """INSERT INTO classic_teacher_labels(
                   state_id,search_regime,filtering_mode,locking_mode,
                   classic_utility,anchor_utility,eligible,metadata_json
               ) VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(state_id) DO UPDATE SET
                   search_regime=excluded.search_regime,
                   filtering_mode=excluded.filtering_mode,
                   locking_mode=excluded.locking_mode,
                   classic_utility=excluded.classic_utility,
                   anchor_utility=excluded.anchor_utility,
                   eligible=excluded.eligible,
                   metadata_json=excluded.metadata_json""",
            (
                record["state_id"],
                record["search_regime"],
                record["filtering_mode"],
                record["locking_mode"],
                float(record["classic_utility"]),
                float(record["anchor_utility"]),
                int(bool(record["eligible"])),
                canonical_json(record.get("metadata", {})),
            ),
        )
        self.connection.commit()

    def teacher_rows(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT l.*,s.task_id,s.context_json,s.history_json,
                      t.family,t.specification_json
               FROM classic_teacher_labels l
               JOIN states s ON s.state_id=l.state_id
               JOIN tasks t ON t.task_id=s.task_id
               ORDER BY l.state_id"""
        ).fetchall()
        return [
            {
                **dict(row),
                "eligible": bool(row["eligible"]),
                "metadata": json.loads(row["metadata_json"]),
                "context": json.loads(row["context_json"]),
                "history": json.loads(row["history_json"]),
                "space": json.loads(row["specification_json"]).get("space", {}),
            }
            for row in rows
        ]

    def state_records(
        self,
        task_ids: set[str] | None = None,
        acquisition_round: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if acquisition_round is not None:
            clauses.append("s.acquisition_round=?")
            parameters.append(int(acquisition_round))
        query = (
            "SELECT s.*,t.family,t.specification_json FROM states s "
            "JOIN tasks t ON t.task_id=s.task_id"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY s.state_id"
        rows = self.connection.execute(query, parameters).fetchall()
        return [
            {
                **dict(row),
                "history": json.loads(row["history_json"]),
                "context": json.loads(row["context_json"]),
                "space": json.loads(row["specification_json"]).get("space", {}),
                "task_specification": json.loads(row["specification_json"]),
            }
            for row in rows
            if task_ids is None or row["task_id"] in task_ids
        ]
