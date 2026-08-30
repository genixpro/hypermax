import json
import math
import sqlite3

import numpy as np
import pytest

from hypermax.algorithms.atpe2.controller import ATPE2Action, ATPE2Controller
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from research.atpe2.database import ResearchStore, SCHEMA_VERSION
from research.atpe2.labels import label_state
from research.atpe2.iterate import summarize_iteration
from research.atpe2.compare import compare_reports
from research.atpe2 import generate as generate_module
from research.atpe2.generate import _third_repeat_candidates, action_candidates, rollout_action
from research.atpe2.acquisition import (
    acquisition_lineage,
    canonical_action as canonical_structured_action,
    structured_action_candidates,
)
from research.atpe2.factorized import (
    derive_factorized_rows,
    robust_support_artifact,
)
from research.atpe2.factorized_screen import assess_five_quick_screens
from research.atpe2.merge_databases import merge_database_shards
from research.atpe2.select_factorized import (
    select_behavioral_variants,
    select_manifest_variants,
)
from research.atpe2.fit_grid import BASE_BRANCHES, declared_jobs
from research.atpe2.train import grouped_split
from research.atpe2.train_factorized import train_factorized
from research.atpe2.gates import (
    configurations_equivalent,
    gate2_report,
    incumbent_metrics,
    paired_bootstrap_interval,
)
from research.atpe2.evidence import (
    boundary_diagnostics,
    structurally_inactive_locking_knobs,
)
from research.atpe2.synthetic import factorized_training_tasks, pilot_tasks
from research.atpe2.configspace_adapter import ConfigSpaceAdapter
from research.atpe2.validate import _clone_ensemble
from research.atpe2 import external as external_module
from research.atpe2.external import ExternalStore, run_trace


def test_research_store_is_resumable_and_retains_every_configuration(tmp_path):
    database = tmp_path / "research.sqlite"
    with ResearchStore(database) as store:
        store.upsert_task("task", "smooth", {"space": {}}, "generator-v1")
        state_id = store.upsert_state(
            "task",
            "random",
            "lineage",
            10,
            [],
            {"successful_count": 10},
            acquisition_metadata={"implementation_version": "test"},
        )
        record = {
            "state_id": state_id,
            "task_id": "task",
            "generator_version": "generator-v1",
            "history_lineage": "lineage",
            "configuration": {"gamma": 0.25},
            "rollout_seed": 7,
            "repeat_index": 0,
            "incumbent_curve": [5, 4, 3, 2, 1],
            "failures": [],
            "timing": {"seconds": 1},
            "dependency_versions": {"python": "test"},
            "utility": 0.5,
            "acquisition_role": "group_isolated_search_sobol",
            "varied_groups": ["search"],
        }
        assert store.add_evaluation(record) is True
        assert store.add_evaluation(record) is False
        assert len(store.state_evaluations(state_id)) == 1
        second = dict(record)
        second["configuration"] = {"gamma": 0.5}
        assert store.add_evaluation(second) is True
        rows = store.training_rows()
        # State payloads are parsed once and shared across rollout rows.
        assert rows[0]["history"] is rows[1]["history"]
        assert rows[0]["context"] is rows[1]["context"]
        assert rows[0]["acquisition_role"] == "group_isolated_search_sobol"
        assert rows[0]["varied_groups"] == ["search"]
        assert rows[0]["acquisition_metadata"]["implementation_version"] == "test"
        assert json.loads(store.connection.execute("SELECT value_json FROM metadata WHERE key='schema_version'").fetchone()[0]) == SCHEMA_VERSION

    with ResearchStore(database) as reopened:
        assert len(reopened.state_evaluations(state_id)) == 2


def test_research_store_migrates_v1_database_without_losing_rows(tmp_path):
    database = tmp_path / "old.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, family TEXT NOT NULL,
            specification_json TEXT NOT NULL, generator_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE states (
            state_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            trajectory_method TEXT NOT NULL, trajectory_lineage TEXT NOT NULL,
            checkpoint INTEGER NOT NULL, history_json TEXT NOT NULL,
            context_json TEXT NOT NULL, acquisition_round INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            UNIQUE(task_id, trajectory_lineage, checkpoint, acquisition_round)
        );
        CREATE TABLE evaluations (
            evaluation_id TEXT PRIMARY KEY, state_id TEXT NOT NULL,
            task_id TEXT NOT NULL, generator_version TEXT NOT NULL,
            history_lineage TEXT NOT NULL, configuration_json TEXT NOT NULL,
            configuration_hash TEXT NOT NULL, rollout_seed INTEGER NOT NULL,
            repeat_index INTEGER NOT NULL, incumbent_curve_json TEXT NOT NULL,
            failures_json TEXT NOT NULL, timing_json TEXT NOT NULL,
            dependency_versions_json TEXT NOT NULL, utility REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state_id, configuration_hash, rollout_seed, repeat_index)
        );
        """
    )
    connection.close()

    with ResearchStore(database) as store:
        state_columns = {
            row["name"] for row in store.connection.execute("PRAGMA table_info(states)")
        }
        evaluation_columns = {
            row["name"] for row in store.connection.execute("PRAGMA table_info(evaluations)")
        }
        assert "acquisition_metadata_json" in state_columns
        assert {"acquisition_role", "varied_groups_json"}.issubset(evaluation_columns)


def test_disjoint_database_shards_merge_resumably(tmp_path):
    sources = []
    for index in range(2):
        source = tmp_path / f"shard-{index}.sqlite"
        sources.append(source)
        with ResearchStore(source) as store:
            task_id = f"task-{index}"
            store.upsert_task(task_id, "smooth", {"space": {}}, "generator")
            state_id = store.upsert_state(
                task_id, "tpe", f"lineage-{index}", 20, [], {}, 0
            )
            store.add_evaluation(
                {
                    "state_id": state_id,
                    "task_id": task_id,
                    "generator_version": "generator",
                    "history_lineage": f"lineage-{index}",
                    "configuration": {"gamma": 0.25},
                    "rollout_seed": index,
                    "repeat_index": 0,
                    "incumbent_curve": [1.0] * 5,
                    "failures": [],
                    "timing": {},
                    "dependency_versions": {},
                    "utility": 0.5,
                }
            )
            store.mark_state_complete(state_id)
    destination = tmp_path / "authoritative.sqlite"
    first = merge_database_shards(destination, sources)
    second = merge_database_shards(destination, sources)

    assert first["evaluations_inserted"] == 2
    assert second["evaluations_inserted"] == 0
    with ResearchStore(destination) as store:
        assert len(store.training_rows()) == 2


def test_continuation_and_teacher_artifacts_are_resumable(tmp_path):
    database = tmp_path / "later.sqlite"
    with ResearchStore(database) as store:
        store.upsert_task(
            "task",
            "smooth",
            {"task_id": "task", "family": "smooth", "space": {}, "seed": 1},
            "generator",
        )
        state_id = store.upsert_state("task", "tpe", "lineage", 25, [], {}, 1)
        record = {
            "state_id": state_id,
            "configuration": ATPE2Action.anchor().__dict__,
            "acquisition_role": "neutral_anchor",
            "varied_groups": [],
            "rollout_seed": 7,
            "repeat_index": 0,
            "forced_curve": [5, 4, 3, 2, 1],
            "continuation_curve": [1] * 10,
            "failures": [],
            "provisional_manifest_sha256": "a" * 64,
        }
        assert store.add_continuation(record)
        assert not store.add_continuation(record)
        continuation = store.continuation_rows()[0]
        store.set_continuation_utilities(continuation["continuation_id"], 0.5, 0.75)
        store.upsert_teacher_label(
            {
                "state_id": state_id,
                "search_regime": "balanced",
                "filtering_mode": "none",
                "locking_mode": "fixed:top",
                "classic_utility": 0.5,
                "anchor_utility": 0.5,
                "eligible": True,
                "metadata": {"runtime_executes_classic_atpe": False},
            }
        )

        continuation = store.continuation_rows()[0]
        teacher = store.teacher_rows()[0]
        assert continuation["utility_15"] == pytest.approx(0.75)
        assert len(continuation["forced_curve"] + continuation["continuation_curve"]) == 15
        assert teacher["eligible"] is True
        assert teacher["metadata"]["runtime_executes_classic_atpe"] is False


def test_scale_free_label_ranks_each_paired_step():
    records = [
        {"configuration_hash": "a", "rollout_seed": 1, "repeat_index": 0, "incumbent_curve": [5, 4, 3, 2, 1]},
        {"configuration_hash": "b", "rollout_seed": 1, "repeat_index": 0, "incumbent_curve": [6, 6, 6, 6, 6]},
        {"configuration_hash": "c", "rollout_seed": 1, "repeat_index": 0, "incumbent_curve": [None, None, None, None, None]},
    ]
    labels = label_state(records)
    assert labels[("a", 1, 0)] == 1.0
    assert labels[("b", 1, 0)] == 0.5
    assert labels[("c", 1, 0)] == 0.0


def test_research_action_acquisition_has_64_sobol_points_plus_anchor():
    actions = action_candidates(seed=19)
    assert len(actions) == 65
    assert actions[-1].gamma == 0.25
    assert actions[-1].n_ei_candidates == 24
    assert all(0.1 <= action.gamma <= 2.0 for action in actions)
    assert all(8 <= action.n_ei_candidates <= 64 for action in actions[:-1])


def test_research_action_acquisition_obeys_evidence_expanded_ranges():
    actions = action_candidates(
        seed=21,
        action_ranges={
            "gamma": {
                "type": "number",
                "minimum": 2.0,
                "maximum": 3.0,
                "scale": "linear",
            }
        },
    )
    assert all(2.0 <= action.gamma <= 3.0 for action in actions[:-1])
    assert actions[-1] == ATPE2Action.anchor()


def test_structured_round_one_has_exact_roles_isolation_and_deterministic_refill():
    class Ensemble:
        is_factorized = False
        action_names = ATPE2Action.NAMES

        def predict(self, matrix):
            # Force deterministic ties so canonical tie-breaking is exercised.
            return np.zeros(len(matrix)), np.zeros(len(matrix))

    context = np.zeros(32)
    first = structured_action_candidates(
        19, ATPE2Action.anchor(), Ensemble(), context
    )
    second = structured_action_candidates(
        19, ATPE2Action.anchor(), Ensemble(), context
    )
    keys = [canonical_structured_action(record.action) for record in first]
    roles = [record.role for record in first]

    assert len(first) == len(set(keys)) == 65
    assert [record.lineage_record() for record in first] == [
        record.lineage_record() for record in second
    ]
    assert roles.count("neutral_anchor") == 1
    assert roles.count("provisional_policy") == 1
    assert sum(role.startswith("group_isolated_search") for role in roles) == 8
    assert sum(role.startswith("group_isolated_history") for role in roles) == 12
    assert sum(role.startswith("group_isolated_locking") for role in roles) == 12
    assert sum(role.startswith("two_group") for role in roles) == 18
    assert roles.count("full_action_regime") == 8
    assert roles.count("highest_disagreement") == 5
    isolated_keys = {
        group: {
            tuple(getattr(record.action, name) for name in names)
            for record in first
            if record.varied_groups == (group,)
        }
        | {tuple(getattr(ATPE2Action.anchor(), name) for name in names)}
        for group, names in {
            "search": ("gamma", "n_ei_candidates"),
            "history": ("history_keep_fraction", "history_recency_bias", "history_quality_bias"),
            "locking": ("secondary_fraction", "secondary_lock_probability", "secondary_top_fraction"),
        }.items()
    }
    for record in first:
        if len(record.varied_groups) == 1:
            group = record.varied_groups[0]
            neutral = ATPE2Action.anchor()
            for other_group in {"search", "history", "locking"} - {group}:
                names = {
                    "search": ("gamma", "n_ei_candidates"),
                    "history": ("history_keep_fraction", "history_recency_bias", "history_quality_bias"),
                    "locking": ("secondary_fraction", "secondary_lock_probability", "secondary_top_fraction"),
                }[other_group]
                assert record.action.vector_for(names).tolist() == neutral.vector_for(names).tolist()
        if record.role.startswith("two_group") or record.role == "full_action_regime":
            for group in record.varied_groups:
                names = {
                    "search": ("gamma", "n_ei_candidates"),
                    "history": ("history_keep_fraction", "history_recency_bias", "history_quality_bias"),
                    "locking": ("secondary_fraction", "secondary_lock_probability", "secondary_top_fraction"),
                }[group]
                assert tuple(getattr(record.action, name) for name in names) in isolated_keys[group]


def test_structured_lineage_hashes_manifest_and_ordered_mapping():
    class Ensemble:
        is_factorized = False
        action_names = ATPE2Action.NAMES

        def predict(self, matrix):
            return np.zeros(len(matrix)), np.arange(len(matrix), dtype=float)

    records = structured_action_candidates(
        7, ATPE2Action(gamma=0.5), Ensemble(), np.zeros(32)
    )
    first = acquisition_lineage(records, "a" * 64)
    changed_manifest = acquisition_lineage(records, "b" * 64)
    changed_order = acquisition_lineage(list(reversed(records)), "a" * 64)

    assert first != changed_manifest
    assert first != changed_order
    assert len(first["ordered_role_action_mapping"]) == 65


def test_factorized_targets_exclude_third_repeats_and_decompose_interactions():
    anchor = ATPE2Action.anchor()
    search = ATPE2Action(gamma=0.5)
    history = ATPE2Action(history_keep_fraction=0.5)
    combined = ATPE2Action(gamma=0.5, history_keep_fraction=0.5)
    configurations = [
        (anchor, "neutral_anchor", (), 0.50),
        (search, "group_isolated_search_sobol", ("search",), 0.60),
        (history, "group_isolated_history_sobol", ("history",), 0.55),
        (combined, "two_group_search_history", ("search", "history"), 0.70),
    ]
    rows = []
    for action, role, varied, utility in configurations:
        for repeat in range(3):
            rows.append(
                {
                    "state_id": "state",
                    "task_id": "task",
                    "family": "smooth",
                    "trajectory_lineage": "lineage",
                    "checkpoint": 20,
                    "acquisition_round": 1,
                    "context": {},
                    "history": [],
                    "space": {},
                    "configuration": action.__dict__,
                    "acquisition_role": role,
                    "varied_groups": varied,
                    "repeat_index": repeat,
                    "utility": -999.0 if repeat == 2 else utility,
                    "failures": [] if action != combined else [{"step": 1}],
                }
            )
    derived = derive_factorized_rows(rows)
    pair = next(row for row in derived if row["configuration"] == combined.__dict__)

    assert pair["utility"] == pytest.approx(0.70)
    assert pair["additive_prediction"] == pytest.approx(0.65)
    assert pair["residual_target"] == pytest.approx(0.05)
    assert pair["failed_step_fraction"] == pytest.approx(0.2)
    assert pair["catastrophic_label"] == 1


def test_support_artifact_is_deterministic_stratified_and_finite():
    matrix = np.arange(600 * 4, dtype=float).reshape(600, 4)
    tasks = [f"task-{index % 12}" for index in range(600)]
    first = robust_support_artifact(matrix, tasks, seed=3)
    second = robust_support_artifact(matrix, tasks, seed=3)

    assert first == second
    assert first["representative_count"] == 512
    assert np.all(np.isfinite(first["representatives"]))
    assert np.all(np.asarray(first["scale"]) > 0)
    assert np.all(np.diff(first["calibration"]["distance"]) >= 0)


@pytest.mark.slow
def test_factorized_miniature_database_fit_runtime_and_reload(tmp_path):
    class JointStub:
        is_factorized = False
        action_names = ATPE2Action.NAMES

        def predict(self, matrix):
            return np.zeros(len(matrix)), np.zeros(len(matrix))

    database = tmp_path / "miniature.sqlite"
    space = {"type": "object", "properties": {}}
    context = {
        "__feature_schema_version": "atpe2-context-v4",
        **generate_module.ATPE2ContextV4FeatureExtractor().extract(space, [], []),
    }
    with ResearchStore(database) as store:
        for task_index in range(6):
            task_id = f"task-{task_index}"
            store.upsert_task(
                task_id,
                "hierarchical" if task_index < 3 else "smooth",
                {"task_id": task_id, "family": "smooth", "space": space, "seed": task_index},
                "miniature",
            )
            for round_index in (0, 1):
                state_id = store.upsert_state(
                    task_id,
                    "tpe",
                    f"lineage-{task_index}-{round_index}",
                    20,
                    [],
                    context,
                    round_index,
                )
                if round_index == 0:
                    records = [
                        (action, "neutral_anchor" if action == ATPE2Action.anchor() else "round_zero_joint_sobol", () if action == ATPE2Action.anchor() else ("search", "history", "locking"))
                        for action in action_candidates(task_index)
                    ]
                else:
                    records = [
                        (record.action, record.role, record.varied_groups)
                        for record in structured_action_candidates(
                            task_index,
                            ATPE2Action.anchor(),
                            JointStub(),
                            np.zeros(32),
                        )
                    ]
                for action, role, varied in records:
                    vector = action.vector()
                    utility = float(
                        np.clip(
                            0.60
                            - 0.02 * abs(action.gamma - 0.5)
                            - 0.001 * abs(action.n_ei_candidates - 32)
                            + 0.01 * action.history_quality_bias
                            - 0.02 * action.secondary_lock_probability
                            + 0.005 * action.gamma * action.history_keep_fraction,
                            0.0,
                            1.0,
                        )
                    )
                    for repeat in range(2):
                        store.add_evaluation(
                            {
                                "state_id": state_id,
                                "task_id": task_id,
                                "generator_version": "miniature",
                                "history_lineage": f"lineage-{task_index}-{round_index}",
                                "configuration": action.__dict__,
                                "rollout_seed": task_index * 100 + repeat,
                                "repeat_index": repeat,
                                "incumbent_curve": [1.0 - utility] * 5,
                                "failures": [],
                                "timing": {},
                                "dependency_versions": {},
                                "utility": utility,
                                "acquisition_role": role,
                                "varied_groups": varied,
                            }
                        )
    output = tmp_path / "model"
    manifest = train_factorized(
        database,
        output,
        seed=5,
        branch_name="context_v4",
        trees=8,
        leaf_index=0,
    )
    reproduced = train_factorized(
        database,
        tmp_path / "model-reproduced",
        seed=5,
        branch_name="context_v4",
        trees=8,
        leaf_index=0,
    )
    loaded = ATPE2ModelEnsemble.load(output / "manifest.json")
    controller = ATPE2Controller(9, ensemble=loaded)
    action = controller.action_for(space, [], [])

    assert manifest["schema_version"] == "atpe2-factorized-manifest-v1"
    assert len(manifest["members"]) == 8
    assert reproduced["model_artifact_sha256"] == manifest["model_artifact_sha256"]
    assert reproduced["score_composition"] == manifest["score_composition"]
    assert reproduced["support_artifact"]["sha256"] == manifest["support_artifact"]["sha256"]
    assert loaded.research_ready
    assert isinstance(action, ATPE2Action)
    assert controller.last_record["decision_mode"] == "adaptive_model"
    assert controller.last_record["fallback_reason"] is None
    reproduced_support = tmp_path / "model-reproduced" / "support.json"
    reproduced_support.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ATPE2ModelEnsemble.load(tmp_path / "model-reproduced" / "manifest.json")


def test_five_quick_assessment_enforces_seed_family_classic_and_overhead_gates():
    def report(seed, improvement, regret, overhead):
        rows = []
        for family_index, family in enumerate(("smooth", "rugged", "interaction", "hierarchical")):
            common = {
                "task": f"{family}-{seed}",
                "family": family,
                "trajectory_lineage": f"lineage-{family}-{seed}",
                "checkpoint": 25,
                "seed": seed * 100 + family_index,
            }
            rows.extend(
                [
                    {
                        **common,
                        "optimizer": "atpe2",
                        "normalized_regret": regret,
                        "scale_normalized_improvement_auc": improvement,
                    },
                    {
                        **common,
                        "optimizer": "hyperopt_tpe",
                        "normalized_regret": 0.0,
                        "scale_normalized_improvement_auc": 0.0,
                    },
                    {
                        **common,
                        "optimizer": "legacy_atpe",
                        "normalized_regret": 0.0,
                        "scale_normalized_improvement_auc": 0.1,
                    },
                ]
            )
        return {
            "seed": seed,
            "rows": rows,
            "summary": {
                "screen_passed": True,
                "checks": {"all_atpe2_blocks_have_adaptive_decisions": True},
                "optimizer_overhead": {
                    "atpe2": {"median_block_seconds": overhead}
                },
            },
        }

    candidates = [report(seed, 0.06, -0.01, 1.2) for seed in (7, 19, 23, 31, 47)]
    incumbents = [report(seed, 0.0, 0.0, 1.0) for seed in (7, 19, 23, 31, 47)]
    assessment = assess_five_quick_screens(candidates, incumbents)

    assert assessment["passed"]
    assert assessment["classic_gap"]["deficit_closure_fraction"] >= 0.5
    assert assessment["controller_overhead_ratio"] == pytest.approx(1.2)


def test_variant_pruning_rejects_failed_gates_and_caps_both_stages(tmp_path):
    manifests = []
    for index in range(15):
        manifest = {
            "model_version": f"model-{index}",
            "model_artifact_sha256": f"{index:064d}",
            "architecture_branch": "context_v4",
            "component_gates": {"passed": index != 14},
            "score_grid": [
                {"held_out_stable_policy_quality": float(index)}
            ],
        }
        manifests.append((tmp_path / f"manifest-{index}.json", manifest))
    selected = select_manifest_variants(manifests)
    quality = {
        row["model_artifact_sha256"]: row["held_out_stable_policy_quality"]
        for row in selected
    }
    behavioral = select_behavioral_variants(selected, quality)

    assert len(selected) == 12
    assert all(row["model_version"] != "model-14" for row in selected)
    assert len(behavioral) == 6


def test_declared_grid_is_bounded_and_shardable_without_duplicates():
    jobs = declared_jobs(BASE_BRANCHES)

    assert len(jobs) == 3 + 5 * 3 * 9
    assert len({json.dumps(job, sort_keys=True) for job in jobs}) == len(jobs)
    assert {job["trees"] for job in jobs} == {8, 12, 16}


def test_third_repeat_selection_can_be_recomputed_from_paired_rows_only():
    paired = []
    for index in range(10):
        for repeat in range(2):
            paired.append(
                {
                    "configuration_hash": str(index),
                    "repeat_index": repeat,
                    "incumbent_curve": [float(index + repeat)] * 5,
                }
            )
    selected = _third_repeat_candidates(paired)
    resumed = paired + [
        {
            "configuration_hash": key,
            "repeat_index": 2,
            "incumbent_curve": [-1e9] * 5,
        }
        for key in selected
    ]
    assert _third_repeat_candidates(
        [row for row in resumed if row["repeat_index"] < 2]
    ) == selected


def test_offline_rollout_retains_optimizer_exceptions_as_five_failed_trials(
    monkeypatch,
):
    class BrokenOptimizer:
        def __init__(self, *args, **kwargs):
            self.controller = type("Controller", (), {})()

        def recommendNextParameters(self, *args, **kwargs):
            raise RuntimeError("broken optimizer")

    monkeypatch.setattr(generate_module, "ATPE2Optimizer", BrokenOptimizer)
    task = pilot_tasks(3)[0]
    curve, failures, timing = rollout_action(
        task,
        [{"status": "ok", "loss": -2.0}],
        ATPE2Action.anchor(),
        rollout_seed=17,
    )
    assert curve == [-2.0] * 5
    assert len(failures) == 5
    assert all("optimizer_exception:RuntimeError" in row["error"] for row in failures)
    assert timing["seconds"] >= 0


def test_boundary_diagnostics_expand_only_task_consistent_good_edges():
    rows = []
    for task in range(5):
        for index in range(20):
            rows.append(
                {
                    "task_id": f"task-{task}",
                    "state_id": f"state-{task}",
                    "utility": 1.0 if index < 5 else 0.0,
                    "configuration": {
                        "gamma": 0.1 if index < 5 else 1.0,
                        "n_ei_candidates": 24,
                        "history_keep_fraction": 1.0,
                        "history_recency_bias": 0.0,
                        "history_quality_bias": 0.0,
                        "secondary_fraction": 0.3,
                        "secondary_lock_probability": 0.5,
                        "secondary_top_fraction": 0.5,
                    },
                }
            )
    diagnostics, ranges = boundary_diagnostics(rows)
    assert diagnostics["gamma"]["expanded_lower"] is True
    assert ranges["gamma"]["minimum"] == 0.05
    assert diagnostics["secondary_fraction"]["expanded_lower"] is False


def test_neutral_locking_gate_prunes_every_behaviorally_inert_control():
    assert structurally_inactive_locking_knobs({"secondary_fraction"}) == [
        "secondary_lock_probability",
        "secondary_top_fraction",
    ]
    assert structurally_inactive_locking_knobs(
        {"secondary_lock_probability"}
    ) == ["secondary_fraction", "secondary_top_fraction"]


def test_ablation_overrides_keep_the_trained_action_schema_compatible():
    base = ATPE2ModelEnsemble.load_default()
    manifest = json.loads(json.dumps(base.manifest))
    manifest["status"] = "candidate"
    candidate = ATPE2ModelEnsemble(manifest, [object()] * 8)
    ablation = _clone_ensemble(candidate, {"gamma": 0.25})
    assert ablation.schema_compatible
    assert ablation.research_ready
    actions = ATPE2Controller(8, ensemble=ablation)._candidate_actions(3)
    assert {action.gamma for action in actions} == {0.25}


def test_grouped_split_never_leaks_tasks_or_lineages():
    rows = []
    for task in range(6):
        for lineage in range(2):
            for row in range(2):
                rows.append({"task_id": f"task-{task}", "trajectory_lineage": f"lineage-{task}-{lineage}", "family": "family"})
    split = grouped_split(rows, seed=4)
    task_sets = {
        name: {rows[index]["task_id"] for index in indices}
        for name, indices in split.items()
    }
    assert task_sets["train"].isdisjoint(task_sets["validation"])
    assert task_sets["train"].isdisjoint(task_sets["test"])
    assert task_sets["validation"].isdisjoint(task_sets["test"])


def test_grouped_split_stratifies_each_family_when_possible():
    rows = [
        {"task_id": f"{family}-{task}", "trajectory_lineage": f"{family}-{task}", "family": family}
        for family in ("smooth", "rugged")
        for task in range(6)
    ]
    split = grouped_split(rows, seed=14)
    for indices in split.values():
        assert {rows[index]["family"] for index in indices} == {"smooth", "rugged"}


def test_external_metric_keeps_failures_and_fixed_endpoints_finite():
    rows = [
        {
            "benchmark": "test",
            "scenario": "scenario",
            "family": "family",
            "task": "task",
            "optimizer": "atpe2",
            "seed": 1,
            "trial": trial,
            "status": "fail" if trial == 1 else "ok",
            "loss": None if trial == 1 else 100 - trial,
            "optimum": 0,
            "normalization_worst": 100,
        }
        for trial in range(1, 101)
    ]
    metric = incumbent_metrics(rows)[0]
    assert metric["failures"] == 1
    assert metric["regret"]["25"] == 0.75
    assert math.isfinite(metric["auc_incumbent_regret"])


def test_external_initial_design_comparison_allows_only_tiny_numeric_rounding():
    reference = {"x": 0.6666632854092085, "category": "a", "nested": [1, 2.0]}
    smac_serialized = {
        "x": 0.6666632854092,
        "category": "a",
        "nested": [1, 2.0000000000001],
    }
    assert configurations_equivalent(reference, smac_serialized)
    assert not configurations_equivalent(reference, {**smac_serialized, "x": 0.6666})
    assert not configurations_equivalent(reference, {**smac_serialized, "category": "b"})


def test_external_export_streams_valid_deterministic_json(tmp_path):
    database = tmp_path / "external.sqlite"
    output = tmp_path / "external.json"
    identity = {
        "benchmark": "yahpo",
        "scenario": "lcbench",
        "family": "lcbench",
        "task": "lcbench/1",
        "optimizer": "random",
        "seed": "test0",
    }
    with ExternalStore(database) as store:
        run_id = store.begin(identity)
        store.add(run_id, 0, {**identity, "trial": 0, "configuration": {"x": 1}})
        store.mark_complete(run_id)
        assert store.export(output) == 1
    assert json.loads(output.read_text())[0]["configuration"] == {"x": 1}


def test_external_runner_retains_optimizer_ask_failures(tmp_path, monkeypatch):
    class BrokenEngine:
        def ask(self):
            raise RuntimeError("broken ask")

    class Ensemble:
        manifest = {
            "schema_version": "test",
            "model_version": "test",
            "_source_manifest_sha256": "0" * 64,
        }

    monkeypatch.setattr(
        external_module,
        "make_engine",
        lambda *args, **kwargs: BrokenEngine(),
    )
    identity = {
        "benchmark": "yahpo",
        "scenario": "lcbench",
        "family": "lcbench",
        "task": "lcbench/1",
        "optimizer": "random",
        "seed": "test0",
    }
    with ExternalStore(tmp_path / "failures.sqlite") as store:
        run_trace(
            store,
            identity,
            adapter=None,
            initial=[],
            evaluate=lambda configuration, trial: (0.0, {}),
            optimum=0.0,
            worst=1.0,
            ensemble=Ensemble(),
            study_seed=7,
            budget=3,
        )
        run_id = store.begin(identity)
        rows = store.rows(run_id)
    assert len(rows) == 3
    assert all(row["status"] == "fail" for row in rows)
    assert all(row["configuration"] == {} for row in rows)


def test_paired_bootstrap_is_seeded_and_directional():
    first = paired_bootstrap_interval([0.1, 0.2, 0.3], seed=9, repeats=1000)
    second = paired_bootstrap_interval([0.1, 0.2, 0.3], seed=9, repeats=1000)
    assert first == second
    assert first[0] > 0


def test_gate2_rejects_reduced_smoke_protocol():
    rows = [
        {
            "task": "smooth-smoke",
            "family": "smooth",
            "trajectory_method": "tpe",
            "trajectory_lineage": "lineage",
            "checkpoint": 12,
            "seed": 1,
            "optimizer": optimizer,
            "five_step_utility": 0.5,
            "normalized_regret": 0.0,
        }
        for optimizer in ("atpe2", "hyperopt_tpe")
    ]
    report = gate2_report(
        rows,
        {"backward_knockout": {"retained_knobs": ["gamma"]}},
        [0, 1],
        [
            "filtering",
            "locking",
            "adaptive_gamma_candidates",
            "uncertainty_penalty",
            "full_vs_pruned_schema",
            "legacy_vs_robust_features",
        ],
        seed=3,
    )
    assert report["passed"] is False
    assert report["checks"]["full_held_out_protocol_complete"] is False


def test_iteration_summary_is_directional_and_requires_adaptive_diversity():
    rows = []
    for task_index in range(2):
        for optimizer in ("random", "hyperopt_tpe", "legacy_atpe", "atpe2"):
            rows.append(
                {
                    "task": f"task-{task_index}",
                    "family": "smooth",
                    "trajectory_lineage": f"lineage-{task_index}",
                    "checkpoint": 12,
                    "seed": task_index,
                    "optimizer": optimizer,
                    "five_step_utility": 0.5,
                    "rank_utility_auc": 0.5,
                    "scale_normalized_improvement_auc": 0.1,
                    "normalized_regret": 0.0,
                    "optimizer_overhead_seconds": 0.01,
                    "controller_decisions": (
                        [
                            {
                                "decision_mode": "adaptive_model",
                                "fallback_reason": None,
                                "outside_training_envelope": False,
                                "configuration": {
                                    "gamma": 0.5 + task_index,
                                },
                            }
                        ]
                        if optimizer == "atpe2"
                        else []
                    ),
                }
            )

    summary = summarize_iteration(rows, seed=11)

    assert summary["screen_passed"] is True
    assert summary["adaptive_telemetry"]["decision_count"] == 2
    assert summary["adaptive_telemetry"]["distinct_configurations"] == 2


def test_iteration_comparison_uses_absolute_paired_improvement_not_rank_scores():
    families = ("smooth", "rugged", "interaction", "hierarchical")

    def report(candidate_delta=0.0):
        return {
            "seed": 7,
            "model_manifest_sha256": f"model-{candidate_delta}",
            "summary": {
                "optimizer_overhead": {
                    "atpe2": {"median_block_seconds": 1.0 + candidate_delta}
                }
            },
            "rows": [
                {
                    "task": f"{family}-task",
                    "family": family,
                    "trajectory_method": "tpe",
                    "trajectory_lineage": f"{family}-lineage",
                    "checkpoint": 25,
                    "seed": index,
                    "optimizer": "atpe2",
                    "initial_incumbent": 10.0,
                    "incumbent_curve": [9.0 - candidate_delta],
                    "scale_normalized_improvement_auc": 0.1 + candidate_delta,
                }
                for index, family in enumerate(families)
            ],
        }

    comparison = compare_reports(report(), report(0.05), seed=9)

    assert comparison["performance_family_candidate_minus_baseline"]["wins"] == 4
    assert comparison["performance_family_candidate_minus_baseline"]["mean_delta"] == pytest.approx(0.05)
    assert comparison["task_median_95_percent_interval"] == pytest.approx((0.05, 0.05))
    assert comparison["exact_incumbent_curve_matches"] == 0


def test_synthetic_suite_contains_zero_plateau_and_irrelevant_dimensions():
    tasks = pilot_tasks(5)
    zero = next(task for task in tasks if task.family == "constant_zero")
    params = {"x0": 0.2, "x1": 1.0, "x2": 1, "irrelevant": -1.0}
    assert zero.evaluate(params, 10, 0)["loss"] == 0.0
    smooth = next(task for task in tasks if task.family == "smooth")
    first = smooth.evaluate(params, 10, 0)["loss"]
    params["irrelevant"] = 1.0
    second = smooth.evaluate(params, 10, 0)["loss"]
    assert first == second


def test_factorized_training_task_mix_is_192_with_40_percent_hierarchical():
    tasks = factorized_training_tasks(17)
    counts = {
        family: sum(task.family == family for task in tasks)
        for family in {task.family for task in tasks}
    }

    assert len(tasks) == 192
    assert counts["hierarchical"] == 77
    assert set(counts.values()) == {23, 77}


def test_configspace_adapter_preserves_nested_conditional_activity():
    ConfigSpace = pytest.importorskip("ConfigSpace")
    from ConfigSpace.conditions import EqualsCondition, InCondition
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    space = ConfigSpace.ConfigurationSpace(seed=4)
    learner = CategoricalHyperparameter("learner", ["linear", "tree"])
    booster = CategoricalHyperparameter("booster", ["gbtree", "dart"])
    depth = UniformIntegerHyperparameter("depth", 1, 8)
    dropout = UniformFloatHyperparameter("dropout", 0.0, 1.0)
    space.add([learner, booster, depth, dropout])
    space.add(EqualsCondition(booster, learner, "tree"))
    space.add(InCondition(depth, booster, ["gbtree", "dart"]))
    space.add(EqualsCondition(dropout, booster, "dart"))
    adapter = ConfigSpaceAdapter(space)

    dart = {"learner": "tree", "booster": "dart", "depth": 4, "dropout": 0.3}
    structured = adapter.to_structured(dart)
    assert adapter.to_flat(structured) == dart
    assert structured["learner"]["booster"]["dropout"] == 0.3

    linear = {"learner": "linear"}
    structured_linear = adapter.to_structured(linear)
    assert adapter.to_flat(structured_linear) == linear
    assert "booster" not in structured_linear["learner"]
