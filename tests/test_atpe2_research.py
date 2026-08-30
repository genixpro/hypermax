import json
import math

import pytest

from hypermax.algorithms.atpe2.controller import ATPE2Action, ATPE2Controller
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from research.atpe2.database import ResearchStore, SCHEMA_VERSION
from research.atpe2.labels import label_state
from research.atpe2.iterate import summarize_iteration
from research.atpe2 import generate as generate_module
from research.atpe2.generate import _third_repeat_candidates, action_candidates, rollout_action
from research.atpe2.train import grouped_split
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
from research.atpe2.synthetic import pilot_tasks
from research.atpe2.configspace_adapter import ConfigSpaceAdapter
from research.atpe2.validate import _clone_ensemble
from research.atpe2 import external as external_module
from research.atpe2.external import ExternalStore, run_trace


def test_research_store_is_resumable_and_retains_every_configuration(tmp_path):
    database = tmp_path / "research.sqlite"
    with ResearchStore(database) as store:
        store.upsert_task("task", "smooth", {"space": {}}, "generator-v1")
        state_id = store.upsert_state("task", "random", "lineage", 10, [], {"successful_count": 10})
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
        assert json.loads(store.connection.execute("SELECT value_json FROM metadata WHERE key='schema_version'").fetchone()[0]) == SCHEMA_VERSION

    with ResearchStore(database) as reopened:
        assert len(reopened.state_evaluations(state_id)) == 2


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
