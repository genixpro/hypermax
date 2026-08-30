import json
import math

import numpy as np
import pytest

from hypermax.algorithms.atpe2.controller import ATPE2Action, ATPE2Controller
from hypermax.algorithms.atpe2.features import (
    ATPE2ContextV4FeatureExtractor,
    ATPE2FeatureExtractor,
    describe_space,
    parameter_relevance,
)
from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble
from hypermax.algorithms.atpe2.legacy_features import LegacyATPEFeatureExtractor
from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer
from hypermax.optimizer import Optimizer


MIXED_SPACE = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "min": -3, "max": 3},
        "rounded": {"type": "number", "min": 0, "max": 8, "rounding": 1},
        "log": {"type": "number", "min": 0.01, "max": 10, "scaling": "logarithmic"},
        "normal": {"type": "number", "mode": "normal", "mean": 0, "stddev": 1},
        "kind": {"enum": ["a", "b", "c"]},
        "branch": {
            "oneOf": [
                {"type": "object", "properties": {"left": {"type": "number", "min": -1, "max": 1}}},
                {"type": "object", "properties": {"right": {"enum": ["r", "s"]}}},
            ]
        },
    },
}


def result(index, loss, status="ok"):
    branch = index % 2
    return {
        "trial": index,
        "status": status,
        "loss": loss,
        "x": float(index % 7 - 3),
        "rounded": index % 9,
        "log": 0.01 * (index + 1),
        "normal": float(index % 5 - 2),
        "kind": ["a", "b", "c"][index % 3],
        "branch": branch,
        "branch.0.left": float(index) if branch == 0 else "",
        "branch.1.right": ["r", "s"][index % 2] if branch == 1 else "",
    }


@pytest.mark.parametrize(
    "losses",
    [
        [-4, -2, 0, 3, 8],
        [0, 0, 0, 0, 0],
        [1, 1, 2, 2, 2],
        [1e-300, 2e-300, 3e-300, 4e-300, 5e-300],
        [-1e300, -5e299, 0, 5e299, 1e300],
    ],
)
def test_features_are_finite_for_pathological_losses(losses):
    rows = [result(index, loss) for index, loss in enumerate(losses)]
    rows += [result(90, None, "fail"), result(91, float("nan"))]
    features = ATPE2FeatureExtractor().extract(MIXED_SPACE, rows, [{"params": {}}])

    assert list(features) == list(ATPE2FeatureExtractor.FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features.values())
    assert features["failed_count"] == 2
    assert features["pending_count"] == 1


def test_features_are_translation_and_positive_scale_invariant():
    losses = np.asarray([-8.0, -3.0, 0.0, 2.0, 9.0, 10.0, 10.0, 12.0, 15.0, 20.0, 21.0, 30.0])
    base = ATPE2FeatureExtractor().extract(MIXED_SPACE, [result(i, value) for i, value in enumerate(losses)])
    transformed = ATPE2FeatureExtractor().extract(
        MIXED_SPACE, [result(i, 1e6 * value - 12345) for i, value in enumerate(losses)]
    )

    assert base.keys() == transformed.keys()
    assert np.allclose(list(base.values()), list(transformed.values()), rtol=1e-10, atol=1e-10)
    tiny = ATPE2FeatureExtractor().extract(
        MIXED_SPACE, [result(i, 1e-300 * value) for i, value in enumerate(losses)]
    )
    huge = ATPE2FeatureExtractor().extract(
        MIXED_SPACE, [result(i, 1e300 * value) for i, value in enumerate(losses)]
    )
    assert np.allclose(list(base.values()), list(tiny.values()), rtol=1e-10, atol=1e-10)
    assert np.allclose(list(base.values()), list(huge.values()), rtol=1e-10, atol=1e-10)


def test_context_v4_appends_scale_free_local_summaries():
    losses = [8.0, 7.0, 7.5, 6.0, 5.0, 5.5, 4.0, 3.0, 3.5, 2.0, 1.0, 1.5]
    rows = [result(index, loss) for index, loss in enumerate(losses)]
    extractor = ATPE2ContextV4FeatureExtractor()
    original = extractor.extract(MIXED_SPACE, rows)
    transformed = extractor.extract(
        MIXED_SPACE,
        [result(index, -17.0 + 1e6 * loss) for index, loss in enumerate(losses)],
    )

    assert extractor.version == "atpe2-context-v4"
    assert tuple(original) == extractor.FEATURE_NAMES
    assert np.all(np.isfinite(list(original.values())))
    for name in extractor.V4_FEATURE_NAMES:
        assert original[name] == pytest.approx(transformed[name])


def test_parameter_relevance_is_conservative_for_small_or_constant_activity():
    descriptors = describe_space(MIXED_SPACE)
    relevance = parameter_relevance(descriptors, [result(index, index) for index in range(4)])
    assert set(relevance) == {descriptor.name for descriptor in descriptors}
    assert all(value == 1.0 for value in relevance.values())


def test_weighted_filter_keeps_exact_ceiling_without_replacement_and_is_seeded():
    rows = [result(index, float(index)) for index in range(20)]
    action = ATPE2Action(history_keep_fraction=0.26, history_recency_bias=2, history_quality_bias=-1)
    first = ATPE2Controller(17).filter_history(rows, action, boundary=4)
    second = ATPE2Controller(17).filter_history(rows, action, boundary=4)
    other = ATPE2Controller(18).filter_history(rows, action, boundary=4)

    assert len(first) == math.ceil(0.26 * len(rows))
    assert len({row["trial"] for row in first}) == len(first)
    assert [row["trial"] for row in first] == [row["trial"] for row in second]
    assert [row["trial"] for row in first] != [row["trial"] for row in other]


def test_conditional_and_categorical_locking_only_uses_compatible_observations():
    rows = [result(index, float((index * 7) % 13)) for index in range(30)]
    controller = ATPE2Controller(9)
    action = ATPE2Action(
        secondary_fraction=1.0,
        secondary_lock_probability=1.0,
        secondary_top_fraction=0.5,
    )
    locked = controller.locking_values(MIXED_SPACE, rows, action, boundary=6)

    assert locked.get("kind") in {None, "a", "b", "c"}
    if locked.get("branch") == 0:
        assert "branch.1.right" not in locked
    if locked.get("branch") == 1:
        assert "branch.0.left" not in locked


def test_categorical_locking_and_tiny_top_fraction_never_use_an_empty_pool():
    space = {
        "type": "object",
        "properties": {"category": {"enum": ["a", "b", "c"]}},
    }
    rows = [
        {
            "trial": index,
            "status": "ok",
            "loss": 0.0,
            "category": ["a", "b", "c"][index % 3],
        }
        for index in range(15)
    ]
    locked = ATPE2Controller(77).locking_values(
        space,
        rows,
        ATPE2Action(
            secondary_fraction=1.0,
            secondary_lock_probability=1.0,
            secondary_top_fraction=0.02,
        ),
        boundary=3,
    )
    assert locked["category"] in {"a", "b", "c"}


def test_controller_caches_configuration_for_five_completed_trials():
    controller = ATPE2Controller(1)
    rows = [result(index, float(index)) for index in range(4)]
    first = controller.action_for(MIXED_SPACE, rows, [])
    second = controller.action_for(MIXED_SPACE, rows + [result(4, 4.0)], [])
    third = controller.action_for(MIXED_SPACE, rows + [result(4, 4.0), result(5, 5.0)], [])

    assert second == third
    assert len(controller.records) == 2
    assert controller.records[-1]["fallback_reason"] is None
    assert controller.records[-1]["decision_mode"] == "adaptive_model"
    assert json.loads(json.dumps(controller.last_record))["seed"] == 1


def test_fitted_candidate_scores_1024_joint_sobol_actions_plus_anchor():
    class StubEnsemble:
        manifest = {
            "status": "candidate",
            "model_version": "stub",
            "feature_schema_version": ATPE2FeatureExtractor.version,
            "feature_schema_hash": "stub-feature",
            "action_schema_hash": "stub-action",
            "action_ranges": {},
            "validation_selected_constants": {},
        }
        action_names = ATPE2Action.NAMES
        runtime_candidate_count = 1024
        research_ready = True
        schema_compatible = True
        uncertainty_penalty = 0.5
        @staticmethod
        def in_envelope(context):
            return False

        @staticmethod
        def predict(matrix):
            gamma_column = len(ATPE2FeatureExtractor.FEATURE_NAMES)
            return matrix[:, gamma_column], np.zeros(len(matrix))

    controller = ATPE2Controller(91, ensemble=StubEnsemble())
    assert len(controller._candidate_actions(2)) == 1025
    rows = [result(index, float(index)) for index in range(20)]
    selected = controller.action_for(MIXED_SPACE, rows, [])

    assert selected.gamma > 1.99
    assert controller.last_record["fallback_reason"] is None
    assert controller.last_record["decision_mode"] == "adaptive_model"
    assert controller.last_record["outside_training_envelope"] is True
    assert controller.last_record["expected_utility"] == pytest.approx(selected.gamma)


def test_manifest_controls_runtime_candidate_count_without_changing_the_anchor():
    ensemble = ATPE2ModelEnsemble.load_default()
    manifest = json.loads(json.dumps(ensemble.manifest))
    manifest["runtime_candidate_count"] = 256
    candidate = ATPE2ModelEnsemble(manifest, ensemble.models)

    actions = ATPE2Controller(92, ensemble=candidate)._candidate_actions(2)

    assert len(actions) == 257
    assert actions[-1] == ATPE2Action.anchor()


def test_factorized_runtime_learns_inactive_groups_and_never_falls_back():
    class FactorizedStub:
        is_factorized = True
        research_ready = True
        action_names = ATPE2Action.NAMES
        feature_extractor_class = ATPE2FeatureExtractor
        manifest = {
            "runtime": {
                "group_proposals": 17,
                "retained_per_group": 6,
                "beam_width": 64,
                "max_supported_distance": 1.0,
            },
            "score_composition": {
                "group_weights": {"search": 1.0, "history": 1.0, "locking": 1.0},
                "residual_weight": 0.5,
                "downside_weight": 0.5,
                "support_weight": 0.25,
            },
            "feature_schema_version": ATPE2FeatureExtractor.version,
            "model_version": "factorized-test",
        }

        def predict_group(self, group, matrix):
            widths = {"search": 2, "history": 3, "locking": 3}
            anchor = {
                "search": ATPE2Action.anchor().vector_for(("gamma", "n_ei_candidates")),
                "history": ATPE2Action.anchor().vector_for(
                    ("history_keep_fraction", "history_recency_bias", "history_quality_bias")
                ),
                "locking": ATPE2Action.anchor().vector_for(
                    ("secondary_fraction", "secondary_lock_probability", "secondary_top_fraction")
                ),
            }[group]
            values = matrix[:, -widths[group] :]
            score = -np.sum((values - anchor) ** 2, axis=1)
            return score, np.full(len(matrix), 0.05)

        def predict_residual(self, matrix):
            return np.zeros(len(matrix)), np.full(len(matrix), 0.02)

        def predict_downside(self, matrix):
            return {
                "harm": np.zeros(len(matrix)),
                "variability": np.zeros(len(matrix)),
                "failure": np.zeros(len(matrix)),
                "catastrophic": np.zeros(len(matrix)),
            }

        def support_cost(self, matrix):
            return np.zeros(len(matrix))

        def in_envelope(self, context):
            return True

    controller = ATPE2Controller(33, ensemble=FactorizedStub())
    selected = controller.action_for(MIXED_SPACE, [], [])

    assert selected == ATPE2Action.anchor()
    assert controller.last_record["decision_mode"] == "adaptive_model"
    assert controller.last_record["fallback_reason"] is None
    assert controller.last_record["candidate_count"] <= 65
    assert all(
        decision["inactive"]
        for decision in controller.last_record["group_decisions"].values()
    )
    assert math.isfinite(controller.last_record["support_cost"])


def test_non_power_of_two_runtime_candidate_count_is_schema_incompatible():
    ensemble = ATPE2ModelEnsemble.load_default()
    manifest = json.loads(json.dumps(ensemble.manifest))
    manifest["runtime_candidate_count"] = 300

    assert ATPE2ModelEnsemble(manifest, ensemble.models).schema_compatible is False


def test_atpe2_samples_mixed_conditional_space_and_is_same_seed_deterministic():
    first = ATPE2Optimizer(seed=42)
    second = ATPE2Optimizer(seed=42)
    different = ATPE2Optimizer(seed=43)
    samples_a = [first.recommendNextParameters(MIXED_SPACE, [], []) for _ in range(3)]
    samples_b = [second.recommendNextParameters(MIXED_SPACE, [], []) for _ in range(3)]
    samples_c = [different.recommendNextParameters(MIXED_SPACE, [], []) for _ in range(3)]

    assert samples_a == samples_b
    assert samples_a != samples_c
    for sample in samples_a:
        assert sample["kind"] in {"a", "b", "c"}
        assert sample["branch"]["$index"] in {0, 1}


def test_atpe2_ignores_failed_missing_and_nonfinite_history():
    rows = [result(0, -1.0), result(1, 0.0), result(2, None, "fail"), result(3, float("nan"))]
    del rows[0]["normal"]  # Historical CSV rows may predate a parameter column.
    sample = ATPE2Optimizer(seed=2).recommendNextParameters(MIXED_SPACE, rows, [])
    assert -3 <= sample["x"] <= 3
    assert math.isfinite(sample["normal"])


def test_atpe2_records_effective_history_and_locking_telemetry():
    rows = [result(index, float(index)) for index in range(20)]
    optimizer = ATPE2Optimizer(seed=2)
    optimizer.recommendNextParameters(MIXED_SPACE, rows, [])

    record = optimizer.controller.last_record
    assert record["successful_history_count"] == 20
    assert 1 <= record["filtered_history_count"] <= 20
    assert record["locked_parameters"] == sorted(record["locked_parameters"])


def test_frozen_manifest_and_models_load_from_package_resources():
    ensemble = ATPE2ModelEnsemble.load_default()
    assert ensemble.ready is True
    assert ensemble.manifest["status"] == "frozen"
    assert len(ensemble.models) == 8
    assert ensemble.manifest["validation_evidence"]["gate2_passed"] is True
    assert ensemble.schema_compatible


def test_manifest_schema_requires_an_exact_retained_or_fixed_knob_partition():
    ensemble = ATPE2ModelEnsemble.load_default()
    malformed = json.loads(json.dumps(ensemble.manifest))
    malformed["retained_action_names"] = list(ATPE2Action.NAMES[:-1])
    assert ATPE2ModelEnsemble(malformed, []).schema_compatible is False

    malformed["validation_selected_constants"] = {
        "secondary_top_fraction": 1.0
    }
    # The hash still describes the old schema, so merely adding the missing
    # constant cannot bypass the schema checksum.
    assert ATPE2ModelEnsemble(malformed, []).schema_compatible is False


def test_legacy_ablation_feature_layout_matches_historical_models_and_is_finite():
    from hypermax.algorithms.atpe_optimizer import ATPEOptimizer

    assert list(LegacyATPEFeatureExtractor.FEATURE_NAMES) == sorted(
        ATPEOptimizer.atpeModelFeatureKeys
    )
    rows = [result(index, float(index - 10)) for index in range(20)]
    features = LegacyATPEFeatureExtractor().extract(MIXED_SPACE, rows)
    assert len(features) == 113
    assert all(math.isfinite(value) for value in features.values())


def test_historical_percentile_index_regression_uses_valid_percentiles():
    values = np.arange(1.0, 101.0)
    minimizing = LegacyATPEFeatureExtractor._distribution(values, best_high=False)
    maximizing = LegacyATPEFeatureExtractor._distribution(values, best_high=True)

    assert minimizing["best_percentile25_ratio"] == pytest.approx(1.0 / 25.75)
    assert maximizing["best_percentile25_ratio"] == pytest.approx(100.0 / 75.25)


@pytest.mark.parametrize("method", ["atpe2", "abh2"])
def test_optimizer_accepts_new_compatibility_methods(tmp_path, method):
    config = {
        "hyperparameters": {"type": "object", "properties": {"x": {"type": "number", "min": -1, "max": 1}}},
        "function": {"type": "python_function", "func": lambda params: params["x"] ** 2},
        "search": {"method": method, "iterations": 1, "seed": 3, "min_budget": 1, "max_budget": 3},
        "results": {"directory": str(tmp_path / "results"), "graphs": False},
        "ui": {"enabled": False},
    }
    optimizer = Optimizer(config)
    sample = optimizer.sampleNext()
    assert -1 <= sample["x"] <= 1
    if method == "abh2":
        assert "$budget" in sample


def test_invalid_seed_is_rejected(tmp_path):
    config = {
        "hyperparameters": {"type": "object", "properties": {}},
        "function": {"type": "python_function", "func": lambda params: 0},
        "search": {"method": "atpe2", "iterations": 1, "seed": True},
        "results": {"directory": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="search.seed"):
        Optimizer(config)


def test_atpe2_runtime_writes_controller_telemetry(tmp_path):
    config = {
        "hyperparameters": {"type": "object", "properties": {"x": {"type": "number", "min": -1, "max": 1}}},
        "function": {"type": "python_function", "func": lambda params: params["x"] ** 2},
        "search": {"method": "atpe2", "iterations": 1, "seed": 33},
        "results": {"directory": str(tmp_path / "results"), "graphs": False},
        "ui": {"enabled": False},
    }
    optimizer = Optimizer(config)
    optimizer.runOptimizationThread()
    optimizer.thread.join(timeout=20)
    telemetry = json.loads((tmp_path / "results_0" / "atpe2_controller.json").read_text())
    assert telemetry[0]["seed"] == 33
    assert telemetry[0]["fallback_reason"] is None
    assert telemetry[0]["decision_mode"] == "adaptive_model"
