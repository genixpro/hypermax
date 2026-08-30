"""ATPE2 optimizer: robust history handling around a jointly-scored controller."""

from __future__ import annotations

import functools
import hashlib
import json
from typing import Any

import hyperopt
import numpy as np

from .optimization_algorithm_base import OptimizationAlgorithmBase
from .atpe2.controller import ATPE2Controller
from .atpe2.features import successful_results
from hypermax.hyperparameter import Hyperparameter


class ATPE2Optimizer(OptimizationAlgorithmBase):
    def __init__(self, seed: int = 0, ensemble=None):
        self.seed = int(seed)
        self.controller = ATPE2Controller(self.seed, ensemble=ensemble)
        self.sample_count = 0
        self.lastATPEParameters = None
        self.lastLockedParameters = []
        self.atpeParamDetails = None

    def recommendNextParameters(
        self,
        hyperparameterSpace: dict[str, Any],
        results: list[dict[str, Any]],
        currentTrials: list[dict[str, Any]],
        lockedValues: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action = self.controller.action_for(hyperparameterSpace, results, currentTrials)
        boundary = len(results) // 5
        history = self.controller.filter_history(results, action, boundary)
        adaptive_locks = self.controller.locking_values(hyperparameterSpace, history, action, boundary)
        if self.controller.last_record is not None:
            self.controller.last_record.update(
                {
                    "successful_history_count": len(successful_results(results)),
                    "filtered_history_count": len(history),
                    "locked_parameters": sorted(adaptive_locks),
                }
            )
        locks = dict(adaptive_locks)
        locks.update(lockedValues or {})
        params = self._sample_tpe(
            hyperparameterSpace,
            history,
            locks,
            gamma=action.gamma,
            n_ei_candidates=action.n_ei_candidates,
        )
        self.lastATPEParameters = {
            "gamma": action.gamma,
            "n_ei_candidates": action.n_ei_candidates,
            "history_keep_fraction": action.history_keep_fraction,
            "history_recency_bias": action.history_recency_bias,
            "history_quality_bias": action.history_quality_bias,
            "secondary_fraction": action.secondary_fraction,
            "secondary_lock_probability": action.secondary_lock_probability,
            "secondary_top_fraction": action.secondary_top_fraction,
        }
        self.lastLockedParameters = sorted(adaptive_locks)
        self.atpeParamDetails = self.controller.last_record
        self.sample_count += 1
        return params

    def _sample_tpe(self, space_config, results, locked_values, gamma, n_ei_candidates):
        canonical_results = [dict(result, status=hyperopt.STATUS_OK, loss=float(result["loss"])) for result in results]
        flat_parameters = Hyperparameter(space_config).getFlatParameters()
        for result in canonical_results:
            for parameter in flat_parameters:
                result.setdefault(parameter.name, "")
        trials = self.convertResultsToTrials(space_config, canonical_results)
        # ``Hyperparameter.getFlatParameters`` intentionally omits hp.choice
        # selector nodes for legacy compatibility.  Add their observations only
        # on the ATPE2 path so conditional history is represented faithfully.
        selectors = []

        def collect_selectors(node, path=""):
            alternatives = node.get("anyOf", node.get("oneOf"))
            if alternatives is not None:
                if path:
                    selectors.append((path, node.get("name", "root." + path)))
                for index, child in enumerate(alternatives):
                    collect_selectors(child, f"{path}.{index}" if path else str(index))
            elif node.get("type") == "object":
                for key, child in node.get("properties", {}).items():
                    collect_selectors(child, f"{path}.{key}" if path else key)

        collect_selectors(space_config)
        for index, (result, trial) in enumerate(zip(canonical_results, trials._dynamic_trials)):
            for result_name, hyperopt_name in selectors:
                value = result.get(result_name)
                if value in {None, ""}:
                    trial["misc"]["idxs"][hyperopt_name] = []
                    trial["misc"]["vals"][hyperopt_name] = []
                else:
                    trial["misc"]["idxs"][hyperopt_name] = [index]
                    trial["misc"]["vals"][hyperopt_name] = [int(value)]
        trials.refresh()
        space = Hyperparameter(space_config).createHyperoptSpace(locked_values)
        params: dict[str, Any] = {}

        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0.5, "status": hyperopt.STATUS_OK}

        seed_payload = json.dumps([self.seed, self.sample_count, len(results)], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little")
        hyperopt.fmin(
            fn=sample,
            space=space,
            algo=functools.partial(
                hyperopt.tpe.suggest,
                n_EI_candidates=int(n_ei_candidates),
                gamma=float(gamma),
            ),
            max_evals=len(trials.trials) + 1,
            trials=trials,
            rstate=np.random.default_rng(seed),
            show_progressbar=False,
        )
        return params
