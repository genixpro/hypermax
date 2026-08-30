"""Deterministic structured round-one acquisition for factorized ATPE2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.stats import qmc

from hypermax.algorithms.atpe2.controller import ACTION_GROUPS, ATPE2Action
from hypermax.algorithms.atpe2.controller_schema import ACTION_DOMAINS


ACQUISITION_IMPLEMENTATION_VERSION = "atpe2-structured-acquisition-v1"


@dataclass(frozen=True)
class AcquisitionCandidate:
    action: ATPE2Action
    role: str
    varied_groups: tuple[str, ...]

    def lineage_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "varied_groups": list(self.varied_groups),
            "action": asdict(self.action),
        }


def canonical_action(action: ATPE2Action) -> str:
    return json.dumps(asdict(action), sort_keys=True, separators=(",", ":"))


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sample_value(name: str, coordinate: float, ranges: dict[str, dict[str, Any]]) -> float | int:
    domain = {**ACTION_DOMAINS[name], **ranges.get(name, {})}
    minimum, maximum = float(domain["minimum"]), float(domain["maximum"])
    if domain.get("scale") == "log":
        value = math.exp(
            math.log(minimum)
            + float(coordinate) * (math.log(maximum) - math.log(minimum))
        )
    else:
        value = minimum + float(coordinate) * (maximum - minimum)
    if domain.get("type") == "integer":
        return int(np.clip(round(value), int(minimum), int(maximum)))
    return float(value)


def _group_action(
    group: str,
    coordinates: np.ndarray,
    ranges: dict[str, dict[str, Any]],
) -> ATPE2Action:
    values = asdict(ATPE2Action.anchor())
    for name, coordinate in zip(ACTION_GROUPS[group], coordinates):
        values[name] = _sample_value(name, float(coordinate), ranges)
    return ATPE2Action(**values)


def _combine(groups: dict[str, ATPE2Action]) -> ATPE2Action:
    values = asdict(ATPE2Action.anchor())
    for group, action in groups.items():
        for name in ACTION_GROUPS[group]:
            values[name] = getattr(action, name)
    return ATPE2Action(**values)


def _sobol(seed: int, dimension: int, count: int) -> np.ndarray:
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    power = int(math.ceil(math.log2(max(1, count))))
    return sampler.random_base2(power)[:count]


def _isolated_candidates(
    seed: int, ranges: dict[str, dict[str, Any]]
) -> dict[str, list[AcquisitionCandidate]]:
    counts = {"search": 8, "history": 12, "locking": 12}
    output: dict[str, list[AcquisitionCandidate]] = {}
    for group_index, (group, names) in enumerate(ACTION_GROUPS.items()):
        candidates: list[AcquisitionCandidate] = []
        for name_index, name in enumerate(names):
            for edge, coordinate in (("low", 0.0), ("high", 1.0)):
                values = asdict(ATPE2Action.anchor())
                values[name] = _sample_value(name, coordinate, ranges)
                candidates.append(
                    AcquisitionCandidate(
                        ATPE2Action(**values),
                        f"group_isolated_{group}_contrast_{name}_{edge}",
                        (group,),
                    )
                )
        sobol_count = counts[group] - len(candidates)
        points = _sobol(seed + 10_000 * (group_index + 1), len(names), sobol_count)
        candidates.extend(
            AcquisitionCandidate(
                _group_action(group, point, ranges),
                f"group_isolated_{group}_sobol",
                (group,),
            )
            for point in points
        )
        output[group] = candidates
    return output


def _model_disagreement(
    ensemble: Any,
    context: np.ndarray,
    actions: list[ATPE2Action],
) -> np.ndarray:
    if not getattr(ensemble, "is_factorized", False):
        matrix = np.column_stack(
            (
                np.repeat(context[None, :], len(actions), axis=0),
                np.asarray(
                    [action.vector_for(ensemble.action_names) for action in actions]
                ),
            )
        )
        return ensemble.predict(matrix)[1]
    uncertainty = np.zeros(len(actions), dtype=float)
    for group, names in ACTION_GROUPS.items():
        matrix = np.column_stack(
            (
                np.repeat(context[None, :], len(actions), axis=0),
                np.asarray([action.vector_for(names) for action in actions]),
            )
        )
        uncertainty += ensemble.predict_group(group, matrix)[1]
    full = np.column_stack(
        (
            np.repeat(context[None, :], len(actions), axis=0),
            np.asarray([action.vector() for action in actions]),
        )
    )
    uncertainty += ensemble.predict_residual(full)[1]
    return uncertainty


def structured_action_candidates(
    seed: int,
    provisional_action: ATPE2Action,
    ensemble: Any,
    context: np.ndarray,
    action_ranges: dict[str, dict[str, Any]] | None = None,
) -> list[AcquisitionCandidate]:
    """Return the ordered, state-local 65-slot round-one design.

    Slot roles are fixed before deterministic de-duplication.  When a proposed
    action collides (including a neutral provisional policy), its slot is
    refilled from a role-constrained seeded Sobol pool and retains its role.
    """

    ranges = action_ranges or {}
    isolated = _isolated_candidates(seed, ranges)
    candidates: list[AcquisitionCandidate] = [
        AcquisitionCandidate(ATPE2Action.anchor(), "neutral_anchor", ()),
        AcquisitionCandidate(
            provisional_action, "provisional_policy", tuple(ACTION_GROUPS)
        ),
    ]
    for group in ACTION_GROUPS:
        candidates.extend(isolated[group])

    group_actions = {
        group: [candidate.action for candidate in isolated[group]]
        for group in ACTION_GROUPS
    }
    pairs = (("search", "history"), ("search", "locking"), ("history", "locking"))
    for pair_index, pair in enumerate(pairs):
        left, right = pair
        for index in range(6):
            action = _combine(
                {
                    left: group_actions[left][index % len(group_actions[left])],
                    right: group_actions[right][
                        (index * 5 + pair_index + 1) % len(group_actions[right])
                    ],
                }
            )
            candidates.append(
                AcquisitionCandidate(
                    action, f"two_group_{left}_{right}", tuple(pair)
                )
            )

    for index in range(8):
        selected = {
            group: group_actions[group][
                (index * (group_index + 2) + group_index) % len(group_actions[group])
            ]
            for group_index, group in enumerate(ACTION_GROUPS)
        }
        candidates.append(
            AcquisitionCandidate(
                _combine(selected), "full_action_regime", tuple(ACTION_GROUPS)
            )
        )

    disagreement_pool = [
        ATPE2Action(
            **{
                name: _sample_value(name, point[index], ranges)
                for index, name in enumerate(ATPE2Action.NAMES)
            }
        )
        for point in _sobol(seed + 90_000, len(ATPE2Action.NAMES), 256)
    ]
    disagreement = _model_disagreement(ensemble, context, disagreement_pool)
    disagreement_order = sorted(
        range(len(disagreement_pool)),
        key=lambda index: (-float(disagreement[index]), canonical_action(disagreement_pool[index])),
    )
    candidates.extend(
        AcquisitionCandidate(
            disagreement_pool[index],
            "highest_disagreement",
            tuple(ACTION_GROUPS),
        )
        for index in disagreement_order[:5]
    )
    if len(candidates) != 65:
        raise AssertionError(f"structured acquisition created {len(candidates)} slots")

    seen: set[str] = set()
    output: list[AcquisitionCandidate] = []
    for slot_index, candidate in enumerate(candidates):
        action = candidate.action
        key = canonical_action(action)
        refill_index = 0
        while key in seen:
            varied = candidate.varied_groups or tuple(ACTION_GROUPS)
            if len(varied) >= 2 and (
                candidate.role.startswith("two_group")
                or candidate.role == "full_action_regime"
            ):
                # Interaction targets must retain exact isolated-group matches
                # after refill so their additive decomposition stays defined.
                action = _combine(
                    {
                        group: group_actions[group][
                            (slot_index + refill_index * (group_index + 2))
                            % len(group_actions[group])
                        ]
                        for group_index, group in enumerate(varied)
                    }
                )
            else:
                names = tuple(
                    name for group in varied for name in ACTION_GROUPS[group]
                )
                point = _sobol(
                    seed + 1_000_000 + slot_index * 10_000 + refill_index,
                    len(names),
                    1,
                )[0]
                values = asdict(ATPE2Action.anchor())
                for name, coordinate in zip(names, point):
                    values[name] = _sample_value(name, coordinate, ranges)
                action = ATPE2Action(**values)
            key = canonical_action(action)
            refill_index += 1
            if refill_index > 4096:
                raise RuntimeError("Could not deterministically refill structured acquisition")
        seen.add(key)
        output.append(
            AcquisitionCandidate(action, candidate.role, candidate.varied_groups)
        )
    if len(output) != 65 or len(seen) != 65:
        raise AssertionError("structured acquisition must contain exactly 65 unique actions")
    return output


def acquisition_lineage(
    candidates: list[AcquisitionCandidate], provisional_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "implementation_version": ACQUISITION_IMPLEMENTATION_VERSION,
        "provisional_manifest_sha256": provisional_manifest_sha256,
        "ordered_role_action_mapping": [
            candidate.lineage_record() for candidate in candidates
        ],
    }
