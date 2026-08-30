"""Versioned synthetic objectives covering the ATPE2 correctness envelope."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np


GENERATOR_VERSION = "atpe2-synthetic-v1"


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    family: str
    space: dict[str, Any]
    seed: int
    dimension: int = 4
    noise: float = 0.0
    failure_probability: float = 0.0
    affine_scale: float = 1.0
    affine_shift: float = 0.0
    plateau: float | None = None

    def specification(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "space": self.space,
            "seed": self.seed,
            "dimension": self.dimension,
            "noise": self.noise,
            "failure_probability": self.failure_probability,
            "affine_scale": self.affine_scale,
            "affine_shift": self.affine_shift,
            "plateau": self.plateau,
        }

    @classmethod
    def from_specification(cls, specification: dict[str, Any]) -> "SyntheticTask":
        return cls(
            task_id=specification["task_id"],
            family=specification["family"],
            space=specification["space"],
            seed=int(specification["seed"]),
            dimension=int(specification.get("dimension", 4)),
            noise=float(specification.get("noise", 0.0)),
            failure_probability=float(specification.get("failure_probability", 0.0)),
            affine_scale=float(specification.get("affine_scale", 1.0)),
            affine_shift=float(specification.get("affine_shift", 0.0)),
            plateau=specification.get("plateau"),
        )

    def evaluate(self, params: dict[str, Any], rollout_seed: int, step: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, rollout_seed, step]))
        if rng.random() < self.failure_probability:
            return {"status": "fail", "loss": None, "error": "synthetic_failure"}
        flat = _flatten(params)
        numeric = np.asarray([float(value) for value in flat.values() if isinstance(value, (int, float)) and not isinstance(value, bool)])
        categorical = [value for value in flat.values() if isinstance(value, str)]
        x = numeric[: self.dimension]
        if x.size < self.dimension:
            x = np.pad(x, (0, self.dimension - x.size))
        shift_rng = np.random.default_rng(self.seed)
        shift = shift_rng.uniform(-0.5, 0.5, self.dimension)
        z = x - shift
        rotation, _ = np.linalg.qr(shift_rng.normal(size=(self.dimension, self.dimension)))
        z = rotation @ z
        if self.family == "smooth":
            base = float(np.sum(z * z))
        elif self.family == "rugged":
            base = float(10 * self.dimension + np.sum(z * z - 10 * np.cos(2 * math.pi * z)))
        elif self.family == "interaction":
            base = float(np.sum(z * z) + np.sum(z[:-1] * z[1:]) if len(z) > 1 else np.sum(z * z))
        elif self.family == "hierarchical":
            category_cost = sum((sum(map(ord, str(value))) % 11) / 20 for value in categorical)
            base = float(np.sum(z * z) + category_cost)
        else:
            base = float(np.sum(np.abs(z)))
        base += float(rng.normal(0, self.noise))
        transformed = self.affine_scale * base + self.affine_shift
        if self.plateau:
            transformed = round(transformed / self.plateau) * self.plateau
        return {"status": "ok", "loss": float(transformed)}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key, child in value.items():
        if key.startswith("$"):
            continue
        output.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
    return output


def pilot_tasks(seed: int) -> list[SyntheticTask]:
    numeric_space = {
        "type": "object",
        "properties": {
            "x0": {"type": "number", "min": -5, "max": 5},
            "x1": {"type": "number", "min": 0.01, "max": 10, "scaling": "logarithmic"},
            "x2": {"type": "number", "min": -3, "max": 3, "rounding": 1},
            "irrelevant": {"type": "number", "min": -1, "max": 1},
        },
    }
    mixed_space = {
        "type": "object",
        "properties": {
            "model": {
                "oneOf": [
                    {"type": "object", "properties": {"depth": {"type": "number", "mode": "randint", "max": 12}}},
                    {"type": "object", "properties": {"kernel": {"enum": ["linear", "rbf", "poly"]}, "c": {"type": "number", "min": 0.01, "max": 100, "scaling": "logarithmic"}}},
                ]
            },
            "category": {"enum": ["a", "b", "c"]},
        },
    }
    variants = [
        ("smooth", numeric_space, 3, 0.0, 0.0, 1.0, -10.0, None),
        ("rugged", numeric_space, 3, 0.1, 0.0, 1.0, 0.0, None),
        ("interaction", numeric_space, 3, 0.0, 0.05, 1e9, 0.0, None),
        ("hierarchical", mixed_space, 2, 0.05, 0.1, 2.0, -5.0, 0.25),
        ("plateau", numeric_space, 3, 0.0, 0.0, 1.0, 0.0, 1.0),
        ("constant_zero", numeric_space, 3, 0.0, 0.0, 0.0, 0.0, None),
    ]
    return [
        SyntheticTask(f"{family}-{seed}-{index}", family, space, seed + index, dimension, noise, failure, scale, shift, plateau)
        for index, (family, space, dimension, noise, failure, scale, shift, plateau) in enumerate(variants)
    ]


def synthetic_tasks(seed: int, instances_per_family: int = 4) -> list[SyntheticTask]:
    """Create independent task identities while preserving every task family."""
    if instances_per_family < 1:
        raise ValueError("instances_per_family must be positive")
    templates = pilot_tasks(seed)
    tasks = []
    for template_index, template in enumerate(templates):
        for instance in range(instances_per_family):
            task_seed = seed + template_index * 100_000 + instance * 1_009
            tasks.append(
                replace(
                    template,
                    task_id=f"{template.family}-{seed}-{instance}",
                    seed=task_seed,
                )
            )
    return tasks


def factorized_training_tasks(seed: int, total: int = 192) -> list[SyntheticTask]:
    """Return the predeclared factorized mix (about 40% hierarchical).

    For the protocol size of 192 this is 77 hierarchical tasks and 23 from
    each of the five remaining families.  The largest-remainder allocation is
    deterministic for other miniature/test sizes as well.
    """
    if total < 6:
        raise ValueError("factorized task mix requires at least one task per family")
    templates = {task.family: task for task in pilot_tasks(seed)}
    families = sorted(family for family in templates if family != "hierarchical")
    hierarchical_count = int(round(0.40 * total))
    remaining = total - hierarchical_count
    base, extra = divmod(remaining, len(families))
    counts = {
        "hierarchical": hierarchical_count,
        **{
            family: base + int(index < extra)
            for index, family in enumerate(families)
        },
    }
    tasks: list[SyntheticTask] = []
    family_order = ["hierarchical", *families]
    for family_index, family in enumerate(family_order):
        template = templates[family]
        for instance in range(counts[family]):
            task_seed = seed + family_index * 1_000_000 + instance * 1_009
            tasks.append(
                replace(
                    template,
                    task_id=f"{family}-{seed}-factorized-{instance}",
                    seed=task_seed,
                )
            )
    return tasks
