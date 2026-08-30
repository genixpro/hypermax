"""Cycle-free serialization of the ATPE2 action schema."""

from __future__ import annotations

from typing import Any, Iterable


ACTION_SCHEMA_VERSION = "atpe2-action-v1"
ACTION_DOMAINS = {
    "gamma": {"type": "number", "minimum": 0.1, "maximum": 2.0, "scale": "linear"},
    "n_ei_candidates": {"type": "integer", "minimum": 8, "maximum": 64, "scale": "log"},
    "history_keep_fraction": {"type": "number", "minimum": 0.25, "maximum": 1.0, "scale": "linear"},
    "history_recency_bias": {"type": "number", "minimum": -2.0, "maximum": 2.0, "scale": "linear"},
    "history_quality_bias": {"type": "number", "minimum": -2.0, "maximum": 2.0, "scale": "linear"},
    "secondary_fraction": {"type": "number", "minimum": 0.0, "maximum": 0.75, "scale": "linear"},
    "secondary_lock_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0, "scale": "linear"},
    "secondary_top_fraction": {"type": "number", "minimum": 0.02, "maximum": 1.0, "scale": "linear"},
}


def action_schema_definition(
    retained_names: Iterable[str],
    fixed_constants: dict[str, Any],
    ranges: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retained_names = list(retained_names)
    ranges = ranges or {}
    return {
        "version": ACTION_SCHEMA_VERSION,
        "retained_names": retained_names,
        "domains": {
            name: {
                key: ranges.get(name, ACTION_DOMAINS[name]).get(
                    key, ACTION_DOMAINS[name][key]
                )
                for key in ("type", "minimum", "maximum", "scale")
            }
            for name in retained_names
        },
        "fixed_constants": dict(sorted(fixed_constants.items())),
    }
