"""Scale-free paired five-step utility labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import rankdata


def label_state(records: list[dict[str, Any]]) -> dict[tuple[str, int, int], float]:
    """Rank incumbents within each state/seed/step and average percentile ranks.

    Curves are minimized, while utility is maximized; therefore the best incumbent
    receives utility one. Failed/missing steps are treated as positive infinity.
    """

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["rollout_seed"])].append(record)
    output: dict[tuple[str, int, int], float] = {}
    for seed, paired in by_seed.items():
        step_scores = [[] for _ in range(5)]
        for step in range(5):
            values = np.asarray(
                [
                    float(record["incumbent_curve"][step])
                    if record["incumbent_curve"][step] is not None
                    else np.inf
                    for record in paired
                ]
            )
            finite = np.isfinite(values)
            ranks = np.full(len(values), len(values), dtype=float)
            if finite.any():
                ranks[finite] = rankdata(values[finite], method="average")
            utilities = 1.0 - (ranks - 1.0) / max(1, len(values) - 1)
            for index, utility in enumerate(utilities):
                step_scores[step].append(float(utility))
        for index, record in enumerate(paired):
            key = (record["configuration_hash"], seed, int(record["repeat_index"]))
            output[key] = float(np.mean([step_scores[step][index] for step in range(5)]))
    return output
