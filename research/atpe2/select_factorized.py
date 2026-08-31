"""Deterministic pruning of bounded factorized architecture variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAX_MANIFEST_VARIANTS = 12
MAX_BEHAVIORAL_VARIANTS = 6


def select_manifest_variants(
    manifests: list[tuple[Path, dict[str, Any]]], limit: int = MAX_MANIFEST_VARIANTS
) -> list[dict[str, Any]]:
    eligible = []
    for path, manifest in manifests:
        gates = manifest.get("component_gates", {})
        if not gates.get("passed", False):
            continue
        quality = float(
            manifest.get("score_grid", [{}])[0].get(
                "held_out_stable_policy_quality", float("-inf")
            )
        )
        eligible.append(
            {
                "manifest": str(path),
                "model_version": manifest.get("model_version"),
                "branch": manifest.get("architecture_branch"),
                "held_out_stable_policy_quality": quality,
                "model_artifact_sha256": manifest.get("model_artifact_sha256"),
            }
        )
    eligible.sort(
        key=lambda row: (
            -row["held_out_stable_policy_quality"],
            str(row["branch"]),
            str(row["model_artifact_sha256"]),
        )
    )
    return eligible[: min(limit, MAX_MANIFEST_VARIANTS)]


def select_behavioral_variants(
    manifest_selection: list[dict[str, Any]],
    fresh_policy_quality: dict[str, float],
    limit: int = MAX_BEHAVIORAL_VARIANTS,
) -> list[dict[str, Any]]:
    rescored = []
    for row in manifest_selection:
        identity = str(row["model_artifact_sha256"])
        if identity not in fresh_policy_quality:
            raise ValueError(f"Missing fresh multi-step quality for {identity}")
        rescored.append(
            {**row, "fresh_multi_step_policy_quality": float(fresh_policy_quality[identity])}
        )
    rescored.sort(
        key=lambda row: (
            -row["fresh_multi_step_policy_quality"],
            -row["held_out_stable_policy_quality"],
            row["model_artifact_sha256"],
        )
    )
    return rescored[: min(limit, MAX_BEHAVIORAL_VARIANTS)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--fresh-policy-quality", type=Path)
    args = parser.parse_args(argv)
    selected = select_manifest_variants(
        [(path, json.loads(path.read_text())) for path in args.manifests]
    )
    report: dict[str, Any] = {
        "schema_version": "atpe2-factorized-variant-selection-v1",
        "manifest_only": selected,
    }
    if args.fresh_policy_quality:
        quality = json.loads(args.fresh_policy_quality.read_text())
        quality = quality.get("quality_by_model_artifact_sha256", quality)
        report["behavioral"] = select_behavioral_variants(selected, quality)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
