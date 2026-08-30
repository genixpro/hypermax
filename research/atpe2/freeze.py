"""Publish a Gate-2-approved research ensemble into the runtime package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from hypermax.algorithms.atpe2.model import ATPE2ModelEnsemble


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_runtime(
    candidate_manifest: Path,
    gate2_report: Path,
    pruning_report: Path,
    runtime_directory: Path,
    seed: int,
) -> dict[str, Any]:
    ensemble = ATPE2ModelEnsemble.load(candidate_manifest)
    if not ensemble.research_ready or ensemble.manifest.get("status") != "candidate":
        raise ValueError("Candidate ensemble is not fitted or schema-compatible")
    gate_payload = json.loads(gate2_report.read_text())
    gate = gate_payload.get("gate2", gate_payload)
    if gate.get("gate") != 2 or not gate.get("passed"):
        raise ValueError("Runtime publication requires a passing Gate 2 report")
    if gate_payload.get("schema_version") != "atpe2-synthetic-validation-v1":
        raise ValueError("Runtime publication requires a complete validation artifact")
    artifact_hashes = gate_payload.get("artifact_hashes", {})
    if artifact_hashes.get("pruned_manifest_sha256") != _sha256(candidate_manifest):
        raise ValueError("Gate 2 evidence was produced for a different candidate")
    if artifact_hashes.get("pruning_report_sha256") != _sha256(pruning_report):
        raise ValueError("Gate 2 evidence was produced with different pruning evidence")
    pruning = json.loads(pruning_report.read_text())
    knockout = pruning.get("backward_knockout", {})
    if knockout.get("retained_knobs") != ensemble.manifest.get("retained_action_names"):
        raise ValueError("Candidate action schema does not match pruning evidence")
    if knockout.get("fixed_constants", {}) != ensemble.manifest.get(
        "validation_selected_constants", {}
    ):
        raise ValueError("Candidate constants do not match pruning evidence")
    if set(ensemble.manifest.get("dataset_aggregation_rounds", [])) != {0, 1}:
        raise ValueError(
            "Candidate must contain round zero plus exactly one dataset-aggregation round"
        )

    model_card = candidate_manifest.parent / "MODEL_CARD.md"
    manifest = dict(ensemble.manifest)
    manifest["status"] = "frozen"
    manifest["publication_seed"] = int(seed)
    gate_hash = _sha256(gate2_report)
    pruning_hash = _sha256(pruning_report)
    candidate_card_hash = _sha256(model_card) if model_card.exists() else None
    runtime_card = None
    if model_card.exists():
        runtime_card = model_card.read_text(encoding="utf-8")
        runtime_card = runtime_card.replace(
            "Status: research candidate. Runtime publication requires a passing Gate 2 report\n"
            "and is performed by `python -m research.atpe2.freeze`.",
            "Status: Gate-2-approved frozen runtime ensemble.",
        )
        runtime_card = runtime_card.replace(
            "This candidate predicts scale-free five-step controller utility for ATPE2.",
            "This frozen ensemble predicts scale-free five-step controller utility for ATPE2.",
        ).replace(
            "A candidate must not replace\n"
            "the packaged runtime ensemble until the held-out synthetic gate passes.",
            "Its exact pruning and held-out synthetic evidence were verified before publication.",
        )
        runtime_card += (
            "\n## Validation approval\n\n"
            f"- Gate 2 report SHA-256: `{gate_hash}`\n"
            f"- Knob-pruning report SHA-256: `{pruning_hash}`\n"
        )
    manifest["validation_evidence"] = {
        "gate2_passed": True,
        "gate2_report_sha256": gate_hash,
        "pruning_report_sha256": pruning_hash,
        "candidate_model_card_sha256": candidate_card_hash,
        "runtime_model_card_sha256": (
            hashlib.sha256(runtime_card.encode("utf-8")).hexdigest()
            if runtime_card is not None
            else None
        ),
    }
    runtime_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="atpe2-freeze-", dir=runtime_directory.parent
    ) as staging_name:
        staging = Path(staging_name)
        for artifact in manifest["models"]:
            source = candidate_manifest.parent / artifact["file"]
            if _sha256(source) != artifact["sha256"]:
                raise ValueError(f"Checksum mismatch for {source.name}")
            shutil.copy2(source, staging / source.name)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if runtime_card is not None:
            (staging / "MODEL_CARD.md").write_text(runtime_card, encoding="utf-8")
        # Publish the manifest last so readers never observe a new manifest
        # pointing at model files that have not yet been replaced.
        for source in sorted(
            staging.iterdir(), key=lambda path: (path.name == "manifest.json", path.name)
        ):
            os.replace(source, runtime_directory / source.name)
    published = ATPE2ModelEnsemble.load(runtime_directory / "manifest.json")
    if not published.ready:
        raise RuntimeError("Published runtime ensemble failed its load check")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("gate2_report", type=Path)
    parser.add_argument("pruning_report", type=Path)
    parser.add_argument("runtime_directory", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            freeze_runtime(
                args.candidate_manifest,
                args.gate2_report,
                args.pruning_report,
                args.runtime_directory,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
