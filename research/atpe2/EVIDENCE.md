# ATPE2 evidence ledger

This ledger summarizes the current ATPE2 evidence. Exact report, model-manifest,
and split hashes are recorded in [`published/evidence.json`](published/evidence.json);
the independently grouped task and trajectory split is in
[`published/splits.json`](published/splits.json). Third-party benchmark data is
not redistributed.

> **Gate 3 invalidation (adaptive-only policy v2):** the previously published
> external result was produced by a controller whose hard context-envelope
> fallback selected the standard-TPE anchor at every external decision boundary.
> Its identical ATPE2/TPE curves therefore measured the fallback, not adaptive
> ATPE2. That Gate 3 result and its hashes remain only as a historical record and
> are not valid evidence for the current `atpe2-adaptive-only-v2` policy. A fresh
> external run must pass adaptive-decision telemetry checks before this gate can
> be reported as complete.

## Gate 0 — correctness

- 56 unit and regression tests pass on Python 3.13.
- Coverage includes negative, zero, constant, duplicate, missing, failed, and
  extremely scaled losses; mixed and conditional parameter types; compatible
  selector/descendant locking; affine loss invariance; finite features;
  determinism; serialization; package resources; and the historical zero-loss,
  invalid-access, NaN-correlation, percentile-index, and normal/randint bugs.
- The legacy `hypermax/atpe_models` directory is unchanged.

## Gate 1 — three-hour local pilot

- Passed on the M3 MacBook Air with 8 selected workers and 10,811.22 seconds of
  collection after the required 1/2/4/6/8-worker benchmarks.
- Collected 6,069 state blocks, 881,128 rollout records, and 4,405,640 objective
  evaluations at 407.51 evaluations/second.
- Peak selected-worker benchmark memory was 1.433 GiB, below the 18 GiB cap.
- The SQLite database was 1,281,794,048 bytes; the eight-model pilot fit took
  84.87 seconds. A 10,000-state main set was projected at 4.95 hours and 2.11 GB.

## Gate 2 — held-out synthetic validation

- Re-run and passed under the adaptive-only policy using the full 24-task
  protocol: six task families, four held-out instances
  per family, four trajectory methods, three checkpoints, five paired rollout
  seeds, and all declared ablations.
- The paired 95% bootstrap interval for ATPE2 minus Hyperopt TPE five-step
  utility was `[0.0, 0.0]`; no family exceeded 5% median normalized-regret
  degradation.
- Backward evidence pruning retained `gamma` and `history_keep_fraction`.
  Candidate count, recency/quality biases, and all locking controls are fixed to
  their validation-selected neutral values.
- Efficiency validation reduced each ensemble member from 300 to 8 trees and
  runtime acquisition from 1,024 to 32 Sobol actions. The complete Gate 2 still
  passed: the paired utility interval remained `[0.0, 0.0]`, and the worst
  family degradation was 1.67%, below the 5% limit.
- The final robust model validation MAE was 0.06186098 over 10,454 rows, versus
  0.06225695 for the former 300-tree model. Packaged model data fell from
  7,414,588 to 223,980 bytes; unloaded single-process controller latency fell
  from 121.84 ms to 1.65 ms median on the pilot machine.
- The model was
  retrained once on acquisition rounds 0 and 1 after the single prescribed
  dataset-aggregation deployment.

## Gate 3 — external benchmark study

- **Invalidated; replacement currently paused.** The former 10,290-run result exercised the
  standard-TPE anchor at every ATPE2 boundary because all external contexts were
  rejected by the training-envelope guard.
- The replacement protocol requires every ATPE2 trace from trial 5 onward to
  contain an `adaptive_model` decision and rejects any fallback reason.
- A 899-run partial YAHPO database belongs to the superseded 300-tree policy and
  must not be resumed or mixed with the newly frozen 8-tree policy.
- No external performance or publication conclusion is currently claimed for
  adaptive-only ATPE2.

## Current development findings

The evolving diagnosis, iteration results, runtime measurements, limitations,
and next research questions are maintained in [`FINDINGS.md`](FINDINGS.md).
Those observations are directional only; this ledger remains authoritative for
the formal evidence-gate status.

## Benchmark provenance

- HPO-B v3 protocol and split metadata: [HPO-B repository](https://github.com/machinelearningnuremberg/HPO-B).
- YAHPO scenarios and fixed-fidelity definitions: [YAHPO Gym scenarios](https://slds-lmu.github.io/yahpo_gym/scenarios.html).
- YAHPO data sources include Auto-PyTorch (Zimmer, Lindauer, and Hutter, 2021),
  the OpenML hyperparameter data of Binder, Pfisterer, and Bischl (2020), and
  NAS-Bench-301 (Siems et al., 2020), as requested by the benchmark package.
