# ATPE2 research pipeline

This directory contains the research-only dependencies and artifacts for ATPE2.
It is not included in the runtime package.

All commands require an explicit seed. The SQLite store uses WAL mode and unique
state/evaluation identities, so interrupted collection can be resumed without
duplicating rollouts.

The required order is pilot, full-schema pruning evidence, pruned main
collection, one deployed-controller aggregation round, final training, held-out
validation, guarded runtime publication, and only then external testing.

For the development inner loop, run the small synthetic iteration screen. It
uses two reusable development tasks from each family, paired histories and
seeds, ten-trial incumbent curves at three history depths, 8,640 objective
evaluations, and only Random Search, Hyperopt TPE, legacy ATPE, and ATPE2.
Performance scoring uses continuous scale-normalized improvement AUC on the four
non-degenerate families; plateau and constant-zero remain correctness checks.
Its thresholds are deliberately broad and its JSON is marked
`evidence_eligible: false`; it can reject obvious regressions but can never
replace Gate 2 or Gate 3:

```bash
python -m research.atpe2.iterate hypermax/atpe2_models/manifest.json research/atpe2/iteration.json --seed 7 --workers 6
```

Gate 1 can be exercised with a short smoke run before the required three-hour
run. The real gate benchmarks 1, 2, 4, 6, and 8 processes for three minutes
each, selects the fastest under 18 GiB peak RSS, and keeps collecting until the
three-hour total is reached:

```bash
python -m research.atpe2.pilot research/atpe2/pilot.sqlite research/atpe2/pilot-smoke.json --seed 1 --benchmark-seconds 5 --total-seconds 30
python -m research.atpe2.pilot research/atpe2/pilot.sqlite research/atpe2/gate1.json --seed 1 --benchmark-seconds 180 --total-seconds 10800
```

Train the eight-knob pilot ensemble and produce the required grouped importance,
partial-dependence, interaction, and iterative backward-knockout evidence:

```bash
python -m research.atpe2.train research/atpe2/pilot.sqlite research/atpe2/full --seed 1
python -m research.atpe2.evidence research/atpe2/pilot.sqlite research/atpe2/full/manifest.json research/atpe2/pruning.json --seed 1
```

Generate the main dataset with that frozen action schema, then fit the first
pruned controller. Generation evaluates 64 Sobol actions plus the TPE anchor,
uses two paired rollout seeds, and adds the specified adaptive third repeats.
Training refuses fewer than 30 labeled rows and writes eight LightGBM text
models plus checksummed manifest, split manifest, and model card:

```bash
python -m research.atpe2.generate research/atpe2/main.sqlite --seed 2 --instances-per-family 4 --action-schema research/atpe2/pruning.json --workers 8
python -m research.atpe2.train research/atpe2/main.sqlite research/atpe2/pruned-round-0 --seed 2 --action-schema research/atpe2/pruning.json
```

Tree count and runtime Sobol resolution are versioned policy parameters. Smaller
variants can be trained with `--n-estimators` and
`--runtime-candidate-count`; the latter must be a power of two. These controls
affect runtime inference only. Offline acquisition always preserves its separate
64-action-plus-anchor labeling protocol. A smaller policy must pass the same
complete Gate 2 before publication.

After the first ensemble is trained, collect its visited states once and retrain:

```bash
python -m research.atpe2.generate research/atpe2/main.sqlite --seed 3 --acquisition-round 1 --model-manifest research/atpe2/pruned-round-0/manifest.json --action-schema research/atpe2/pruning.json --workers 8
python -m research.atpe2.train research/atpe2/main.sqlite research/atpe2/pruned-round-1 --seed 2 --action-schema research/atpe2/pruning.json
```

Removed knobs are fixed to the recorded neutral constants. The final candidate
manifest records both retained names and constants; only `freeze` marks that
candidate as approved for runtime publication.

Run held-out synthetic validation with every trajectory family and checkpoint,
the executable behavior ablations, and a separately trained legacy-feature model:

```bash
python -m research.atpe2.train research/atpe2/main.sqlite research/atpe2/legacy-features --seed 2 --action-schema research/atpe2/pruning.json --feature-schema legacy
python -m research.atpe2.validate research/atpe2/pruned-round-1/manifest.json research/atpe2/pruning.json research/atpe2/gate2.json --seed 4 --full-manifest research/atpe2/full/manifest.json --legacy-feature-manifest research/atpe2/legacy-features/manifest.json --workers 4
```

Only a passing Gate 2 report can publish the runtime ensemble:

```bash
python -m research.atpe2.freeze research/atpe2/pruned-round-1/manifest.json research/atpe2/gate2.json research/atpe2/pruning.json hypermax/atpe2_models --seed 4
```

## External benchmark environment

Install the research stack separately from Hypermax. YAHPO Gym 1.0.2 declares
an obsolete ConfigSpace upper bound, so install it without dependencies after
the compatible shared stack:

```bash
python -m pip install -r research/atpe2/requirements.txt
python -m pip install --no-deps yahpo-gym==1.0.2
```

Point the runners at locally downloaded, version-pinned benchmark data. Full
runs are locked until the supplied manifest contains passing Gate 2 evidence.
Both runners are resumable through the same SQLite trace database.

```bash
python -m research.atpe2.external yahpo research/atpe2/external.sqlite /path/to/yahpo_data hypermax/atpe2_models/manifest.json --workers 8 --seed 5
python -m research.atpe2.external hpob research/atpe2/external.sqlite /path/to/hpob-data /path/to/HPO-B/hpob-data/meta-test-tasks-per-space.json hypermax/atpe2_models/manifest.json --workers 2 --seed 5
python -m research.atpe2.gates research/atpe2/external.sqlite research/atpe2/gate3.json --model-manifest hypermax/atpe2_models/manifest.json --seed 1
python -m research.atpe2.external export research/atpe2/external.sqlite research/atpe2/external-traces.json --seed 5
```

After Gate 3, publish the compact reports, exact grouped split, and hashes for
the retained raw database/export. Third-party benchmark data itself is never
copied into the publication directory:

```bash
python -m research.atpe2.publish research/atpe2/published gate1.json pruning.json gate2.json gate3.json pruned-round-1/splits.json hypermax/atpe2_models/manifest.json external.sqlite external-traces.json --seed 6
```

The JSON Schemas for offline rollout records and external trace rows live in
`schemas/`. Training emits `manifest.json`, `splits.json`, eight checksummed
models, and `MODEL_CARD.md`.

`gates.py` turns complete synthetic comparison rows into the Gate 2 decision and
validates external traces before Gate 3 analysis. A reduced smoke validation is
always marked incomplete. External analysis refuses to run unless a model
manifest carries passing Gate 2 evidence, all six optimizers and all five seeds
are present, HPO-B rows use its test split, and all four prescribed YAHPO
scenarios are included. Failed trials remain in the normalized-regret traces.

The checked-in runtime manifest is the guarded, checksummed frozen ensemble
produced after held-out validation, pruning, and the single dataset-aggregation
round passed Gate 2. Re-running `freeze.py` will still refuse publication unless
the candidate, pruning evidence, and Gate 2 report have exact matching hashes.
