# ATPE2 research pipeline

This directory contains the research-only dependencies and artifacts for ATPE2.
It is not included in the runtime package.

All commands require an explicit seed. The SQLite store uses WAL mode and unique
state/evaluation identities, so interrupted collection can be resumed without
duplicating rollouts.

## Factorized-controller cycle

The factorized cycle is intentionally separate from the frozen joint-policy
publication flow documented below. It preserves the public eight-knob action,
uses `atpe2-context-v4`, and never falls back to vanilla TPE at runtime.

Create a new database from a pristine round-zero snapshot. This command refuses
any source containing a previous round-one acquisition:

```bash
python -m research.atpe2.derive_database pristine-round-0.sqlite factorized.sqlite
```

Collect structured round one with 192 source tasks (77 hierarchical and 23 from
each remaining family), checkpoints 20/50/100, and the provisional manifest.
Every state contains exactly 65 unique actions: anchor, provisional policy, 32
group-isolated actions, 18 two-group interactions, 8 full regimes, and 5
highest-disagreement actions. The database records every role and varied group,
plus the ordered mapping, provisional-manifest hash, and acquisition version in
state lineage.

```bash
python -m research.atpe2.generate factorized.sqlite --seed 3 \
  --checkpoints 20 50 100 --task-count 192 --acquisition-round 1 \
  --model-manifest provisional/manifest.json --workers 4
```

Fit one of the bounded factorized branches. Authoritative fits require a clean
committed source snapshot. Tree and leaf choices are restricted to the declared
grid; group/residual/downside/support weights are selected from the held-out
score grid and written into the manifest.

```bash
python -m research.atpe2.train_factorized factorized.sqlite candidate \
  --seed 5 --branch context_v4 --trees 12 --leaf-index 0 \
  --require-clean-source
```

`manifest.json` contains eight task-bootstrap members, each with three group
rankers, an optional residual model, and downside heads. It also checksums the
512-representative support artifact and records component-gate diagnostics.
Candidates that do not pass those gates must not enter behavioral screens.

The quick ladder uses seeds 7, 19, 23, 31, and 47. Aggregate the five candidate
and incumbent reports only after every individual report is present:

```bash
python -m research.atpe2.factorized_screen five-quick.json \
  --candidate candidate-7.json candidate-19.json candidate-23.json candidate-31.json candidate-47.json \
  --incumbent incumbent-7.json incumbent-19.json incumbent-23.json incumbent-31.json incumbent-47.json
```

This cycle stops after the existing pre-Gate-2 standard assessment. Do not run
Gate 2, freeze, publish, or Gate 3 without a new decision.

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

Compare two reports from identical tasks, histories, and seeds with the paired
absolute improvement-AUC comparator. Unlike within-report rank utility and
normalized regret, this remains meaningful when the competing ATPE2 policy
changes:

```bash
python -m research.atpe2.compare baseline.json candidate.json comparison.json --seed 7
```

An experimental eight-knob candidate can be fit without changing the frozen
runtime policy by omitting `--action-schema`. Keep these artifacts under the
ignored research artifact directory until the full collection, aggregation,
and validation sequence passes:

```bash
python -m research.atpe2.train research/atpe2/pilot.sqlite research/atpe2/.artifacts/full-eight-knob --seed 1 --n-estimators 16 --runtime-candidate-count 64
python -m research.atpe2.iterate research/atpe2/.artifacts/full-eight-knob/manifest.json research/atpe2/.artifacts/full-eight-knob-screen.json --seed 7 --workers 6
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

## Factorized eight-knob cycle

The factorized cycle starts from a new database and keeps the public eight-knob
action intact. Round zero uses the exact 192-task family mix and can be split
deterministically across Linux hosts. Merge is idempotent, so interrupted shards
may be resumed and merged again:

```bash
python -m research.atpe2.generate round0-shard.sqlite --seed 20260830 --checkpoints 20 50 100 --task-count 192 --task-shard-index 0 --task-shard-count 3 --workers 4
python -m research.atpe2.merge_databases factorized.sqlite round0-shard-0.sqlite round0-shard-1.sqlite round0-shard-2.sqlite
```

Generate structured round one from the merged database with the provisional
policy manifest. Every state has exactly 65 unique actions with acquisition-role
and varied-group metadata in its lineage:

```bash
python -m research.atpe2.generate factorized.sqlite --seed 20260830 --checkpoints 20 50 100 --acquisition-round 1 --model-manifest provisional/manifest.json --workers 4
```

The declared base grid is bounded and shardable. It contains the structured
joint control plus factorized no-residual, residual, downside, support, and
context-v4 branches. Failed component gates are removed before fresh policy
selection; at most twelve manifests and six behavioral candidates survive:

```bash
python -m research.atpe2.fit_grid factorized.sqlite fits --seed 20260830 --shard-index 0 --shard-count 3
python -m research.atpe2.select_factorized selection.json fits/*/manifest.json
```

After manifest-only pruning, run the fixed held-out five-step quality protocol
on no more than twelve manifests. Feed its artifact-keyed quality mapping back
into selection to advance no more than six candidates to the seed-7 smoke:

```bash
python -m research.atpe2.fresh_policy fresh-policy.json selected/*/manifest.json --seed 20260830 --workers 4
python -m research.atpe2.select_factorized selection.json selected/*/manifest.json --fresh-policy-quality fresh-policy.json
```

Generate each seed's comparator cache once on the Linux screen host, then pass
that immutable cache to every candidate. With a cache, `iterate` computes only
ATPE2 rows and validates the protocol, seed, and environment before pairing:

```bash
python -m research.atpe2.cache_comparators comparators-7.json --seed 7 --workers 6
python -m research.atpe2.iterate candidate/manifest.json candidate-7.json --seed 7 --workers 6 --comparator-cache comparators-7.json
```

Targeted hierarchy/interaction screens use separate environment-locked caches
and the same family filter for comparator and candidate runs:

```bash
python -m research.atpe2.cache_comparators targeted-19.json --seed 19 --families hierarchical interaction
python -m research.atpe2.iterate winner/manifest.json targeted-winner-19.json --seed 19 --comparator-cache targeted-19.json --families hierarchical interaction
```

Continuation and classic-teacher collection are deliberately unavailable until
a factorized manifest reports passing component gates. Continuation uses 24
training-only tasks at checkpoints 25 and 50 and stores paired five- and
fifteen-step targets from the same rollouts. Teacher labels store classic modes
only for training; runtime loads small probability predictors and never imports
or executes classic ATPE:

```bash
python -m research.atpe2.continuation factorized.sqlite winner/manifest.json --seed 20260830 --workers 4
python -m research.atpe2.teacher factorized.sqlite winner/manifest.json --seed 20260830
python -m research.atpe2.fit_grid factorized.sqlite later-fits --seed 20260830 --branches multi_horizon teacher multi_horizon_teacher
```

This cycle stops after the standard pre-Gate-2 assessment. Do not freeze,
publish, execute Gate 2, or start external Gate 3 without a new decision.
