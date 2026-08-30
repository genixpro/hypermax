# ATPE2 current findings

Status: development findings as of 2026-08-30. These observations guide
iteration; they are not publication claims. Only a fresh complete Gate 2 and an
untouched external Gate 3 can support research-grade conclusions.

## Why the first external ATPE2 result tied TPE

The tie had an exact implementation cause, not merely weak adaptation. Every
sampled HPO-B and YAHPO context was outside the narrow synthetic training
envelope. The former controller treated that diagnostic as a hard gate and
selected the standard-TPE anchor at every five-trial boundary. ATPE2 and
Hyperopt TPE consequently produced identical configurations and identical
losses throughout the invalidated study.

The common envelope violations were structural rather than small numerical
drift. HPO-B spaces had no integer parameter where training had always seen one.
YAHPO spaces frequently had many more numeric, integer, categorical, and
conditional parameters, greater effective dimension and cardinality, deeper
branches, and lower conditional activity coverage than the training tasks.

The old Gate 3 result therefore measured fallback-only TPE behavior and is not
evidence about adaptive ATPE2.

## Adaptive-only runtime correction

Runtime policy `atpe2-adaptive-only-v2` removes warm-up, training-envelope,
uncertainty, and non-finite fallback branches. Every controller boundary now
scores the joint action set with the fitted ensemble. Training-envelope status
is retained only as telemetry. Missing or incompatible models and non-finite
predictions raise explicit errors.

The TPE anchor remains one ordinarily scored action in the candidate set; it is
not a fallback. On a matched post-fix `lcbench` run, ATPE2 and TPE shared the
five-trial initial design and then differed on all 90 remaining configurations
and losses. This confirms that the adaptive path is behaviorally active.

## Model and inference efficiency

The original adaptive-only ensemble used eight 300-tree LightGBM regressors and
scored 1,024 Sobol actions plus the anchor per boundary. Since validation had
retained only `gamma` and `history_keep_fraction`, this was unnecessarily
expensive.

Efficiency variants were retrained against the same data, grouped split, seed,
and two-knob action schema. The smallest tested variant was accepted only after
the complete 24-task, 1,440-block Gate 2 passed.

| Measurement | Former policy | Current policy |
| --- | ---: | ---: |
| Trees per ensemble member | 300 | 8 |
| Runtime Sobol actions | 1,024 | 32 |
| Model artifact bytes | 7,414,588 | 223,980 |
| Median unloaded controller latency | 121.84 ms | 1.65 ms |
| Held-out model MAE | 0.06225695 | 0.06186098 |

The complete Gate 2 interval against TPE remained `[0.0, 0.0]`; the worst
synthetic-family degradation was 1.67%, below the declared 5% bound. All runtime
decisions remain adaptive. Offline data acquisition is deliberately separate
and still evaluates 64 Sobol actions plus the anchor when producing labels.

## Lightweight development benchmark

The first fast screen completed in seconds but produced too many ties to guide
algorithm changes. Version 2 now uses:

- two fixed development tasks from each of six synthetic families;
- random, TPE, and space-filled ATPE2 histories;
- history checkpoints at 10, 25, and 50 trials;
- two paired rollout seeds and ten-trial incumbent curves;
- Random Search, Hyperopt TPE, legacy ATPE, and ATPE2;
- 216 paired blocks and 8,640 objective evaluations;
- scale-normalized improvement AUC and paired family regret effects;
- constant-zero and plateau tasks as correctness checks, not comparative
  performance families.

The report is always marked `evidence_eligible: false`. Repeated development on
these fixed tasks can overfit them, so the screen can reject obvious regressions
but cannot approve publication. Runtime has ranged from about 125 to 204 seconds
on the pilot machine depending on concurrent load.

### Seed-7 result

Higher utility/improvement is better; lower normalized regret is better.

| Optimizer | Mean rank-utility AUC | Mean normalized regret |
| --- | ---: | ---: |
| Legacy ATPE | 0.5806 | 0.2271 |
| ATPE2 | 0.5024 | 0.3394 |
| Hyperopt TPE | 0.4888 | 0.3669 |
| Random Search | 0.4282 | 0.4668 |

ATPE2 beat TPE on 44 rank-utility blocks, tied 135, and lost 37. The directional
95% interval for scale-normalized improvement-AUC difference was
`[0.0, 0.00971892]`. This weakly favors ATPE2 but does not show a decisive
aggregate advantage.

Paired mean normalized-regret differences for ATPE2 minus TPE were:

| Family | Paired mean regret difference | Direction |
| --- | ---: | --- |
| Rugged | -0.2302 | ATPE2 better |
| Interaction | -0.0310 | ATPE2 better |
| Smooth | +0.0314 | ATPE2 worse |
| Hierarchical | +0.0648 | ATPE2 worse |

All four paired family medians were zero, so the effects come from a subset of
blocks rather than a universal shift. Legacy ATPE is strongest on this synthetic
development distribution but is roughly 2.7 times slower than ATPE2. Legacy
ATPE's earlier external underperformance also demonstrates that this development
distribution is not a substitute for external evidence.

## Controller behavior and current limitations

The seed-7 v2 run recorded 432 adaptive decisions and 345 distinct
configurations. No fallback was taken. About 42.6% of decisions were outside the
recorded training envelope. The 50-trial checkpoint deliberately tests beyond
the training envelope's maximum successful-history count of 40, while
hierarchical spaces also exceed several structural ranges.

Current development evidence suggests:

1. Adaptation is active and no longer aliases standard TPE.
2. ATPE2 is modestly better than TPE in aggregate on the development screen.
3. Rugged and interacting numerical objectives are the strongest area.
4. Hierarchical spaces and longer histories are the clearest weaknesses.
5. The synthetic development distribution favors legacy ATPE and does not
   reliably predict the external ranking.
6. Broader hierarchical/conditional training states and histories beyond 40
   trials are higher-priority experiments than further inference compression.

## External-study runtime and status

The invalidated fallback-only Gate 3 remains historical evidence only. A later
899-run partial YAHPO database belongs to the superseded 300-tree adaptive-only
policy and must not be resumed or mixed with the current eight-tree policy.

Direct timings across all four YAHPO scenario types project the six-optimizer
study at about 20.9 single-worker hours. Measured multiprocessing scaling on the
M3 machine was about 3.2x, producing an estimated wall time of roughly 6.5 hours
and a practical range of 6–8 hours. The smaller ensemble greatly accelerates
ATPE2 itself, but 600 YAHPO evaluations and five comparison optimizers per
task-seed job dominate full-study time.

The next research-grade sequence is:

1. iterate on the lightweight synthetic screen;
2. freeze a candidate without using external test results for tuning;
3. rerun the complete held-out synthetic Gate 2;
4. start a fresh YAHPO/HPO-B database bound to that exact frozen manifest;
5. make no performance claim until the complete Gate 3 report passes its
   predeclared statistical checks.
