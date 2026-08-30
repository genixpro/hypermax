# ATPE2 model card

Status: Gate-2-approved frozen runtime ensemble.

## Model

- Ensemble: eight task-bootstrap LightGBM regressors
- Trees per ensemble member: 8
- Runtime Sobol candidates per decision: 32 plus the TPE anchor
- Artifact identity: `eac474343d04ac898e301bb032d52149295ce21df1fa638d9bc7e7d2ce22d0a0`
- Feature schema: `atpe2-context-v2`
- Action schema: `atpe2-action-v1`
- Retained actions: gamma, history_keep_fraction
- Fixed constants: `{"history_quality_bias": 0.0, "history_recency_bias": 0.0, "n_ei_candidates": 24, "secondary_fraction": 0.0, "secondary_lock_probability": 0.0, "secondary_top_fraction": 1.0}`
- Validation MAE: 0.06186098 over 10454 rows
- Uncertainty penalty: 0.9000
- Validation uncertainty p95 (diagnostic only): 0.00715996

## Data and evaluation boundaries

- Training data SHA-256: `a156170dca52cc504c564a9ba53e9fb649c71e4cb1068ca6333c8ce6597b243d`
- Grouping unit: whole task and trajectory lineage
- Train/validation/test task counts: 12/6/6
- Dataset-aggregation rounds present: [0, 1]
- External HPO-B/YAHPO data was not used for model fitting or calibration.

## Intended use and limitations

This frozen ensemble predicts scale-free five-step controller utility for ATPE2. It is
not a multi-fidelity controller. `abh2` is a compatibility path through the
existing scheduler and carries no performance claim. Publication is guarded by
`research.atpe2.freeze`, which requires exact matching pruning and passing Gate 2
evidence.

## Validation approval

- Gate 2 report SHA-256: `572d303a391a33bdcf57b7409f6230a6928b77665dd8f6b8c443a3c8040fe5c1`
- Knob-pruning report SHA-256: `70732e8a83cf9aed44ec8ba91ed72b9e95420efb1449a3badcf36f7649396d17`
