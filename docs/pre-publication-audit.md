# Pre-Publication Integrity Audit

This audit was completed before repository cleanup and publication. Its purpose
was to trace every reported result back to the local official NASA C-MAPSS FD002
files, identify unsupported claims, and correct methodological or documentation
discrepancies.

This record covers the completed Phase 1 project. Phase 2 was added afterward as
an explicitly exploratory extension and has its own locked protocol, automated
tests, generated result file, and independent prediction-consistency checks.

## Checks completed

- Confirmed the three local FD002 files match the original downloaded copies by
  SHA-256 hash.
- Confirmed 53,759 training rows across 260 units, 33,991 test rows across 259
  units, and 259 official endpoint RUL values.
- Re-ran the structural audit for field counts, numeric parsing, unit/cycle
  uniqueness, cycle continuity, RUL alignment, and non-negative integer RUL
  labels.
- Re-ran preparation, both model arms, interval evaluation, paired bootstrap
  analysis, regime diagnostics, and dashboard generation from the raw files.
- Recomputed aggregate metrics directly from the engine-level prediction file
  and confirmed they agree with the evaluation JSON.
- Confirmed the dashboard embeds the current evaluation, analysis, and prediction
  records, with no interval-nesting violations.
- Searched source code, documentation, and dashboard text for invented physical
  sensor meanings, operational claims, secrets, and stale result values.
- Cross-checked dataset structure and scope against the official NASA dataset
  page and the supplied dataset readme.

## Correction made during the audit

The first interval implementation used every row from each held-out calibration
trajectory. Rows within one engine trajectory are correlated, so that approach
incorrectly treated many observations from the same engine as independent and
gave longer trajectories more weight.

The final implementation uses one deterministic, uniformly sampled observed
cycle from each of the 52 held-out calibration engines. All interval results and
dashboard data were regenerated. Point-prediction MAE and RMSE were unaffected.
This correction is also recorded in the methodology revision history.

Separately fitted quantile models could also yield non-nested intervals. A
deterministic post-processing step now guarantees that each 95% interval contains
its 90% interval and each 90% interval contains its 80% interval. This affects
interval coherence only, not point predictions.

## Claims supported by this project

- In this fixed FD002 comparison, the condition-aware representation has a
  0.65-cycle lower MAE than the global representation.
- A paired 10,000-resample engine-level bootstrap gives a 95% interval from
  -0.17 to +1.48 cycles for that observed MAE improvement, so the aggregate
  advantage is modest and inconclusive.
- Both interval methods over-cover on the official test endpoints and are
  described as conservative rather than exactly calibrated.
- Regimes are anonymous clusters of the three operating-setting columns. They
  are not assigned physical or operational meanings.

## Limitations that remain

- FD002 is simulated and does not establish performance on real equipment.
- The evaluation uses one fixed development/calibration split and one model
  family; it is not a broad model-selection study.
- The official test-truncation sampling mechanism is not documented in the
  supplied files. Exchangeability between sampled calibration pseudo-endpoints
  and official test endpoints cannot be fully verified.
- The calibration set has only 52 independent engine trajectories.
- Regime and RUL-band analyses were added after the primary 90% comparison and
  are explicitly exploratory.
- NASA's download page does not specify a reuse license. The raw files therefore
  remain excluded from the repository and must be downloaded from NASA.

No fabricated rows, labels, physical sensor identities, maintenance outcomes,
cost estimates, or deployment claims were found in the audited project.
