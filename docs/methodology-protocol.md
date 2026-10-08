# Modelling and Evaluation Protocol

This document records the original comparison and the later, explicitly labelled
exploratory extensions and pre-publication corrections.

## Question

Does a condition-aware feature representation improve point-prediction error and
the empirical coverage of adjusted 90% prediction intervals, relative to a
global representation, on the official held-out FD002 test units?

## Prediction target and evaluation unit

For each training row, remaining useful life is calculated only from observed
run-to-failure data:

`RUL = final cycle for that training unit - current cycle`.

The official test set is evaluated only at the final observed row of each test
unit. Its matching value in `RUL_FD002.txt` is the true remaining cycle count at
that endpoint. This gives one prediction and one interval per test unit.

## Data separation

The official test files remain untouched during development and calibration.

Training units, rather than individual rows, are split deterministically
using a documented random seed into a development set and a calibration set.
Keeping all rows from a unit together prevents the same degradation trajectory
from appearing on both sides of the split.

All learned transformations, including sensor summaries and operating regimes,
are fit using the development units only. The unchanged transformations are then
applied to calibration and official test data.

For interval adjustment, one observed cycle is sampled uniformly and
deterministically from each of the 52 calibration engines (seed 42). This produces
one calibration score per independent engine trajectory instead of treating many
correlated rows from the same engine as independent observations. The sampled
calibration RUL values range from 5 to 273 cycles, with a median of 91.

## Two representations

Both arms will use the same model family, training-unit split, features available
at prediction time, and evaluation procedure.

| Arm | Representation |
| --- | --- |
| Global baseline | Raw anonymous operational settings and sensor values, with any scaling fit on development units only. |
| Condition-aware | Six regimes inferred from the three operational settings of development units only. The learned regime rule is persisted and applied unchanged to all later data. The model receives the regime label and within-regime sensor deviations calculated from development-set regime summaries. |

The six inferred regimes are an analysis device, not an assertion that individual
raw setting triples are literal NASA condition identifiers. They are only used
because NASA documents six FD002 operating conditions.

## Intervals and calibration

The primary interval method uses three otherwise identical histogram
gradient-boosting models to estimate the 5th, 50th, and 95th conditional
quantiles. The lower and upper models form an initial 90% interval. Their interval
is expanded by a split-conformal adjustment learned from the 52 calibration
pseudo-endpoints before official test coverage is calculated.

An exploratory extension adds 80% and 95% intervals. Across all three levels,
seven quantile models are fit per representation at quantiles 0.025, 0.05, 0.10,
0.50, 0.90, 0.95, and 0.975. Each interval level receives its own calibration
adjustment.
Because the quantile models are fit independently, a deterministic monotonic
post-processing step ensures that every 95% interval contains its 90% interval,
and every 90% interval contains its 80% interval. This step can only widen an
outer interval; it cannot improve point predictions.

The official test set provides labels only for each unit's final observed row.
The calibration pseudo-endpoints approximate that endpoint-style evaluation, but
NASA does not document the exact test-truncation sampling mechanism. Therefore,
the finite-sample conformal interpretation depends on an exchangeability
assumption that cannot be fully verified here. Empirical test coverage is always
reported alongside the nominal level.

The project reports empirical coverage, mean interval width, and interval
width by pre-defined RUL bands. It may call an interval "90% calibrated" only if
the measured test coverage is reported beside that claim. If coverage materially
misses 90%, the result will be described as miscalibrated.

## Metrics

For each arm on the official test endpoints, report:

- MAE and RMSE for the central prediction;
- empirical coverage of the nominal 90% interval;
- mean interval width;
- the same metrics by inferred regime and by pre-defined RUL band, where group
  size is disclosed;
- counts of excluded rows, transformations, and any failed checks.
- a paired engine-level bootstrap interval for the overall MAE difference.

No RUL capping, physical sensor interpretation, or operational decision rule is
permitted unless it is separately justified, implemented, and documented before
results are inspected.

## Result discipline

The project compares two representations on one simulated benchmark. A better
score supports only that narrow comparison. It does not establish real-world
maintenance performance, safety, financial benefit, or transfer to another fleet.

## Revision history

- **Original primary analysis:** global versus condition-aware point prediction
  and a nominal 90% interval were defined before the exploratory additions. The
  official test endpoints were then used for the reported comparison.
- **Exploratory extension:** 80% and 95% intervals, regime diagnostics, RUL-band
  diagnostics, and dashboard views were added after the first 90% result was
  reviewed. They are not presented as pre-registered primary outcomes.
- **Pre-publication audit correction:** calibration originally used every row of
  each held-out calibration trajectory. That approach overweights long engines
  and treats correlated rows as if they were independent. It was replaced with
  one deterministic pseudo-endpoint per calibration engine. Point predictions
  were unaffected; all interval results and dashboard data were regenerated.
- **Interval-coherence correction:** independently fitted quantiles produced some
  crossing prediction intervals. A deterministic nesting step was added before
  final reporting, and all output was regenerated. This was a post-hoc integrity
  correction, not part of the original primary protocol.
