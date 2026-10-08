# Phase 2 Temporal Protocol

## Status and purpose

Phase 2 is an explicitly exploratory extension designed after the Phase 1
official-test results were inspected. It does not replace or retroactively alter
the audited Phase 1 comparison.

The Phase 2 question is:

> Does trailing trajectory information improve FD002 RUL prediction, and does a
> learned Temporal CNN later add value beyond transparent engineered history?

Method 3 establishes the engineered-history comparator. Method 4 will later use
the same information boundary with a Temporal CNN.

## Method 3 representation

At every prediction cycle, Method 3 uses only that cycle and up to the preceding
29 cycles from the same engine. The maximum trailing window is therefore 30
observations.

The existing condition-aware transformer is fitted without validation or test
data. For each of the 21 anonymous sensor deviations from its learned regime
mean, Method 3 calculates:

- trailing mean;
- trailing population standard deviation;
- ordinary least-squares slope against observed cycle number;
- change from the first to the last observation in the trailing window;
- change over at most the most recent five cycles.

The number of observations in the current trailing window is also included.
Windows at the beginning of a trajectory use the observations that actually
exist; no synthetic sensor values are created. Current-cycle condition-aware
features remain present. Sensor channels retain anonymous names, and no trend is
assigned a physical interpretation.

## Development-only validation

The original Phase 1 split first reserves 52 complete training engines for later
interval calibration. The remaining 208 development engines are split again by
complete engine using seed 2026:

- 166 Phase 2 model-training engines;
- 42 Phase 2 validation engines.

Five observed cycles are sampled without replacement from every validation
engine using seed 2026. Metrics are paired on the same 210 pseudo-endpoints.
Because every engine contributes five validation records, each validation engine
has equal influence.

All available training-cycle endpoints are used for fitting, but sample weights
are chosen so every training engine has the same total weight. This prevents
longer trajectories from dominating merely because they generate more endpoints.

## Controlled comparison

Both validation arms use the same histogram gradient-boosting median-regression
model and fixed Phase 1 hyperparameters:

1. `phase2_snapshot`: current-cycle condition-aware features;
2. `phase2_engineered_history`: the same features plus the locked Method 3
   trailing statistics.

No architecture or feature choice is tuned against official FD002 test labels.
After this protocol is implemented and tested, both arms are refitted on all 208
development engines and evaluated once on the 259 official test endpoints. Those
results remain exploratory because Phase 1 test results were already known when
Phase 2 was designed.

## Metrics and interpretation

Method 3 reports MAE, RMSE, mean error, a paired engine-level bootstrap interval
for the MAE difference, and error by true-RUL band. The uncapped RUL target is
retained for comparability with Phase 1.

Method 3 is useful even if it does not improve error: it isolates whether simple
historical summaries explain any later Temporal CNN gain. No uncertainty model is
added at this stage; interval work follows only after the point-prediction
comparison is stable.

## Method 4 locked Temporal CNN

Method 4 is fixed before its validation or official-test result is inspected.
Each endpoint is represented by the current cycle and up to 29 preceding cycles
from the same engine. Every time step contains the same 52 condition-aware
current-cycle features used by the Phase 2 snapshot comparator, standardized
using model-training engines only. A binary channel distinguishes observed rows
from left padding. No synthetic sensor values are interpreted as observations.

The network has four causal one-dimensional convolution blocks with kernel size
3, 32 hidden channels, dilations 1, 2, 4, and 8, ReLU activations, residual
connections, and dropout 0.10. One convolution per block gives a receptive field
of 31 cycles. The final time-step representation feeds a 16-unit hidden layer and
one uncapped RUL output.

Training uses engine-balanced absolute-error loss, Adam with learning rate
0.001 and weight decay 0.0001, batches of 256, at most 60 epochs, and early
stopping after eight epochs without development-validation MAE improvement. The
best validation epoch is then fixed; the final model is retrained from scratch on
all 208 Phase 1 development engines for exactly that many epochs before one
exploratory official-test evaluation.

The TCN is compared primarily with Method 3 on the exact same validation and
official-test endpoints. GRU, LSTM, window-length searches, architecture sweeps,
RUL capping, and test-driven hyperparameter changes are out of scope.
