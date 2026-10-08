# FD004 held-out comparison protocol

Recorded before parsing FD004 test RUL labels. All four methods are trained anew
on FD004. Features, boosting parameters, TCN architecture, 30-cycle window,
uncapped target, and equal-engine training weights carry forward from FD002.

Seed 42 reserves 20% of training engines for possible later calibration. Seed
2026 divides the remaining engines into 80% fitting and 20% validation engines.
Five observed endpoints per validation engine are selected with seed 2026.
TCN training uses the existing maximum 60 epochs and patience eight; the best
validation epoch is fixed before final retraining. Final models use all remaining
development engines. No FD004 test labels are loaded until final models are fit.

Report MAE, RMSE, mean error and paired engine bootstrap intervals (10,000
resamples, seed 2026) for condition-aware versus global, history versus
condition-aware, and TCN versus history. Bootstrap intervals describe this fixed
experiment, not variation across training seeds. No test-driven changes follow.
FD004 remains a same-simulator replication, not real-equipment validation.
