# Phase 2 Method 3 Results

These results implement the locked engineered-history design in
[the Phase 2 protocol](phase-2-temporal-protocol.md). Phase 2 was designed after
the Phase 1 official-test results were inspected, so the official-test section
below is exploratory rather than confirmatory.

## Development-only validation

The controlled validation compares two median-regression models trained on the
same 166 engines with equal total weight per engine. Both use the condition-aware
current-cycle representation; Method 3 adds the fixed 30-cycle history features.
Five endpoints from each of 42 complete validation engines produce 210 paired
predictions per arm.

| Representation | Features | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: | ---: |
| Phase 2 snapshot | 52 | 28.77 | 40.45 | -11.25 |
| Engineered 30-cycle history | 158 | 26.15 | 38.49 | -10.06 |

The paired engine-average MAE improvement is **2.62 cycles**. A 10,000-resample
engine-level bootstrap interval is **1.00 to 4.26 cycles**, and Method 3 improves
average absolute error for 61.9% of validation engines.

The validation endpoints intentionally span each training trajectory and are not
distributed like the official test endpoints. Their absolute MAE values should
therefore not be compared directly with Phase 1 test MAE. The paired difference
between representations is the relevant validation result.

## Exploratory official-test result

After the representation was fixed, both arms were refitted on all 208 Phase 1
development engines and evaluated on the same 259 official test endpoints.

The `phase2_snapshot` arm is **not** the Phase 1 global baseline. It is the
52-feature condition-aware snapshot representation refitted under the Phase 2
equal-engine-weighting rule. Phase 1 reported 20.36 MAE for the global snapshot
and 19.71 MAE for the condition-aware snapshot. The small shift from 19.71 to
19.57 comes from the changed training weights. The controlled Method 3 comparison
is 19.57 versus 17.27 because those two Phase 2 arms use identical fitting rules
and differ only by the historical features.

| Representation | Features | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: | ---: |
| Phase 2 snapshot | 52 | 19.57 | 27.32 | -2.82 |
| Engineered 30-cycle history | 158 | 17.27 | 25.56 | -1.04 |

The exploratory paired MAE improvement is **2.30 cycles**, with a 10,000-resample
engine-level bootstrap interval of **0.95 to 3.65 cycles**. Method 3 has lower
absolute error for 57.9% of official test engines.

| True-RUL range | Test engines | Snapshot MAE | History MAE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| <=25 | 58 | 5.90 | 3.13 | +2.77 |
| 26–50 | 30 | 9.77 | 7.76 | +2.01 |
| 51–100 | 79 | 21.81 | 18.41 | +3.41 |
| >100 | 92 | 29.47 | 28.31 | +1.15 |

The improvement occurs in every reported RUL band in this run, with the largest
observed difference in the 51–100 range. This is a descriptive exploratory
finding on one simulated benchmark, not evidence of real-equipment performance.

## Interpretation and next role

Method 3 establishes that trailing history contains useful predictive information
beyond the refitted current-cycle comparator under this protocol. It now becomes
the essential baseline for Method 4. A Temporal CNN will be valuable only if it
adds something beyond these transparent summaries, not merely if it beats the
older Phase 1 snapshot result.

Uncertainty intervals have not been added to Method 3 yet. The Phase 2 protocol
deliberately stabilizes point prediction before extending interval estimation.
