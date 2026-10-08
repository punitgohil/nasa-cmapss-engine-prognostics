# FD004 replication results

The four methods were frozen in [the protocol](fd004-protocol.md) before FD004
test labels were parsed. Each was fitted anew with equal total weight per engine.
The TCN stopping epoch was selected using training-engine validation only.

## Source-count discrepancy

NASA's supplied README lists 248 training and 249 test trajectories. The actual
local raw files contain **249 training trajectories (61,249 rows)** and **248
test trajectories (41,214 rows)**, with exactly **248 matching RUL labels**.
The previously recorded hashes match these files. No engines were discarded;
the evaluation follows the observed files rather than the reversed README counts.

The split contains 199 development engines and 50 reserved engines. Development
validation uses 159 fitting engines and 40 validation engines, with five observed
endpoints per validation engine. Final fitting uses all 199 development engines.

## Validation versus official test

| Method | Validation MAE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: |
| Global snapshot | 32.38 | 22.40 | 31.58 |
| Condition-aware snapshot | 30.69 | 21.67 | 30.57 |
| Engineered history | 30.99 | **18.05** | **27.93** |
| Temporal CNN | **26.94** | 21.69 | 33.56 |

Validation selected epoch 11 for the TCN. Although it led on validation, its
advantage did not carry through to official test endpoints. Validation samples
span run-to-failure training trajectories, while official endpoints follow an
undocumented truncation process. That difference and the single training split
are plausible explanations to investigate, not established causes.

## Paired official-test comparisons

Positive improvement means the added method reduced MAE. Intervals resample test
engines 10,000 times with seed 2026; they do not capture training-seed variation.

| Comparison | MAE improvement | Bootstrap 95% interval |
| --- | ---: | --- |
| Condition-aware over global | +0.73 | -0.48 to +1.91 |
| Engineered history over condition-aware | +3.63 | +2.09 to +5.17 |
| TCN over engineered history | -3.64 | -5.53 to -1.81 |

Engineered history performs best on this held-out subset. Condition-awareness
alone gives a small uncertain improvement. The frozen TCN falls behind engineered
history, reinforcing the conclusion that neural complexity did not add reliable
value under the tested protocol. These are results from one simulated family,
one split and one initialization; they do not establish deployment performance.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_fd004 --raw-directory "PATH_TO_NASA_FILES"
.\.venv\Scripts\python.exe -m src.verify_fd004 --raw-directory "PATH_TO_NASA_FILES"
```

Results are stored in `data/processed/fd004_results.json` and paired predictions
in `data/processed/fd004_predictions.csv`. The verifier uses the supplied raw
folder for source-label checks. Phase 1 and FD002 Phase 2 artifacts
remain separate; the dashboard now includes the combined result table.
