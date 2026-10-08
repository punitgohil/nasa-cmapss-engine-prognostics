# First Reproducible Evaluation

This record describes one fixed comparison run. It is not a claim about real
engines, maintenance operations, or deployment performance.

## Fixed protocol

- Dataset: NASA C-MAPSS FD002 simulated benchmark.
- Development split: 208 complete training trajectories.
- Calibration split: 52 complete training trajectories, contributing one
  deterministic uniformly sampled pseudo-endpoint each.
- Official evaluation: 259 final observations from the official test set.
- Primary model: histogram gradient boosting at the 5th, 50th, and 95th
  quantiles. Exploratory 80% and 95% intervals add four more quantile models.
- Interval adjustment: one calibration score per held-out engine trajectory.
- Seed: 42.

## Overall results

| Representation | MAE | RMSE | Empirical 90% coverage | Mean interval width |
| --- | ---: | ---: | ---: | ---: |
| Global baseline | 20.36 | 27.92 | 94.21% | 99.36 cycles |
| Condition-aware | 19.71 | 27.12 | 93.44% | 94.68 cycles |

The condition-aware representation improved MAE by 0.65 cycles and RMSE by
0.80 cycles in this single fixed comparison. Its average 90% interval was 4.68
cycles narrower. Both interval procedures are conservative on the official test
endpoints: empirical coverage is above the nominal 90% level.

The paired per-engine MAE improvement is 0.65 cycles, with a 10,000-resample
bootstrap 95% interval of -0.17 to +1.48 cycles. Condition-aware prediction has
lower absolute error for 52.5% of test engines. Because the bootstrap interval
crosses zero, this run does not establish a decisive overall advantage.

## Important nuance

The condition-aware representation was not superior in every pre-defined true
RUL band. For example, its error was higher in the `<=25` band in this run.
The aggregate improvement is therefore not evidence that it is uniformly better
at every stage of degradation.

## Reproduce locally

From the project folder, with the local environment active:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.audit_data --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.prepare_data --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.evaluate_models --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.analyze_results --raw-directory data/raw --skip-figures
.\.venv\Scripts\python.exe -m src.build_dashboard_data
```

Generated JSON files remain local under `data/processed/`. The raw NASA data and
generated results are excluded from Git by design.

See the methodology revision history for the distinction between the original
90% analysis, exploratory additions, and pre-publication corrections.
