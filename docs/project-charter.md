# Project Charter

## Title

Engine RUL: Snapshots, History, and Temporal CNNs on NASA C-MAPSS FD002 and FD004

## Purpose

Build a reproducible analysis that evaluates whether condition-aware preprocessing improves Remaining Useful Life (RUL) prediction accuracy and empirical prediction-interval coverage on a simulated engine-degradation benchmark.

## Research question

Does condition-aware preprocessing improve RUL prediction accuracy and empirical
coverage of 90% prediction intervals compared with a global preprocessing
baseline? Later exploratory work may examine additional interval levels without
recasting them as part of the original primary question.

## Scope

- NASA C-MAPSS FD002 primary study and FD004 replication;
- exploratory engineered-history and Temporal CNN comparison;
- reproducible ingestion and data-integrity audit;
- global and condition-aware preprocessing paths;
- held-out accuracy and empirical interval-coverage evaluation;
- paired engine-level uncertainty analysis for the observed MAE difference;
- interactive evaluation dashboard.

## Boundaries

- The benchmark is simulated and is not a deployed maintenance system.
- Anonymous sensor channels will not be assigned physical meanings.
- The project will not invent dropouts, physical limits, fleet details, maintenance records, costs, or business impact.
- A prediction interval will only be described as calibrated when held-out empirical coverage supports it.
- Negative findings will be reported.
- Exploratory additions and post-hoc corrections will be labelled as such.

## Success criteria

Another person can reproduce the audit and evaluation from the official NASA download, understand the data limitations, and inspect the comparison between preprocessing approaches.
