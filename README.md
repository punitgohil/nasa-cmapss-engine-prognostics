# Engine RUL: Snapshots, History, and Temporal CNNs

A reproducible comparison of Remaining Useful Life prediction on NASA C-MAPSS
FD002 and FD004 simulated engine data.

## Results

| Experiment | Method | Test MAE | Test RMSE |
| --- | --- | ---: | ---: |
| FD002 Phase 1 | Global snapshot | 20.36 | 27.92 |
| FD002 Phase 1 | Condition-aware snapshot | 19.71 | 27.12 |
| FD002 Phase 2 | Condition-aware snapshot | 19.57 | 27.32 |
| FD002 Phase 2 | Engineered history | **17.27** | 25.56 |
| FD002 Phase 2 | Temporal CNN | 17.32 | **25.44** |
| FD004 | Global snapshot | 22.40 | 31.58 |
| FD004 | Condition-aware snapshot | 21.67 | 30.57 |
| FD004 | Engineered history | **18.05** | **27.93** |
| FD004 | Temporal CNN | 21.69 | 33.56 |

Engineered history achieves the strongest test MAE on both subsets. The TCN ties
it on FD002 and falls behind on FD004 despite leading during validation.
Condition-awareness alone gives small uncertain gains.

Phase 2 and FD004 give every engine equal total training weight. Phase 1 weighted
rows equally, explaining the small 19.71-to-19.57 shift. Compare methods within
the same experiment.

## Dashboard

Open `dashboard/index.html` in a browser. The included bundle requires no Python,
server, API key, or cloud account. It shows the complete experiment comparison
and Phase 1 prediction inspection, regime diagnostics, and interval evaluation.
Temporal methods currently report point predictions only.

## Methods

1. Global: current cycle, three operational settings, and 21 anonymous sensors.
2. Condition-aware: six training-derived regimes and within-regime deviations.
3. Engineered history: trailing mean, standard deviation, slope, window change,
   and recent change over up to 30 cycles.
4. Temporal CNN: causal convolutions with dilations 1, 2, 4, 8 and a padding mask.

Complete engines are separated for development validation. All transformations
are learned from fitting engines. Paired bootstrap intervals resample engines.

FD002 Phase 2 is exploratory because earlier test results were already known.
FD004 features, architecture, and training rules were fixed before test labels
were opened; its stopping epoch was selected using training-engine validation.
One split and initialization do not establish training-seed robustness or
real-equipment performance. Phase 1 intervals over-cover; unknown test truncation
and only 52 calibration engines limit calibration claims.

## Reproduce (Windows PowerShell)

Verified with Python 3.9.13 and pinned CPU dependencies. Other environments have
not been verified. No paid services are required.

Download the archive from [NASA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
Put `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`,
`train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` in `data/raw/`.
Raw data is excluded from Git.

The observed FD004 files contain 249 training engines, 248 test engines, and 248
RUL labels, reversing the counts in NASA's supplied README. No units were dropped.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.audit_data --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.prepare_data --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.evaluate_models --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.analyze_results --raw-directory data/raw --skip-figures
.\.venv\Scripts\python.exe -m src.evaluate_temporal_features --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.evaluate_tcn --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.evaluate_fd004 --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.verify_fd004 --raw-directory data/raw
.\.venv\Scripts\python.exe -m src.build_dashboard_data
```

Run in the listed order from the project folder. TCN training takes several
minutes on CPU. Local outputs remain in `data/processed/`. Published summaries
are retained in `results/` and the dashboard contains derived outputs.

## Documentation

- [Sources and provenance](docs/sources-and-provenance.md)
- [Data contract](docs/data-contract.md)
- [Phase 1 protocol](docs/methodology-protocol.md) and [findings](docs/analysis-findings.md)
- [Phase 2 protocol](docs/phase-2-temporal-protocol.md)
- [Engineered history](docs/phase-2-method3-results.md)
- [Temporal CNN](docs/phase-2-tcn-results.md)
- [FD004 protocol](docs/fd004-protocol.md) and [results](docs/fd004-results.md)
- [Integrity audit](docs/pre-publication-audit.md)

## Layout

```text
dashboard/       static dashboard and published derived data
docs/            protocols, findings, provenance, audit history
results/         published evaluation summaries
src/             data preparation, features, evaluation, dashboard generation
tests/           integrity, temporal leakage, and causality checks
data/raw/        local NASA files (ignored)
data/processed/  generated local predictions (ignored)
.venv/           local Python environment (ignored)
```

Sensor channels remain anonymous. No maintenance savings, physical sensor
interpretations, safety guarantees, or real-fleet results are claimed.
