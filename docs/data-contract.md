# FD002 Data Contract

## Authoritative source

NASA C-MAPSS Turbofan Engine Degradation Simulation dataset, FD002 subset. Raw
files are obtained from the official NASA download and remain outside version
control. Links and local checksums are recorded in
[sources and provenance](sources-and-provenance.md).

## Required input files

| File | Verified purpose | Verified structure |
| --- | --- | --- |
| `train_FD002.txt` | Run-to-failure trajectories for model development | 53,759 rows; 260 units; 26 whitespace-separated numeric fields per row |
| `test_FD002.txt` | Truncated trajectories for held-out evaluation | 33,991 rows; 259 units; 26 whitespace-separated numeric fields per row |
| `RUL_FD002.txt` | True remaining cycles for the final observation of each test unit | 259 values, aligned to the 259 test units |

## Row schema

| Position | Name | Type | Meaning |
| --- | --- | --- | --- |
| 1 | `unit_id` | integer | Engine/unit identifier |
| 2 | `cycle` | integer | Operational cycle index |
| 3–5 | `op_setting_1`–`op_setting_3` | float | Anonymous operational settings |
| 6–26 | `sensor_1`–`sensor_21` | float | 21 anonymous measurement columns |

The settings and sensor channels must not be assigned physical units or engineering meanings not supplied by NASA.

## Observed integrity checks

- All checked FD002 rows contain 26 fields.
- Training cycles are continuous within each unit.
- Test cycles are continuous within each unit.
- No continuity violations were observed in the local source copy.
- Exact triples of operational-setting values vary within the dataset. They must not
  be treated as six literal condition labels.

## Implemented ingestion checks

The pipeline verifies field count, numeric parsing, finite values, integer unit and
cycle identifiers, unique unit/cycle pairs, cycle continuity, RUL count and type,
and column variance. It reports observed results rather than creating synthetic
faults. Model evaluation refuses to proceed when this structural audit fails.

## Condition-aware methodology constraint

NASA documents six operating conditions for FD002, but exact operational-setting
triples vary. The condition-aware transformation infers six regimes from the
development-set operational settings only and applies the fitted rule unchanged
to calibration and test data. This prevents test-data leakage and avoids
mistaking raw numeric tuples for documented condition identities.

`RUL_FD002.txt` contains values without unit identifiers. The implementation maps
its row order to sorted test unit IDs only after verifying that those IDs are the
contiguous sequence 1 through 259. This makes the alignment assumption explicit
and fail-fast rather than silent.
