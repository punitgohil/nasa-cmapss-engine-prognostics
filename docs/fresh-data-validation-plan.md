# Fresh-Data Validation Plan

**Completed:** the frozen FD004 comparison has now been evaluated. Test labels
were first parsed after final model fitting. See [results](fd004-results.md).
The original planning statements below record the pre-evaluation state; their
248-train/249-test counts repeat NASA's README, while the observed files actually
contain 249 training and 248 test engines.

The Temporal CNN is developed on FD002 despite the fact that prior Phase 1 and
Method 3 official-test results are already known. Its FD002 result is therefore
exploratory. A stronger follow-up requires data whose labels have not influenced
any design decision.

## Immediate blind follow-up: FD004

FD004 is the preferred next dataset after the FD002 Temporal CNN is frozen. NASA
documents 248 training trajectories, 249 test trajectories, six operating
conditions, and two fault modes. It is comparable to FD002 while adding a second
fault mode. The local NASA archive already contains its train, test, and RUL
files, but this project has not loaded or summarized the FD004 RUL values.

To preserve that state, only filenames, byte sizes, and SHA-256 hashes have been
recorded. File contents and RUL values have not been parsed:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `train_FD004.txt` | 10,350,705 | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| `test_FD004.txt` | 6,957,759 | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| `RUL_FD004.txt` | 1,084 | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |

Before those labels are opened, a separate FD004 protocol will freeze:

- the global snapshot comparator;
- the condition-aware snapshot comparator;
- Method 3 engineered history;
- the Temporal CNN architecture and training rule carried forward from FD002;
- dataset-specific development/calibration splits determined without test labels;
- identical uncapped-RUL metrics and disclosure rules.

FD004 is a same-simulator external check, not evidence about real engines.

## Later independent extension: N-CMAPSS

NASA's Prognostics Center of Excellence also publishes the newer “Aircraft
Engine Run-to-Failure Dataset under real flight conditions,” commonly called
N-CMAPSS. It uses C-MAPSS with realistic flight conditions and is a stronger
external benchmark, but its files and schema are substantially larger and differ
from FD002/FD004. It should be handled as a separate data-engineering phase after
the blind FD004 experiment, not silently forced through the current loader.

Primary NASA sources:

- [C-MAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
