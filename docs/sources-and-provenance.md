# Sources and Provenance

## Primary sources

- [NASA Open Data: CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Saxena, Goebel, Simon, and Eklund (2008), Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214)

The NASA dataset page and the README included with the downloaded archive state
that FD002 contains 260 training trajectories, 259 test trajectories, six
operating conditions, one fault mode, three operational-setting columns, and 21
measurement columns in positions 6 through 26. The training series run to failure;
the test series stop before failure and have a separate RUL vector.

## Local-file verification

The project-local files were compared byte-for-byte by SHA-256 with the original
files retained from the downloaded archive. The copies match.

| File | SHA-256 |
| --- | --- |
| `train_FD002.txt` | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `test_FD002.txt` | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `RUL_FD002.txt` | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |

The structural audit independently observes 53,759 training rows, 33,991 test
rows, 260 training units, 259 test units, 26 numeric fields per trajectory row,
and 259 RUL values.

## Redistribution decision

NASA's Open Data resource currently lists the dataset license as unspecified.
For that reason, this repository does not redistribute the raw text files. Users
are directed to NASA's official download and can compare their local checksums
with the values above. The dashboard bundle contains model outputs and aggregate
analysis, not the anonymous raw sensor measurements.
