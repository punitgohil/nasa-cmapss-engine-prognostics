# Analysis Findings

These findings describe the fixed NASA C-MAPSS FD002 experiment. They do not
establish performance on real equipment.

## Why the aggregate improvement is modest

The inferred regimes are not arbitrary-looking or poorly populated. On a fixed
5,000-row development sample, six clusters achieve a silhouette score of 0.997,
the strongest score among the tested choices from 2 through 10 clusters. Every
regime contains at least 6,450 development rows, and the official test endpoint
groups contain 32 to 69 engines.

The most plausible interpretation is model redundancy: the global nonlinear
model already receives all three operational-setting columns and can learn many
condition-dependent splits directly. Explicit regime indicators and within-regime
sensor deviations therefore add some useful structure, but much of the available
operating-condition signal was already accessible to the baseline. This is an
inference from the design and observed results, not a proven causal mechanism.

The aggregate difference is uncertain. A paired engine-level bootstrap with
10,000 resamples estimates the 0.65-cycle MAE improvement with a 95% interval of
-0.17 to +1.48 cycles; condition-aware prediction has lower absolute error for
52.5% of test engines. Because the interval includes zero, the observed overall
gain should be described as modest and inconclusive rather than definitive.

## Effect by true-RUL range

| True-RUL range | Test engines | Global MAE | Condition-aware MAE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| <=25 | 58 | 5.42 | 6.20 | -0.78 |
| 26–50 | 30 | 10.72 | 10.77 | -0.05 |
| 51–100 | 79 | 24.86 | 22.89 | +1.97 |
| >100 | 92 | 29.06 | 28.40 | +0.66 |

Condition-awareness helps most in the 51–100 range. It does not help close to
failure in this run; below or equal to 25 true cycles, the global baseline has
the lower MAE.

## Effect by inferred regime

| Regime | Test engines | Global MAE | Condition-aware MAE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 0 | 69 | 16.91 | 16.15 | +0.76 |
| 1 | 39 | 22.46 | 20.67 | +1.79 |
| 2 | 42 | 21.73 | 21.88 | -0.15 |
| 3 | 32 | 21.51 | 21.37 | +0.14 |
| 4 | 41 | 22.34 | 22.31 | +0.03 |
| 5 | 36 | 19.84 | 18.51 | +1.33 |

The gain is concentrated in regimes 1 and 5. Regime 2 becomes slightly worse,
while the observed differences in regimes 3 and 4 are very small. The project
must therefore describe the effect as context-dependent rather than universal.

## Interval trade-off

| Representation | Desired coverage | Actual coverage | Mean width (cycles) |
| --- | ---: | ---: | ---: |
| Global | 80% | 86.49% | 80.89 |
| Global | 90% | 94.21% | 99.36 |
| Global | 95% | 97.30% | 119.04 |
| Condition-aware | 80% | 87.26% | 77.72 |
| Condition-aware | 90% | 93.44% | 94.68 |
| Condition-aware | 95% | 96.91% | 111.21 |

Condition-aware intervals are narrower at all three nominal coverage levels. Both
methods over-cover on the official test endpoints, especially at the 80% level,
so the intervals are conservative rather than exactly calibrated. Independently
fitted quantile models are post-processed so higher-coverage intervals always
contain lower-coverage intervals.

Calibration uses one pseudo-endpoint from each of 52 held-out training engines.
This avoids row-level pseudo-replication, but the small calibration sample and
unknown official test-truncation mechanism remain important limitations.
