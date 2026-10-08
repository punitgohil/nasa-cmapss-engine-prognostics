# Phase 2 Temporal CNN Results

These results follow the locked Method 4 design in
[the Phase 2 protocol](phase-2-temporal-protocol.md). The architecture and
training rules were written before the TCN validation or official-test results
were inspected. Phase 2 remains exploratory because earlier FD002 models had
already exposed the official test labels.

## Locked model

- 30-cycle input sequence;
- 52 condition-aware features at every observed time step plus one mask channel;
- four causal convolution blocks with kernel size 3 and dilations 1, 2, 4, 8;
- 31-cycle receptive field;
- 32 hidden channels, residual connections, ReLU, and 0.10 dropout;
- engine-balanced absolute-error training on uncapped RUL;
- validation-only early stopping with a maximum of 60 epochs and patience 8.

Validation selected epoch 11. The final model was then initialized again and
trained on all 208 development engines for exactly 11 epochs.

## Development validation

All three values below use the same 42 validation engines and 210 fixed
pseudo-endpoints. Snapshot and Method 3 results are carried forward from the
controlled Method 3 experiment.

| Representation | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: |
| Phase 2 condition-aware snapshot | 28.77 | 40.45 | -11.25 |
| Method 3 engineered history | 26.15 | 38.49 | -10.06 |
| Temporal CNN | **25.16** | **36.63** | -6.15 |

The TCN's observed MAE gain over Method 3 is 0.99 cycles. Its paired
engine-level bootstrap interval is **-1.14 to +3.06 cycles**, and it improves
average error for 61.9% of validation engines. Because the interval crosses
zero, validation does not establish a reliable advantage over Method 3.

## Exploratory official test

| Representation | MAE | RMSE | Mean error |
| --- | ---: | ---: | ---: |
| Phase 2 condition-aware snapshot | 19.57 | 27.32 | -2.82 |
| Method 3 engineered history | **17.27** | 25.56 | -1.04 |
| Temporal CNN | 17.32 | **25.44** | -0.34 |

The TCN is 0.05 MAE cycles worse than Method 3 and 0.12 RMSE cycles better.
Its paired MAE improvement over Method 3 is -0.05 cycles, with a bootstrap
interval of **-1.25 to +1.17 cycles**. It has lower absolute error for 49.8% of
test engines. These values describe a practical tie, not TCN superiority.

## Conclusion

Phase 2 supports the value of temporal information: both historical approaches
substantially outperform the controlled snapshot comparator. The locked TCN does
not demonstrate additional value beyond the transparent engineered statistics.
Method 3 remains the best MAE model in this experiment, while the TCN has a very
small RMSE advantage.

No architecture search will be performed against FD002 test results. The honest
follow-up carried the frozen methods to previously untouched FD004 labels. See
[the completed FD004 results](fd004-results.md).

The complete TCN run was repeated and reproduced the selected epoch and MAE
values exactly. The exported comparison contains 938 complete records with no
missing values or duplicated model/engine/cycle keys.
