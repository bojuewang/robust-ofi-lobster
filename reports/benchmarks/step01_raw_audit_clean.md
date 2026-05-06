# Step 01 Raw LOBSTER Audit + Cleaning Benchmark

Overall status: **PASS**

## Time-grid contract
- Continuous trading window: `[34200, 57600)` seconds after midnight
- Bin width: `0.5` seconds
- Number of bins: `46800`
- Chronological split: `32760 / 7020 / 7020`
- Any-asset event bins: `46290`

## Ticker audit
| ticker | msg rows | orderbook file | first ts | last ts | event bins | carry-forward bins | sample min spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMZN | 269748 | exists | 34200.017460 | 57599.959360 | 29801 | 16999 | 0.02 |
| AAPL | 400391 | exists | 34200.004241 | 57599.913118 | 34800 | 12000 | 0.01 |
| GOOG | 147916 | exists | 34200.015105 | 57599.871751 | 22678 | 24122 | 0.01 |
| INTC | 624040 | exists | 34200.005743 | 57599.948442 | 36086 | 10714 | 0.01 |
| MSFT | 668765 | exists | 34200.013994 | 57599.907797 | 37005 | 9795 | 0.01 |

## Benchmark checks
| check | status | observed | expected |
|---|---|---|---|
| AMZN_zip_found | PASS | `data/raw/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip` | `exists` |
| AMZN_orderbook_member_exists | PASS | `AMZN_2012-06-21_34200000_57600000_orderbook_10.csv` | `orderbook csv exists` |
| AMZN_expected_message_row_count | PASS | `269748` | `269748` |
| AMZN_timestamp_start | PASS | `34200.017459617` | `[34200.0, 34201.0)` |
| AMZN_timestamp_end | PASS | `57599.95935965` | `(57599.0, 57600.0)` |
| AMZN_event_bins | PASS | `29801` | `29801` |
| AMZN_orderbook_has_40_columns | PASS | `40` | `40` |
| AMZN_sample_spread_positive | PASS | `0.020000000000010232` | `> 0` |
| AMZN_sample_best_sizes_positive | PASS | `True` | `True` |
| AAPL_zip_found | PASS | `data/raw/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip` | `exists` |
| AAPL_orderbook_member_exists | PASS | `AAPL_2012-06-21_34200000_57600000_orderbook_10.csv` | `orderbook csv exists` |
| AAPL_expected_message_row_count | PASS | `400391` | `400391` |
| AAPL_timestamp_start | PASS | `34200.004241176` | `[34200.0, 34201.0)` |
| AAPL_timestamp_end | PASS | `57599.913117637` | `(57599.0, 57600.0)` |
| AAPL_event_bins | PASS | `34800` | `34800` |
| AAPL_orderbook_has_40_columns | PASS | `40` | `40` |
| AAPL_sample_spread_positive | PASS | `0.009999999999990905` | `> 0` |
| AAPL_sample_best_sizes_positive | PASS | `True` | `True` |
| GOOG_zip_found | PASS | `data/raw/LOBSTER_SampleFile_GOOG_2012-06-21_10.zip` | `exists` |
| GOOG_orderbook_member_exists | PASS | `GOOG_2012-06-21_34200000_57600000_orderbook_10.csv` | `orderbook csv exists` |
| GOOG_expected_message_row_count | PASS | `147916` | `147916` |
| GOOG_timestamp_start | PASS | `34200.015105074` | `[34200.0, 34201.0)` |
| GOOG_timestamp_end | PASS | `57599.871751084` | `(57599.0, 57600.0)` |
| GOOG_event_bins | PASS | `22678` | `22678` |
| GOOG_orderbook_has_40_columns | PASS | `40` | `40` |
| GOOG_sample_spread_positive | PASS | `0.009999999999990905` | `> 0` |
| GOOG_sample_best_sizes_positive | PASS | `True` | `True` |
| INTC_zip_found | PASS | `data/raw/LOBSTER_SampleFile_INTC_2012-06-21_10.zip` | `exists` |
| INTC_orderbook_member_exists | PASS | `INTC_2012-06-21_34200000_57600000_orderbook_10.csv` | `orderbook csv exists` |
| INTC_expected_message_row_count | PASS | `624040` | `624040` |
| INTC_timestamp_start | PASS | `34200.005742728` | `[34200.0, 34201.0)` |
| INTC_timestamp_end | PASS | `57599.948441634` | `(57599.0, 57600.0)` |
| INTC_event_bins | PASS | `36086` | `36086` |
| INTC_orderbook_has_40_columns | PASS | `40` | `40` |
| INTC_sample_spread_positive | PASS | `0.00999999999999801` | `> 0` |
| INTC_sample_best_sizes_positive | PASS | `True` | `True` |
| MSFT_zip_found | PASS | `data/raw/LOBSTER_SampleFile_MSFT_2012-06-21_10.zip` | `exists` |
| MSFT_orderbook_member_exists | PASS | `MSFT_2012-06-21_34200000_57600000_orderbook_10.csv` | `orderbook csv exists` |
| MSFT_expected_message_row_count | PASS | `668765` | `668765` |
| MSFT_timestamp_start | PASS | `34200.01399412` | `[34200.0, 34201.0)` |
| MSFT_timestamp_end | PASS | `57599.907796528` | `(57599.0, 57600.0)` |
| MSFT_event_bins | PASS | `37005` | `37005` |
| MSFT_orderbook_has_40_columns | PASS | `40` | `40` |
| MSFT_sample_spread_positive | PASS | `0.00999999999999801` | `> 0` |
| MSFT_sample_best_sizes_positive | PASS | `True` | `True` |
| any_asset_event_bins | PASS | `46290` | `46290` |

Gate passed: `.pipeline_state/step01_raw_audit.PASS` should exist.
Next module should be `02_feature_build_500ms.sh`, but do not run it until this report is reviewed.