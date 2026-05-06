# Step 02 500ms Feature Build Benchmark

Overall status: **PASS**

## Output contract
- Tickers: `AMZN, AAPL, GOOG, INTC, MSFT`
- Time bins: `46800`
- Assets: `5`
- LOB levels: `10`
- Feature dimension: `51`
- Feature tensor: `X_raw.shape = (46800, 5, 51)`
- Standardized tensor: `X_std.shape = (46800, 5, 51)`
- Output NPZ: `data/processed/features_500ms_step02.npz`

## Ticker feature summaries
| ticker | event bins | carry-forward bins | min spread | median spread | nonzero OFI entries | nonzero message-flow bins | invalid deep-level cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMZN | 29801 | 16999 | 0.01 | 0.12 | 166061 | 29801 | 0 |
| AAPL | 34800 | 12000 | 0.01 | 0.15 | 231480 | 34800 | 0 |
| GOOG | 22678 | 24122 | 0.01 | 0.25 | 134038 | 22678 | 0 |
| INTC | 36086 | 10714 | 0.01 | 0.01 | 108376 | 36086 | 0 |
| MSFT | 37005 | 9795 | 0.01 | 0.01 | 125739 | 37005 | 0 |

## Standardization audit
- Nonconstant standardized train features: `255`
- Max absolute train mean after standardization: `1.26942e-05`
- Max absolute train std error after standardization: `5.55388e-08`
- Near-constant train feature count: `0`

## Target audit
| tau | valid samples per asset | train | validation | test | max abs log-return | max abs spread-normalized return |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46799 | 32759 | 7019 | 7019 | 0.00254452 | 14.5019 |
| 2 | 46798 | 32758 | 7018 | 7018 | 0.00254637 | 14.5019 |
| 10 | 46790 | 32750 | 7010 | 7010 | 0.00256222 | 19.0002 |
| 20 | 46780 | 32740 | 7000 | 7000 | 0.0032253 | 24.501 |

## Benchmark checks
| check | status | observed | expected |
|---|---|---|---|
| step01_gate_exists | PASS | `.pipeline_state/step01_raw_audit.PASS` | `exists` |
| time_bins | PASS | `46800` | `46800` |
| chronological_split | PASS | `(32760, 7020, 7020)` | `(32760, 7020, 7020)` |
| feature_dimension_contract | PASS | `51` | `51` |
| AMZN_X_shape | PASS | `(46800, 51)` | `(46800, 51)` |
| AMZN_event_bins | PASS | `29801` | `29801` |
| AMZN_spread_positive | PASS | `0.0099999998` | `> 0` |
| AMZN_mid_positive | PASS | `220.55` | `> 0` |
| AMZN_X_finite | PASS | `True` | `True` |
| AAPL_X_shape | PASS | `(46800, 51)` | `(46800, 51)` |
| AAPL_event_bins | PASS | `34800` | `34800` |
| AAPL_spread_positive | PASS | `0.0099999998` | `> 0` |
| AAPL_mid_positive | PASS | `577.47998` | `> 0` |
| AAPL_X_finite | PASS | `True` | `True` |
| GOOG_X_shape | PASS | `(46800, 51)` | `(46800, 51)` |
| GOOG_event_bins | PASS | `22678` | `22678` |
| GOOG_spread_positive | PASS | `0.0099999998` | `> 0` |
| GOOG_mid_positive | PASS | `563.76501` | `> 0` |
| GOOG_X_finite | PASS | `True` | `True` |
| INTC_X_shape | PASS | `(46800, 51)` | `(46800, 51)` |
| INTC_event_bins | PASS | `36086` | `36086` |
| INTC_spread_positive | PASS | `0.0099999998` | `> 0` |
| INTC_mid_positive | PASS | `26.615` | `> 0` |
| INTC_X_finite | PASS | `True` | `True` |
| MSFT_X_shape | PASS | `(46800, 51)` | `(46800, 51)` |
| MSFT_event_bins | PASS | `37005` | `37005` |
| MSFT_spread_positive | PASS | `0.0099999998` | `> 0` |
| MSFT_mid_positive | PASS | `30.065001` | `> 0` |
| MSFT_X_finite | PASS | `True` | `True` |
| X_raw_shape | PASS | `(46800, 5, 51)` | `(46800, 5, 51)` |
| any_asset_event_bins | PASS | `46290` | `46290` |
| X_raw_all_finite | PASS | `True` | `True` |
| X_std_shape | PASS | `(46800, 5, 51)` | `(46800, 5, 51)` |
| X_std_all_finite | PASS | `True` | `True` |
| standardized_train_mean_close_to_zero | PASS | `1.2694195e-05` | `< 1e-4` |
| standardized_train_std_close_to_one | PASS | `5.5538826e-08` | `< 1e-4` |
| target_tau_1_logret_finite | PASS | `True` | `True` |
| target_tau_1_spread_return_finite | PASS | `True` | `True` |
| target_tau_1_tail_nan | PASS | `tail NaN` | `tail NaN` |
| target_tau_2_logret_finite | PASS | `True` | `True` |
| target_tau_2_spread_return_finite | PASS | `True` | `True` |
| target_tau_2_tail_nan | PASS | `tail NaN` | `tail NaN` |
| target_tau_10_logret_finite | PASS | `True` | `True` |
| target_tau_10_spread_return_finite | PASS | `True` | `True` |
| target_tau_10_tail_nan | PASS | `tail NaN` | `tail NaN` |
| target_tau_20_logret_finite | PASS | `True` | `True` |
| target_tau_20_spread_return_finite | PASS | `True` | `True` |
| target_tau_20_tail_nan | PASS | `tail NaN` | `tail NaN` |
| output_npz_exists | PASS | `data/processed/features_500ms_step02.npz` | `nonempty file` |
| metadata_json_exists | PASS | `data/processed/features_500ms_step02_metadata.json` | `nonempty file` |
| ticker_summary_csv_exists | PASS | `data/processed/features_500ms_step02_ticker_summary.csv` | `nonempty file` |

Gate passed: `.pipeline_state/step02_feature_build.PASS` should exist.
Next module should be `03_graph_build_and_baseline.sh`, but do not run it until this report is reviewed.
