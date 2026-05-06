# Step 03 Graph Build + ERM Baseline Benchmark

Overall status: **PASS**

## Input contract
- Feature NPZ: `data/processed/features_500ms_step02.npz`
- NPZ standardized key: `X_std`
- Raw feature tensor: `(46800, 5, 51)`
- Standardized feature tensor: `(46800, 5, 51)`
- Event mask: `(46800, 5)`
- Tickers: `AMZN, AAPL, GOOG, INTC, MSFT`
- Time bins/assets/features: `46800 / 5 / 51`

## Graph summaries
| graph | min | max | mean_abs | nonzero_edges |
|---|---|---|---|---|
| return_signed | 0 | 0.171938 | 0.091919 | 20 |
| return_abs | 0 | 0.171938 | 0.091919 | 20 |
| return_signed_mask | 0 | 0.101536 | 0.045101 | 20 |
| return_abs_mask | 0 | 0.101536 | 0.045101 | 20 |
| ofi_signed | -0.007447 | 0.028413 | 0.008474 | 20 |
| ofi_abs | 0 | 0.028413 | 0.008474 | 20 |
| ofi_signed_mask | -0.004397 | 0.010851 | 0.003736 | 20 |
| ofi_abs_mask | 0 | 0.010851 | 0.003736 | 20 |
| cross_ofi_signed | -0.00099409 | 0.013188 | 0.004478 | 20 |
| cross_ofi_abs | 0 | 0.013188 | 0.004478 | 20 |
| cross_ofi_signed_mask | -0.000480598 | 0.007110 | 0.002075 | 20 |
| cross_ofi_abs_mask | 0 | 0.007110 | 0.002075 | 20 |

## Best validation-selected model by horizon
| tau | model | graph | alpha | val_mse | mse | mae | directional_accuracy | sharpe_proxy_no_tc |
|---|---|---|---|---|---|---|---|---|
| 1 | lasso_own | none | 1e-06 | 1.50086e-09 | 2.929e-09 | 1.72549e-05 | 0.602310 | 0.225876 |
| 2 | graph_linear_ridge | return_abs | 0.001000 | 3.06471e-09 | 5.93836e-09 | 3.30375e-05 | 0.595855 | 0.269888 |
| 10 | graph_linear_ridge | return_abs | 10.000000 | 1.65465e-08 | 3.17499e-08 | 0.000107269 | 0.575196 | 0.347969 |
| 20 | graph_linear_ridge | return_abs | 10.000000 | 3.51101e-08 | 6.60782e-08 | 0.000172053 | 0.551388 | 0.313968 |

## All baseline results
| tau | model | graph | alpha | val_mse | mse | mae | directional_accuracy | sharpe_proxy_no_tc |
|---|---|---|---|---|---|---|---|---|
| 1 | ridge_own | none | 1.000000 | 1.50397e-09 | 2.93694e-09 | 1.84836e-05 | 0.599657 | 0.220935 |
| 1 | lasso_own | none | 1e-06 | 1.50086e-09 | 2.929e-09 | 1.72549e-05 | 0.602310 | 0.225876 |
| 1 | graph_linear_ridge | return_abs | 1.000000 | 1.50116e-09 | 2.94591e-09 | 1.92747e-05 | 0.604963 | 0.226935 |
| 1 | graph_linear_ridge | ofi_abs | 1.000000 | 1.50181e-09 | 2.94337e-09 | 1.90763e-05 | 0.603559 | 0.226493 |
| 1 | graph_linear_ridge | cross_ofi_abs | 1.000000 | 1.50144e-09 | 2.94714e-09 | 1.91589e-05 | 0.603403 | 0.226857 |
| 1 | graph_linear_ridge | return_abs_mask | 1.000000 | 1.50117e-09 | 2.94706e-09 | 1.93069e-05 | 0.605588 | 0.225810 |
| 1 | graph_linear_ridge | ofi_abs_mask | 1.000000 | 1.50154e-09 | 2.94431e-09 | 1.91193e-05 | 0.600593 | 0.226054 |
| 1 | graph_linear_ridge | cross_ofi_abs_mask | 1.000000 | 1.50123e-09 | 2.94867e-09 | 1.92086e-05 | 0.605588 | 0.225996 |
| 2 | ridge_own | none | 0.001000 | 3.07282e-09 | 5.91929e-09 | 3.15971e-05 | 0.599817 | 0.269023 |
| 2 | lasso_own | none | 1e-06 | 3.07329e-09 | 5.90648e-09 | 3.05448e-05 | 0.599208 | 0.276604 |
| 2 | graph_linear_ridge | return_abs | 0.001000 | 3.06471e-09 | 5.93836e-09 | 3.30375e-05 | 0.595855 | 0.269888 |
| 2 | graph_linear_ridge | ofi_abs | 1.000000 | 3.06779e-09 | 5.90514e-09 | 3.23107e-05 | 0.598903 | 0.277564 |
| 2 | graph_linear_ridge | cross_ofi_abs | 0.001000 | 3.0674e-09 | 5.93823e-09 | 3.26367e-05 | 0.596972 | 0.267525 |
| 2 | graph_linear_ridge | return_abs_mask | 0.001000 | 3.06567e-09 | 5.94188e-09 | 3.31314e-05 | 0.597379 | 0.269116 |
| 2 | graph_linear_ridge | ofi_abs_mask | 1.000000 | 3.06759e-09 | 5.90474e-09 | 3.23903e-05 | 0.601138 | 0.277202 |
| 2 | graph_linear_ridge | cross_ofi_abs_mask | 0.001000 | 3.06734e-09 | 5.93874e-09 | 3.27144e-05 | 0.597582 | 0.266920 |
| 10 | ridge_own | none | 0.001000 | 1.66102e-08 | 3.17727e-08 | 0.000104436 | 0.581716 | 0.361571 |
| 10 | lasso_own | none | 1e-06 | 1.66106e-08 | 3.16253e-08 | 0.000103415 | 0.582843 | 0.381762 |
| 10 | graph_linear_ridge | return_abs | 10.000000 | 1.65465e-08 | 3.17499e-08 | 0.000107269 | 0.575196 | 0.347969 |
| 10 | graph_linear_ridge | ofi_abs | 1.000000 | 1.65968e-08 | 3.18022e-08 | 0.000106213 | 0.575294 | 0.350565 |
| 10 | graph_linear_ridge | cross_ofi_abs | 0.001000 | 1.65962e-08 | 3.198e-08 | 0.000106798 | 0.573775 | 0.335011 |
| 10 | graph_linear_ridge | return_abs_mask | 10.000000 | 1.65495e-08 | 3.17452e-08 | 0.000107407 | 0.576373 | 0.345095 |
| 10 | graph_linear_ridge | ofi_abs_mask | 1.000000 | 1.65898e-08 | 3.17755e-08 | 0.000106361 | 0.576667 | 0.349289 |
| 10 | graph_linear_ridge | cross_ofi_abs_mask | 0.001000 | 1.65908e-08 | 3.19594e-08 | 0.000106965 | 0.576275 | 0.336668 |
| 20 | ridge_own | none | 0.001000 | 3.51594e-08 | 6.54614e-08 | 0.000166812 | 0.567224 | 0.338705 |
| 20 | lasso_own | none | 1e-06 | 3.51591e-08 | 6.52114e-08 | 0.000165745 | 0.569673 | 0.354162 |
| 20 | graph_linear_ridge | return_abs | 10.000000 | 3.51101e-08 | 6.60782e-08 | 0.000172053 | 0.551388 | 0.313968 |
| 20 | graph_linear_ridge | ofi_abs | 10.000000 | 3.52569e-08 | 6.58093e-08 | 0.000170137 | 0.557714 | 0.326469 |
| 20 | graph_linear_ridge | cross_ofi_abs | 0.001000 | 3.51803e-08 | 6.63593e-08 | 0.000170919 | 0.549592 | 0.293845 |
| 20 | graph_linear_ridge | return_abs_mask | 10.000000 | 3.51156e-08 | 6.60903e-08 | 0.000172191 | 0.551143 | 0.309604 |
| 20 | graph_linear_ridge | ofi_abs_mask | 10.000000 | 3.52496e-08 | 6.58251e-08 | 0.000170381 | 0.559388 | 0.321074 |
| 20 | graph_linear_ridge | cross_ofi_abs_mask | 0.001000 | 3.51653e-08 | 6.64477e-08 | 0.000171309 | 0.549347 | 0.288173 |

## Benchmark checks
| check | status | observed | expected |
|---|---|---|---|
| step02_gate_exists | PASS | `.pipeline_state/step02_feature_build.PASS` | `exists` |
| xstd_key_found | PASS | `X_std` | `standardized feature tensor key` |
| X_std_shape | PASS | `(46800, 5, 51)` | `(46800, 5, 51)` |
| event_mask_shape | PASS | `(46800, 5)` | `(46800, 5)` |
| X_std_finite | PASS | `True` | `True` |
| event_mask_binary | PASS | `[0.0, 1.0]` | `{0,1}` |
| tau1_target_exists | PASS | `[1, 2, 10, 20]` | `tau=1 target available` |
| tau_1_target_exists | PASS | `[1, 2, 10, 20]` | `tau=1` |
| tau_1_target_shape | PASS | `(46800, 5)` | `(46800, 5)` |
| tau_2_target_exists | PASS | `[1, 2, 10, 20]` | `tau=2` |
| tau_2_target_shape | PASS | `(46800, 5)` | `(46800, 5)` |
| tau_10_target_exists | PASS | `[1, 2, 10, 20]` | `tau=10` |
| tau_10_target_shape | PASS | `(46800, 5)` | `(46800, 5)` |
| tau_20_target_exists | PASS | `[1, 2, 10, 20]` | `tau=20` |
| tau_20_target_shape | PASS | `(46800, 5)` | `(46800, 5)` |
| graph_return_signed_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_return_signed_finite | PASS | `True` | `True` |
| graph_return_signed_diag_zero | PASS | `0.0` | `0` |
| graph_return_abs_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_return_abs_finite | PASS | `True` | `True` |
| graph_return_abs_diag_zero | PASS | `0.0` | `0` |
| graph_return_signed_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_return_signed_mask_finite | PASS | `True` | `True` |
| graph_return_signed_mask_diag_zero | PASS | `0.0` | `0` |
| graph_return_abs_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_return_abs_mask_finite | PASS | `True` | `True` |
| graph_return_abs_mask_diag_zero | PASS | `0.0` | `0` |
| graph_ofi_signed_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_ofi_signed_finite | PASS | `True` | `True` |
| graph_ofi_signed_diag_zero | PASS | `0.0` | `0` |
| graph_ofi_abs_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_ofi_abs_finite | PASS | `True` | `True` |
| graph_ofi_abs_diag_zero | PASS | `0.0` | `0` |
| graph_ofi_signed_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_ofi_signed_mask_finite | PASS | `True` | `True` |
| graph_ofi_signed_mask_diag_zero | PASS | `0.0` | `0` |
| graph_ofi_abs_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_ofi_abs_mask_finite | PASS | `True` | `True` |
| graph_ofi_abs_mask_diag_zero | PASS | `0.0` | `0` |
| graph_cross_ofi_signed_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_cross_ofi_signed_finite | PASS | `True` | `True` |
| graph_cross_ofi_signed_diag_zero | PASS | `0.0` | `0` |
| graph_cross_ofi_abs_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_cross_ofi_abs_finite | PASS | `True` | `True` |
| graph_cross_ofi_abs_diag_zero | PASS | `0.0` | `0` |
| graph_cross_ofi_signed_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_cross_ofi_signed_mask_finite | PASS | `True` | `True` |
| graph_cross_ofi_signed_mask_diag_zero | PASS | `0.0` | `0` |
| graph_cross_ofi_abs_mask_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_cross_ofi_abs_mask_finite | PASS | `True` | `True` |
| graph_cross_ofi_abs_mask_diag_zero | PASS | `0.0` | `0` |
| baseline_results_nonempty | PASS | `32` | `> 0` |
| baseline_mse_finite | PASS | `count=32` | `all finite` |
| baseline_da_range | PASS | `min=0.5493, max=0.6056` | `[0,1]` |

Gate passed: `.pipeline_state/step03_graph_baseline.PASS` should exist.
Next module should be `04_feature_robust_pgda.sh`, but do not run it until this report is reviewed.