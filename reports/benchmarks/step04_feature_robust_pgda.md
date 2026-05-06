# Step 04 Feature-Robust PGDA Benchmark

Overall status: **PASS**

## Input contract
- Feature tensor: `(46800, 5, 51)` from key `X_std`
- Graph file: `data/processed/graphs_step03.npz`
- Default graph: `return_abs`
- Training grid: `rho_X=[0.0, 0.05, 0.1, 0.2]`, `R_adv=[1, 3]`, epochs `8`, batch size `512`

## Best validation-selected model by horizon
| tau | model | rho_x_train | r_adv | val_mse | clean_mse | clean_da | sensitivity_X | eval_rho_0.10_deg_mse | eval_rho_0.10_deg_da |
|---|---|---|---|---|---|---|---|---|---|
| 1 | graph_linear_sgd_erm | 0 | 1 | 1.01746e-07 | 1.74783e-07 | 0.557827 | 7.95262e-10 | 0 | 0 |
| 2 | graph_linear_feature_pgda | 0.050000 | 3 | 1.74062e-07 | 2.97238e-07 | 0.561008 | 2.29105e-09 | 0 | 0 |
| 10 | graph_linear_feature_pgda | 0.100000 | 3 | 1.04088e-06 | 1.7464e-06 | 0.550098 | 2.35945e-08 | 0 | 0 |
| 20 | graph_linear_sgd_erm | 0 | 1 | 1.81213e-06 | 2.96527e-06 | 0.543347 | 5.31721e-08 | 0 | 0 |

## ERM vs best robust degradation comparison
| tau | erm_clean_mse | robust_clean_mse | erm_deg_mse_rho_0.10 | robust_deg_mse_rho_0.10 | deg_mse_improvement | erm_sensitivity_X | robust_sensitivity_X | sensitivity_improvement | robust_choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.74783e-07 | 1.79775e-07 | 0 | 0 | 0 | 7.95262e-10 | 8.06499e-10 | -1.12366e-11 | rho=0.05,R=1 |
| 2 | 4.35711e-07 | 4.57676e-07 | 0 | 0 | 0 | 2.77284e-09 | 2.83743e-09 | -6.45921e-11 | rho=0.05,R=1 |
| 10 | 2.58373e-06 | 2.05916e-06 | 0 | 0 | 0 | 2.87009e-08 | 2.56393e-08 | 3.06156e-09 | rho=0.05,R=1 |
| 20 | 2.96527e-06 | 3.66628e-06 | 0 | 0 | 0 | 5.31721e-08 | 5.90076e-08 | -5.83545e-09 | rho=0.05,R=1 |

## All feature-robust results
| tau | rho_x_train | r_adv | val_mse | clean_mse | clean_da | sensitivity_X | eval_rho_0.05_deg_mse | eval_rho_0.10_deg_mse |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 1.01746e-07 | 1.74783e-07 | 0.557827 | 7.95262e-10 | 0 | 0 |
| 1 | 0.050000 | 1 | 1.04643e-07 | 1.79775e-07 | 0.557827 | 8.06499e-10 | 0 | 0 |
| 1 | 0.050000 | 3 | 1.25923e-07 | 2.16442e-07 | 0.557827 | 8.85352e-10 | 0 | 0 |
| 1 | 0.100000 | 1 | 1.2262e-07 | 2.10749e-07 | 0.557827 | 8.72658e-10 | 0 | 0 |
| 1 | 0.100000 | 3 | 1.07259e-07 | 1.84282e-07 | 0.557827 | 8.17137e-10 | 0 | 0 |
| 1 | 0.200000 | 1 | 1.40561e-07 | 2.41672e-07 | 0.557827 | 9.34682e-10 | 0 | 0 |
| 1 | 0.200000 | 3 | 1.40351e-07 | 2.4131e-07 | 0.557827 | 9.33624e-10 | 0 | 0 |
| 2 | 0 | 1 | 2.54692e-07 | 4.35711e-07 | 0.561008 | 2.77284e-09 | 0 | 0 |
| 2 | 0.050000 | 1 | 2.67473e-07 | 4.57676e-07 | 0.561008 | 2.83743e-09 | 0 | 0 |
| 2 | 0.050000 | 3 | 1.74062e-07 | 2.97238e-07 | 0.561008 | 2.29105e-09 | 0 | 0 |
| 2 | 0.100000 | 1 | 1.76867e-07 | 3.02053e-07 | 0.561008 | 2.31214e-09 | 0 | 0 |
| 2 | 0.100000 | 3 | 2.66514e-07 | 4.56028e-07 | 0.561008 | 2.83668e-09 | 0 | 0 |
| 2 | 0.200000 | 1 | 2.20104e-07 | 3.76289e-07 | 0.561008 | 2.57809e-09 | 0 | 0 |
| 2 | 0.200000 | 3 | 2.05682e-07 | 3.5152e-07 | 0.561008 | 2.49291e-09 | 0 | 0 |
| 10 | 0 | 1 | 1.53676e-06 | 2.58373e-06 | 0.550098 | 2.87009e-08 | 0 | 0 |
| 10 | 0.050000 | 1 | 1.22619e-06 | 2.05916e-06 | 0.550098 | 2.56393e-08 | 0 | 0 |
| 10 | 0.050000 | 3 | 1.86226e-06 | 3.1339e-06 | 0.550098 | 3.16706e-08 | 0 | 0 |
| 10 | 0.100000 | 1 | 1.79303e-06 | 3.01686e-06 | 0.550098 | 3.10473e-08 | 0 | 0 |
| 10 | 0.100000 | 3 | 1.04088e-06 | 1.7464e-06 | 0.550098 | 2.35945e-08 | 0 | 0 |
| 10 | 0.200000 | 1 | 1.62609e-06 | 2.73468e-06 | 0.550098 | 2.95649e-08 | 0 | 0 |
| 10 | 0.200000 | 3 | 1.50693e-06 | 2.53332e-06 | 0.550098 | 2.84183e-08 | 0 | 0 |
| 20 | 0 | 1 | 1.81213e-06 | 2.96527e-06 | 0.543347 | 5.31721e-08 | 0 | 0 |
| 20 | 0.050000 | 1 | 2.23799e-06 | 3.66628e-06 | 0.543347 | 5.90076e-08 | 0 | 0 |
| 20 | 0.050000 | 3 | 2.12421e-06 | 3.47893e-06 | 0.543347 | 5.75857e-08 | 0 | 0 |
| 20 | 0.100000 | 1 | 2.71949e-06 | 4.4595e-06 | 0.543347 | 6.51994e-08 | 0 | 0 |
| 20 | 0.100000 | 3 | 2.22462e-06 | 3.64425e-06 | 0.543347 | 5.88972e-08 | 0 | 0 |
| 20 | 0.200000 | 1 | 1.83995e-06 | 3.01104e-06 | 0.543347 | 5.34384e-08 | 0 | 0 |
| 20 | 0.200000 | 3 | 1.86084e-06 | 3.04542e-06 | 0.543347 | 5.37848e-08 | 0 | 0 |

## Benchmark checks
| check | status | observed | expected |
|---|---|---|---|
| step03_gate_exists | PASS | `.pipeline_state/step03_graph_baseline.PASS` | `exists` |
| X_shape | PASS | `(46800, 5, 51)` | `(46800, 5, 51)` |
| graph_key_exists | PASS | `return_abs` | `in ['coactivity_train', 'phi_bar', 'return_signed', 'return_abs', 'return_signed_mask', 'return_abs_mask', 'ofi_signed', 'ofi_abs', 'ofi_signed_mask', 'ofi_abs_mask', 'cross_ofi_signed', 'cross_ofi_abs', 'cross_ofi_signed_mask', 'cross_ofi_abs_mask']` |
| graph_shape | PASS | `(5, 5)` | `(5, 5)` |
| graph_diag_zero | PASS | `0.0` | `0` |
| targets_available | PASS | `[1, 2, 10, 20]` | `[1, 2, 10, 20]` |
| results_nonempty | PASS | `28` | `> 0` |
| all_core_metrics_finite | PASS | `168/168 finite` | `all finite` |
| clean_da_range | PASS | `min=0.5433, max=0.5610` | `[0,1]` |
| degradation_nonnegative_mostly | PASS | `100.00%` | `>= 90%` |
| has_feature_robust_models | PASS | `rho_x > 0 present` | `True` |

Gate passed: `.pipeline_state/step04_feature_robust_pgda.PASS` should exist.
Next module should be `05_graph_robust_pgda.sh`, but do not run it until this report is reviewed.