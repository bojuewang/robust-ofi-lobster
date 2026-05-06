# Step 06 Feature/Graph Robustness Summary Benchmark

Overall status: **PASS**

## Final conclusion
Graph-robust training gives the most consistent robustness improvement in the current experiment: MSE degradation under graph perturbation decreases for all tested horizons, and graph sensitivity decreases for all tested horizons.

## Final ablation table
| tau | feature_deg_mse_improvement_0.1 | feature_robust_choice | graph_deg_mse_improvement_0.1 | graph_sensitivity_improvement | graph_robust_choice |
|---|---|---|---|---|---|
| 1.000000 | -2.39226e-10 | rho=0.05,R=3 | 2.09721e-12 | 1.53701e-11 | rho=0.1,R=1 |
| 2.000000 | -1.86677e-09 | rho=0.05,R=1 | 6.04494e-12 | 6.86322e-11 | rho=0.2,R=1 |
| 10.000000 | 6.94136e-08 | rho=0.2,R=3 | 1.14795e-10 | 9.38278e-10 | rho=0.2,R=3 |
| 20.000000 | 7.28205e-08 | rho=0.2,R=1 | 2.4567e-10 | 2.52479e-09 | rho=0.2,R=3 |

## Generated artifacts
- `reports/final/robust_ofi_final_summary.md`
- `reports/final/final_ablation_table.csv`
- `reports/figures/step06_degradation_improvement.png`
- `reports/figures/step06_clean_mse_tradeoff.png`
- `reports/figures/step06_graph_sensitivity_improvement.png`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `scripts/run_pipeline_to_step06.sh`

## Benchmark checks
| check | status | observed | expected |
|---|---|---|---|
| step03_status_pass | PASS | `PASS` | `PASS` |
| step04b_status_pass | PASS | `PASS` | `PASS` |
| step05_status_pass | PASS | `PASS` | `PASS` |
| all_tau_in_step03 | PASS | `[1, 2, 10, 20]` | `[1, 2, 10, 20]` |
| all_tau_in_step04b | PASS | `[1, 2, 10, 20]` | `[1, 2, 10, 20]` |
| all_tau_in_step05 | PASS | `[1, 2, 10, 20]` | `[1, 2, 10, 20]` |
| final_rows_count | PASS | `4` | `4` |
| graph_improvement_all_tau | PASS | `['2.09721e-12', '6.04494e-12', '1.14795e-10', '2.4567e-10']` | `positive for all tau` |
| feature_improvement_some_tau | PASS | `['-2.39226e-10', '-1.86677e-09', '6.94136e-08', '7.28205e-08']` | `positive for at least one tau` |
| plots_created | PASS | `plots created` | `matplotlib plots saved` |

Gate passed: `.pipeline_state/step06_summary.PASS` should exist.
Next recommended module: `07_github_packaging.sh` if you want an automatic commit-ready repository package.