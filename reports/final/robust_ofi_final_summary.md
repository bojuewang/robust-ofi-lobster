# Robust OFI Prediction: Final Pipeline Summary

## Experimental contract

- Data: five LOBSTER sample tickers, ten order-book levels.
- Time grid: 500ms bins over the regular trading window, giving `T = 46800` bins.
- Feature tensor: `X_std.shape = (46800, 5, 51)`.
- Split: chronological `70% / 15% / 15%` train/validation/test.
- Core comparison: ERM baseline vs feature-robust PGDA vs graph-robust PGDA.

## Main ablation table

| tau | step03_best_model | step03_best_graph | step03_clean_mse | step03_directional_accuracy | feature_deg_mse_improvement_0.1 | feature_robust_choice | graph_deg_mse_improvement_0.1 | graph_sensitivity_improvement | graph_robust_choice |
|---|---|---|---|---|---|---|---|---|---|
| 1.000000 | lasso_own | none | 2.929e-09 | 0.602310 | -2.39226e-10 | rho=0.05,R=3 | 2.09721e-12 | 1.53701e-11 | rho=0.1,R=1 |
| 2.000000 | graph_linear_ridge | return_abs | 5.93836e-09 | 0.595855 | -1.86677e-09 | rho=0.05,R=1 | 6.04494e-12 | 6.86322e-11 | rho=0.2,R=1 |
| 10.000000 | graph_linear_ridge | return_abs | 3.17499e-08 | 0.575196 | 6.94136e-08 | rho=0.2,R=3 | 1.14795e-10 | 9.38278e-10 | rho=0.2,R=3 |
| 20.000000 | graph_linear_ridge | return_abs | 6.60782e-08 | 0.551388 | 7.28205e-08 | rho=0.2,R=1 | 2.4567e-10 | 2.52479e-09 | rho=0.2,R=3 |

## Interpretation

1. The Step 03 ERM baselines establish a reproducible prediction baseline with finite MSE and directional accuracy above 50% across all tested horizons.
2. Calibrated Step 04b shows that feature perturbations are meaningful; feature-robust training helps mainly at longer horizons, but can sacrifice clean accuracy.
3. Step 05 gives the clearest robustness result: graph-robust training reduces graph-perturbation MSE degradation for every tested horizon and reduces graph sensitivity in all horizons.
4. The strongest report-level conclusion is therefore: in this five-asset one-day LOBSTER sample, robustness to cross-asset graph misspecification is more stable than feature-level adversarial robustness.

## Conclusion by horizon

| tau | feature_gain_positive | graph_gain_positive | clean_tradeoff_feature | clean_tradeoff_graph | verdict |
|---|---|---|---|---|---|
| 1.000000 | 0 | 1.000000 | 5.52935e-11 | 8.46923e-12 | graph robust improves; feature robust does not |
| 2.000000 | 0 | 1.000000 | 2.75761e-10 | -2.59739e-11 | graph robust improves; feature robust does not |
| 10.000000 | 1.000000 | 1.000000 | 9.59562e-08 | -4.22403e-10 | both feature and graph robustness improve degradation |
| 20.000000 | 1.000000 | 1.000000 | 3.60462e-07 | -6.01055e-10 | both feature and graph robustness improve degradation |

## Figures

- `reports/figures/step06_degradation_improvement.png`
- `reports/figures/step06_clean_mse_tradeoff.png`
- `reports/figures/step06_graph_sensitivity_improvement.png`

## Recommended final-report wording

> Robust optimization does not uniformly improve clean prediction error. Instead, its value appears in degradation control under structured perturbations. Feature-level robust training gives mixed results and is most useful at longer horizons, while graph-level robust training consistently reduces sensitivity and MSE degradation under cross-asset graph perturbations. This supports the proposal's thesis that robust optimization is useful as a stability mechanism rather than simply as a nominal accuracy booster.

## Limitations

- The experiment uses one trading day of public LOBSTER samples, so regime-shift conclusions are preliminary.
- The graph is fixed from the training period; rolling graph experiments should be added if more time or data are available.
- The PnL score is only a sanity check, not a production backtest.
- Feature-robust PGDA is sensitive to perturbation calibration; normalized adversarial evaluation is needed to avoid false zero-degradation conclusions.
