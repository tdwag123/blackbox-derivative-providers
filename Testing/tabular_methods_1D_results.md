# 1D tabular method comparison

This compares the derivative-provider methods inside the same 1D nonlinear FEM solve.

Each method supplies `q`, `dq/ds`, and `dq/dT` to the Newton residual/tangent assembly.

Training data are reduced by chopping each dataset into a 17x17 `(s,T)` grid and keeping at most 1 row per occupied cell.

Only Analytic evaluates the exact constitutive law. FiniteDiff now finite-differences a noisy tabular interpolant built from the same chopped training data.

Accuracy is measured using held-out noisy CSV rows for `q_obs` residuals and random clean points for derivative errors against the known clean derivative functions.

The main flux accuracy column is `test_obs_q_RMSE/noise`: RMSE to held-out noisy observations divided by the pointwise noise scale `sigma_eff = sigma * max(1, abs(q_true))`. Values near 1 mean the residual is about the noise floor.

Derivative accuracy columns compare to clean true derivatives. They use `sigma_eff / (sqrt(2) h)` with the chopped-grid spacing in the relevant direction.

Physics checks are evaluated on the same clean random points. Entropy violations count `q*s > tol`, which indicates anti-diffusion. Derivative violations count `dq/ds > tol`, equivalently negative tangent conductivity. Violation columns are percentages of evaluation points.

FEM cost columns include method build time, Newton steps, total solve wall-clock time, flux-provider call count, total flux-provider evaluation time, non-flux solve time, and average flux-provider call time.

Methods tested: CubicSpline, PCHIP, Smooth+PCHIP, RBF, SavGol, KISS-GP, FiniteDiff, and Analytic.





## nonlinear_high_noise

Dataset: `Data/NoisyDeterministicOracles/datasets/nonlinear_high_noise.csv`

Parameters: `k0=1.0`, `alpha=0.5`, `beta=0.2`, `sigma=0.1`

Training rows after grid chop: `289` of `3000` (`17x17`, max `1` per cell)

Noisy held-out test rows: `500`; clean random evaluation points: `500`

Provider build time, including GP training: `1.94 s`

FEM plot: `Data/Images/tabular_methods_fem_solutions_nonlinear_high_noise.png`

### Accuracy metrics

| method | test_obs_q_RMSE/noise | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE/noise |
| --- | --- | --- | --- |
| Analytic | 1.0517e+00 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 1.1722e+00 | 6.3323e-01 | 4.1501e-01 |
| KISS-GP | 1.3139e+00 | 9.4778e-01 | 3.8954e-01 |
| RFF | 1.3160e+00 | 1.9178e+00 | 1.7921e+00 |
| FiniteDiff | 1.7978e+00 | 1.0114e+00 | 9.6966e-01 |
| CubicSpline | 2.1030e+00 | 2.6282e+00 | 3.0097e+00 |
| RBF | 1.0347e+01 | 4.4808e+00 | 3.6209e+00 |

### Physics checks

| method | entropy_violation_% | worst_entropy_violation | deriv_violation_% | worst_deriv_violation |
| --- | --- | --- | --- | --- |
| Analytic | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| KISS-GP | 4.0000e-01 | 1.1288e-04 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 1.0000e+00 | 3.1677e-03 | 0.0000e+00 | 0.0000e+00 |
| RFF | 1.6000e+00 | 4.2558e-03 | 1.4000e+00 | 7.9353e+00 |
| FiniteDiff | 2.0000e+00 | 2.7382e-02 | 0.0000e+00 | 0.0000e+00 |
| CubicSpline | 2.6000e+00 | 5.6064e-02 | 5.8000e+00 | 7.7323e+00 |
| RBF | 2.6800e+01 | 2.2580e+00 | 2.9000e+01 | 8.2432e+00 |

### FEM results

| method | status | build_s | newton_steps | flux_calls | final_residual | rel_solution_err | solve_total_s | flux_eval_s | nonflux_s | avg_flux_eval_us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FiniteDiff | not converged | 8.8187e-04 | 12 | 5200 | 1.2294e-01 | 4.4072e-03 | 4.9028e-01 | 4.4761e-01 | 4.2672e-02 | 8.6079e+01 |
| CubicSpline | not converged | 1.6049e-03 | 35 | 43680 | 6.1506e-02 | 8.3846e-03 | 6.3953e-01 | 3.5489e-01 | 2.8464e-01 | 8.1249e+00 |
| Analytic | ok | 0.0000e+00 | 5 | 720 | 2.9847e-14 | 0.0000e+00 | 5.4387e-03 | 6.5275e-04 | 4.7860e-03 | 9.0659e-01 |
| RFF | ok | 2.1302e-02 | 6 | 880 | 1.6988e-13 | 8.1088e-03 | 9.0075e-02 | 8.1778e-02 | 8.2970e-03 | 9.2930e+01 |
| SavGol | ok | 1.4375e-05 | 8 | 1200 | 3.2543e-10 | 8.5329e-03 | 1.6668e-01 | 1.5773e-01 | 8.9435e-03 | 1.3145e+02 |
| KISS-GP | ok | 1.8517e+00 | 5 | 720 | 4.5737e-14 | 2.3940e-02 | 1.1082e+00 | 1.1007e+00 | 7.4700e-03 | 1.5288e+03 |
| RBF | ok | 1.6983e-02 | 6 | 880 | 1.7040e-13 | 4.7911e-02 | 2.3007e-02 | 1.6917e-02 | 6.0897e-03 | 1.9224e+01 |

### Quick read

- Best converged method by solution error: `Analytic` with relative solution error `0.000e+00`.
- Non-converged or failed methods: `FiniteDiff`, `CubicSpline`.








## nonlinear_low_noise

Dataset: `Data/NoisyDeterministicOracles/datasets/nonlinear_low_noise.csv`

Parameters: `k0=0.7`, `alpha=0.2`, `beta=0.05`, `sigma=0.02`

Training rows after grid chop: `289` of `3000` (`17x17`, max `1` per cell)

Noisy held-out test rows: `500`; clean random evaluation points: `500`

Provider build time, including GP training: `0.38 s`

FEM plot: `Data/Images/tabular_methods_fem_solutions_nonlinear_low_noise.png`

### Accuracy metrics

| method | test_obs_q_RMSE/noise | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE/noise |
| --- | --- | --- | --- |
| Analytic | 1.0076e+00 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 1.0658e+00 | 5.5584e-01 | 4.0475e-01 |
| RFF | 1.1158e+00 | 9.7727e-01 | 1.4120e+00 |
| KISS-GP | 2.0877e+00 | 3.5262e+00 | 1.6849e+00 |
| FiniteDiff | 4.8741e+00 | 3.3549e+00 | 2.8961e+00 |
| CubicSpline | 5.8611e+00 | 9.6454e+00 | 9.3180e+00 |
| RBF | 2.8410e+01 | 1.8115e+01 | 1.6858e+01 |

### Physics checks

| method | entropy_violation_% | worst_entropy_violation | deriv_violation_% | worst_deriv_violation |
| --- | --- | --- | --- | --- |
| Analytic | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| KISS-GP | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| RFF | 2.0000e-01 | 5.4179e-05 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 2.0000e-01 | 3.7291e-05 | 0.0000e+00 | 0.0000e+00 |
| FiniteDiff | 2.0000e+00 | 3.3287e-02 | 0.0000e+00 | 0.0000e+00 |
| CubicSpline | 2.0000e+00 | 4.5530e-02 | 3.4000e+00 | 2.9054e+00 |
| RBF | 2.0600e+01 | 6.3135e-01 | 2.4200e+01 | 2.3327e+00 |

### FEM results

| method | status | build_s | newton_steps | flux_calls | final_residual | rel_solution_err | solve_total_s | flux_eval_s | nonflux_s | avg_flux_eval_us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RBF | not converged | 2.5265e-03 | 14 | 20800 | 5.4163e-02 | 5.7675e-02 | 5.3518e-01 | 3.9587e-01 | 1.3931e-01 | 1.9032e+01 |
| Analytic | ok | 0.0000e+00 | 5 | 720 | 2.1713e-14 | 0.0000e+00 | 5.2838e-03 | 6.4900e-04 | 4.6348e-03 | 9.0139e-01 |
| SavGol | ok | 8.0000e-06 | 6 | 880 | 2.6836e-10 | 2.8787e-03 | 1.2193e-01 | 1.1538e-01 | 6.5485e-03 | 1.3112e+02 |
| RFF | ok | 7.0210e-03 | 5 | 720 | 1.7543e-14 | 3.6017e-03 | 7.3826e-02 | 6.7095e-02 | 6.7316e-03 | 9.3187e+01 |
| FiniteDiff | ok | 1.1292e-04 | 23 | 3760 | 5.5978e-09 | 8.4635e-03 | 3.2614e-01 | 2.9880e-01 | 2.7342e-02 | 7.9468e+01 |
| KISS-GP | ok | 3.2044e-01 | 5 | 720 | 2.0597e-14 | 1.2076e-02 | 1.0880e+00 | 1.0808e+00 | 7.1279e-03 | 1.5012e+03 |
| CubicSpline | ok | 5.5167e-05 | 8 | 1360 | 1.1290e-14 | 1.3344e-02 | 2.0111e-02 | 1.1137e-02 | 8.9735e-03 | 8.1891e+00 |

### Quick read

- Best converged method by solution error: `Analytic` with relative solution error `0.000e+00`.
- Non-converged or failed methods: `RBF`.










## linear_medium_noise

Dataset: `Data/NoisyDeterministicOracles/datasets/linear_medium_noise.csv`

Parameters: `k0=1.5`, `alpha=0.0`, `beta=0.0`, `sigma=0.05`

Training rows after grid chop: `289` of `3000` (`17x17`, max `1` per cell)

Noisy held-out test rows: `500`; clean random evaluation points: `500`

Provider build time, including GP training: `0.39 s`

FEM plot: `Data/Images/tabular_methods_fem_solutions_linear_medium_noise.png`

### Accuracy metrics

| method | test_obs_q_RMSE/noise | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE/noise |
| --- | --- | --- | --- |
| Analytic | 9.1274e-01 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 1.0083e+00 | 3.8454e-01 | 4.6361e-01 |
| RFF | 1.0369e+00 | 1.1230e+00 | 1.1115e+00 |
| KISS-GP | 1.2366e+00 | 9.7475e-01 | 3.7953e-01 |
| FiniteDiff | 2.2487e+00 | 1.2398e+00 | 1.1093e+00 |
| CubicSpline | 2.7418e+00 | 3.4771e+00 | 3.6197e+00 |
| RBF | 7.3389e+00 | 5.1996e+00 | 4.3862e+00 |

### Physics checks

| method | entropy_violation_% | worst_entropy_violation | deriv_violation_% | worst_deriv_violation |
| --- | --- | --- | --- | --- |
| Analytic | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| KISS-GP | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| FiniteDiff | 2.0000e-01 | 1.2356e-04 | 0.0000e+00 | 0.0000e+00 |
| SavGol | 4.0000e-01 | 5.8813e-04 | 0.0000e+00 | 0.0000e+00 |
| RFF | 4.0000e-01 | 1.7612e-04 | 4.0000e-01 | 3.9382e-01 |
| CubicSpline | 6.0000e-01 | 1.3914e-03 | 4.8000e+00 | 1.2659e+00 |
| RBF | 9.2000e+00 | 8.9569e-02 | 1.8000e+01 | 2.1265e+00 |

### FEM results

| method | status | build_s | newton_steps | flux_calls | final_residual | rel_solution_err | solve_total_s | flux_eval_s | nonflux_s | avg_flux_eval_us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SavGol | not converged | 1.1625e-05 | 24 | 36560 | 7.9928e-03 | 6.8907e-03 | 5.0860e+00 | 4.8138e+00 | 2.7222e-01 | 1.3167e+02 |
| Analytic | ok | 0.0000e+00 | 2 | 240 | 2.3728e-14 | 0.0000e+00 | 1.7824e-03 | 2.2937e-04 | 1.5530e-03 | 9.5571e-01 |
| RFF | ok | 8.6038e-03 | 5 | 720 | 1.9309e-14 | 3.5788e-03 | 7.1223e-02 | 6.4626e-02 | 6.5966e-03 | 8.9758e+01 |
| KISS-GP | ok | 3.3072e-01 | 4 | 560 | 6.5799e-11 | 5.9026e-03 | 8.5465e-01 | 8.4908e-01 | 5.5658e-03 | 1.5162e+03 |
| FiniteDiff | ok | 1.3483e-04 | 25 | 4000 | 6.1157e-09 | 6.5669e-03 | 3.4500e-01 | 3.1579e-01 | 2.9209e-02 | 7.8948e+01 |
| CubicSpline | ok | 6.6584e-05 | 7 | 1280 | 8.0712e-13 | 7.1820e-03 | 1.8978e-02 | 1.0542e-02 | 8.4361e-03 | 8.2356e+00 |
| RBF | ok | 1.9036e-03 | 4 | 560 | 4.8413e-09 | 3.3799e-02 | 1.4554e-02 | 1.0732e-02 | 3.8220e-03 | 1.9165e+01 |

### Quick read

- Best converged method by solution error: `Analytic` with relative solution error `0.000e+00`.
- Non-converged or failed methods: `SavGol`.

