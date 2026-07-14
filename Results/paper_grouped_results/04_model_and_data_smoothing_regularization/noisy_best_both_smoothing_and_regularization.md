# Best Rows With Both Data Smoothing And Internal Regularization

Source sweep: `Results/model_and_data_regularization_sweep_noisy`.

This report keeps only rows where both of these are true:

- internal/model regularization is on: `+ridge=...`
- input-grid data smoothing is on: `+reg=gradient:...` or `+reg=laplacian:...`

This is meant to compare against the separate reports where only one knob was used.

Selection rule: converged row with the fewest Newton steps; ties use lower relative solution error, then fewer flux calls.

## Overall Best Per Dataset

| dataset | method_type | method | lambda_alpha | smoothing_type | smoothing_strength | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% | build_s | solve_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linear_medium_noise` | `KISS-GP Ridge` | `kissgp+ridge=1e-1+reg=laplacian:0.1` | 1.0000e-01 | `laplacian` | 1.0000e-01 | `ok` | 3.0000e+00 | 2.3506e-03 | 5.6000e+02 | 2.9688e-12 | 2.1607e-01 | 1.7861e+00 | 1.6761e-01 | 3.0734e-01 | 1.2677e+00 | 2.7700e-01 | 5.7017e-01 | 4.0000e-01 | 0.0000e+00 | 8.3695e-01 | 3.6200e+00 |
| `nonlinear_low_noise` | `RFF Ridge` | `ridge_rff+ridge=1e-1+reg=gradient:0.3` | 1.0000e-01 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 2.1876e-03 | 5.6000e+02 | 5.1563e-09 | 2.2674e-01 | 3.7319e+00 | 2.3082e-01 | 4.1139e-01 | 4.4718e+00 | 2.9919e-01 | 1.6295e+00 | 6.0000e-01 | 0.0000e+00 | 8.5476e-02 | 1.2815e-01 |
| `nonlinear_high_noise` | `RFF Frequency Weighted` | `penalty_rff+ridge=1e-6+reg=gradient:0.3` | 1.0000e-06 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 7.0299e-03 | 5.6000e+02 | 6.8187e-09 | 9.7334e-01 | 1.4483e+00 | 6.8749e-01 | 1.2571e+00 | 1.0581e+00 | 1.3546e+00 | 5.6242e-01 | 1.4000e+00 | 0.0000e+00 | 2.7313e-01 | 9.0337e-02 |

## Best Per Dataset And Method Type

| dataset | method_type | method | lambda_alpha | smoothing_type | smoothing_strength | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% | build_s | solve_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linear_medium_noise` | `KISS-GP Ridge` | `kissgp+ridge=1e-1+reg=laplacian:0.1` | 1.0000e-01 | `laplacian` | 1.0000e-01 | `ok` | 3.0000e+00 | 2.3506e-03 | 5.6000e+02 | 2.9688e-12 | 2.1607e-01 | 1.7861e+00 | 1.6761e-01 | 3.0734e-01 | 1.2677e+00 | 2.7700e-01 | 5.7017e-01 | 4.0000e-01 | 0.0000e+00 | 8.3695e-01 | 3.6200e+00 |
| `linear_medium_noise` | `RFF Ridge` | `ridge_rff+ridge=1e-6+reg=laplacian:0.01` | 1.0000e-06 | `laplacian` | 1.0000e-02 | `ok` | 3.0000e+00 | 4.2817e-03 | 5.6000e+02 | 1.8136e-10 | 2.1012e-01 | 1.8052e+00 | 1.7754e-01 | 2.5731e-01 | 1.0613e+00 | 5.0816e-01 | 1.0460e+00 | 4.0000e-01 | 0.0000e+00 | 3.2523e-01 | 2.6068e-01 |
| `linear_medium_noise` | `RFF Frequency Weighted` | `penalty_rff+ridge=1e-1+reg=gradient:0.03` | 1.0000e-01 | `gradient` | 3.0000e-02 | `ok` | 3.0000e+00 | 4.4617e-03 | 5.6000e+02 | 2.4177e-14 | 1.9540e-01 | 1.5461e+00 | 1.4773e-01 | 1.6591e-01 | 6.8431e-01 | 1.6194e-01 | 3.3333e-01 | 4.0000e-01 | 0.0000e+00 | 1.2670e+00 | 1.6651e-01 |
| `linear_medium_noise` | `MLP Ridge` | `mlp+ridge=1e-2+reg=gradient:0.03` | 1.0000e-02 | `gradient` | 3.0000e-02 | `ok` | 3.0000e+00 | 7.9569e-03 | 5.6000e+02 | 5.7065e-11 | 2.4694e-01 | 2.4832e+00 | 2.0568e-01 | 3.2840e-01 | 1.3545e+00 | 2.4944e-01 | 5.1345e-01 | 1.2000e+00 | 0.0000e+00 | 8.3890e+00 | 5.5100e+00 |
| `linear_medium_noise` | `RBF Ridge` | `rbf+ridge=1e-1+reg=laplacian:0.01` | 1.0000e-01 | `laplacian` | 1.0000e-02 | `ok` | 3.0000e+00 | 1.8865e-02 | 5.6000e+02 | 5.7743e-09 | 7.3171e-01 | 9.1945e+00 | 7.1760e-01 | 1.5235e+00 | 6.2838e+00 | 2.0315e+00 | 4.1816e+00 | 1.4200e+01 | 2.5000e+01 | 1.3271e-01 | 7.7816e-02 |
| `linear_medium_noise` | `Matern GP Monotone Internal Ridge` | `materngpmonotone_unregularized+ridge=1e-1+reg=gradient:0.3` | 1.0000e-01 | `gradient` | 3.0000e-01 | `ok` | 4.0000e+00 | 6.2370e-03 | 7.2000e+02 | 5.9868e-13 | 2.2195e-01 | 1.8631e+00 | 1.8066e-01 | 2.9346e-01 | 1.2104e+00 | 5.0781e-01 | 1.0453e+00 | 0.0000e+00 | 0.0000e+00 | 2.9903e-01 | 1.8588e-01 |
| `nonlinear_high_noise` | `RFF Frequency Weighted` | `penalty_rff+ridge=1e-6+reg=gradient:0.3` | 1.0000e-06 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 7.0299e-03 | 5.6000e+02 | 6.8187e-09 | 9.7334e-01 | 1.4483e+00 | 6.8749e-01 | 1.2571e+00 | 1.0581e+00 | 1.3546e+00 | 5.6242e-01 | 1.4000e+00 | 0.0000e+00 | 2.7313e-01 | 9.0337e-02 |
| `nonlinear_high_noise` | `KISS-GP Ridge` | `kissgp+ridge=1e-6+reg=gradient:0.3` | 1.0000e-06 | `gradient` | 3.0000e-01 | `ok` | 4.0000e+00 | 2.1770e-03 | 7.2000e+02 | 2.8927e-14 | 1.0373e+00 | 1.3924e+00 | 7.9675e-01 | 1.4072e+00 | 1.1844e+00 | 1.0254e+00 | 4.2572e-01 | 4.0000e-01 | 0.0000e+00 | 3.4837e-01 | 2.2142e+00 |
| `nonlinear_high_noise` | `Matern GP Monotone Internal Ridge` | `materngpmonotone_unregularized+ridge=1e-6+reg=laplacian:0.1` | 1.0000e-06 | `laplacian` | 1.0000e-01 | `ok` | 4.0000e+00 | 5.4374e-03 | 7.2000e+02 | 7.0680e-09 | 9.5531e-01 | 1.6416e+00 | 7.5197e-01 | 1.5086e+00 | 1.2698e+00 | 3.6881e+00 | 1.5312e+00 | 2.0000e+00 | 2.0000e-01 | 3.1720e-01 | 1.7942e-01 |
| `nonlinear_high_noise` | `RFF Ridge` | `ridge_rff+ridge=1e-4+reg=gradient:0.3` | 1.0000e-04 | `gradient` | 3.0000e-01 | `ok` | 4.0000e+00 | 6.3639e-03 | 7.2000e+02 | 3.9309e-14 | 9.7717e-01 | 1.4272e+00 | 6.8722e-01 | 1.2308e+00 | 1.0359e+00 | 1.2014e+00 | 4.9880e-01 | 1.2000e+00 | 0.0000e+00 | 1.0317e-01 | 1.6023e-01 |
| `nonlinear_high_noise` | `MLP Ridge` | `mlp+ridge=1e-6+reg=gradient:0.03` | 1.0000e-06 | `gradient` | 3.0000e-02 | `ok` | 4.0000e+00 | 7.7302e-03 | 7.2000e+02 | 1.0802e-13 | 9.2038e-01 | 1.5765e+00 | 5.9979e-01 | 1.4047e+00 | 1.1823e+00 | 1.4274e+00 | 5.9265e-01 | 2.8000e+00 | 6.0000e-01 | 4.1016e+00 | 7.0045e+00 |
| `nonlinear_high_noise` | `RBF Ridge` | `rbf+ridge=1e-2+reg=gradient:0.03` | 1.0000e-02 | `gradient` | 3.0000e-02 | `ok` | 7.0000e+00 | 9.4058e-02 | 1.3600e+03 | 1.0193e-09 | 6.6814e+01 | 3.0981e+02 | 6.3975e+01 | 1.0814e+02 | 9.1014e+01 | 2.1169e+02 | 8.7891e+01 | 5.1200e+01 | 5.1400e+01 | 7.0757e-02 | 7.8902e-02 |
| `nonlinear_low_noise` | `RFF Ridge` | `ridge_rff+ridge=1e-1+reg=gradient:0.3` | 1.0000e-01 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 2.1876e-03 | 5.6000e+02 | 5.1563e-09 | 2.2674e-01 | 3.7319e+00 | 2.3082e-01 | 4.1139e-01 | 4.4718e+00 | 2.9919e-01 | 1.6295e+00 | 6.0000e-01 | 0.0000e+00 | 8.5476e-02 | 1.2815e-01 |
| `nonlinear_low_noise` | `RFF Frequency Weighted` | `penalty_rff+ridge=1e-2+reg=gradient:0.3` | 1.0000e-02 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 2.3777e-03 | 5.6000e+02 | 6.5170e-09 | 2.2746e-01 | 3.6284e+00 | 2.2961e-01 | 2.8440e-01 | 3.0915e+00 | 2.1008e-01 | 1.1443e+00 | 8.0000e-01 | 0.0000e+00 | 4.4029e-01 | 8.9724e-02 |
| `nonlinear_low_noise` | `KISS-GP Ridge` | `kissgp+ridge=1e-6+reg=gradient:0.3` | 1.0000e-06 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 5.9772e-03 | 5.6000e+02 | 3.7075e-09 | 2.3878e-01 | 3.3569e+00 | 2.4241e-01 | 4.0474e-01 | 4.3995e+00 | 2.9325e-01 | 1.5972e+00 | 8.0000e-01 | 0.0000e+00 | 3.5054e-01 | 1.7059e+00 |
| `nonlinear_low_noise` | `MLP Ridge` | `mlp+ridge=1e-1+reg=gradient:0.3` | 1.0000e-01 | `gradient` | 3.0000e-01 | `ok` | 3.0000e+00 | 1.9391e-02 | 5.6000e+02 | 4.7333e-14 | 4.6110e-01 | 7.3556e+00 | 4.7096e-01 | 7.1435e-01 | 7.7651e+00 | 6.3900e-01 | 3.4804e+00 | 1.2000e+00 | 0.0000e+00 | 4.0473e+00 | 5.4115e+00 |
| `nonlinear_low_noise` | `Matern GP Monotone Internal Ridge` | `materngpmonotone_unregularized+ridge=1e-1+reg=laplacian:0.1` | 1.0000e-01 | `laplacian` | 1.0000e-01 | `ok` | 4.0000e+00 | 5.6425e-03 | 8.0000e+02 | 8.5669e-09 | 2.3593e-01 | 4.3708e+00 | 2.3759e-01 | 4.0147e-01 | 4.3640e+00 | 8.1677e-01 | 4.4487e+00 | 1.6000e+00 | 2.0000e-01 | 3.5388e-01 | 2.1139e-01 |
| `nonlinear_low_noise` | `RBF Ridge` | `rbf+ridge=1e-1+reg=laplacian:0.01` | 1.0000e-01 | `laplacian` | 1.0000e-02 | `ok` | 6.0000e+00 | 2.5408e-02 | 1.0400e+03 | 6.0011e-12 | 9.3477e-01 | 3.3461e+01 | 9.2460e-01 | 2.0064e+00 | 2.1810e+01 | 2.5988e+00 | 1.4155e+01 | 2.4400e+01 | 2.9400e+01 | 7.3316e-02 | 6.6372e-02 |
