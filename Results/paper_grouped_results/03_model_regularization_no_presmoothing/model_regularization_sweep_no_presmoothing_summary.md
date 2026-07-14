# Model-Regularization Sweep, No Data Pre-Smoothing

No `+reg=...` input-grid pre-smoothing was used. These are internal/model regularization settings only.

Lambda/alpha values tested: `1e-6`, `1e-4`, `1e-2`, `1e-1`.

Method families:

- `ridge_rff`: Random Fourier features with sklearn Ridge `alpha`
- `penalty_rff`: frequency-weighted RFF penalty `alpha`
- `kissgp`: KISS-GP L2/ridge penalty on trainable parameters
- `rbf`: extra SciPy RBF diagonal smoothing
- `materngpmonotone_unregularized`: internal monotone-GP Tikhonov strength
- `mlp`: MLP L2 weight penalty

## Overall Best Per Dataset

| dataset | family | method | lambda_alpha | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% | build_s | solve_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linear_no_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-4` | 1.0000e-04 | `ok` | 2.0000e+00 | 8.2625e-08 | 4.0000e+02 | 4.4278e-09 | 2.9010e-06 | -- | 3.0028e-06 | 1.2641e-05 | -- | 3.8425e-06 | -- | 0.0000e+00 | 0.0000e+00 | 5.0744e-01 | 9.8339e-02 |
| `linear_medium_noise` | `ridge_rff` | `ridge_rff+ridge=1e-1` | 1.0000e-01 | `ok` | 3.0000e+00 | 1.4820e-03 | 5.6000e+02 | 3.5904e-12 | 1.4307e-01 | 1.0412e+00 | 7.7832e-02 | 2.7365e-01 | 1.1287e+00 | 3.1816e-01 | 6.5490e-01 | 2.0000e-01 | 0.0000e+00 | 7.9072e-02 | 1.2140e-01 |
| `nonlinear_no_noise` | `ridge_rff` | `ridge_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 4.0000e+00 | 5.1417e-05 | 7.2000e+02 | 3.7724e-14 | 3.9825e-03 | -- | 2.6845e-03 | 2.0063e-02 | -- | 2.5686e-02 | -- | 0.0000e+00 | 0.0000e+00 | 9.7311e-02 | 1.5934e-01 |
| `nonlinear_low_noise` | `penalty_rff` | `penalty_rff+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 1.6526e-03 | 5.6000e+02 | 5.9464e-09 | 5.2292e-02 | 1.0917e+00 | 2.7045e-02 | 1.2167e-01 | 1.3225e+00 | 1.4878e-01 | 8.1033e-01 | 0.0000e+00 | 0.0000e+00 | 2.9624e-01 | 9.1915e-02 |
| `nonlinear_high_noise` | `ridge_rff` | `ridge_rff+ridge=1e-4` | 1.0000e-04 | `ok` | 4.0000e+00 | 7.4529e-03 | 7.2000e+02 | 3.1717e-10 | 7.5130e-01 | 1.1943e+00 | 3.0724e-01 | 1.2078e+00 | 1.0166e+00 | 2.0736e+00 | 8.6093e-01 | 1.0000e+00 | 0.0000e+00 | 2.7115e-01 | 3.1181e-01 |

## Best Per Dataset And Family

| dataset | family | method | lambda_alpha | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% | build_s | solve_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linear_no_noise` | `ridge_rff` | `ridge_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 2.0000e+00 | 3.7401e-05 | 4.0000e+02 | 2.5211e-11 | 8.4477e-04 | -- | 9.3222e-04 | 6.2567e-03 | -- | 5.3895e-03 | -- | 0.0000e+00 | 0.0000e+00 | 1.8940e-01 | 9.1844e-02 |
| `linear_no_noise` | `penalty_rff` | `penalty_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 2.0000e+00 | 2.9676e-05 | 4.0000e+02 | 1.5572e-11 | 3.7856e-04 | -- | 4.5888e-04 | 3.0144e-03 | -- | 2.2425e-03 | -- | 0.0000e+00 | 0.0000e+00 | 4.9800e-01 | 6.3307e-02 |
| `linear_no_noise` | `kissgp` | `kissgp+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 4.3498e-03 | 5.6000e+02 | 7.2972e-11 | 7.9224e-02 | -- | 8.5116e-02 | 2.3655e-01 | -- | 1.9180e-01 | -- | 0.0000e+00 | 0.0000e+00 | 3.5755e-01 | 1.6984e+00 |
| `linear_no_noise` | `rbf` | `rbf+ridge=1e-6` | 1.0000e-06 | `ok` | 3.0000e+00 | 3.1279e-02 | 5.6000e+02 | 6.4980e-09 | 6.2419e-01 | -- | 6.3527e-01 | 1.3147e+00 | -- | 1.9801e+00 | -- | 8.4000e+00 | 1.8200e+01 | 6.7846e-02 | 3.1730e-02 |
| `linear_no_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-4` | 1.0000e-04 | `ok` | 2.0000e+00 | 8.2625e-08 | 4.0000e+02 | 4.4278e-09 | 2.9010e-06 | -- | 3.0028e-06 | 1.2641e-05 | -- | 3.8425e-06 | -- | 0.0000e+00 | 0.0000e+00 | 5.0744e-01 | 9.8339e-02 |
| `linear_no_noise` | `mlp` | `mlp+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 1.0372e-02 | 5.6000e+02 | 3.3952e-10 | 1.2726e-01 | -- | 1.3299e-01 | 3.2881e-01 | -- | 2.8886e-01 | -- | 8.0000e-01 | 0.0000e+00 | 3.9553e+00 | 5.2924e+00 |
| `linear_medium_noise` | `ridge_rff` | `ridge_rff+ridge=1e-1` | 1.0000e-01 | `ok` | 3.0000e+00 | 1.4820e-03 | 5.6000e+02 | 3.5904e-12 | 1.4307e-01 | 1.0412e+00 | 7.7832e-02 | 2.7365e-01 | 1.1287e+00 | 3.1816e-01 | 6.5490e-01 | 2.0000e-01 | 0.0000e+00 | 7.9072e-02 | 1.2140e-01 |
| `linear_medium_noise` | `penalty_rff` | `penalty_rff+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 2.8498e-03 | 5.6000e+02 | 3.4787e-14 | 1.2710e-01 | 9.7090e-01 | 5.0865e-02 | 1.0332e-01 | 4.2617e-01 | 1.7351e-01 | 3.5716e-01 | 0.0000e+00 | 0.0000e+00 | 5.0555e-01 | 9.2148e-02 |
| `linear_medium_noise` | `kissgp` | `kissgp+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 4.6491e-03 | 5.6000e+02 | 2.2989e-11 | 1.5978e-01 | 1.0970e+00 | 9.6296e-02 | 2.5300e-01 | 1.0435e+00 | 2.0358e-01 | 4.1906e-01 | 0.0000e+00 | 0.0000e+00 | 3.4378e-01 | 1.7438e+00 |
| `linear_medium_noise` | `rbf` | `rbf+ridge=1e-6` | 1.0000e-06 | `ok` | 3.0000e+00 | 3.3799e-02 | 5.6000e+02 | 4.8415e-09 | 6.6231e-01 | 7.3389e+00 | 6.4637e-01 | 1.2606e+00 | 5.1996e+00 | 2.1309e+00 | 4.3862e+00 | 9.2000e+00 | 1.8000e+01 | 6.8542e-02 | 3.2254e-02 |
| `linear_medium_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-1` | 1.0000e-01 | `ok` | 5.0000e+00 | 4.6508e-03 | 9.6000e+02 | 1.5032e-12 | 1.6958e-01 | 1.3305e+00 | 1.5587e-01 | 2.7207e-01 | 1.1222e+00 | 2.3631e+00 | 4.8641e+00 | 6.0000e-01 | 2.0000e-01 | 3.7404e-01 | 2.3624e-01 |
| `linear_medium_noise` | `mlp` | `mlp+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 1.0879e-02 | 5.6000e+02 | 6.9389e-10 | 1.8051e-01 | 1.6628e+00 | 1.3187e-01 | 3.3087e-01 | 1.3647e+00 | 2.7367e-01 | 5.6332e-01 | 8.0000e-01 | 0.0000e+00 | 4.0955e+00 | 5.3040e+00 |
| `nonlinear_no_noise` | `ridge_rff` | `ridge_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 4.0000e+00 | 5.1417e-05 | 7.2000e+02 | 3.7724e-14 | 3.9825e-03 | -- | 2.6845e-03 | 2.0063e-02 | -- | 2.5686e-02 | -- | 0.0000e+00 | 0.0000e+00 | 9.7311e-02 | 1.5934e-01 |
| `nonlinear_no_noise` | `penalty_rff` | `penalty_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 4.0000e+00 | 8.3856e-05 | 7.2000e+02 | 5.2013e-14 | 2.3695e-03 | -- | 1.9015e-03 | 1.4756e-02 | -- | 1.2705e-02 | -- | 0.0000e+00 | 0.0000e+00 | 3.9584e-01 | 1.1404e-01 |
| `nonlinear_no_noise` | `kissgp` | `kissgp+ridge=1e-1` | 1.0000e-01 | `ok` | 4.0000e+00 | 1.1917e-02 | 7.2000e+02 | 4.7569e-14 | 3.7437e-01 | -- | 3.8494e-01 | 1.5614e+00 | -- | 1.3726e+00 | -- | 0.0000e+00 | 0.0000e+00 | 4.0150e-01 | 2.3860e+00 |
| `nonlinear_no_noise` | `rbf` | `rbf+ridge=1e-4` | 1.0000e-04 | `failed: ZeroDivisionError` | -- | -- | 7.6000e+03 | -- | 2.7499e+00 | -- | 2.6989e+00 | 5.0416e+00 | -- | 9.1706e+00 | -- | 2.5000e+01 | 2.7600e+01 | 7.0134e-02 | 4.5338e-01 |
| `nonlinear_no_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-1` | 1.0000e-01 | `failed: RuntimeError` | -- | -- | 6.8000e+03 | -- | 5.2508e-04 | -- | 3.1491e-04 | 2.8512e-03 | -- | 3.1192e-03 | -- | 0.0000e+00 | 0.0000e+00 | 6.1510e-01 | 1.6916e+00 |
| `nonlinear_no_noise` | `mlp` | `mlp+ridge=1e-2` | 1.0000e-02 | `ok` | 4.0000e+00 | 1.5217e-02 | 7.2000e+02 | 5.3538e-14 | 4.0830e-01 | -- | 4.1024e-01 | 1.2946e+00 | -- | 1.4297e+00 | -- | 1.4000e+00 | 0.0000e+00 | 8.1950e+00 | 6.7984e+00 |
| `nonlinear_low_noise` | `ridge_rff` | `ridge_rff+ridge=1e-4` | 1.0000e-04 | `ok` | 4.0000e+00 | 3.3235e-03 | 7.2000e+02 | 2.4213e-14 | 5.0173e-02 | 1.0802e+00 | 2.2814e-02 | 6.8699e-02 | 7.4676e-01 | 1.3292e-01 | 7.2396e-01 | 2.0000e-01 | 0.0000e+00 | 9.1475e-02 | 1.6658e-01 |
| `nonlinear_low_noise` | `penalty_rff` | `penalty_rff+ridge=1e-2` | 1.0000e-02 | `ok` | 3.0000e+00 | 1.6526e-03 | 5.6000e+02 | 5.9464e-09 | 5.2292e-02 | 1.0917e+00 | 2.7045e-02 | 1.2167e-01 | 1.3225e+00 | 1.4878e-01 | 8.1033e-01 | 0.0000e+00 | 0.0000e+00 | 2.9624e-01 | 9.1915e-02 |
| `nonlinear_low_noise` | `kissgp` | `kissgp+ridge=1e-1` | 1.0000e-01 | `ok` | 4.0000e+00 | 1.0123e-02 | 7.2000e+02 | 2.0703e-14 | 1.4245e-01 | 2.4853e+00 | 1.3435e-01 | 5.0944e-01 | 5.5377e+00 | 4.7441e-01 | 2.5839e+00 | 4.0000e-01 | 0.0000e+00 | 3.4592e-01 | 2.2192e+00 |
| `nonlinear_low_noise` | `rbf` | `rbf+ridge=1e-4` | 1.0000e-04 | `failed: RuntimeError` | -- | -- | 1.8000e+04 | -- | 9.2562e-01 | 2.8415e+01 | 9.0573e-01 | 1.6664e+00 | 1.8114e+01 | 3.0964e+00 | 1.6865e+01 | 2.0600e+01 | 2.4200e+01 | 6.7717e-02 | 1.0528e+00 |
| `nonlinear_low_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-1` | 1.0000e-01 | `ok` | 4.0000e+00 | 2.6431e-03 | 7.2000e+02 | 6.2408e-12 | 6.1088e-02 | 1.2333e+00 | 4.2650e-02 | 1.9202e-01 | 2.0873e+00 | 3.3071e-01 | 1.8012e+00 | 4.0000e-01 | 0.0000e+00 | 2.7239e-01 | 1.8453e-01 |
| `nonlinear_low_noise` | `mlp` | `mlp+ridge=1e-1` | 1.0000e-01 | `ok` | 3.0000e+00 | 2.3376e-02 | 5.6000e+02 | 1.9634e-13 | 3.8620e-01 | 7.1528e+00 | 3.9949e-01 | 6.5427e-01 | 7.1120e+00 | 6.4220e-01 | 3.4978e+00 | 1.0000e+00 | 0.0000e+00 | 3.9789e+00 | 1.0037e+01 |
| `nonlinear_high_noise` | `ridge_rff` | `ridge_rff+ridge=1e-4` | 1.0000e-04 | `ok` | 4.0000e+00 | 7.4529e-03 | 7.2000e+02 | 3.1717e-10 | 7.5130e-01 | 1.1943e+00 | 3.0724e-01 | 1.2078e+00 | 1.0166e+00 | 2.0736e+00 | 8.6093e-01 | 1.0000e+00 | 0.0000e+00 | 2.7115e-01 | 3.1181e-01 |
| `nonlinear_high_noise` | `penalty_rff` | `penalty_rff+ridge=1e-6` | 1.0000e-06 | `ok` | 4.0000e+00 | 7.7197e-03 | 7.2000e+02 | 9.3869e-13 | 7.7411e-01 | 1.2368e+00 | 3.8237e-01 | 1.7365e+00 | 1.4615e+00 | 3.1853e+00 | 1.3225e+00 | 1.2000e+00 | 8.0000e-01 | 1.2715e+00 | 2.2798e-01 |
| `nonlinear_high_noise` | `kissgp` | `kissgp+ridge=1e-1` | 1.0000e-01 | `ok` | 4.0000e+00 | 1.9837e-02 | 7.2000e+02 | 5.1359e-14 | 8.2285e-01 | 1.3559e+00 | 4.4531e-01 | 1.7341e+00 | 1.4595e+00 | 1.4437e+00 | 5.9941e-01 | 4.0000e-01 | 0.0000e+00 | 1.1883e+00 | 5.1101e+00 |
| `nonlinear_high_noise` | `rbf` | `rbf+ridge=1e-6` | 1.0000e-06 | `ok` | 5.0000e+00 | 4.7911e-02 | 8.8000e+02 | 1.4848e-13 | 2.6987e+00 | 1.0347e+01 | 2.6535e+00 | 5.3237e+00 | 4.4808e+00 | 8.7212e+00 | 3.6209e+00 | 2.6800e+01 | 2.9000e+01 | 1.2969e-01 | 1.2326e-01 |
| `nonlinear_high_noise` | `materngpmonotone_unregularized` | `materngpmonotone_unregularized+ridge=1e-2` | 1.0000e-02 | `ok` | 1.4000e+01 | 1.6023e-02 | 6.4000e+03 | 5.0547e-10 | 9.2303e-01 | 1.8059e+00 | 6.3310e-01 | 3.4469e+00 | 2.9012e+00 | 5.1939e+00 | 2.1564e+00 | 3.4000e+00 | 8.4000e+00 | 6.6884e-01 | 3.3323e+00 |
| `nonlinear_high_noise` | `mlp` | `mlp+ridge=1e-2` | 1.0000e-02 | `ok` | 4.0000e+00 | 1.0931e-02 | 7.2000e+02 | 5.8927e-14 | 8.7646e-01 | 1.5272e+00 | 5.0292e-01 | 1.4880e+00 | 1.2524e+00 | 1.5281e+00 | 6.3444e-01 | 1.4000e+00 | 0.0000e+00 | 6.9437e+00 | 1.3226e+01 |
