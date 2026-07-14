# GP Monotone Family After Coordinate Fix

Newton tolerance: 1e-8 in Testing/comparison.py.

Methods tested:

- materngpmonotone_unregularized: no pre-smoothing, no internal Tikhonov
- materngpmonotone_regularized: no pre-smoothing, internal GP Tikhonov
- maternGPMonotone+reg=...: harness pre-smoothed q-grid, no internal Tikhonov

## Best Converged Row Per Dataset

| dataset | method | regularization_type | regularization_strength | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_no_noise | materngpmonotone_unregularized | none | 0.0000e+00 | ok | 2.0000e+00 | 8.2597e-08 | 4.0000e+02 | 5.8705e-09 | 2.9013e-06 | -- | 3.0027e-06 | 1.2641e-05 | -- | 3.8425e-06 | -- | 0.0000e+00 | 0.0000e+00 |
| linear_medium_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 4.0000e+00 | 6.2417e-03 | 7.2000e+02 | 6.1093e-13 | 2.2199e-01 | 1.8633e+00 | 1.8072e-01 | 2.9342e-01 | 1.2102e+00 | 5.0788e-01 | 1.0454e+00 | 0.0000e+00 | 0.0000e+00 |
| nonlinear_no_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 5.0000e+00 | 3.7131e-03 | 9.6000e+02 | 2.6045e-12 | 6.9719e-01 | -- | 7.1558e-01 | 1.3358e+00 | -- | 2.0486e+00 | -- | 1.0000e+00 | 0.0000e+00 |
| nonlinear_low_noise | materngpmonotone_regularized | none | 0.0000e+00 | ok | 4.0000e+00 | 2.6527e-03 | 7.2000e+02 | 3.0561e-12 | 6.1155e-02 | 1.2336e+00 | 4.2696e-02 | 1.9105e-01 | 2.0767e+00 | 3.3093e-01 | 1.8024e+00 | 4.0000e-01 | 0.0000e+00 |
| nonlinear_high_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 4.0000e+00 | 5.4374e-03 | 7.2000e+02 | 7.0682e-09 | 9.5531e-01 | 1.6416e+00 | 7.5197e-01 | 1.5086e+00 | 1.2698e+00 | 3.6881e+00 | 1.5312e+00 | 2.0000e+00 | 2.0000e-01 |

## All Rows

| dataset | method | regularization_type | regularization_strength | status | newton_steps | rel_solution_err | flux_calls | final_residual | test_obs_q_RMSE | test_obs_q_RMSE/noise | clean_q_RMSE | clean_dq_ds_RMSE | clean_dq_ds_RMSE/noise | clean_dq_dT_RMSE | clean_dq_dT_RMSE/noise | entropy_violation_% | deriv_violation_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_no_noise | materngpmonotone_unregularized | none | 0.0000e+00 | ok | 2.0000e+00 | 8.2597e-08 | 4.0000e+02 | 5.8705e-09 | 2.9013e-06 | -- | 3.0027e-06 | 1.2641e-05 | -- | 3.8425e-06 | -- | 0.0000e+00 | 0.0000e+00 |
| linear_no_noise | materngpmonotone_regularized | none | 0.0000e+00 | ok | 2.0000e+00 | 8.2633e-08 | 4.0000e+02 | 5.6360e-09 | 2.9012e-06 | -- | 3.0029e-06 | 1.2641e-05 | -- | 3.8426e-06 | -- | 0.0000e+00 | 0.0000e+00 |
| linear_no_noise | maternGPMonotone+reg=gradient:0.03 | gradient | 3.0000e-02 | ok | 8.0000e+00 | 1.7757e-02 | 1.7600e+03 | 2.3840e-13 | 2.0029e-01 | -- | 2.1583e-01 | 6.1563e-01 | -- | 1.4734e+00 | -- | 1.8000e+00 | 1.4000e+00 |
| linear_no_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 4.0000e+00 | 9.8455e-03 | 7.2000e+02 | 1.2534e-09 | 1.8118e-01 | -- | 1.8957e-01 | 3.4543e-01 | -- | 7.6741e-01 | -- | 8.0000e-01 | 0.0000e+00 |
| linear_no_noise | maternGPMonotone+reg=laplacian:0.01 | laplacian | 1.0000e-02 | ok | 8.0000e+00 | 1.6311e-02 | 1.5200e+03 | 8.9499e-13 | 1.9709e-01 | -- | 2.1234e-01 | 5.8145e-01 | -- | 1.4031e+00 | -- | 1.6000e+00 | 1.2000e+00 |
| linear_no_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 4.0000e+00 | 8.4863e-03 | 7.2000e+02 | 1.4385e-12 | 1.7093e-01 | -- | 1.8087e-01 | 3.3290e-01 | -- | 7.6466e-01 | -- | 8.0000e-01 | 0.0000e+00 |
| linear_medium_noise | materngpmonotone_unregularized | none | 0.0000e+00 | ok | 5.0000e+00 | 4.6604e-03 | 9.6000e+02 | 1.8247e-12 | 1.6972e-01 | 1.3314e+00 | 1.5616e-01 | 2.7279e-01 | 1.1252e+00 | 2.3651e+00 | 4.8682e+00 | 6.0000e-01 | 2.0000e-01 |
| linear_medium_noise | materngpmonotone_regularized | none | 0.0000e+00 | ok | 5.0000e+00 | 4.6594e-03 | 9.6000e+02 | 1.5993e-12 | 1.6971e-01 | 1.3314e+00 | 1.5613e-01 | 2.7272e-01 | 1.1249e+00 | 2.3649e+00 | 4.8678e+00 | 6.0000e-01 | 2.0000e-01 |
| linear_medium_noise | maternGPMonotone+reg=gradient:0.03 | gradient | 3.0000e-02 | ok | 5.0000e+00 | 5.4681e-03 | 9.6000e+02 | 9.9214e-11 | 2.5502e-01 | 2.5025e+00 | 2.1522e-01 | 6.8700e-01 | 2.8336e+00 | 1.3758e+00 | 2.8320e+00 | 4.0000e-01 | 2.0000e+00 |
| linear_medium_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 4.0000e+00 | 6.2417e-03 | 7.2000e+02 | 6.1093e-13 | 2.2199e-01 | 1.8633e+00 | 1.8072e-01 | 2.9342e-01 | 1.2102e+00 | 5.0788e-01 | 1.0454e+00 | 0.0000e+00 | 0.0000e+00 |
| linear_medium_noise | maternGPMonotone+reg=laplacian:0.01 | laplacian | 1.0000e-02 | ok | 5.0000e+00 | 5.0156e-03 | 9.6000e+02 | 1.0847e-12 | 2.5184e-01 | 2.4582e+00 | 2.1206e-01 | 6.5432e-01 | 2.6988e+00 | 1.3059e+00 | 2.6881e+00 | 4.0000e-01 | 1.2000e+00 |
| linear_medium_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 4.0000e+00 | 6.9429e-03 | 7.2000e+02 | 3.0986e-13 | 2.1761e-01 | 1.9453e+00 | 1.7702e-01 | 3.4501e-01 | 1.4230e+00 | 6.8390e-01 | 1.4077e+00 | 0.0000e+00 | 0.0000e+00 |
| nonlinear_no_noise | materngpmonotone_unregularized | none | 0.0000e+00 | failed: RuntimeError | -- | -- | 7.5200e+03 | -- | 5.2448e-04 | -- | 3.1447e-04 | 2.8460e-03 | -- | 3.1163e-03 | -- | 0.0000e+00 | 0.0000e+00 |
| nonlinear_no_noise | materngpmonotone_regularized | none | 0.0000e+00 | failed: RuntimeError | -- | -- | 1.4800e+04 | -- | 5.2454e-04 | -- | 3.1451e-04 | 2.8465e-03 | -- | 3.1166e-03 | -- | 0.0000e+00 | 0.0000e+00 |
| nonlinear_no_noise | maternGPMonotone+reg=gradient:0.03 | gradient | 3.0000e-02 | ok | 6.0000e+00 | 7.8525e-03 | 1.2000e+03 | 9.7243e-10 | 7.5531e-01 | -- | 7.6864e-01 | 1.9428e+00 | -- | 3.8023e+00 | -- | 2.4000e+00 | 2.0000e+00 |
| nonlinear_no_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 5.0000e+00 | 3.7131e-03 | 9.6000e+02 | 2.6045e-12 | 6.9719e-01 | -- | 7.1558e-01 | 1.3358e+00 | -- | 2.0486e+00 | -- | 1.0000e+00 | 0.0000e+00 |
| nonlinear_no_noise | maternGPMonotone+reg=laplacian:0.01 | laplacian | 1.0000e-02 | ok | 6.0000e+00 | 8.2689e-03 | 1.2000e+03 | 1.8420e-11 | 7.5341e-01 | -- | 7.6300e-01 | 1.8722e+00 | -- | 3.6677e+00 | -- | 1.8000e+00 | 1.2000e+00 |
| nonlinear_no_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 5.0000e+00 | 4.8308e-03 | 9.6000e+02 | 1.1661e-12 | 6.8665e-01 | -- | 6.9124e-01 | 1.2941e+00 | -- | 2.3249e+00 | -- | 1.4000e+00 | 2.0000e-01 |
| nonlinear_low_noise | materngpmonotone_unregularized | none | 0.0000e+00 | ok | 4.0000e+00 | 2.6538e-03 | 7.2000e+02 | 4.8057e-12 | 6.1164e-02 | 1.2337e+00 | 4.2701e-02 | 1.9095e-01 | 2.0756e+00 | 3.3095e-01 | 1.8026e+00 | 4.0000e-01 | 0.0000e+00 |
| nonlinear_low_noise | materngpmonotone_regularized | none | 0.0000e+00 | ok | 4.0000e+00 | 2.6527e-03 | 7.2000e+02 | 3.0561e-12 | 6.1155e-02 | 1.2336e+00 | 4.2696e-02 | 1.9105e-01 | 2.0767e+00 | 3.3093e-01 | 1.8024e+00 | 4.0000e-01 | 0.0000e+00 |
| nonlinear_low_noise | maternGPMonotone+reg=gradient:0.03 | gradient | 3.0000e-02 | ok | 6.0000e+00 | 9.9881e-03 | 1.1200e+03 | 2.2114e-12 | 2.6385e-01 | 5.5459e+00 | 2.6816e-01 | 6.6919e-01 | 7.2742e+00 | 1.3952e+00 | 7.5991e+00 | 2.4000e+00 | 1.6000e+00 |
| nonlinear_low_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 5.0000e+00 | 6.2987e-03 | 9.6000e+02 | 8.2541e-13 | 2.4172e-01 | 4.2231e+00 | 2.4602e-01 | 4.1440e-01 | 4.5045e+00 | 7.2007e-01 | 3.9220e+00 | 1.6000e+00 | 0.0000e+00 |
| nonlinear_low_noise | maternGPMonotone+reg=laplacian:0.01 | laplacian | 1.0000e-02 | ok | 5.0000e+00 | 9.9621e-03 | 9.6000e+02 | 1.6106e-11 | 2.6225e-01 | 5.4749e+00 | 2.6568e-01 | 6.3752e-01 | 6.9299e+00 | 1.3397e+00 | 7.2966e+00 | 2.0000e+00 | 1.0000e+00 |
| nonlinear_low_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 4.0000e+00 | 5.6562e-03 | 8.0000e+02 | 7.7781e-09 | 2.3633e-01 | 4.3745e+00 | 2.3806e-01 | 4.0162e-01 | 4.3657e+00 | 8.1675e-01 | 4.4485e+00 | 1.6000e+00 | 2.0000e-01 |
| nonlinear_high_noise | materngpmonotone_unregularized | none | 0.0000e+00 | failed: ZeroDivisionError | -- | -- | 1.1360e+04 | -- | 9.2387e-01 | 1.8119e+00 | 6.3489e-01 | 3.4556e+00 | 2.9085e+00 | 5.1977e+00 | 2.1580e+00 | 3.4000e+00 | 8.6000e+00 |
| nonlinear_high_noise | materngpmonotone_regularized | none | 0.0000e+00 | ok | 1.4000e+01 | 1.6023e-02 | 6.4000e+03 | 5.0547e-10 | 9.2303e-01 | 1.8059e+00 | 6.3310e-01 | 3.4469e+00 | 2.9012e+00 | 5.1939e+00 | 2.1564e+00 | 3.4000e+00 | 8.4000e+00 |
| nonlinear_high_noise | maternGPMonotone+reg=gradient:0.03 | gradient | 3.0000e-02 | ok | 8.0000e+00 | 6.1049e-03 | 2.0800e+03 | 2.9263e-12 | 1.0391e+00 | 2.0165e+00 | 8.5560e-01 | 2.5452e+00 | 2.1422e+00 | 5.6885e+00 | 2.3618e+00 | 4.2000e+00 | 4.6000e+00 |
| nonlinear_high_noise | maternGPMonotone+reg=gradient:0.3 | gradient | 3.0000e-01 | ok | 5.0000e+00 | 3.5399e-03 | 8.8000e+02 | 9.1388e-13 | 9.8515e-01 | 1.5548e+00 | 7.4980e-01 | 1.5350e+00 | 1.2919e+00 | 2.9041e+00 | 1.2057e+00 | 1.4000e+00 | 0.0000e+00 |
| nonlinear_high_noise | maternGPMonotone+reg=laplacian:0.01 | laplacian | 1.0000e-02 | ok | 8.0000e+00 | 5.6811e-03 | 1.7600e+03 | 3.0118e-13 | 1.0268e+00 | 1.9820e+00 | 8.4865e-01 | 2.3933e+00 | 2.0144e+00 | 5.5542e+00 | 2.3060e+00 | 3.8000e+00 | 4.0000e+00 |
| nonlinear_high_noise | maternGPMonotone+reg=laplacian:0.1 | laplacian | 1.0000e-01 | ok | 4.0000e+00 | 5.4374e-03 | 7.2000e+02 | 7.0682e-09 | 9.5531e-01 | 1.6416e+00 | 7.5197e-01 | 1.5086e+00 | 1.2698e+00 | 3.6881e+00 | 1.5312e+00 | 2.0000e+00 | 2.0000e-01 |
