# Paper Result Groups

This folder groups the result CSVs and summaries used by the current writeup. The source CSV headers have been normalized so data-side preprocessing columns are named `smoothing_type` and `smoothing_strength`. Internal model penalties remain `model_regularization_type` and `model_regularization_strength`.

## 01_noisy_best_newton_baseline

Use for the initial noisy surrogate/best-Newton tables.

CSV files:

- `linear_no_noise.csv`
- `linear_medium_noise.csv`
- `nonlinear_no_noise.csv`
- `nonlinear_high_noise.csv`
- `nonlinear_low_noise.csv`

Summary:

- `best_newton_latex_tables.md`

Notes:

- This is the most complete noisy baseline/best-Newton source.
- Earlier manually assembled tables may be incomplete for some methods, especially where older runs lacked full RMSE or timing columns.

## 02_no_noise_no_smoothing_baselines

Use for no-noise, no-smoothing baseline tables.

CSV files:

- `linear_no_noise.csv`
- `nonlinear_no_noise.csv`

Summary:

- `no_noise_no_smoothing_tables.md`

Notes:

- This group intentionally excludes RFF, FDIntegrated, FDMatern, Smooth+PCHIP, and smoothing variants.

## 03_model_regularization_no_presmoothing

Use for model-regularized tables with no input-data smoothing.

CSV files:

- `linear_medium_noise.csv`
- `nonlinear_high_noise.csv`
- `nonlinear_low_noise.csv`

Summaries:

- `noisy_best_by_model_regularization_type.md`
- `model_regularization_sweep_no_presmoothing_summary.md`

Notes:

- This group is for internal/model regularization only, such as RFF Ridge, RFF Frequency Weighted, KISS-GP Ridge, RBF Ridge, Matern GP Monotone Ridge, and MLP Ridge.
- The full summary includes no-noise and noisy rows, so the no-noise CSVs are included here too.

## 04_model_and_data_smoothing_regularization

Use for tables where both model regularization and input-data smoothing are active.

CSV files:

- `linear_medium_noise.csv`
- `nonlinear_high_noise.csv`
- `nonlinear_low_noise.csv`

Summaries:

- `noisy_best_both_smoothing_and_regularization.md`
- `noisy_best_combined_model_and_data_regularization.md`

Notes:

- `noisy_best_both_smoothing_and_regularization.md` keeps only rows with both internal/model regularization and data smoothing active.
- `noisy_best_combined_model_and_data_regularization.md` compares all rows from that sweep, including rows where smoothing is `none`.

## 05_gp_monotone_variants

Use for checking the GP-monotone family naming and behavior after the coordinate-pairing fix.

CSV files:

- `linear_no_noise.csv`
- `linear_medium_noise.csv`
- `nonlinear_no_noise.csv`
- `nonlinear_high_noise.csv`
- `nonlinear_low_noise.csv`

Summary:

- `gp_monotone_family_after_coordinate_fix_summary.md`

Notes:

- This group distinguishes plain monotone GP, internally Tikhonov-regularized monotone GP, and input-data-smoothed monotone GP.
- It should be used to sanity-check any Overleaf table that mentions `MaternGPR+Monotonicity`, `Matern GP Monotone Ridge`, or `Matern GP Monotone Internal Ridge`.
