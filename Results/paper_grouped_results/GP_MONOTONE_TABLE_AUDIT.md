# GP Monotone Table Audit

The Overleaf draft uses several similar names for GP-based methods. They are not all the same experiment.

## Naming Map

| Overleaf name | Current method/spec meaning | Source group |
| --- | --- | --- |
| `KISS-GP` | Standard KISS-GP provider, no internal ridge unless the method name contains `+ridge=...` | `01_noisy_best_newton_baseline`, `02_no_noise_no_smoothing_baselines` |
| `KISS-GP Ridge` | KISS-GP with internal ridge/L2 penalty, optionally also with data smoothing if `+reg=...` appears | `03_model_regularization_no_presmoothing`, `04_model_and_data_smoothing_regularization` |
| `MaternGPR+Monotonicity` | Older monotone Matern GP comparison block in Overleaf. Exact values in the pasted draft were not found in the current grouped CSVs. Treat as legacy/stale unless the original CSV is recovered. | not currently reproducible from grouped CSVs |
| `Matern GP Monotone Ridge` | Monotone GP with internal Tikhonov/ridge strength and no input-data smoothing | `03_model_regularization_no_presmoothing` |
| `Matern GP Monotone Internal Ridge` | Monotone GP with internal ridge; in the model-and-data tables this also has input-data smoothing via `+reg=gradient:...` or `+reg=laplacian:...` | `04_model_and_data_smoothing_regularization` |
| `maternGPMonotone+reg=...` | Monotone GP fit after input-grid data smoothing. In the older no-internal-Tikhonov run, this was smoothing only. In the model-and-data sweep, this may also include `+ridge=...`. | `05_gp_monotone_variants`, `04_model_and_data_smoothing_regularization` |

## Main Findings

1. The `MaternGPR+Monotonicity` tables in the Overleaf "Further GP Experiments" section appear to be from an older run. I searched for exact values such as `7.0263e-01`, `4.7829e-01`, and `1.3460e+00`; they do not appear in the current grouped result CSVs or summaries.

2. The current reproducible monotone-GP family source is `05_gp_monotone_variants/gp_monotone_family_after_coordinate_fix_summary.md`.

3. The model-regularized tables use rows like `materngpmonotone_unregularized+ridge=...`. The raw method name looks contradictory, but in the current provider logic a positive `+ridge=...` activates internal GP Tikhonov. The friendly table name `Matern GP Monotone Ridge` is therefore semantically right.

4. In the combined model-and-data report, not every best row has data smoothing. Check `smoothing_type`: if it is `none`, that row belongs in the combined comparison, not in the "both smoothing and regularization" table.

5. For the "both smoothing and regularization" tables, use `noisy_best_both_smoothing_and_regularization.md`, not `noisy_best_combined_model_and_data_regularization.md`.
