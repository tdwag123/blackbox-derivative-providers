# FD Then Smooth Sweep Results

Sweep date: 2026-07-10

Constraint: do not smooth or regularize `q` before finite differences.

Pipeline tested:

1. Build raw structured `q(s,T)` table.
2. Compute finite-difference derivative fields.
3. Smooth derivative fields with candidate kernels.
4. Repair boundary derivative values.
5. Integrate the smoothed derivatives back to a compatible `q_hat`.
6. Test the resulting flux law in `Testing/comparison.py`.

Best overall candidate across `linear_medium_noise`, `nonlinear_high_noise`,
and `nonlinear_low_noise`:

```text
Method: FDIntegratedEpanechnikov
Kernel: Epanechnikov
Kernel width: 15
Endpoint repair: polynomial, width 7 on each side
q fidelity during integration: 0.03
```

Official `17 x 17` comparison results from `Results/bello_7`:

```text
linear_medium_noise:  ok, 3 Newton steps, rel_solution_err = 0.004519698923395095
nonlinear_high_noise: ok, 5 Newton steps, rel_solution_err = 0.0023153531293318274
nonlinear_low_noise:  ok, 4 Newton steps, rel_solution_err = 0.00400857939188982
```

High-noise specialist:

```text
Method: FDIntegratedTricube
Kernel: tricube
Kernel width: 17
Endpoint repair: polynomial, width 8 on each side
q fidelity during integration: 0.1
nonlinear_high_noise rel_solution_err = 0.0006559207775241158
```

Finer-grid check:

The grid was temporarily changed to `25 x 25`, `Testing/comparison.py` was run,
and then `Testing/data.py` was restored to `17 x 17`.

`FDIntegratedEpanechnikov` on `25 x 25`, from `Results/bello_8`:

```text
linear_medium_noise:  ok, 4 Newton steps, rel_solution_err = 0.006368802977260927
nonlinear_high_noise: ok, 6 Newton steps, rel_solution_err = 0.016693405316122994
nonlinear_low_noise:  ok, 5 Newton steps, rel_solution_err = 0.017717104148202618
```

Takeaway:

`FDIntegratedEpanechnikov` is the best verified `17 x 17` all-dataset setting
from the sweep. The same width/fidelity does not transfer cleanly to `25 x 25`,
so finer grids should get their own width/fidelity sweep.
