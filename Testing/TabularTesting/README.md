## 1D Global Surrogates Testing

#### Contents:
- `comparison.py`: Main driver for tabular flux-law methods
- `data.py`: Dataset preparation helpers for tabular flux-law experiments
- `providers.py`: Interface with tabular data flux law providers
- `tabular_models.py`

#### Dependencies:
- pytorch
- gpytorch
- numpy
- pandas
- jax
- probably more idk

#### Adding models for evaluation:
Currently supported: Analytic, CubicSpline, PCHIP, RBF, KISS-GP, SavGol, FiniteDiff, MLP
1. The module for each flux law provider should provide the function  `evaluate` which takes in arrays for s, T and returns q, dq_ds, and dq_dT.
2. Add provider to the if statements in `providers.py`. Will need to initialize the provider and wrap as `scaled_flux` or `unscaled_flux` depending on how model takes in data.
3. Add provider key to methods = [] in `comparison.py`.

#### Running evaluation:
- Set methods, datasets, and experiment name variables in `comparison.py`
- Run python3 comparison.py
