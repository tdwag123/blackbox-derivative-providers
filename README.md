# RIPS-LLNL Black-Box Diffusion Code

This repository contains the code for the RIPS-LLNL multidimensional black-box diffusion project. It includes the Newton/FEM solver, the 2D/3D diffusion oracle, the Gaussian-process-based flux surrogate methods, and the comparison script used to run the submitted experiments.

## Disclaimer

We want our RIPS sponsors to be aware of the nature of software developed by RIPS project teams. IPAM does not regard RIPS software as anything more than a prototype developed as a proof-of-concept only, and it is never developed for commercial use nor is it warranted by IPAM in any way. Here are some points to remember:

1. Software developed by a RIPS project team that appears to have been created wholly by a project team, may in fact contain proprietary codes borrowed from other sources; the sponsor must assume all risk for using such software.
2. IPAM makes every effort to discourage misuse of proprietary software by RIPS project participants; IPAM cannot be held responsible for such misuse.
3. As participants in an academic program, RIPS students will at times be permitted to use software that cannot be used by sponsors without a license.
4. Any restriction required by the sponsor on the use of special software, or platform needed to run the software, should be declared by the sponsor at the time of negotiating the project Work Statement. Otherwise the project team is free to choose software solutions as they see fit.

## Contents

- `Basic/newton_nd.py`  
  Rectangular-grid Newton/FEM solve for 1D, 2D, and 3D problems.

- `Data/bboracle3dDiffusion.py`  
  Two- and three-dimensional nonlinear diffusion oracle. The black-box interface returns only flux values.

- `Methods/baseGP.py`  
  Multidimensional Matern GP flux surrogate.

- `Methods/directionalMonotoneGP.py`  
  Multidimensional monotone GP flux surrogate.

- `Methods/rff_baseGP.py`  
  Random Fourier feature version of the multidimensional base GP.

- `Methods/isotropicMonotoneGP.py`  
  Isotropic monotone GP version retained from the most recent isotropic results.

- `Testing/bb3d_providers.py`  
  Local adaptive black-box providers for `baseGP`, `monotoneGP`, and `rff_baseGP`.

- `Testing/bb3d_comparison.py`  
  Runs the provider methods inside `Basic.newton_nd.NM` and writes CSV results.

## Methods

The comparison provider currently supports:

- `bb3d_basegp`
- `bb3d_monotonegp`
- `bb3d_rff_basegp`
- `analytic`, used as the reference flux law

The provider is local: it samples the black-box oracle near the states visited by Newton, fits a small surrogate, and uses the fitted model to return fluxes and derivatives.

## Example

```python
from Testing.bb3d_comparison import comparison

comparison(
    "documentedcode_example",
    methods=[
        "bb3d_basegp",
        "bb3d_monotonegp",
        "bb3d_rff_basegp",
    ],
    oracle_configs=["nonlinear_no_noise"],
    dim=2,
    n_per_axis=10,
    noisy=False,
    seed=0,
)
```

For quick smoke tests, smaller local-cache settings are useful:

```python
methods = [
    "bb3d_basegp+initial_points_per_dim=3+max_points_per_dim=4+max_refinements_per_eval=0+variance_tolerance=1e9+optimize_hyperparameters=false",
    "bb3d_monotonegp+initial_points_per_dim=3+max_points_per_dim=4+max_refinements_per_eval=0+variance_tolerance=1e9+optimize_hyperparameters=false+n_virtual_per_axis=2+ep_max_iter=2",
    "bb3d_rff_basegp+initial_points_per_dim=3+max_points_per_dim=4+max_refinements_per_eval=0+variance_tolerance=1e9+n_rff_features=32",
]
```

Results are written to `Results/<experiment_name>/`.
