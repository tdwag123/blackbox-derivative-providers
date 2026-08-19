"""Scalar nonlinear diffusion oracle retained for reference.

The active 2D/3D comparison uses ``bboracle3dDiffusion.py``. This file keeps the
older one-dimensional flux law and parameter names in the branch for comparison
with earlier experiments.
"""

import numpy as np


ORACLE_CONFIGS = {
    "nonlinear_high_noise": {
        "k_0": 1.0,
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0.10,
    },
    "nonlinear_low_noise": {
        "k_0": 0.7,
        "alpha": 0.2,
        "beta": 0.05,
        "sigma": 0.02,
    },
    "linear_medium_noise": {
        "k_0": 1.5,
        "alpha": 0.0,
        "beta": 0.0,
        "sigma": 0.05,
    },
    "nonlinear_no_noise": {
        "k_0": 1.0,
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0.0,
    },
    "linear_no_noise": {
        "k_0": 1.5,
        "alpha": 0.0,
        "beta": 0.0,
        "sigma": 0.0,
    },
}


def physical_flux(s, T, k_0, alpha, beta):
    """One-dimensional version of q(s, T) = -k(s, T) s."""
    return -(k_0 * (1.0 + alpha * T**2) + beta * s**2) * s


def physical_flux_derivatives(s, T, k_0, alpha, beta):
    """Exact derivatives used for diagnostics or analytic reference models."""
    q_s = -k_0 * (1.0 + alpha * T**2) - 3.0 * beta * s**2
    q_T = -2.0 * k_0 * alpha * T * s
    return q_s, q_T


def make_diffusion_oracle(config="nonlinear_high_noise", seed=None, noisy=True):
    """Create a scalar q-only oracle with optional multiplicative noise."""
    if config not in ORACLE_CONFIGS:
        raise ValueError(f"Unknown oracle config: {config}")

    params = ORACLE_CONFIGS[config]
    rng = np.random.default_rng(seed)

    k_0 = params["k_0"]
    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]

    def oracle(s, T, return_full=False):
        s = np.asarray(s, dtype=float)
        T = np.asarray(T, dtype=float)

        q_true = physical_flux(s, T, k_0, alpha, beta)
        q_s, q_T = physical_flux_derivatives(s, T, k_0, alpha, beta)

        if noisy:
            # Noise scales with flux magnitude but has a floor near zero flux.
            noise_std = sigma * np.maximum(1.0, np.abs(q_true))
            noise = noise_std * rng.standard_normal(np.shape(q_true))
            q = q_true + noise
        else:
            noise = np.zeros_like(q_true, dtype=float)
            q = q_true

        if return_full:
            return {
                "q": q,
                "q_true": q_true,
                "q_s": q_s,
                "q_T": q_T,
                "noise": noise,
                "k_0": k_0,
                "alpha": alpha,
                "beta": beta,
                "sigma": sigma,
                "config": config,
            }

        return q

    return oracle
