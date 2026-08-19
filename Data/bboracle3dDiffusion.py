"""Two- and three-dimensional nonlinear heat-flux oracle.

The comparison code treats this as a black box: normal provider calls receive
only ``q``. Exact derivatives are available here only for the analytic reference
model and diagnostic checks.
"""

from __future__ import annotations

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


def physical_flux(
    grad_T: np.ndarray,
    T: np.ndarray,
    k_0: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Evaluate q(g, T) = -[k_0(1 + alpha T^2) + beta |g|^2] g."""
    grad = np.asarray(grad_T, dtype=float)
    temperature = np.asarray(T, dtype=float)

    # The coefficient depends on both temperature and gradient magnitude.
    grad_norm_sq = np.sum(grad**2, axis=-1)
    coefficient = k_0 * (1.0 + alpha * temperature**2) + beta * grad_norm_sq
    return -coefficient[..., np.newaxis] * grad


def physical_flux_derivatives(
    grad_T: np.ndarray,
    T: np.ndarray,
    k_0: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact diagnostic derivatives dq/dg and dq/dT."""
    grad = np.asarray(grad_T, dtype=float)
    temperature = np.asarray(T, dtype=float)
    dim = grad.shape[-1]

    grad_norm_sq = np.sum(grad**2, axis=-1)
    coefficient = k_0 * (1.0 + alpha * temperature**2) + beta * grad_norm_sq
    identity = np.eye(dim, dtype=float)
    outer = grad[..., :, np.newaxis] * grad[..., np.newaxis, :]

    A = -(coefficient[..., np.newaxis, np.newaxis] * identity + 2.0 * beta * outer)
    b = -2.0 * k_0 * alpha * temperature[..., np.newaxis] * grad
    return A, b


def make_diffusion_oracle(
    config: str = "nonlinear_high_noise",
    dim: int = 3,
    seed: int | None = None,
    noisy: bool = True,
):
    """Create a fixed nonlinear heat-flux oracle with interface (grad_T, T) -> q."""
    if config not in ORACLE_CONFIGS:
        raise ValueError(f"Unknown oracle config: {config}")
    if dim not in (2, 3):
        raise ValueError(f"dim must be either 2 or 3, got {dim}.")

    params = ORACLE_CONFIGS[config]
    rng = np.random.default_rng(seed)

    k_0 = params["k_0"]
    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]

    def oracle(grad_T: np.ndarray, T: np.ndarray, return_full: bool = False):
        grad = np.asarray(grad_T, dtype=float)
        temperature = np.asarray(T, dtype=float)

        if grad.shape == () or grad.shape[-1] != dim:
            raise ValueError(
                f"grad_T final axis must have length dim={dim}, got shape {grad.shape}."
            )

        q_true = physical_flux(grad, temperature, k_0, alpha, beta)
        A, b = physical_flux_derivatives(grad, temperature, k_0, alpha, beta)

        if noisy:
            # The oracle noise is proportional to flux size with a unit floor.
            noise_std = sigma * np.maximum(1.0, np.abs(q_true))
            noise = noise_std * rng.standard_normal(q_true.shape)
            q = q_true + noise
        else:
            noise = np.zeros_like(q_true, dtype=float)
            q = q_true

        if return_full:
            return {
                "q": q,
                "q_true": q_true,
                "A": A,
                "b": b,
                "noise": noise,
                "k_0": k_0,
                "alpha": alpha,
                "beta": beta,
                "sigma": sigma,
                "config": config,
                "dim": dim,
            }

        return q

    return oracle
