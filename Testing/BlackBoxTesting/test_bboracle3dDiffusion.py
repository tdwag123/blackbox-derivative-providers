from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Data" / "BlackBoxOracle"))

from bboracle3dDiffusion import (
    make_diffusion_oracle,
    physical_flux,
    physical_flux_derivatives,
)


def test_2d_single_point_flux_shape() -> None:
    oracle = make_diffusion_oracle(config="nonlinear_no_noise", dim=2, noisy=False)
    q = oracle(np.array([0.4, -0.2]), 1.2)
    assert q.shape == (2,)


def test_3d_single_point_flux_shape() -> None:
    oracle = make_diffusion_oracle(config="nonlinear_no_noise", dim=3, noisy=False)
    q = oracle(np.array([0.4, -0.2, 0.7]), 1.2)
    assert q.shape == (3,)


def test_batched_3d_flux_shape() -> None:
    oracle = make_diffusion_oracle(config="nonlinear_no_noise", dim=3, noisy=False)
    grad_T = np.array([[0.4, -0.2, 0.7], [0.1, 0.3, -0.5]])
    T = np.array([1.2, -0.4])
    q = oracle(grad_T, T)
    assert q.shape == (2, 3)


def test_linear_case_is_negative_k0_grad() -> None:
    oracle = make_diffusion_oracle(config="linear_no_noise", dim=3, noisy=False)
    grad_T = np.array([[0.4, -0.2, 0.7], [0.1, 0.3, -0.5]])
    T = np.array([1.2, -0.4])
    q = oracle(grad_T, T)
    np.testing.assert_allclose(q, -1.5 * grad_T)


def test_A_matches_centered_finite_differences() -> None:
    grad_T = np.array([0.4, -0.2, 0.7])
    T = 1.2
    k_0 = 1.0
    alpha = 0.5
    beta = 0.2
    step = 1.0e-6

    A, _ = physical_flux_derivatives(grad_T, T, k_0, alpha, beta)
    finite_difference_A = np.zeros((3, 3))

    for j in range(3):
        delta = np.zeros(3)
        delta[j] = step
        q_plus = physical_flux(grad_T + delta, T, k_0, alpha, beta)
        q_minus = physical_flux(grad_T - delta, T, k_0, alpha, beta)
        finite_difference_A[:, j] = (q_plus - q_minus) / (2.0 * step)

    np.testing.assert_allclose(A, finite_difference_A, rtol=1.0e-8, atol=1.0e-10)


def test_b_matches_centered_finite_difference() -> None:
    grad_T = np.array([0.4, -0.2, 0.7])
    T = 1.2
    k_0 = 1.0
    alpha = 0.5
    beta = 0.2
    step = 1.0e-6

    _, b = physical_flux_derivatives(grad_T, T, k_0, alpha, beta)
    q_plus = physical_flux(grad_T, T + step, k_0, alpha, beta)
    q_minus = physical_flux(grad_T, T - step, k_0, alpha, beta)
    finite_difference_b = (q_plus - q_minus) / (2.0 * step)

    np.testing.assert_allclose(b, finite_difference_b, rtol=1.0e-8, atol=1.0e-10)
