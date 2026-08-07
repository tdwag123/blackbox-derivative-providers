from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Data" / "BlackBoxOracle"))

from bboracledarcyforchheimer import DarcyForchheimerOracle


def test_single_and_batch_shapes() -> None:
    oracle = DarcyForchheimerOracle()
    velocity = np.array([0.01, -0.02, 0.03])
    velocities = np.array([[0.01, -0.02, 0.03], [0.0, 0.04, -0.01]])

    assert oracle.evaluate(velocity).shape == (3,)
    assert oracle.evaluate_batch(velocities).shape == (2, 3)
    np.testing.assert_allclose(oracle.evaluate_batch(velocities)[0], oracle.evaluate(velocity))


def test_zero_velocity_returns_zero_force() -> None:
    oracle = DarcyForchheimerOracle()
    np.testing.assert_allclose(oracle.evaluate(np.zeros(3)), np.zeros(3))
    np.testing.assert_allclose(oracle.evaluate_batch(np.zeros((4, 3))), np.zeros((4, 3)))


def test_odd_symmetry() -> None:
    oracle = DarcyForchheimerOracle()
    velocity = np.array([0.12, -0.03, 0.07])
    np.testing.assert_allclose(oracle.evaluate(-velocity), -oracle.evaluate(velocity))


def test_positive_dissipation() -> None:
    oracle = DarcyForchheimerOracle()
    velocities = np.array(
        [
            [0.12, -0.03, 0.07],
            [-0.05, 0.08, 0.02],
            [0.0, 0.0, 0.0],
            [0.01, 0.02, -0.04],
        ]
    )
    forces = oracle.evaluate_batch(velocities)
    dissipation = np.einsum("ij,ij->i", velocities, forces)
    assert np.all(dissipation >= -1.0e-12)


def test_repeated_evaluations_are_deterministic() -> None:
    oracle = DarcyForchheimerOracle()
    velocity = np.array([0.02, -0.01, 0.05])

    first = oracle.evaluate(velocity)
    second = oracle.evaluate(velocity)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "bad_velocity",
    [np.array([1.0, 2.0]), np.zeros((1, 3)), np.array([1.0, np.nan, 3.0])],
)
def test_invalid_single_inputs_raise_clear_errors(bad_velocity: np.ndarray) -> None:
    oracle = DarcyForchheimerOracle()
    with pytest.raises(ValueError, match="velocity"):
        oracle.evaluate(bad_velocity)


@pytest.mark.parametrize(
    "bad_velocities",
    [np.zeros(3), np.zeros((2, 2)), np.array([[1.0, 2.0, np.inf]])],
)
def test_invalid_batch_inputs_raise_clear_errors(bad_velocities: np.ndarray) -> None:
    oracle = DarcyForchheimerOracle()
    with pytest.raises(ValueError, match="velocities"):
        oracle.evaluate_batch(bad_velocities)


def test_rotation_permeability_and_forchheimer_tensors_are_valid() -> None:
    oracle = DarcyForchheimerOracle()
    eps = 0.40
    diameters = np.array([2.0e-3, 1.5e-3, 1.0e-3])
    expected_k = eps**3 * diameters**2 / (150.0 * (1.0 - eps) ** 2)
    expected_beta = 1.75 * (1.0 - eps) / (eps**3 * diameters)

    rz = np.array(
        [
            [np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0)), 0.0],
            [np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(np.deg2rad(20.0)), 0.0, np.sin(np.deg2rad(20.0))],
            [0.0, 1.0, 0.0],
            [-np.sin(np.deg2rad(20.0)), 0.0, np.cos(np.deg2rad(20.0))],
        ]
    )
    expected_r = rz @ ry

    np.testing.assert_allclose(oracle.R, expected_r)
    np.testing.assert_allclose(oracle.R.T @ oracle.R, np.eye(3), atol=1.0e-15)
    np.testing.assert_allclose(oracle.K, expected_r @ np.diag(expected_k) @ expected_r.T)
    np.testing.assert_allclose(
        oracle.C, expected_r @ np.diag(expected_beta ** (2.0 / 3.0)) @ expected_r.T
    )
    assert np.all(np.linalg.eigvalsh(oracle.K) > 0.0)
    assert np.all(np.linalg.eigvalsh(oracle.C) > 0.0)


def test_principal_direction_reduction() -> None:
    oracle = DarcyForchheimerOracle()
    speeds = np.array([0.03, -0.02, 0.04])

    for i, speed in enumerate(speeds):
        principal_velocity = np.zeros(3)
        principal_velocity[i] = speed
        lab_velocity = oracle.R @ principal_velocity
        principal_force = oracle.R.T @ oracle.evaluate(lab_velocity)
        expected = np.zeros(3)
        expected[i] = oracle.mu / oracle.k[i] * speed + oracle.rho * oracle.beta[i] * abs(speed) * speed
        np.testing.assert_allclose(principal_force, expected, rtol=1.0e-12, atol=1.0e-8)
