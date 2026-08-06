"""Deterministic anisotropic Darcy-Forchheimer constitutive oracle."""

from __future__ import annotations

import numpy as np


class DarcyForchheimerOracle:
    """Map superficial velocity in 3D to pressure-driving force."""

    rho: float = 998.2
    mu: float = 1.002e-3
    porosity: float = 0.40
    diameters: np.ndarray = np.array([2.0e-3, 1.5e-3, 1.0e-3], dtype=float)

    def __init__(self) -> None:
        eps = self.porosity
        d = self.diameters

        self.k = eps**3 * d**2 / (150.0 * (1.0 - eps) ** 2)
        self.beta = 1.75 * (1.0 - eps) / (eps**3 * d)
        self.R = self._rotation_z(np.deg2rad(30.0)) @ self._rotation_y(np.deg2rad(20.0))
        self.K = self.R @ np.diag(self.k) @ self.R.T
        self.C = self.R @ np.diag(self.beta ** (2.0 / 3.0)) @ self.R.T
        self.K_inv = self.R @ np.diag(1.0 / self.k) @ self.R.T

    def evaluate(self, velocity: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        velocity:
            Array with shape (3,), containing [ux, uy, uz] in m/s.

        Returns
        -------
        force:
            Array with shape (3,), containing [Fx, Fy, Fz] in Pa/m.
        """
        u = self._validate_vector(velocity, "velocity")
        cu = self.C @ u
        speed_metric = np.sqrt(max(float(u @ cu), 0.0))
        darcy = self.mu * (self.K_inv @ u)
        forchheimer = self.rho * speed_metric * cu
        return darcy + forchheimer

    def evaluate_batch(self, velocities: np.ndarray) -> np.ndarray:
        """Accept shape (n_points, 3) and return shape (n_points, 3)."""
        u = self._validate_batch(velocities)
        cu = u @ self.C.T
        speed_metric = np.sqrt(np.maximum(np.einsum("ij,ij->i", u, cu), 0.0))
        darcy = self.mu * (u @ self.K_inv.T)
        forchheimer = self.rho * speed_metric[:, np.newaxis] * cu
        return darcy + forchheimer

    @staticmethod
    def _rotation_y(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

    @staticmethod
    def _rotation_z(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    @staticmethod
    def _validate_batch(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"velocities must have shape (n_points, 3), got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("velocities must contain only finite values.")
        return array
