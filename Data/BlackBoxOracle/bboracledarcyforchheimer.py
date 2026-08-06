"""Deterministic anisotropic Darcy-Forchheimer constitutive oracle.

Darcy-Forchheimer model, in words
---------------------------------
This module describes how hard it is for water to flow through one fixed
porous material. The input is the local superficial velocity

    u = [ux, uy, uz],

and the output is the pressure-driving force per unit volume

    F = [Fx, Fy, Fz] = -grad(p).

Small velocities are mostly governed by Darcy's law: the force needed to
push the fluid is proportional to velocity. Larger velocities need an
extra Forchheimer correction: the force grows like velocity times speed,
so it behaves quadratically along a principal material direction.

The constitutive law implemented here is

    F(u) = mu K^{-1} u + rho sqrt(u^T C u) C u.

The K coefficients and the B coefficients are computed via these equations:
    k_i = epsilon^3 d_i^2 / [150 (1 - epsilon)^2]
    beta_i = 1.75 (1 - epsilon) / [epsilon^3 d_i]
    
Here ``K`` is the anisotropic permeability tensor and ``C`` is the
anisotropic Forchheimer tensor. "Anisotropic" means the material resists
flow differently in different directions, like wood being easier to split
with the grain than across it. The material has three principal directions
with scalar values ``k_i`` and ``beta_i``; these are rotated into the
global xyz coordinates by

    R = Rz(30 degrees) Ry(20 degrees),
    K = R diag(k1, k2, k3) R^T,
    C = R diag(beta1^(2/3), beta2^(2/3), beta3^(2/3)) R^T.

Along a principal material direction, the vector law reduces exactly to

    F_i = (mu / k_i) u_i + rho beta_i |u_i| u_i.

This class is only the exact physical oracle ``u -> F(u)``. It does not
return derivatives.
"""

from __future__ import annotations

import numpy as np


class DarcyForchheimerOracle:
    """Map superficial velocity in 3D to pressure-driving force."""

    # ------------------------------------------------------------------
    # Fixed fluid and material parameters.
    #
    # These are intentionally internal to the oracle. The caller supplies
    # only a velocity; this object represents one fixed porous material.
    # ------------------------------------------------------------------
    rho: float = 998.2
    mu: float = 1.002e-3
    porosity: float = 0.40
    diameters: np.ndarray = np.array([2.0e-3, 1.5e-3, 1.0e-3], dtype=float)

    def __init__(self) -> None:
        """Build the fixed anisotropic tensors used by every evaluation."""
        eps = self.porosity
        d = self.diameters

        # Principal-direction Darcy permeability and Forchheimer coefficients.
        # These are the scalar values seen in the material's own coordinate frame.
        self.k = eps**3 * d**2 / (150.0 * (1.0 - eps) ** 2)
        self.beta = 1.75 * (1.0 - eps) / (eps**3 * d)

        # Rotate the principal material axes into the lab/global frame.
        # NumPy trigonometric functions use radians, so the degrees are converted.
        self.R = self._rotation_z(np.deg2rad(30.0)) @ self._rotation_y(np.deg2rad(20.0))

        # Tensor form of the anisotropic material law in the global frame.
        # K controls the linear Darcy resistance; C controls the nonlinear term.
        self.K = self.R @ np.diag(self.k) @ self.R.T
        self.C = self.R @ np.diag(self.beta ** (2.0 / 3.0)) @ self.R.T

        # Precompute K^{-1} once because it is used at every quadrature point.
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
        # ----- Input validation -------------------------------------------------
        u = self._validate_vector(velocity, "velocity")

        # ----- Nonlinear anisotropic metric ------------------------------------
        # C u is the directionally weighted velocity. The scalar sqrt(u^T C u)
        # plays the role of anisotropic speed in the Forchheimer term.
        cu = self.C @ u
        speed_metric = np.sqrt(max(float(u @ cu), 0.0))

        # ----- Darcy contribution ---------------------------------------------
        # Linear resistance: mu K^{-1} u. This dominates as |u| approaches zero.
        darcy = self.mu * (self.K_inv @ u)

        # ----- Forchheimer contribution ---------------------------------------
        # Nonlinear resistance: rho sqrt(u^T C u) C u. This grows quadratically
        # along principal material directions.
        forchheimer = self.rho * speed_metric * cu

        # ----- Total pressure-driving force -----------------------------------
        return darcy + forchheimer

    def evaluate_batch(self, velocities: np.ndarray) -> np.ndarray:
        """Accept shape (n_points, 3) and return shape (n_points, 3)."""
        # The batch version applies the same formula to many local velocities,
        # e.g. velocities sampled at FEM quadrature points.
        u = self._validate_batch(velocities)

        # Row-wise equivalent of C @ u for each velocity vector.
        cu = u @ self.C.T

        # Row-wise u^T C u, clamped at zero to avoid tiny negative roundoff.
        speed_metric = np.sqrt(np.maximum(np.einsum("ij,ij->i", u, cu), 0.0))

        # Row-wise Darcy and Forchheimer contributions.
        darcy = self.mu * (u @ self.K_inv.T)
        forchheimer = self.rho * speed_metric[:, np.newaxis] * cu
        return darcy + forchheimer

    # ------------------------------------------------------------------
    # Rotation helpers.
    #
    # These construct standard right-handed 3D rotation matrices. They are
    # kept small and explicit so the material orientation is easy to audit.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Validation helpers.
    #
    # Keeping these separate makes the public methods read like the physics:
    # validate input, compute Darcy term, compute Forchheimer term, return sum.
    # ------------------------------------------------------------------
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
