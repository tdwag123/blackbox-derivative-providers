"""
Black-box oracle for anisotropic Darcy-Forchheimer flow.

Given a pressure gradient grad_p in R^d (d = 2 or 3), returns superficial
velocity u satisfying eq (3) in the reference text:

    -grad_p = mu K^{-1} u
              + rho * sqrt(u^T B^(2/3) u) * B^(2/3) u.

The implementation assumes K and B share the same principal material axes,
as in the supplied model summary. This reduces the nonlinear vector inversion
to a monotone scalar root solve.

Noise is applied only to the returned velocity observation. The deterministic
velocity can always be recovered with noise_level="none".

TO DO: look into using scipy optimizer bisector versus manual bisection.

"""

import numpy as np

_NOISE_LEVELS = {
    "none": 0.0,
    "low": 0.005,    # 0.5% relative observation noise
    "medium": 0.02,  # 2%
    "high": 0.05,    # 5%
}

def rotation_2d(theta_deg: float):
    """Return a 2D material-to-global rotation matrix R."""
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def rotation_3d_xyz(
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    gamma_deg: float = 0.0,
):
    """Return R = Rz(gamma) @ Ry(beta) @ Rx(alpha); Columns of R are the principal 
    material axes expressed in global coordinates."""

    a, b, g = np.deg2rad([alpha_deg, beta_deg, gamma_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(a), -np.sin(a)],
        [0.0, np.sin(a),  np.cos(a)],
    ])
    Ry = np.array([
        [ np.cos(b), 0.0, np.sin(b)],
        [0.0,        1.0, 0.0],
        [-np.sin(b), 0.0, np.cos(b)],
    ])
    Rz = np.array([
        [np.cos(g), -np.sin(g), 0.0],
        [np.sin(g),  np.cos(g), 0.0],
        [0.0,        0.0,       1.0],
    ])
    return Rz @ Ry @ Rx


def ergun_principal_parameters(
    hydraulic_lengths_m,
    porosity: float = 0.40,
):
    """
    Build a physically scaled synthetic anisotropic benchmark from Ergun's
    packed-bed coefficients, applied independently along principal directions.

    This directional extension is a benchmark construction, not a claim that
    a particular anisotropic material obeys Ergun independently in each axis.

    Returns
    -------
    k : ndarray
        Principal permeabilities [m^2].
    beta : ndarray
        Principal Forchheimer coefficients [1/m].
    """
    d = np.asarray(hydraulic_lengths_m, dtype=float)
    if d.ndim != 1 or d.size not in (2, 3):
        raise ValueError("hydraulic_lengths_m must contain 2 or 3 values.")
    if np.any(d <= 0.0):
        raise ValueError("All hydraulic lengths must be positive.")
    if not (0.0 < porosity < 1.0):
        raise ValueError("porosity must lie strictly between 0 and 1.")

    eps = float(porosity)
    k = d**2 * eps**3 / (150.0 * (1.0 - eps)**2)
    beta = 1.75 * (1.0 - eps) / (d * eps**3)
    return k, beta


class AnisotropicDarcyForchheimerOracle:
    """
    Configurable 2D/3D black-box map grad(p) -> superficial velocity.

    Parameters
    ----------
    dim : {2, 3}
        Spatial dimension.
    k_principal : array_like
        Principal permeability values [m^2].
    beta_principal : array_like
        Principal Forchheimer coefficients [1/m].
    rotation : ndarray, optional
        Orthogonal material-to-global rotation matrix R.
    mu : float
        Dynamic viscosity [Pa s].
    rho : float
        Fluid density [kg/m^3].
    noise_level : {"none", "low", "medium", "high"}
        Relative Gaussian observation-noise level.
    seed : int, optional
        Random seed for reproducible noisy observations.
    bisection_tol : float
        Relative scalar-root tolerance.
    max_iter : int
        Maximum bisection iterations.
    """

    def __init__(
        self,
        dim: int,
        k_principal,
        beta_principal,
        *,
        rotation=None,
        mu: float = 1.0016e-3,
        rho: float = 998.2,
        noise_level: str = "none",
        seed: int | None = 0,
        bisection_tol: float = 1e-12,
        max_iter: int = 100,
    ):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        self.dim = int(dim)

        self.k = np.asarray(k_principal, dtype=float)
        self.beta = np.asarray(beta_principal, dtype=float)
        if self.k.shape != (self.dim,) or self.beta.shape != (self.dim,):
            raise ValueError("k_principal and beta_principal must have length dim.")
        if np.any(self.k <= 0.0) or np.any(self.beta <= 0.0):
            raise ValueError("All permeability and Forchheimer coefficients must be positive.")
        if mu <= 0.0 or rho <= 0.0:
            raise ValueError("mu and rho must be positive.")

        self.mu = float(mu)
        self.rho = float(rho)

        if rotation is None:
            self.R = np.eye(self.dim)
        else:
            self.R = np.asarray(rotation, dtype=float)
            if self.R.shape != (self.dim, self.dim):
                raise ValueError("rotation must have shape (dim, dim).")
            if not np.allclose(self.R.T @ self.R, np.eye(self.dim), atol=1e-10):
                raise ValueError("rotation must be orthogonal.")

        noise_level = str(noise_level).lower()
        if noise_level not in _NOISE_LEVELS:
            raise ValueError(f"noise_level must be one of {tuple(_NOISE_LEVELS)}.")
        self.noise_level = noise_level
        self.noise_fraction = _NOISE_LEVELS[noise_level]
        self.rng = np.random.default_rng(seed)

        self.bisection_tol = float(bisection_tol)
        self.max_iter = int(max_iter)

        # Principal-coordinate coefficients:
        # a_i = mu/k_i, c_i = beta_i^(2/3).
        self.a = self.mu / self.k
        self.c = self.beta ** (2.0 / 3.0)

        # Full tensors, useful for verification / downstream FEM code.
        self.K = self.R @ np.diag(self.k) @ self.R.T
        self.B = self.R @ np.diag(self.beta) @ self.R.T
        self.B23 = self.R @ np.diag(self.c) @ self.R.T

    def _solve_one(self, grad_p: np.ndarray):
        """
        Deterministically invert one pressure-gradient vector.
        """
        grad_p = np.asarray(grad_p, dtype=float)
        if grad_p.shape != (self.dim,):
            raise ValueError(f"Each pressure gradient must have shape ({self.dim},).")

        # Driving force in global coordinates and then in material coordinates.
        F_global = -grad_p
        f = self.R.T @ F_global

        if np.linalg.norm(f) == 0.0:
            return np.zeros(self.dim)

        # For fixed eta = sqrt(v^T C v), each principal component is explicit:
        #
        #   v_i(eta) = f_i / (a_i + rho * eta * c_i).
        #
        # eta satisfies
        #
        #   h(eta) = eta^2
        #            - sum_i c_i f_i^2/(a_i + rho*eta*c_i)^2 = 0.
        #
        # h is strictly increasing for eta >= 0. The Darcy solution provides
        # a rigorous upper bracket because adding Forchheimer resistance can
        # only reduce each |v_i|.
        eta_hi = np.sqrt(np.sum(self.c * (f / self.a) ** 2))
        if eta_hi == 0.0:
            return np.zeros(self.dim)

        def h(eta):
            denom = self.a + self.rho * eta * self.c
            return eta * eta - np.sum(self.c * f * f / (denom * denom))

        lo, hi = 0.0, eta_hi
        h_lo = h(lo)
        h_hi = h(hi)

        if h_lo > 0.0 or h_hi < 0.0:
            raise RuntimeError("Failed to bracket the unique scalar Darcy-Forchheimer root.")

        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            h_mid = h(mid)

            if h_mid > 0.0:
                hi = mid
            else:
                lo = mid

            if (hi - lo) <= self.bisection_tol * max(1.0, hi):
                break
        else:
            raise RuntimeError("Scalar Darcy-Forchheimer inversion did not converge.")

        eta = 0.5 * (lo + hi)
        v = f / (self.a + self.rho * eta * self.c)
        return self.R @ v

    def evaluate(self, pressure_gradient, *, return_clean: bool = False):
        """
        Evaluate the black-box oracle.

        Parameters
        ----------
        pressure_gradient : array_like, shape (dim,) or (..., dim)
            Global pressure gradient grad(p) [Pa/m].
        return_clean : bool
            If True, return (u_observed, u_clean).

        Returns
        -------
        u_observed : ndarray
            Superficial velocity [m/s].
        u_clean : ndarray, optional
            Noise-free superficial velocity [m/s].
        """
        g = np.asarray(pressure_gradient, dtype=float)
        if g.shape == (self.dim,):
            clean = self._solve_one(g)
        elif g.ndim >= 2 and g.shape[-1] == self.dim:
            flat = g.reshape(-1, self.dim)
            clean = np.stack([self._solve_one(row) for row in flat], axis=0)
            clean = clean.reshape(g.shape)
        else:
            raise ValueError(
                f"pressure_gradient must have shape ({self.dim},) or (..., {self.dim})."
            )

        if self.noise_fraction == 0.0:
            observed = clean.copy()
        else:
            # Isotropic measurement error in velocity space, scaled by the
            # local velocity RMS. This perturbs observations, not the physical law.
            speed_scale = np.linalg.norm(clean, axis=-1, keepdims=True) / np.sqrt(self.dim)
            sigma = self.noise_fraction * speed_scale
            observed = clean + sigma * self.rng.standard_normal(clean.shape)

        if return_clean:
            return observed, clean
        return observed

    def constitutive_residual(self, pressure_gradient, velocity):
        """
        Return residual of the original global constitutive law.

        A correct clean inverse has residual approximately zero.
        """
        g = np.asarray(pressure_gradient, dtype=float)
        u = np.asarray(velocity, dtype=float)
        F = -g
        inertial_speed = np.sqrt(float(u @ self.B23 @ u))
        rhs = self.mu * np.linalg.solve(self.K, u) + self.rho * inertial_speed * (self.B23 @ u)
        return rhs - F


if __name__ == "__main__":
    # Example physically scaled benchmark:
    # water near 20 C; porosity 0.40; anisotropic hydraulic lengths.
    dim = 3
    k, beta = ergun_principal_parameters(
        hydraulic_lengths_m=[1.0e-3, 0.70e-3, 0.45e-3],
        porosity=0.40,
    )
    R = rotation_3d_xyz(alpha_deg=10.0, beta_deg=20.0, gamma_deg=30.0)

    oracle = AnisotropicDarcyForchheimerOracle(
        dim=dim,
        k_principal=k,
        beta_principal=beta,
        rotation=R,
        noise_level="none",
    )

    grad_p = np.array([1.0e5, -5.0e4, 2.5e4])  # Pa/m
    u = oracle.evaluate(grad_p)
    res = oracle.constitutive_residual(grad_p, u)

    print("k [m^2] =", k)
    print("beta [1/m] =", beta)
    print("grad p [Pa/m] =", grad_p)
    print("u [m/s] =", u)
    print("residual norm =", np.linalg.norm(res))