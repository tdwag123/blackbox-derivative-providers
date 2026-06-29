import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline


class CubicSplineFlux1D:
    """Cubic spline model for a flux table q = q(s)."""

    def __init__(self, s_grid, q_grid, bc_type="not-a-knot"):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.spline = CubicSpline(self.s_grid, self.q_grid, bc_type=bc_type)

    def evaluate(self, s_q):
        s_q = np.asarray(s_q, dtype=float)
        q_g = self.spline(s_q)
        a_g = self.spline(s_q, 1)
        b_g = np.zeros_like(a_g)
        return q_g, a_g, b_g


class CubicSplineFluxST:
    """Tensor-product cubic spline model for a 1D FEM flux table q = q(s, T)."""

    def __init__(self, s_grid, T_grid, q_grid, kx=3, ky=3):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if self.q_grid.shape != expected_shape:
            raise ValueError(
                f"q_grid must have shape {expected_shape}, got {self.q_grid.shape}"
            )

        # RectBivariateSpline is the spline for 2D interpolation
        self.spline = RectBivariateSpline(
            self.s_grid,
            self.T_grid,
            self.q_grid,
            kx=kx,
            ky=ky,
        )

    def evaluate(self, s_q, T_q):
        """Return q_g, a_g = dq/ds, and b_g = dq/dT at quadrature states."""
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        q_g = self.spline.ev(s_q, T_q)
        a_g = self.spline.ev(s_q, T_q, dx=1, dy=0)
        b_g = self.spline.ev(s_q, T_q, dx=0, dy=1)

        return q_g, a_g, b_g


def example_problem():
    def q_true(s, T):
        return -((1.0 + 0.25 * T**2) + 0.05 * s**2) * s

    def dq_ds_true(s, T):
        return -(1.0 + 0.25 * T**2) - 0.15 * s**2

    def dq_dT_true(s, T):
        return -0.5 * T * s

    s_grid = np.linspace(-2.0, 2.0, 25)
    T_grid = np.linspace(0.0, 3.0, 25)

    S, T = np.meshgrid(s_grid, T_grid, indexing="ij")
    q_grid = q_true(S, T)

    model = CubicSplineFluxST(s_grid, T_grid, q_grid)

    s_q = np.array([-1.25, -0.50, 0.10, 0.80, 1.50])
    T_q = np.array([0.25, 0.75, 1.25, 2.00, 2.75])

    q_g, a_g, b_g = model.evaluate(s_q, T_q)

    q_error = np.linalg.norm(q_g - q_true(s_q, T_q), ord=np.inf)
    a_error = np.linalg.norm(a_g - dq_ds_true(s_q, T_q), ord=np.inf)
    b_error = np.linalg.norm(b_g - dq_dT_true(s_q, T_q), ord=np.inf)

    print("SciPy cubic spline flux test for q = q(s, T)")
    print(f"max q error:     {q_error:.3e}")
    print(f"max dq/ds error: {a_error:.3e}")
    print(f"max dq/dT error: {b_error:.3e}")
    print("\nquadrature point results:")
    print("s_q | T_q | q_g | a_g=dq/ds | b_g=dq/dT")
    for i in range(len(s_q)):
        print(
            f"{s_q[i]: .3f} | {T_q[i]: .3f} | {q_g[i]: .6f} | "
            f"{a_g[i]: .6f} | {b_g[i]: .6f}"
        )


if __name__ == "__main__":
    example_problem()