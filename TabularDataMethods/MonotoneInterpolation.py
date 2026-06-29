import numpy as np
from scipy.interpolate import PchipInterpolator


class PchipFlux1D:
    """Monotone cubic interpolation model for a flux table q = q(s)."""

    def __init__(self, s_grid, q_grid, extrapolate=False):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.interp = PchipInterpolator(
            self.s_grid,
            self.q_grid,
            extrapolate=extrapolate,
        )

    def evaluate(self, s_q):
        s_q = np.asarray(s_q, dtype=float)
        q_g = self.interp(s_q)
        a_g = self.interp(s_q, nu=1)
        b_g = np.zeros_like(a_g)
        return q_g, a_g, b_g


class PchipFluxST:
    """Tensor-product PCHIP model for a 1D FEM flux table q = q(s, T)."""

    def __init__(self, s_grid, T_grid, q_grid, extrapolate=False):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.extrapolate = extrapolate

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if self.q_grid.shape != expected_shape:
            raise ValueError(
                f"q_grid must have shape {expected_shape}, got {self.q_grid.shape}"
            )

        self.s_interps = [
            PchipInterpolator(
                self.s_grid,
                self.q_grid[:, j],
                extrapolate=extrapolate,
            )
            for j in range(len(self.T_grid))
        ]

    def evaluate_one(self, s, T):
        q_at_T_grid = np.array([interp(s) for interp in self.s_interps])
        dqds_at_T_grid = np.array([interp(s, nu=1) for interp in self.s_interps])

        T_interp_for_q = PchipInterpolator(
            self.T_grid,
            q_at_T_grid,
            extrapolate=self.extrapolate,
        )
        T_interp_for_dqds = PchipInterpolator(
            self.T_grid,
            dqds_at_T_grid,
            extrapolate=self.extrapolate,
        )

        q_g = T_interp_for_q(T)
        a_g = T_interp_for_dqds(T)
        b_g = T_interp_for_q(T, nu=1)

        return q_g, a_g, b_g

    def evaluate(self, s_q, T_q):
        """Return q_g, a_g = dq/ds, and b_g = dq/dT at quadrature states."""
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        q_g = np.zeros_like(s_q, dtype=float)
        a_g = np.zeros_like(s_q, dtype=float)
        b_g = np.zeros_like(s_q, dtype=float)

        for i in np.ndindex(s_q.shape):
            q_g[i], a_g[i], b_g[i] = self.evaluate_one(s_q[i], T_q[i])

        return q_g, a_g, b_g


def example_problem():
    import matplotlib.pyplot as plt

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

    model = PchipFluxST(s_grid, T_grid, q_grid)

    s_q = np.array([-1.25, -0.50, 0.10, 0.80, 1.50])
    T_q = np.array([0.25, 0.75, 1.25, 2.00, 2.75])

    q_g, a_g, b_g = model.evaluate(s_q, T_q)

    q_error = np.linalg.norm(q_g - q_true(s_q, T_q), ord=np.inf)
    a_error = np.linalg.norm(a_g - dq_ds_true(s_q, T_q), ord=np.inf)
    b_error = np.linalg.norm(b_g - dq_dT_true(s_q, T_q), ord=np.inf)

    print("SciPy PCHIP flux test for q = q(s, T)")
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

    T_slice = 1.5
    s_fine = np.linspace(s_grid[0], s_grid[-1], 300)
    T_fine = np.full_like(s_fine, T_slice)
    q_pchip_fine, _, _ = model.evaluate(s_fine, T_fine)
    q_true_fine = q_true(s_fine, T_fine)
    q_table_slice = q_true(s_grid, T_slice)

    plt.figure(figsize=(10, 5))
    plt.plot(s_fine, q_true_fine, "k--", label="True flux", linewidth=1.5)
    plt.plot(s_fine, q_pchip_fine, "r-", label="PCHIP interpolation", linewidth=2)
    plt.plot(s_grid, q_table_slice, "bo", label="Table samples", markersize=4)
    plt.xlabel("s = dT/dx")
    plt.ylabel("q(s, T)")
    plt.title(f"PCHIP interpolation at T = {T_slice}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pchip_interpolation_test.png", dpi=150)
    print("\nPlot saved as 'pchip_interpolation_test.png'")


if __name__ == "__main__":
    example_problem()
