import numpy as np
from scipy.interpolate import Rbf


class RBFDerivativeProviderST:
    """
    RBF derivative provider for a 1D constitutive table q = phi(s, T).

    uses scipy Rbf
    """

    def __init__(
        self,
        s_data,
        T_data,
        q_data,
        function="gaussian",
        epsilon=None,
        smooth=0.0,
    ):
        self.s_data = np.asarray(s_data, dtype=float)
        self.T_data = np.asarray(T_data, dtype=float)
        self.q_data = np.asarray(q_data, dtype=float)
        self.function = function

        if self.s_data.shape != self.T_data.shape:
            raise ValueError("s_data and T_data must have the same shape")
        if self.s_data.shape != self.q_data.shape:
            raise ValueError("s_data and q_data must have the same shape")

        self.rbf = Rbf(
            self.s_data,
            self.T_data,
            self.q_data,
            function=function,
            epsilon=epsilon,
            smooth=smooth,
        )

        self.epsilon = self.rbf.epsilon
        self.centers = np.column_stack([self.s_data, self.T_data])
        self.weights = np.asarray(self.rbf.nodes, dtype=float)

    def evaluate(self, s_q, T_q):
        """
        Return q_g, a_g = dq/ds, and b_g = dq/dT at quadrature states.
        """
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        q_g = self.rbf(s_q, T_q)
        a_g, b_g = self.derivatives(s_q, T_q)

        return q_g, a_g, b_g

    def derivatives(self, s_q, T_q):
        """
        Evaluate analytic RBF surrogate derivatives.

        For the Gaussian RBF,

            phi(r) = exp(-(r / epsilon)^2)

        and

            d phi / d s = -2 (s - s_j) / epsilon^2 * phi(r)
            d phi / d T = -2 (T - T_j) / epsilon^2 * phi(r)

        for each RBF center (s_j, T_j).
        """
        if self.function != "gaussian":
            raise NotImplementedError(
                "analytic derivatives are currently implemented for function='gaussian'"
            )

        original_shape = np.shape(s_q)
        s_flat = np.asarray(s_q, dtype=float).ravel()
        T_flat = np.asarray(T_q, dtype=float).ravel()

        ds = s_flat[:, None] - self.centers[None, :, 0]
        dT = T_flat[:, None] - self.centers[None, :, 1]
        r2 = ds**2 + dT**2

        phi = np.exp(-r2 / self.epsilon**2)

        dphi_ds = -2.0 * ds / self.epsilon**2 * phi
        dphi_dT = -2.0 * dT / self.epsilon**2 * phi

        a_flat = dphi_ds @ self.weights
        b_flat = dphi_dT @ self.weights

        return a_flat.reshape(original_shape), b_flat.reshape(original_shape)


def example_problem():
    def q_true(s, T):
        return -((1.0 + 0.25 * T**2) + 0.05 * s**2) * s

    def dq_ds_true(s, T):
        return -(1.0 + 0.25 * T**2) - 0.15 * s**2

    def dq_dT_true(s, T):
        return -0.5 * T * s

    s_grid = np.linspace(-2.0, 2.0, 15)
    T_grid = np.linspace(0.0, 3.0, 15)

    S, T = np.meshgrid(s_grid, T_grid, indexing="ij")
    s_data = S.ravel()
    T_data = T.ravel()
    q_data = q_true(s_data, T_data)

    model = RBFDerivativeProviderST(
        s_data,
        T_data,
        q_data,
        function="gaussian",
        epsilon=0.8,
        smooth=0.0,
    )

    s_q = np.array([-1.25, -0.50, 0.10, 0.80, 1.50])
    T_q = np.array([0.25, 0.75, 1.25, 2.00, 2.75])

    q_g, a_g, b_g = model.evaluate(s_q, T_q)

    q_error = np.linalg.norm(q_g - q_true(s_q, T_q), ord=np.inf)
    a_error = np.linalg.norm(a_g - dq_ds_true(s_q, T_q), ord=np.inf)
    b_error = np.linalg.norm(b_g - dq_dT_true(s_q, T_q), ord=np.inf)

    print("SciPy Rbf derivative-provider test for q = q(s, T)")
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
