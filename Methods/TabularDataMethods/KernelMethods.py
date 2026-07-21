import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve


class KernelDerivativeProviderST:
    """
    Kernel ridge derivative provider for a 1D constitutive table q = phi(s, T).

    The fitted function is

        f(x) = sum_j weights[j] K(x, x_j)

    Supported kernels are Gaussian RBF,

        K(x, u) = exp(-||x - u||^2 / epsilon^2).

    and Matern 5/2,

        K(x, u) = (1 + rho + rho^2 / 3) exp(-rho),
        rho = sqrt(5) ||x - u|| / epsilon.

    Ridge regularization follows the RKHS objective

        (1 / n) sum_i (f(x_i) - y_i)^2 + lambda ||f||_K^2,

    which gives the finite system

        (K + n * lambda I) weights = y.
    """

    def __init__(
        self,
        s_data,
        T_data,
        q_data,
        function="gaussian",
        epsilon=None,
        smooth=0.0,
        ridge_strength=0.0,
        jitter=1.0e-10,
    ):
        self.s_data = np.asarray(s_data, dtype=float)
        self.T_data = np.asarray(T_data, dtype=float)
        self.q_data = np.asarray(q_data, dtype=float)
        self.function = function
        self.ridge_strength = float(ridge_strength)
        self.smooth = float(smooth)
        self.jitter = float(jitter)

        if self.s_data.shape != self.T_data.shape:
            raise ValueError("s_data and T_data must have the same shape")
        if self.s_data.shape != self.q_data.shape:
            raise ValueError("s_data and q_data must have the same shape")
        if self.function not in {"gaussian", "matern52"}:
            raise NotImplementedError(
                "kernel ridge derivatives are implemented for 'gaussian' and 'matern52'"
            )

        self.centers = np.column_stack([self.s_data, self.T_data])
        self.n_samples = self.centers.shape[0]
        self.epsilon = self._choose_epsilon(epsilon)
        self.lambda_ = self.ridge_strength + self.smooth

        self.K_train = self._kernel(self.centers, self.centers)
        self.system_matrix = self._regularized_kernel_matrix(self.jitter)
        self.condition_number = float(np.linalg.cond(self.system_matrix))
        self.weights = self._solve_krr_system()

    def _regularized_kernel_matrix(self, jitter):
        diagonal_shift = self.n_samples * self.lambda_ + float(jitter)
        return self.K_train + diagonal_shift * np.eye(self.n_samples)

    def _solve_krr_system(self):
        try:
            factor = cho_factor(self.system_matrix, lower=True, check_finite=False)
            return cho_solve(factor, self.q_data, check_finite=False)
        except LinAlgError:
            return np.linalg.solve(self.system_matrix, self.q_data)

    def _choose_epsilon(self, epsilon):
        if epsilon is not None:
            return float(epsilon)

        if self.n_samples < 2:
            return 1.0

        diffs = self.centers[:, None, :] - self.centers[None, :, :]
        distances = np.sqrt(np.sum(diffs**2, axis=2))
        nonzero_distances = distances[distances > 0.0]
        if nonzero_distances.size == 0:
            return 1.0
        return float(np.median(nonzero_distances))

    def _kernel(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        diffs = X[:, None, :] - Y[None, :, :]
        r2 = np.sum(diffs**2, axis=2)
        if self.function == "gaussian":
            return np.exp(-r2 / self.epsilon**2)

        r = np.sqrt(r2)
        rho = np.sqrt(5.0) * r / self.epsilon
        return (1.0 + rho + rho**2 / 3.0) * np.exp(-rho)

    def evaluate(self, s_q, T_q):
        """
        Return q_g, a_g = dq/ds, and b_g = dq/dT at quadrature states.
        """
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        original_shape = np.shape(s_q)
        Xq = np.column_stack([s_q.ravel(), T_q.ravel()])
        q_g = self._kernel(Xq, self.centers) @ self.weights
        a_g, b_g = self.derivatives(s_q, T_q)

        return q_g.reshape(original_shape), a_g, b_g

    def derivatives(self, s_q, T_q):
        """
        Evaluate analytic kernel surrogate derivatives.

        For the Gaussian RBF,

            phi(r) = exp(-(r / epsilon)^2)

        and

            d phi / d s = -2 (s - s_j) / epsilon^2 * phi(r)
            d phi / d T = -2 (T - T_j) / epsilon^2 * phi(r)

        for each kernel center (s_j, T_j).
        """
        original_shape = np.shape(s_q)
        s_flat = np.asarray(s_q, dtype=float).ravel()
        T_flat = np.asarray(T_q, dtype=float).ravel()

        ds = s_flat[:, None] - self.centers[None, :, 0]
        dT = T_flat[:, None] - self.centers[None, :, 1]
        r2 = ds**2 + dT**2

        if self.function == "gaussian":
            phi = np.exp(-r2 / self.epsilon**2)

            dphi_ds = -2.0 * ds / self.epsilon**2 * phi
            dphi_dT = -2.0 * dT / self.epsilon**2 * phi
        elif self.function == "matern52":
            r = np.sqrt(r2)
            rho = np.sqrt(5.0) * r / self.epsilon
            derivative_factor = (
                -5.0 / (3.0 * self.epsilon**2)
                * (1.0 + rho)
                * np.exp(-rho)
            )
            dphi_ds = derivative_factor * ds
            dphi_dT = derivative_factor * dT
        else:
            raise NotImplementedError(
                "analytic derivatives are implemented for 'gaussian' and 'matern52'"
            )

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

    model = KernelDerivativeProviderST(
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

    print("Gaussian kernel ridge derivative-provider test for q = q(s, T)")
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


RBFDerivativeProviderST = KernelDerivativeProviderST


if __name__ == "__main__":
    example_problem()
