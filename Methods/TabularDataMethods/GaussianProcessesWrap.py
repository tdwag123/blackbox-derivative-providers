import numpy as np
import torch
import gpytorch


class _KISSGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid_size, grid_bounds):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        kiss_kernel = gpytorch.kernels.GridInterpolationKernel(
            base_kernel,
            grid_size=grid_size,
            num_dims=2,
            grid_bounds=grid_bounds,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(kiss_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KISSGPFluxST:
    """
    hehehehaw
    """

    def __init__(
        self,
        s_data,
        T_data,
        q_data,
        grid_size=32,
        training_iter=75,
        learning_rate=0.1,
        ridge_strength=0.0,
        dtype=torch.float64,
    ):
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.ridge_strength = float(ridge_strength)

        s_data = np.asarray(s_data, dtype=float)
        T_data = np.asarray(T_data, dtype=float)
        q_data = np.asarray(q_data, dtype=float)

        if s_data.shape != T_data.shape:
            raise ValueError("s_data and T_data must have the same shape")
        if s_data.shape != q_data.shape:
            raise ValueError("s_data and q_data must have the same shape")

        X = np.column_stack([s_data.ravel(), T_data.ravel()])
        y = q_data.ravel()

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std == 0.0] = 1.0

        self.y_mean = y.mean()
        self.y_std = y.std()
        if self.y_std == 0.0:
            self.y_std = 1.0

        X_hat = (X - self.X_mean) / self.X_std
        y_hat = (y - self.y_mean) / self.y_std

        train_x = torch.as_tensor(X_hat, dtype=dtype, device=self.device)
        train_y = torch.as_tensor(y_hat, dtype=dtype, device=self.device)

        pad = 0.05
        lower = train_x.min(dim=0).values
        upper = train_x.max(dim=0).values
        width = upper - lower
        width[width == 0.0] = 1.0
        grid_bounds = tuple(
            (float(lower[j] - pad * width[j]), float(upper[j] + pad * width[j]))
            for j in range(2)
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device,
            dtype=dtype,
        )
        self.model = _KISSGPModel(
            train_x,
            train_y,
            self.likelihood,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
        ).to(device=self.device, dtype=dtype)

        self.train(training_iter=training_iter, learning_rate=learning_rate)

    def train(self, training_iter=75, learning_rate=0.1):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])
            loss = -mll(output, self.model.train_targets)
            if self.ridge_strength > 0.0:
                ridge_penalty = sum(
                    torch.sum(parameter**2)
                    for parameter in self.model.parameters()
                )
                loss = loss + self.ridge_strength * ridge_penalty
            loss.backward()
            optimizer.step()

    def evaluate(self, s_q, T_q, return_variance=False):
        """
        Return q_g, a_g = dq/ds, and b_g = dq/dT at quadrature states.
        """
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        original_shape = s_q.shape
        X_q = np.column_stack([s_q.ravel(), T_q.ravel()])
        X_hat = (X_q - self.X_mean) / self.X_std

        x_tensor = torch.as_tensor(
            X_hat,
            dtype=self.dtype,
            device=self.device,
        )
        x_tensor.requires_grad_(True)

        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            latent_pred = self.model(x_tensor)
            observed_pred = self.likelihood(latent_pred)
            mean_hat = observed_pred.mean

        grad_hat = torch.autograd.grad(
            mean_hat.sum(),
            x_tensor,
            retain_graph=False,
            create_graph=False,
        )[0]

        q = self.y_mean + self.y_std * mean_hat.detach().cpu().numpy()
        deriv_hat = grad_hat.detach().cpu().numpy()
        a = self.y_std * deriv_hat[:, 0] / self.X_std[0]
        b = self.y_std * deriv_hat[:, 1] / self.X_std[1]

        q = q.reshape(original_shape)
        a = a.reshape(original_shape)
        b = b.reshape(original_shape)

        if return_variance:
            variance = (self.y_std**2) * observed_pred.variance.detach().cpu().numpy()
            return q, a, b, variance.reshape(original_shape)

        return q, a, b


def example_problem():
    import matplotlib.pyplot as plt

    def q_true(s, T):
        base = -((1.0 + 0.20 * T**2) + 0.04 * s**2) * s
        oscillation = 0.35 * np.sin(2.5 * s) * np.exp(-0.5 * (T - 1.4) ** 2)
        transition = -0.45 * np.tanh(3.0 * (s - 0.45)) * np.exp(
            -2.0 * (T - 2.1) ** 2
        )
        return base + oscillation + transition

    def dq_ds_true(s, T):
        base = -(1.0 + 0.20 * T**2) - 0.12 * s**2
        oscillation = 0.875 * np.cos(2.5 * s) * np.exp(-0.5 * (T - 1.4) ** 2)
        u = 3.0 * (s - 0.45)
        sech2 = 1.0 / np.cosh(u) ** 2
        transition = -1.35 * sech2 * np.exp(-2.0 * (T - 2.1) ** 2)
        return base + oscillation + transition

    def dq_dT_true(s, T):
        base = -0.4 * T * s
        oscillation = (
            -0.35
            * (T - 1.4)
            * np.sin(2.5 * s)
            * np.exp(-0.5 * (T - 1.4) ** 2)
        )
        transition = (
            1.8
            * (T - 2.1)
            * np.tanh(3.0 * (s - 0.45))
            * np.exp(-2.0 * (T - 2.1) ** 2)
        )
        return base + oscillation + transition

    rng = np.random.default_rng(0)
    n_data = 900
    s_data = rng.uniform(-2.0, 2.0, n_data)
    T_data = rng.uniform(0.0, 3.0, n_data)
    noise_scale = 0.035 * (1.0 + 0.25 * np.abs(s_data))
    q_data = q_true(s_data, T_data) + noise_scale * rng.standard_normal(n_data)

    model = KISSGPFluxST(
        s_data,
        T_data,
        q_data,
        grid_size=40,
        training_iter=70,
        learning_rate=0.08,
    )

    s_q = np.array([-1.25, -0.50, 0.10, 0.80, 1.50])
    T_q = np.array([0.25, 0.75, 1.25, 2.00, 2.75])

    q_g, a_g, b_g, variance = model.evaluate(s_q, T_q, return_variance=True)

    q_error = np.linalg.norm(q_g - q_true(s_q, T_q), ord=np.inf)
    a_error = np.linalg.norm(a_g - dq_ds_true(s_q, T_q), ord=np.inf)
    b_error = np.linalg.norm(b_g - dq_dT_true(s_q, T_q), ord=np.inf)

    print("KISS-GP derivative-provider test for a richer q = q(s, T)")
    print(f"max q error:     {q_error:.3e}")
    print(f"max dq/ds error: {a_error:.3e}")
    print(f"max dq/dT error: {b_error:.3e}")
    print("\nquadrature point results:")
    print("s_q | T_q | q_g | a_g=dq/ds | b_g=dq/dT | variance")
    for i in range(len(s_q)):
        print(
            f"{s_q[i]: .3f} | {T_q[i]: .3f} | {q_g[i]: .6f} | "
            f"{a_g[i]: .6f} | {b_g[i]: .6f} | {variance[i]: .3e}"
        )

    T_slice = 1.5
    s_fine = np.linspace(-2.0, 2.0, 300)
    T_fine = np.full_like(s_fine, T_slice)
    q_mean, a_mean, b_mean, q_var = model.evaluate(s_fine, T_fine, return_variance=True)
    q_std = np.sqrt(np.maximum(q_var, 0.0))

    slice_width = 0.08
    near_slice = np.abs(T_data - T_slice) < slice_width

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(s_fine, q_true(s_fine, T_fine), "k--", label="True flux", linewidth=1.5)
    axes[0].plot(s_fine, q_mean, "b-", label="KISS-GP mean", linewidth=2)
    axes[0].fill_between(
        s_fine,
        q_mean - 2 * q_std,
        q_mean + 2 * q_std,
        color="tab:blue",
        alpha=0.2,
        label="2 std. dev.",
    )
    axes[0].plot(
        s_data[near_slice],
        q_data[near_slice],
        "k.",
        alpha=0.45,
        label=f"training data near T={T_slice}",
    )
    axes[0].set_ylabel("q(s, T)")
    axes[0].set_title(f"KISS-GP surrogate and derivatives at T = {T_slice}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s_fine, dq_ds_true(s_fine, T_fine), "k--", label="True dq/ds")
    axes[1].plot(s_fine, a_mean, "r-", label="KISS-GP dq/ds", linewidth=2)
    axes[1].set_ylabel("a = dq/ds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(s_fine, dq_dT_true(s_fine, T_fine), "k--", label="True dq/dT")
    axes[2].plot(s_fine, b_mean, "g-", label="KISS-GP dq/dT", linewidth=2)
    axes[2].set_xlabel("s = dT/dx")
    axes[2].set_ylabel("b = dq/dT")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig("kiss_gp_flux_slice.png", dpi=150)
    print("\nPlot saved as 'kiss_gp_flux_slice.png'")


if __name__ == "__main__":
    example_problem()
