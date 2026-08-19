"""Random Fourier feature GP surrogate for multidimensional flux laws.

This is the fast approximate counterpart to ``baseGP.py``. It keeps the same
``GPFluxST`` public interface and supports local posterior updates from new
q-only samples.
"""

import numpy as np

class GPFluxST:
    """Approximate vector-valued GP for q(s, T) and its fitted derivatives."""

    def __init__(self, s_train, T_train, q_train, *,
                 learn_neg_flux=True,
                 jitter=1e-8,
                 kernel_variance=1.0,
                 lengthscale=2.0,
                 noise_variance=1e-2,
                 n_rff_features=500,
                 alpha=0.0,
                 p=2.0,
                 random_state=0,
                 max_cache_size=None):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.jitter = self._validate_nonnegative_float(jitter, "jitter")
        self.kernel_variance = self._validate_variance(kernel_variance, "kernel_variance")
        self.lengthscale = lengthscale
        self.lengthscales = None
        self.noise_variance = self._validate_variance(noise_variance, "noise_variance")
        self.n_rff_features = self._validate_n_rff_features(n_rff_features)
        self.alpha = self._validate_nonnegative_float(alpha, "alpha")
        self.p = self._validate_nonnegative_float(p, "p")
        self.random_state = random_state
        self.max_cache_size = self._validate_max_cache_size(max_cache_size)

        self.omega = None
        self.offset = None

        self.input_dim = None

        self.X_train = None
        self.y_train = None
        self.Phi_train = None
        
        self.posterior_updates = 0
        self.total_points_added = 0
        self.total_points_dropped = 0
        self.fit(s_train, T_train, q_train)

    @staticmethod
    def _validate_nonnegative_float(value, name):
        value = float(value)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value

    @staticmethod
    def _validate_variance(variance, name):
        variance = float(variance)
        if variance <= 0.0 or not np.isfinite(variance):
            raise ValueError(f"{name} must be finite and positive.")
        return variance

    @staticmethod
    def _validate_n_rff_features(n_rff_features):
        n_features = int(n_rff_features)
        if n_features != n_rff_features:
            raise(ValueError("n_rff_features must be integer."))
        if n_features <= 0:
            raise ValueError("n_rff_features must be positive.")
        return n_features

    @staticmethod
    def _validate_lengthscale(lengthscale, input_dim):
        lengthscales = np.asarray(lengthscale, dtype=float).reshape(-1)

        if lengthscales.size == 1:
            lengthscales = np.full(input_dim, lengthscales.item())
        if lengthscales.size != input_dim:
            raise ValueError(
                f"lengthscale must be a positive scalar or length-{input_dim} array."
            )
        if np.any(~np.isfinite(lengthscales)) or np.any(lengthscales <= 0.0):
            raise ValueError("lengthscale entries must be finite and positive.")
        return lengthscales

    @staticmethod
    def _matrix(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]

        if values.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D array.")
        if values.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one point.")
        if values.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        return values

    @classmethod
    def _training_arrays(cls, s, T, q):
        s = cls._matrix(s, "s")
        T = cls._matrix(T, "T")
        q = cls._matrix(q, "q")
        return s, T, q

    def _query_features(self, s, T):
        s = self._matrix(s, "s")
        T = self._matrix(T, "T")

        if s.shape[0] != T.shape[0]:
            raise ValueError("s and T must have same number of rows.")
        if s.shape[1] != self.s_dim:
            raise ValueError(f"s must have {self.s_dim} columns.")
        if T.shape[1] != self.T_dim:
            raise ValueError(f"T must have {self.T_dim} columns.")

        X_raw = np.concatenate((s, T), axis=1)
        X = (X_raw - self.x_mean) / self.x_scale
        Phi = self._feature_map(X)

        return X, Phi

    @staticmethod
    def _validate_max_cache_size(value):
        if value is None:
            return 0
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("max_cache_size must be an integer or None.")

        size = int(value)
        if size != value:
            raise ValueError("max_cache_size must be an integer.")
        if size <= 0:
            raise ValueError("max_cache_size must be positive.")
        return size

    def _sample_rff_frequencies(self):
        """
        Based on notation from Tracy's paper, to approximate Matern-5/2 kernel, we sample weights from Student's t distribution
        p(w) = frac{Gamma((nu+d)/2)} / {(sigma*pi^{d/2}nu^{d/2}Gamma(nu/2))} (1+frac{1}{sigma^2 nu} ||w||^2)^{-(nu+d)/2}
        NOTE: sigma acts as inverse lengthscale, 1/lengthscale
        
        the kernel the above equation approximates is 
        k(x,y)=frac{(sigma sqrt(nu) ||x-y||)^{nu/2}} / {2^{nu/2-1}Gamma(nu/2)} K_{nu/2}(sigma sqrt{nu} ||x-y||)
        
        we want to approximate Matern kernel of the form:
        k(r)=sigma_f^2 (frac{2^{1-v}} / {Gamma(v)}) (sqrt(2v)r)^v * K_v(sqrt(2v)r) where v = 5/2 
        ==> k(r) = sigma_f^2 e^{-sqrt(5)r} (1+sqrt(5)r+{5r^2}{3})

        where r is defined to be sigma*||x-y|| = ||x-y||/lengthscale
        """
        rng = np.random.default_rng(self.random_state)
        nu = 5.0 # here nu is degrees of freedom in Student's t distribution
        sigma = 1.0 / self.lengthscales

        scaled_gaussian = rng.normal(loc=0, scale=sigma, size=(self.n_rff_features, self.input_dim))
        chi2 = rng.chisquare(nu, size=(self.n_rff_features, 1))

        omega = np.sqrt(nu / chi2) * scaled_gaussian
        random_offset = rng.uniform(0, 2 * np.pi, size=(self.n_rff_features))
        return omega, random_offset

    def _feature_map(self, X):
        """
        Constructs RFF feature map with previously sampled frequencies + offset.
        In RBFSampler, phi(x) = sqrt(2/n_rff_features) cos(Wx + offset) where W is random_weights_ and offset is random_offset_.
        However, this assumes variance = 1.0. We add variance under square root so that inner product of feature map
        approximates correct Matern kernel k(r) = sigma_f^2 e^{-sqrt(5)r} (1+sqrt(5)r+{5r^2}{3}).
        """
        if (self.omega is None or self.offset is None):
            raise ValueError("omega and offset undefined.")
        if (self.input_dim is None):
            raise ValueError("input_dim undefined.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_points, {self.input_dim}).")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")
        
        # X @ omega.T has shape (n_points, n_rff_features).
        A = np.cos(X@self.omega.T + self.offset) 
        scale = np.sqrt(2.0 * self.kernel_variance / self.n_rff_features)
        return scale * A

    def _dfeature_map_dx(self, X, dim):
        if self.omega is None or self.offset is None:
            raise ValueError("omega and offset undefined.")
        if (self.input_dim is None):
            raise ValueError("input_dim undefined.")
        if not isinstance(dim, (int, np.integer)):
            raise TypeError("dim must be an integer.")
        if not 0 <= dim < self.input_dim:
            raise ValueError(f"dim must be between 0 and {self.input_dim - 1}.")
        
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n_points, {self.input_dim}).")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        projection = X @ self.omega.T + self.offset
        omega_dim = self.omega[:, dim]

        scale = np.sqrt(2.0 * self.kernel_variance / self.n_rff_features)

        return -scale * np.sin(projection) * omega_dim[None, :]

    def _refresh_posterior(self):
        effective_noise = self.noise_variance + self.jitter

        # Symmetrize before Cholesky to avoid tiny roundoff asymmetry.
        self.feature_gram = 0.5 * (self.feature_gram + self.feature_gram.T)

        # Bayesian linear regression in random-feature space.
        self.training_system = (np.diag(self.feature_precision) + self.feature_gram / effective_noise)
        self.rhs = self.feature_target / effective_noise

        self.posterior_cholesky = np.linalg.cholesky(self.training_system)
        temporary = np.linalg.solve(self.posterior_cholesky, self.rhs)
        self.posterior_mean = np.linalg.solve(
            self.posterior_cholesky.T,
            temporary
        )

    def _rebuild_posterior(self):
        self.Phi_train = self._feature_map(self.X_train)
        self.feature_gram = self.Phi_train.T @ self.Phi_train
        self.feature_target = self.Phi_train.T @ self.y_train
        self._refresh_posterior()
        self.cache_size = int(self.X_train.shape[0])

    def update_posterior(self, s_new, T_new, q_new):
        s_new, T_new, q_new = self._training_arrays(s_new, T_new, q_new)

        if not (s_new.shape[0] == T_new.shape[0] == q_new.shape[0]):
            raise ValueError("New arrays must have the same number of rows.")
        if s_new.shape[1] != self.s_dim:
            raise ValueError(f"s_new must have {self.s_dim} columns.")
        if T_new.shape[1] != self.T_dim:
            raise ValueError(f"T_new must have {self.T_dim} columns.")
        if q_new.shape[1] != self.output_dim:
            raise ValueError(f"q_new must have {self.output_dim} columns.")

        X_raw_new = np.concatenate((s_new, T_new), axis=1)

        X_combined_raw = np.vstack((self.X_cache_raw, X_raw_new))
        q_combined_raw = np.vstack((self.q_cache_raw, q_new))

        # RFF uses FIFO cache eviction. The adaptive provider controls locality.
        overflow = max(0, X_combined_raw.shape[0] - self.max_cache_size)

        if overflow:
            X_combined_raw = X_combined_raw[overflow:]
            q_combined_raw = q_combined_raw[overflow:]

        self.X_cache_raw = X_combined_raw
        self.q_cache_raw = q_combined_raw

        latent_raw = -self.q_cache_raw if self.learn_neg_flux else self.q_cache_raw

        # Keep initial standardization fixed so posterior updates are comparable.
        self.X_train = (self.X_cache_raw - self.x_mean) / self.x_scale
        self.y_train = (latent_raw - self.y_mean[None, :]) / self.y_scale[None, :]

        self._rebuild_posterior()

        n_received = q_new.shape[0]

        self.posterior_updates += 1
        self.total_points_added += n_received
        self.total_points_dropped += overflow

        n_retained_new = min(n_received, self.max_cache_size)

        return {
            "n_added": int(q_new.shape[0]),
            "n_retained_new": int(n_retained_new),
            "n_dropped": int(overflow),
            "cache_size": self.cache_size,
            "posterior_updates": self.posterior_updates
        }

    def fit(self, s_train, T_train, q_train):
        """
        Fits GP model to training data. 
        """
        s_train, T_train, q_train = self._training_arrays(s_train, T_train, q_train)

        if not (s_train.shape[0] == T_train.shape[0] == q_train.shape[0]):
            raise ValueError("Training arrays must have the same number of rows.")

        n_train = q_train.shape[0]

        if self.max_cache_size == 0:
            self.max_cache_size = n_train

        if n_train > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s_train = s_train[keep]
            T_train = T_train[keep]
            q_train = q_train[keep]

        self.s_dim = int(s_train.shape[1])
        self.T_dim = int(T_train.shape[1])
        self.input_dim = self.s_dim + self.T_dim
        self.output_dim = int(q_train.shape[1])
        self.lengthscales = self._validate_lengthscale(self.lengthscale, self.input_dim)

        self.omega, self.offset = self._sample_rff_frequencies()

        self.X_cache_raw = np.concatenate((s_train, T_train), axis=1)
        self.q_cache_raw = q_train.copy()

        X_raw = self.X_cache_raw
        y_raw = -self.q_cache_raw if self.learn_neg_flux else self.q_cache_raw

        # Standardize inputs and targets before sampling random features.
        self.x_mean = X_raw.mean(axis=0) # returns array [mean_s, mean_T]
        self.x_scale = X_raw.std(axis=0) # returns array [std_s, std_T]
        self.x_scale[self.x_scale == 0.0] = 1.0 # if a particular element in x_scale is 0.0, change it to 1.0

        self.y_mean = y_raw.mean(axis=0)
        self.y_scale = y_raw.std(axis=0)
        self.y_scale[self.y_scale == 0.0] = 1.0

        self.X_train = (X_raw - self.x_mean) / self.x_scale
        self.y_train = (y_raw - self.y_mean[None, :]) / self.y_scale[None, :]
        # Larger frequency norms get stronger prior shrinkage when alpha > 0.
        omega_norms = np.linalg.norm(self.omega, axis=1)
        self.feature_precision = (1.0 + self.alpha * omega_norms ** self.p)

        self._rebuild_posterior()
        return self

    def predict(self, s, T, *, return_variance=False):
        """
        Return predicted physical flux and optionally its variance.
        """
        _, Phi = self._query_features(s,T)

        latent_standardized = Phi @ self.posterior_mean
        latent_physical = (
            self.y_mean[None, :] + self.y_scale[None, :] * latent_standardized
        )

        sign = -1.0 if self.learn_neg_flux else 1.0
        mean = sign * latent_physical

        if not return_variance:
            return mean

        solved = np.linalg.solve(self.posterior_cholesky, Phi.T)
        variance_standardized = np.sum(solved**2, axis=0)

        variance_physical = (
            variance_standardized[:, None]
            * self.y_scale[None, :] ** 2
        )
        return mean, variance_physical

    def evaluate(self, s, T, *, return_variance=False):
        """Return flux q, dq/ds, dq/dT, and optional variances."""
        X, Phi = self._query_features(s,T)

        latent_standardized = Phi @ self.posterior_mean
        latent_physical = (
            self.y_mean[None, :] 
            + self.y_scale[None, :] * latent_standardized
        )

        sign = -1.0 if self.learn_neg_flux else 1.0
        mean = sign * latent_physical

        # Derivatives are analytic derivatives of the random-feature map.
        n_points = X.shape[0]
        gradient_standardized = np.empty(
            (n_points, self.output_dim, self.input_dim),
            dtype=float,
        )

        for dim in range(self.input_dim):
            dPhi = self._dfeature_map_dx(X, dim)
            gradient_standardized[:, :, dim] = (dPhi @ self.posterior_mean)

        gradient_physical = (
            sign * self.y_scale[None, :, None] * gradient_standardized / self.x_scale[None, None, :]
        )

        dq_ds = gradient_physical[:, :, :self.s_dim]
        dq_dT = gradient_physical[:, :, self.s_dim:]

        if not return_variance:
            return mean, dq_ds, dq_dT

        # Flux variance from the feature-space posterior covariance.
        solved = np.linalg.solve(self.posterior_cholesky, Phi.T)
        variance_standardized = np.sum(solved**2, axis=0)

        variance_q = (variance_standardized[:, None] * self.y_scale[None, :] ** 2)

        # Derivative variances use the same posterior covariance with dPhi/dx.
        derivative_variance_standardized = np.empty((n_points, self.input_dim),dtype=float)

        for dim in range(self.input_dim):
            dPhi = self._dfeature_map_dx(X, dim)
            solved_derivative = np.linalg.solve(
                self.posterior_cholesky,
                dPhi.T,
            )
            derivative_variance_standardized[:, dim] = np.sum(solved_derivative**2,axis=0)

        derivative_variance_physical = (
            derivative_variance_standardized[:, None, :]
            * self.y_scale[None, :, None] ** 2
            / self.x_scale[None, None, :] ** 2
        )

        variance_dq_ds = derivative_variance_physical[:, :, :self.s_dim]
        variance_dq_dT = derivative_variance_physical[:, :, self.s_dim:]

        return (
            mean,
            dq_ds,
            dq_dT,
            variance_q,
            variance_dq_ds,
            variance_dq_dT,
        )


def smoke_test():
    """Minimal smoke test for bounded FIFO cache eviction."""

    def flux(s, T):
        return -(1.0 + 0.1 * T**2 + 0.05 * s**2) * s

    s_train = np.linspace(-2.0, 2.0, 8)
    T_train = np.linspace(0.5, 2.0, 8)
    q_train = flux(s_train, T_train)

    model = GPFluxST(
        s_train,
        T_train,
        q_train,
        learn_neg_flux=True,
        jitter=1e-8,
        kernel_variance=1.0,
        n_rff_features=500,
        alpha=1e-5,
        p=2.0,
        lengthscale=2.0,
        noise_variance=1e-2,
        random_state=0,
    )

    s_query = np.array([1.4, 1.6, 1.8])
    T_query = np.array([1.4, 1.6, 1.8])

    before = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )

    s_new = np.array([1.7, 1.9])
    T_new = np.array([1.7, 1.9])
    q_new = flux(s_new, T_new)

    update_info = model.update_posterior(
        s_new,
        T_new,
        q_new,
    )

    after = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )

    assert len(before) == 6
    assert len(after) == 6

    for value in before + after:
        assert np.all(np.isfinite(value))

    assert update_info["posterior_updates"] == 1
    assert update_info["cache_size"] == model.max_cache_size
    assert update_info["n_dropped"] == 2
    assert update_info["n_retained_new"] == 2

    (
        q_before,
        dq_ds_before,
        dq_dT_before,
        variance_q_before,
        variance_dq_ds_before,
        variance_dq_dT_before,
    ) = before

    (
        q_after,
        dq_ds_after,
        dq_dT_after,
        variance_q_after,
        variance_dq_ds_after,
        variance_dq_dT_after,
    ) = after

    print("Smoke test passed.")
    print("Update info:", update_info)
    print("q before update:", q_before)
    print("q after update:", q_after)
    print("dq/ds before update:", dq_ds_before)
    print("dq/ds after update:", dq_ds_after)
    print("dq/dT before update:", dq_dT_before)
    print("dq/dT after update:", dq_dT_after)
    print("q variance before:", variance_q_before)
    print("q variance after:", variance_q_after)
    print("dq/ds variance before:", variance_dq_ds_before)
    print("dq/ds variance after:", variance_dq_ds_after)
    print("dq/dT variance before:", variance_dq_dT_before)
    print("dq/dT variance after:", variance_dq_dT_after)


if __name__ == "__main__":
    smoke_test()

    
