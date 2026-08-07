"""
GP with RFF approximation of Matern-5/2 covariance.
"""

import numpy as np
from scipy.linalg import cho_solve

class GPFluxST:
    """
    learn_neg_flux: if True, learns -q(s,T) instead of q(s,T) to enforce accurate monotonicity constraints
    lengthscale: can either be scalar or vector of length 2 to allow for anisotropic lengthscaling
    random_state: if 0, use same frequencies each time so results are reproducible. if None, each run samples different random frequencies.
    """
    def __init__(self, s_train, T_train, q_train, *, 
                 learn_neg_flux=True,
                 jitter=1e-8,
                 kernel_variance=1.0,
                 n_rff_features=500,
                 alpha=0.0,
                 p=2.0,
                 lengthscale=2.0,
                 noise_variance=1e-2,
                 random_state=0
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.jitter = self._validate_nonnegative_float(jitter, "jitter")
        self.kernel_variance = self._validate_kernel_variance(kernel_variance)
        self.n_rff_features = self._validate_n_rff_features(n_rff_features)
        self.alpha = self._validate_nonnegative_float(alpha, "alpha")
        self.p = self._validate_nonnegative_float(p, "p")
        self.lengthscales = self._validate_lengthscale(lengthscale)
        self.noise_variance = self._validate_noise_variance(noise_variance)
        self.random_state = random_state

        self.omega = None
        self.offset = None

        self.X_train = None
        self.y_train = None
        self.Phi_train = None

        self.fit(s_train, T_train, q_train)

    @staticmethod
    def _validate_kernel_variance(kernel_variance):
        variance = float(kernel_variance)
        if variance <= 0.0:
            raise ValueError("kernel_variance must be positive.")
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
    def _validate_nonnegative_float(value, name):
        value = float(value)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value

    @staticmethod
    def _validate_lengthscale(lengthscale):
        lengthscales = np.asarray(lengthscale, dtype=float).reshape(-1)

        if lengthscales.size == 1:
            lengthscales = np.full(2, lengthscales.item())
        if lengthscales.size != 2:
            raise ValueError("lengthscale must be a positive scalar or length-2 array.")

        if np.any(~np.isfinite(lengthscales)) or np.any(lengthscales <= 0.0):
            raise ValueError("lengthscale entries must be finite and positive.")

        return lengthscales

    @staticmethod
    def _validate_noise_variance(noise_variance):
        variance = float(noise_variance)
        if variance <= 0.0 or not np.isfinite(variance):
            raise ValueError("noise_variance must be finite and positive.")
        return variance

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

        scaled_gaussian = rng.normal(loc=0, scale=sigma, size=(self.n_rff_features, 2))
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

        X = np.asarray(X, dtype=float).reshape(-1, 2)
        
        # note: X.shape = (n_points, 2); omega.shape = (n_rff_features, 2); offset.shape = (n_rff_features,)
        # so X@omega.T has shape (n_points, n_rff_features)
        A = np.cos(X@self.omega.T + self.offset) 
        scale = np.sqrt(2.0 * self.kernel_variance / self.n_rff_features)
        return scale * A

    def _dfeature_map_dx(self, X, dim):
        if self.omega is None or self.offset is None:
            raise ValueError("omega and offset undefined.")

        X = np.asarray(X, dtype=float).reshape(-1, 2)

        projection = X @ self.omega.T + self.offset
        omega_dim = self.omega[:, dim]

        scale = np.sqrt(2.0 * self.kernel_variance / self.n_rff_features)

        return -scale * np.sin(projection) * omega_dim[None, :]

    def fit(self, s_train, T_train, q_train):
        self.omega, self.offset = self._sample_rff_frequencies()

        s_train = np.asarray(s_train, dtype=float).reshape(-1)
        T_train = np.asarray(T_train, dtype=float).reshape(-1)
        q_train = np.asarray(q_train, dtype=float).reshape(-1)

        if not (s_train.size == T_train.size == q_train.size):
            raise ValueError("Training arrays must have the same length.")

        X_raw = np.column_stack([s_train, T_train])
        y_raw = -q_train if self.learn_neg_flux else q_train

        # ----------------------------------- standardization ------------------------------------------
        # we standardize GP's inputs and target, making each variable roughly zero-mean and unit scale.
        self.x_mean = X_raw.mean(axis=0) # returns length 2 array [mean_s, mean_T]
        self.x_scale = X_raw.std(axis=0) # returns length 2 array [std_s, std_T]
        self.x_scale[self.x_scale == 0.0] = 1.0 # if a particular element in x_scale is 0.0, change it to 1.0

        self.y_mean = float(y_raw.mean())
        self.y_scale = float(y_raw.std())
        if self.y_scale == 0.0:
            self.y_scale = 1.0

        self.X_train = (X_raw - self.x_mean) / self.x_scale
        self.y_train = (y_raw - self.y_mean) / self.y_scale
        # ----------------------------------- end of standardization ----------------------------------

        self.Phi_train = self._feature_map(self.X_train) # feature matrix for training points

        # we want to solve for posterior mean
        omega_norms = np.linalg.norm(self.omega, axis=1)

        # frequency-weighted regularization!!
        self.feature_precision = (
            1.0 + self.alpha * omega_norms ** self.p
        )
        self.prior_precision = np.diag(self.feature_precision)
        diagonal_variance = self.noise_variance + self.jitter

        self.training_system = (self.prior_precision + (self.Phi_train.T @ self.Phi_train) / diagonal_variance)
        self.rhs = (self.Phi_train.T @ self.y_train) / diagonal_variance

        # self.weight_mean = np.linalg.solve(self.training_system, self.rhs)
        self.training_cholesky = np.linalg.cholesky(self.training_system)
        self.weight_mean = cho_solve(
            (self.training_cholesky, True),
            self.rhs,
            check_finite=False,
        )
        return self

    def evaluate(self, s_query, T_query, return_variance=False):
        """
        Evaluates fitted flux model.
        """
        s, T = np.broadcast_arrays(np.asarray(s_query, dtype=float), np.asarray(T_query, dtype=float))
        output_shape = s.shape
        
        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X = (X_raw - self.x_mean) / self.x_scale
        Phi_query = self._feature_map(X)

        y_standardized = Phi_query @ self.weight_mean

        gradient_standardized = np.empty((X.shape[0], 2), dtype=float)
        for dim in range(2):
            dPhi = self._dfeature_map_dx(X, dim)
            gradient_standardized[:, dim] = dPhi @ self.weight_mean

        y_physical = self.y_mean + self.y_scale * y_standardized
        gradient_physical = (
            self.y_scale * gradient_standardized / self.x_scale[None, :]
        )

        sign = -1.0 if self.learn_neg_flux else 1.0
        q = sign * y_physical
        dq_ds = sign * gradient_physical[:, 0]
        dq_dT = sign * gradient_physical[:, 1]

        if return_variance:
            solved = cho_solve(
                (self.training_cholesky, True),
                Phi_query.T,
                check_finite=False,
            )
        
            variance_standardized = np.sum(Phi_query * solved.T, axis=1)
            variance = self.y_scale**2 * np.maximum(variance_standardized, 0.0)

            return (
                q.reshape(output_shape),
                dq_ds.reshape(output_shape),
                dq_dT.reshape(output_shape),
                variance.reshape(output_shape),
            )

        return (
            q.reshape(output_shape),
            dq_ds.reshape(output_shape),
            dq_dT.reshape(output_shape)
        )

def smoke_test():
    """minimal smoke test for query-local cache eviction"""

    def flux(s, T):
        return -(1.0 + 0.1 * T**2 + 0.05 * s**2) * s
    
    s_train = np.linspace(-2.0, 2.0, 8)
    T_train = np.linspace(0.5, 2.0, 8)
    q_train = flux(s_train, T_train)

    model = GPFluxST(s_train, T_train, q_train, 
                     learn_neg_flux=True,
                     jitter=1e-8,
                     kernel_variance=1.0,
                     n_rff_features=500,
                     alpha=1e-5,
                     p=2.0,
                     lengthscale=2.0,
                     noise_variance=1.0e-2,
                     random_state=0)
    
    s_query = np.array([1.4, 1.6, 1.8])
    T_query = np.array([1.4, 1.6, 1.8])
    q_before, dq_ds_before, dq_dT_before, variance_before = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )

    s_new = np.array([1.7, 1.9])
    T_new = np.array([1.7, 1.9])
    q_new = flux(s_new, T_new)

    q_after, dq_ds_after, dq_dT_after, variance_after = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )

    assert np.all(np.isfinite(q_after))
    assert np.all(np.isfinite(dq_ds_after))
    assert np.all(np.isfinite(dq_dT_after))
    assert np.all(np.isfinite(variance_after))

    print("Smoke test passed.")
    print("q before update:", q_before)
    print("q after update: ", q_after)
    print("dq_ds before update:", dq_ds_before)
    print("dq_ds after update: ", dq_ds_after)
    print("dq_dT before update:", dq_dT_before)
    print("dq_dT after update: ", dq_dT_after)
    print("variance before:", variance_before)
    print("variance after: ", variance_after)


if __name__ == "__main__":
    smoke_test()

    