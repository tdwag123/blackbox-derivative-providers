"""
Base Matern-5/2 GP. Instead of exact Matern kernel, using RFF.
- Regularization allowed. 
- No deriv-monotonicity constraints through EP.
- Direct analytic differentiation through the kernel. 
- 
"""
import numpy as np
from scipy.linalg import cho_solve

class GPFluxST:
    """
    s_train: training values of s.
    T_train: training values of the temperature T. 
    q_train: observed flux values.
    noise_std: assumed std dev of noise in observed flux values q; 0 -> treats observations as exact. larger values smooth data more strongly.
    learn_neg_flux: if True, learns -q(s,T) instead of q(s,T) to enforce accurate monotonicity constraints.
    jitter: added to diagonal of covariance matrix to improve stability. K_stable = K + (jitter)I.
    n_restarts_optimizer: extra attempts to tune GP. 
        if 0, runs once from default starting point; if 2, one initial run + 2 optimizations from different starting points = 3 runs.
        uses same training data with different kernel hyperparameters each time.
    reg_function: 
    kernel_variance: 
    lengthscale:
    noise_variance:
    n_rff_features: number of RFF features
    random_state: if 0, use same frequencies each time so results are reproducible. 
        if None, each run samples different random frequencies.
    """

    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        *,
        noise_std=0.0,
        learn_neg_flux=True,
        jitter=1e-8,
        n_restarts_optimizer=0,
        reg_function=0.0,
        kernel_variance=1.0,
        lengthscale=2.0,
        noise_variance=1.0e-2,
        n_rff_features=500,
        random_state=0
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.jitter = float(jitter)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.reg_function = float(reg_function)
        self.kernel_variance = float(kernel_variance)
        self.lengthscale = lengthscale
        self.noise_variance = float(noise_variance)
        self.n_rff_features = int(n_rff_features)
        self.random_state = random_state

        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    def _sample_rff_frequencies(self, n_features, lengthscales, random_state):
        rng = np.random.default_rng(random_state)
        df = 5.0

        z = rng.normal(size=(n_features, 2))
        chi2 = rng.chisquare(df, size=(n_features, 1))

        omega = z / np.sqrt(chi2 / df)
        omega = omega / lengthscales[None, :]

        return omega

    def _features(self, X):
        projection = X @ self.rff_omega_.T
        scale = np.sqrt(self.variance_ / self.n_rff_features)

        return scale * np.column_stack([
            np.cos(projection),
            np.sin(projection),
        ])

    def _K(self, X, Y):
        """
        Constructing the RFF approximation to the Matern-5/2 covariance matrix.
        """
        return self._features(X) @ self._features(Y).T

    # ---------------------------------------kernel derivatives------------------------------------------------
    def _dfeatures_dx(self, X, dim):
        projection = X @ self.rff_omega_.T
        omega_dim = self.rff_omega_[:, dim]
        scale = np.sqrt(self.variance_ / self.n_rff_features)

        return scale * np.column_stack([
            -np.sin(projection) * omega_dim[None, :],
            np.cos(projection) * omega_dim[None, :],
        ])

    def _dK_dx(self, X, Y, dim):
        return self._dfeatures_dx(X, dim) @ self._features(Y).T
    # --------------------------------------end of kernel derivatives------------------------------------------


    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        """
        Fits GP model to training data. Input coordinates, target fluxes standardized before fitting.
        Depending on learn_neg_flux, either learns q or -q. 

        s_train: training values of s.
        T_train: training values of the temperature T. 
        q_train: observed flux values.
        noise_std: assumed std dev of noise in observed flux values q. retained as diagnostic and not used to
            construct the fitted covariance matrix. default = 0.0.
        """
        s = np.asarray(s_train, dtype=float).reshape(-1)
        T = np.asarray(T_train, dtype=float).reshape(-1)
        q = np.asarray(q_train, dtype=float).reshape(-1)

        if not (s.size == T.size == q.size):
            raise ValueError("Training arrays must have the same length.")
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be nonnegative.")
        if self.reg_function < 0.0:
            raise ValueError("reg_function must be nonnegative.")
        if self.kernel_variance <= 0.0:
            raise ValueError("kernel_variance must be positive.")
        if self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be positive.")
        
        X_raw = np.column_stack([s, T])
        latent_raw = -q if self.learn_neg_flux else q

        # ----------------------------------- standardization ------------------------------------------
        # we standardize GP's inputs and target, making each variable roughly zero-mean and unit scale.
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0

        self.y_mean_ = float(latent_raw.mean())
        self.y_scale_ = float(latent_raw.std())
        if self.y_scale_ == 0.0:
            self.y_scale_ = 1.0

        X = (X_raw - self.x_mean_) / self.x_scale_
        y = (latent_raw - self.y_mean_) / self.y_scale_
        # ----------------------------------- end of standardization ----------------------------------

        # --------------- supplied noise is retained only as a reference diagnostic -------------------
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma = np.full(y.size, sigma.item())
        if sigma.size != y.size:
            raise ValueError("noise_std must be scalar or match the training data.")
        if np.any(sigma < 0.0):
            raise ValueError("noise_std must be nonnegative.")
        self.supplied_noise_variance_standardized_ = (sigma / self.y_scale_) ** 2

        # lengthscale was given in GPFluxST initialization
        lengthscale = np.asarray(self.lengthscale, dtype=float)
        if lengthscale.size == 1:
            lengthscale = np.full(2, lengthscale.item())
        if lengthscale.size != 2 or np.any(lengthscale <= 0.0):
            raise ValueError("lengthscale must be positive scalar or length-2 array.")


        # The following defines GP's signal covariance function k(x,x')=sigma_f^2 k_{Matern-5/2}(x,x')
        # Specifices how strongly the model expects flux values at 2 input points 
        # x = (s, T) and x' = (s', T') to be related.

        self.variance_ = self.kernel_variance
        self.lengthscales_ = lengthscale
        self.learned_noise_variance_ = self.noise_variance
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))

        self.learned_noise_variance_physical_ = float(
            self.learned_noise_variance_ * self.y_scale_**2
        )
        self.learned_noise_std_physical_ = float(
            np.sqrt(self.learned_noise_variance_physical_)
        )

        self.rff_omega_ = self._sample_rff_frequencies(
            self.n_rff_features,
            self.lengthscales_,
            self.random_state,
        )

        self.gp_kernel_ = "RFF approximation to Matern-5/2"
        self.log_marginal_likelihood_ = np.nan

        # final latent-function posterior system: 
        #       --> noise_variance supplies fixed observation variance.
        #       --> reg_function is an additional function-only diagonal regularizer.

        # Builds signal covariance matrix
        K_train = self._K(X, X) 
        diagonal_variance = (self.learned_noise_variance_ + self.reg_function + self.jitter)

        training_system = (K_train + diagonal_variance * np.eye(X.shape[0]))
        L = np.linalg.cholesky(training_system)

        self.alpha_ = cho_solve((L, True), y, check_finite=False)
        self.X_train_ = X
        self.training_cholesky_ = L
        self.training_system_ = training_system
        self.effective_diagonal_variance_ = diagonal_variance
        self.condition_number_ = float(np.linalg.cond(training_system))
        self.alpha_norm_ = float(np.linalg.norm(self.alpha_))

        return self


    def evaluate(self, s_q, T_q, return_variance=False):
        """
        Evaluates fitted flux model.
        """
        s, T = np.broadcast_arrays(np.asarray(s_q, dtype=float), np.asarray(T_q, dtype=float))
        output_shape = s.shape

        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X = (X_raw - self.x_mean_) / self.x_scale_
        latent_standardized = (self._K(X, self.X_train_) @ self.alpha_)
        gradient_standardized = np.empty((X.shape[0], 2), dtype=float)

        for dim in range(2):
            gradient_standardized[:, dim] = (self._dK_dx(X, self.X_train_, dim) @ self.alpha_)

        latent_physical = (self.y_mean_ + self.y_scale_ * latent_standardized)
        gradient_physical = (self.y_scale_ * gradient_standardized / self.x_scale_[None, :])
        
        sign = -1.0 if self.learn_neg_flux else 1.0
        q = sign * latent_physical
        dq_ds = sign * gradient_physical[:, 0]
        dq_dT = sign * gradient_physical[:, 1]

        if return_variance:
            K_query_train = self._K(X, self.X_train_)
            solved = cho_solve(
                (self.training_cholesky_, True),
                K_query_train.T,
                check_finite=False,
            )

            prior_variance = np.sum(self._features(X) ** 2, axis=1)
            variance_standardized = prior_variance - np.sum(K_query_train * solved.T, axis=1)
            variance = self.y_scale_**2 * np.maximum(variance_standardized, 0.0)

            return (
                q.reshape(output_shape),
                dq_ds.reshape(output_shape),
                dq_dT.reshape(output_shape),
                variance.reshape(output_shape),
            )

        return (
            q.reshape(output_shape),
            dq_ds.reshape(output_shape),
            dq_dT.reshape(output_shape),
        )
