"""
Multidimensional Matern-5/2 GP with a bounded, moving posterior cache. 

Point eviction is distance-based.

Training shapes permitted are general:
    s: (n, d_s)
    T: (n, d_T)
    q: (n, d_q)

RMK: d_* indicates dimension of associated variable.

For a query with leading shape B, evaluate returns q with shape B+(d_q,),
dq/ds with shape B+(d_q,d_s), and dq/dT with shape B+(d_q, d_T). 

RMK: The B+(_,_) notation simply means this GP retains the structure of the 
query grid, batch, or array and appends to it the physical output dimensions. 
Note that |B| := number of query/training points.

"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

class GPFluxST:
    def __init__(self, s_train, T_train, q_train, *,
        noise_std=0.0,
        learn_neg_flux=True,
        jitter=1e-8,
        kernel_variance=1.0,
        lengthscale=2.0,
        noise_variance=1.0e-2,
        max_cache_size=None,
        n_rff_features=500,
        alpha=0.0,
        p=2.0,
        random_state=0
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.jitter = self._validate_nonnegative_float(jitter, "jitter")
        self.kernel_variance = self._validate_kernel_variance(kernel_variance)
        self.lengthscale = self._validate_lengthscale(lengthscale)
        self.noise_variance = self._validate_noise_variance(noise_variance)
        self.max_cache_size = 0 if max_cache_size is None else int(max_cache_size)

        self.n_rff_features = self._validate_n_rff_features(n_rff_features)
        self.alpha = self._validate_nonnegative_float(alpha, "alpha")
        self.p = self._validate_nonnegative_float(p, "p")
        self.random_state = random_state

        self.omega = None
        self.offset = None

        self.X_train = None
        self.y_train = None
        self.Phi_train = None

        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    @staticmethod
    def _validate_kernel_variance(kernel_variance):
        variance = float(kernel_variance)
        if variance <= 0.0:
            raise ValueError("kernel_variance must be positive.")
        return variance

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
    def _matrix(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        return values
    
    @classmethod
    def _training_arrays(cls, s, T, q):
        s = cls._matrix(s, "s")
        T = cls._matrix(T, "T")
        q = cls._matrix(q, "q")
        return s, T, q

    @staticmethod
    def _query_features(values, dimension, name):
        values = np.asarray(values, dtype=float)
        if dimension == 1:
            if values.ndim == 0: 
                return values.reshape(1), () 
        return values, values.shape[:-1]
    
    def _query_matrix(self, s, T):
        s, s_shape = self._query_features(s, self.s_dim_, "s")
        T, T_shape = self._query_features(T, self.T_dim_, "T")
        output_shape = np.broadcast_shapes(s_shape, T_shape)
        s = np.broadcast_to(s, output_shape + (self.s_dim_,))
        T = np.broadcast_to(T, output_shape + (self.T_dim_,))
        X_raw = np.concatenate([s.reshape(-1, self.s_dim_), T.reshape(-1, self.T_dim_)], axis=1)
        return X_raw, output_shape

    def _format_values(self, values, output_shape):
        if self.output_dim_ == 1:
            return values[:,0].reshape(output_shape)
        return values.reshape(output_shape + (self.output_dim_,))
    
    def _format_jacobian(self, jacobian, output_shape, input_dimension):
        if self.output_dim_ == 1 and input_dimension == 1:
            return jacobian[:, 0, 0].reshape(output_shape)
        if self.output_dim_ == 1:
            return jacobian[:, 0, :].reshape(output_shape + (input_dimension,))
        if input_dimension == 1:
            return jacobian[:, :, 0].reshape(output_shape + (self.output_dim_,))
        return jacobian.reshape(output_shape + (self.output_dim_, input_dimension))

    def _kernel_parts(self, X, Y):
        X = np.asarray(X, dtype=float).reshape(-1, self.input_dim_)
        Y = np.asarray(Y, dtype=float).reshape(-1, self.input_dim_)
        delta = X[:, None, :] - Y[None, :, :]
        r = np.sqrt(np.sum((delta / self.lengthscales_) ** 2, axis=2))
        return delta, r

    def _K(self, X, Y):
        _, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        return (self.variance_ * (1.0 + a * r + 5.0 * r**2 / 3.0) * np.exp(-a * r))

    def _dK_dx(self, X, Y, dim):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        factor = (-(5.0 / 3.0) * self.variance_ * (1.0 + a * r) * np.exp(-a * r))
        return (factor * delta[:, :, dim] / self.lengthscales_[dim] ** 2)
    
    def _d2K_dx(self, X, Y, dim_x, dim_y):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        lx = self.lengthscales_[dim_x]
        ly = self.lengthscales_[dim_y]
        same = float(dim_x == dim_y)
        diagonal = ((5.0 / 3.0) * (1.0 + a * r) * same / lx**2)
        outer = ((25.0 / 3.0) * delta[:, :, dim_x] * delta[:, :, dim_y] / (lx**2 * ly**2))
        return self.variance_ * np.exp(-a * r) * (diagonal - outer)

    def _standardize_X(self, X_raw):
        return (X_raw - self.x_mean_) / self.x_scale_
    
    def _standardize_q(self, q_raw):
        latent_raw = -q_raw if self.learn_neg_flux else q_raw
        return (latent_raw - self.y_mean_) / self.y_scale_    

    def _refresh_posterior(self):
        self.alpha_ = cho_solve((self.training_cholesky_, True), 
                                self.y_train_,
                                check_finite=False)
        self.alpha_norm_ = float(np.linalg.norm(self.alpha_))
        self.cache_size_ = int(self.X_train_.shape[0])

    def _rebuild_current_posterior(self):
        K_train = self._K(self.X_train_, self.X_train_)
        self.training_system_ = K_train + (self.effective_diagonal_variance_ 
                                           * np.eye(self.X_train_.shape[0]))
        self.training_cholesky_ = np.linalg.cholesky(self.training_system_)
        self._refresh_posterior()

    def _append_block(self, X_new, y_new):
        n_old = self.X_train_.shape[0]
        n_new = X_new.shape[0]
        K_old_new = self._K(self.X_train_, X_new)
        K_new_new = self._K(X_new, X_new)
        K_new_new += (self.effective_diagonal_variance_ * np.eye(n_new))
        V = solve_triangular(self.training_cholesky_, K_old_new, lower=True,
                             check_finite=False)
        schur = K_new_new - V.T @ V
        schur = 0.5 * (schur + schur.T)
        try:
            new_block_cholesky = np.linalg.cholesky(schur)
        except np.linalg.LinAlgError:
            # numerical fallback: still no kernel/noise optimization
            self.X_train_ = np.vstack([self.X_train_, X_new])
            self.y_train_ = np.concatenate([self.y_train_, y_new])
            self._rebuild_current_posterior()
            return
        full_cholesky = np.zeros((n_old + n_new, n_old + n_new), dtype=float)
        full_cholesky[:n_old, :n_old] = self.training_cholesky_
        full_cholesky[n_old:, :n_old] = V.T
        full_cholesky[n_old:, n_old:] = new_block_cholesky
        self.training_system_ = np.block([[self.training_system_, K_old_new],
                                          [K_old_new.T, K_new_new]])
        self.training_cholesky_ = full_cholesky
        self.X_train_ = np.vstack([self.X_train_, X_new])
        self.y_train_ = np.concatenate([self.y_train_, y_new])

    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        s, T, q = self._training_arrays(s_train, T_train, q_train)
        self.s_dim_ = int(s.shape[1])
        self.T_dim_ = int(T.shape[1])
        self.input_dim_ = self.s_dim_ + self.T_dim_
        self.output_dim_ = int(q.shape[1])
        sigma = np.asarray(noise_std, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(q.shape, float(sigma))
        elif sigma.shape == (q.shape[0],):
            sigma = np.repeat(sigma[:, None], self.output_dim_, axis=1)
        else:
            try:
                sigma = np.broadcast_to(sigma, q.shape).copy()
            except ValueError as error:
                raise ValueError("noise_std must be scalar, length n, or broadcastable to q.shape.") from error
        if np.any(sigma < 0.0):
            raise ValueError("noise_std must be nonnegative.")
        if self.max_cache_size == 0:
            self.max_cache_size = int(q.shape[0])
        if q.shape[0] > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s = s[keep]
            T = T[keep]
            q = q[keep]
            sigma = sigma[keep]
        X_raw = np.concatenate([s,T], axis=1)
        latent_raw = -q if self.learn_neg_flux else q
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = latent_raw.mean(axis=0)
        self.y_scale_ = latent_raw.std(axis=0)
        self.y_scale_[self.y_scale_ == 0.0] = 1.0
        X = self._standardize_X(X_raw)
        y = self._standardize_q(q)
        self.supplied_noise_variance_standardized_ = (sigma / self.y_scale_[None, :]) ** 2
        lengthscale = np.asarray(self.lengthscale, dtype=float).reshape(-1)
        if lengthscale.size == 1:
            lengthscale = np.full(self.input_dim_, lengthscale.item())
        if (lengthscale.size != self.input_dim_ or np.any(lengthscale <= 0.0)):
            raise ValueError(f"lengthscale must be a positive scalar or length-{self.input_dim_} array.")
        signal_kernel = ConstantKernel(self.kernel_variance, (1e-4, 1e4)) * Matern(
            length_scale=lengthscale, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        white_kernel = WhiteKernel(noise_level=self.noise_variance, noise_level_bounds=(1e-10, 1e1))
        fitted_gp = GaussianProcessRegressor(kernel=signal_kernel + white_kernel, alpha=self.jitter,
                                             normalize_y=False, n_restarts_optimizer=self.n_restarts_optimizer, 
                                             random_state=0)
        fitted_gp.fit(X, y)
        fitted_signal = fitted_gp.kernel_.k1
        fitted_white = fitted_gp.kernel_.k2
        self.variance_ = float(fitted_signal.k1.constant_value)
        self.lengthscales_ = np.asarray(fitted_signal.k2.length_scale, dtype=float)
        self.learned_noise_variance_ = float(fitted_white.noise_level)
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))
        self.learned_noise_variance_physical_ = self.learned_noise_variance_ * self.y_scale_**2
        self.learned_noise_std_physical_ = np.sqrt(self.learned_noise_variance_physical_)
        self.gp_kernel_ = fitted_gp.kernel_
        self.log_marginal_likelihood_ = float(fitted_gp.log_marginal_likelihood_value_)
        if (self.reg_function > 0.0 and self.reg_function >= self.learned_noise_variance_):
            self.reg_function = min(0.9*self.learned_noise_variance_, self.reg_function*self.learned_noise_variance_)
        self.effective_diagonal_variance_ = (self.learned_noise_variance_ + self.reg_function + self.jitter)
        # cache rows may be in any order; point eviction is based on query distance
        self.X_cache_raw_ = X_raw.copy()
        self.q_cache_raw_ = q.copy()
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._rebuild_current_posterior()
        return self

    def evaluate(self, s_q, T_q, return_variance=False):
        X_raw, output_shape = self._query_matrix(s_q, T_q)
        X = self._standardize_X(X_raw)
        K_query_train = self._K(X, self.X_train_)
        latent_standardized = K_query_train @ self.alpha_
        gradient_standardized = np.empty((X.shape[0], self.output_dim_, self.input_dim_), dtype=float)
        for dim in range(self.input_dim_):
            gradient_standardized[:, :, dim] = self._dK_dx(X, self.X_train_, dim) @ self.alpha_
        latent_physical = self.y_mean_[None, :] + self.y_scale_[None, :] * latent_standardized
        gradient_physical = (self.y_scale_[None, :, None] * gradient_standardized / self.x_scale_[None, None, :])
        sign = -1.0 if self.learn_neg_flux else 1.0
        q = sign * latent_physical
        dq_ds = sign * gradient_physical[:, :, : self.s_dim_]
        dq_dT = sign * gradient_physical[:, :, self.s_dim_ :]
        result = (self._format_values(q, output_shape), self._format_jacobian(dq_ds, output_shape, self.s_dim_),
                  self._format_jacobian(dq_dT, output_shape, self.T_dim_))
        if not return_variance:
            return result
        solved = cho_solve((self.training_cholesky_, True), K_query_train.T, check_finite=False)
        variance_standardized = self.variance_ - np.sum(K_query_train * solved.T, axis=1)
        variance_physical = (np.maximum(variance_standardized, 0.0)[:, None] * self.y_scale_[None, :] ** 2)
        derivative_variance_standardized = np.empty((X.shape[0], self.input_dim_), dtype=float)
        for dim in range(self.input_dim_):
            K_derivative_train = self._dK_dx(X, self.X_train_, dim)
            solved_derivative = cho_solve((self.training_cholesky_, True), K_derivative_train.T, check_finite=False)
            prior_derivative_variance = ((5.0 / 3.0) * self.variance_ / self.lengthscales_[dim] ** 2)
            derivative_variance_standardized[:, dim] = (prior_derivative_variance - np.sum(K_derivative_train * solved_derivative.T, axis=1))
        derivative_variance_standardized = np.maximum(derivative_variance_standardized, 0.0)
        derivative_variance_physical = (derivative_variance_standardized[:, None, :] * self.y_scale_[None, :, None] ** 2 / self.x_scale_[None, None, :] ** 2)
        variance_dq_ds = derivative_variance_physical[:, :, :self.s_dim_]
        variance_dq_dT = derivative_variance_physical[:, :, self.s_dim_:]
        return result + (self._format_values(variance_physical, output_shape), 
                         self._format_jacobian(variance_dq_ds, output_shape, self.s_dim_),
                         self._format_jacobian(variance_dq_dT, output_shape, self.T_dim_))

    def update_posterior(self, s_new, T_new, q_new, *, s_query=None, T_query=None):
        """Update the fixed-hyperparameter posterior using query-local eviction."""
        s, T, q = self._training_arrays(s_new, T_new, q_new)
        if (s.shape[1] != self.s_dim_ or T.shape[1] != self.T_dim_ or q.shape[1] != self.output_dim_):
            raise ValueError("new observations must match the fitted dimensions.")
        X_raw_new = np.concatenate([s, T], axis=1)
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_q(q)
        if s_query is None and T_query is None:
            X_reference_raw = X_raw_new.mean(axis=0, keepdims=True)
        elif s_query is None or T_query is None:
            raise ValueError("s_query and T_query must be supplied together.")
        else:
            candidates, _ = self._query_matrix(s_query, T_query)
            X_reference_raw = candidates.mean(axis=0, keepdims=True)
        X_reference = self._standardize_X(X_reference_raw)
        old_cache = self.X_cache_raw_.copy()
        old_size = self.X_train_.shape[0]
        n_new = X_new.shape[0]
        capacity = self.max_cache_size
        overflow = max(0, old_size + n_new - capacity)
        added = n_new
        dropped = overflow
        evicted_old_points = np.empty((0, self.input_dim_))
        rejected_new_points = np.empty((0, self.input_dim_))
        if overflow == 0:
            self.X_cache_raw_ = np.vstack([self.X_cache_raw_, X_raw_new])
            self.q_cache_raw_ = np.vstack([self.q_cache_raw_, q])
            self._append_block(X_new, y_new)
            self._refresh_posterior()
        elif n_new >= capacity:
            distances = np.linalg.norm((X_new - X_reference) / self.lengthscales_, axis=1)
            keep_new = np.argsort(distances)[:capacity]
            rejected = np.setdiff1d(np.arange(n_new), keep_new)
            evicted_old_points = old_cache
            rejected_new_points = X_raw_new[rejected].copy()
            self.X_cache_raw_ = X_raw_new[keep_new].copy()
            self.q_cache_raw_ = q[keep_new].copy()
            self.X_train_ = X_new[keep_new].copy()
            self.y_train_ = y_new[keep_new].copy()
            added = capacity
            dropped = old_size
        else:
            old_slots = capacity - n_new
            old_distances = np.linalg.norm((self.X_train_ - X_reference) / self.lengthscales_, axis=1)
            keep_old = np.argsort(old_distances)[:old_slots]
            evict_old = np.setdiff1d(np.arange(old_size), keep_old)
            evicted_old_points = old_cache[evict_old].copy()
            self.X_cache_raw_ = np.vstack([self.X_cache_raw_[keep_old], X_raw_new])
            self.q_cache_raw_ = np.vstack([self.q_cache_raw_[keep_old], q])
            self.X_train_ = np.vstack([self.X_train_[keep_old], X_new])
            self.y_train_ = np.vstack([self.y_train_[keep_old], y_new])
        if overflow > 0:
            self._rebuild_current_posterior()
        self.posterior_updates_ += 1
        self.total_points_added_ += int(added)
        self.total_points_dropped_ += int(dropped)
        return {
            "n_added": int(added),
            "n_dropped": int(dropped),
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
            "evicted_old_points": evicted_old_points,
            "rejected_new_points": rejected_new_points,
        }
