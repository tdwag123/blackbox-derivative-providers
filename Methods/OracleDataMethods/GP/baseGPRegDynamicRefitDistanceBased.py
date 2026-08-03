"""
base matern-5/2 GP with a bounded, moving posterior cache. 
- supports function-only regularization.
- observation noise learned once through WhiteKernel.
- direct analytic differentiation through the matern kernel. 
- new black box observations can update the posterior without reoptimizing 
  kernel hyperparameters. 
- when cache is full, observations furthest from current query are removed first. 

***ALGORITHM STRUCTURE ***

(RMK: The "cache" here refers to the posterior prediction vector, not the sampling
cache; fret not.)

initial physical observations
        |
validate and standardize
        |
one-time kernel + WhiteKernel optimization
        |
freeze scaling and hyperparameters
        |
build K + (noise + regularization + jitter)I
        |
Cholesky factor + alpha
        |
evaluate q, dq/ds, dq/dT, variance
        |
external interface obtains new black-box observations
        |
choose a reference query
        |
compute cache overflow
        |
no overflow --> block-Cholesky append
        │
        |__overflow --> retain query-nearest points
                               |
                        rebuild bounded posterior
                               |
                        subsequent evaluate uses new cache

"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

class GPFluxST:
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
        max_cache_size=None,
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.jitter = float(jitter)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.reg_function = float(reg_function)
        self.kernel_variance = float(kernel_variance)
        self.lengthscale = lengthscale
        self.noise_variance = float(noise_variance)
        self.max_cache_size = 0 if max_cache_size is None else int(max_cache_size)
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(s_train, T_train, q_train, noise_std=noise_std)
    
    def _kernel_parts(self, X, Y):
        X = np.asarray(X, dtype=float).reshape(-1,2)
        Y = np.asarray(Y, dtype=float).reshape(-1,2)
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

    @staticmethod
    def _training_arrays(s, T, q):
        s = np.asarray(s, dtype=float).reshape(-1)
        T = np.asarray(T, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)
        if not (s.size == T.size == q.size):
            raise ValueError("s, T, and q must have the same length")
        if s.size == 0:
            raise ValueError("at least one observation is required")
        if not (np.all(np.isfinite(s)) 
                and np.all(np.isfinite(T))
                and np.all(np.isfinite(q))):
            raise ValueError("training values must be finite")
        return s, T, q
    
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
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma = np.full(q.size, sigma.item())
        if sigma.size != q.size or np.any(sigma < 0.0):
            raise ValueError("noise_std must be nonnegative and scalar or data-sized.")
        # by default, moving cache keeps exactly the initial number of
        # observations; once full, adding m points removes m existing points
        # furthest from the query that initiated the update
        if self.max_cache_size == 0:
            self.max_cache_size = int(q.size)
        if q.size > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s = s[keep]
            T = T[keep]
            q = q[keep]
            sigma = sigma[keep]
        X_raw = np.column_stack([s, T])
        latent_raw = -q if self.learn_neg_flux else q
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = float(latent_raw.mean())
        self.y_scale_ = float(latent_raw.std())
        if self.y_scale_ == 0.0:
            self.y_scale_ = 1.0
        X = self._standardize_X(X_raw)
        y = self._standardize_q(q)
        self.supplied_noise_variance_standardized_ = (sigma / self.y_scale_) ** 2
        lengthscale = np.asarray(self.lengthscale, dtype=float).reshape(-1)
        if lengthscale.size == 1:
            lengthscale = np.full(2, lengthscale.item())
        if lengthscale.size != 2 or np.any(lengthscale <= 0.0):
            raise ValueError("lengthscale must be a positive scalar or length-2 array.")
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
        self.learned_noise_variance_physical_ = float(self.learned_noise_variance_ * self.y_scale_**2)
        self.learned_noise_std_physical_ = float(np.sqrt(self.learned_noise_variance_physical_))
        self.gp_kernel_ = fitted_gp.kernel_
        self.log_marginal_likelihood_ = float(fitted_gp.log_marginal_likelihood_value_)
        if (self.reg_function > 0.0 and self.reg_function >= self.learned_noise_variance_):
            self.reg_function *= self.learned_noise_variance_
        self.effective_diagonal_variance_ = (self.learned_noise_variance_ + self.reg_function + self.jitter)
        # cache rows may be in any order; point eviction is based on query distance
        self.X_cache_raw_ = X_raw.copy()
        self.q_cache_raw_ = q.copy()
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self._rebuild_current_posterior()
        return self

    def evaluate(self, s_q, T_q, return_variance=False):
        s, T = np.broadcast_arrays(np.asarray(s_q, dtype=float), np.asarray(T_q, dtype=float))
        output_shape = s.shape
        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X = self._standardize_X(X_raw)
        K_query_train = self._K(X, self.X_train_)
        latent_standardized = K_query_train @ self.alpha_
        gradient_standardized = np.empty((X.shape[0], 2), dtype=float)
        for dim in range(2):gradient_standardized[:, dim] = (self._dK_dx(X, self.X_train_, dim) @ self.alpha_)
        latent_physical = (self.y_mean_ + self.y_scale_ * latent_standardized)
        gradient_physical = (self.y_scale_ * gradient_standardized / self.x_scale_[None, :])
        sign = -1.0 if self.learn_neg_flux else 1.0
        q = sign * latent_physical
        dq_ds = sign * gradient_physical[:, 0]
        dq_dT = sign * gradient_physical[:, 1]
        result = (q.reshape(output_shape), dq_ds.reshape(output_shape), dq_dT.reshape(output_shape))
        if return_variance:
            solved = cho_solve((self.training_cholesky_, True), K_query_train.T, check_finite=False)
            variance_standardized = self.variance_ - np.sum(K_query_train * solved.T, axis=1)
            variance_physical = self.y_scale_**2 * np.maximum(variance_standardized, 0.0)
            return result + (
                variance_physical.reshape(output_shape),
            )
        return result

    def update_posterior(self, s_new, T_new, q_new, *, s_query=None, T_query=None):
        s, T, q = self._training_arrays(s_new, T_new, q_new)
        X_raw_new = np.column_stack([s, T])
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_q(q)
        if s_query is None or T_query is None:
            X_reference_raw = X_raw_new.mean(axis=0, keepdims=True)
        else:
            s_ref, T_ref = np.broadcast_arrays(np.asarray(s_query, dtype=float),
                                               np.asarray(T_query, dtype=float))
            X_reference_raw = np.array([[float(np.mean(s_ref)), float(np.mean(T_ref))]])
        X_reference = self._standardize_X(X_reference_raw)
        old_size = self.X_train_.shape[0]
        n_new = X_new.shape[0]
        capacity = self.max_cache_size
        overflow = max(0, old_size + n_new - capacity)
        added = n_new
        dropped = overflow
        if overflow == 0:
            self.X_cache_raw_ = np.vstack([self.X_cache_raw_, X_raw_new])
            self.q_cache_raw_ = np.concatenate([self.q_cache_raw_, q])
            self._append_block(X_new, y_new)
            self._refresh_posterior()
        else:
            if n_new >= capacity:
                new_distances = np.sqrt(np.sum(((X_new - X_reference) / self.lengthscales_) ** 2, axis=1))
                keep_new = np.argsort(new_distances)[:capacity]
                self.X_cache_raw_ = X_raw_new[keep_new].copy()
                self.q_cache_raw_ = q[keep_new].copy()
                self.X_train_ = X_new[keep_new].copy()
                self.y_train_ = y_new[keep_new].copy()
                added = capacity
                dropped = old_size
            else:
                old_slots = capacity - n_new
                old_distances = np.sqrt(np.sum(((self.X_train_ - X_reference)/self.lengthscales_)**2, axis=1))
                keep_old = np.argsort(old_distances)[:old_slots]
                self.X_cache_raw_ = np.vstack([self.X_cache_raw_[keep_old], X_raw_new])
                self.q_cache_raw_ = np.concatenate([self.q_cache_raw_[keep_old], q])
                self.X_train_ = np.vstack([self.X_train_[keep_old], X_new])
                self.y_train_ = np.concatenate([self.y_train_[keep_old], y_new])
            self._rebuild_current_posterior()
        self.posterior_updates_ += 1
        self.total_points_added_ += int(added)
        self.total_points_dropped_ += int(dropped)
        return {
            "n_added": added,
            "n_dropped": dropped,
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
        }

def main():
    """minimal smoke test for query-local cache eviction"""
    def flux(s, T):
        return -(1.0 + 0.1 * T**2 + 0.05 * s**2) * s
    s_train = np.linspace(-2.0, 2.0, 8)
    T_train = np.linspace(0.5, 2.0, 8)
    q_train = flux(s_train, T_train)
    model = GPFluxST(s_train, T_train, q_train, noise_std=0.02, n_restarts_optimizer=0,
                     max_cache_size=8)
    old_cache = model.X_cache_raw_.copy()
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
    info = model.update_posterior(s_new, T_new, q_new, s_query=1.8, T_query=1.8)
    q_after, dq_ds_after, dq_dT_after, variance_after = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )
    assert info["n_added"] == 2
    assert info["n_dropped"] == 2
    assert info["cache_size"] == 8
    assert np.all(np.isfinite(q_after))
    assert np.all(np.isfinite(dq_ds_after))
    assert np.all(np.isfinite(dq_dT_after))
    assert np.all(np.isfinite(variance_after))
    query = np.array([1.8, 1.8])
    old_standardized = model._standardize_X(old_cache)
    query_standardized = model._standardize_X(query.reshape(1, 2))
    old_distances = np.linalg.norm((old_standardized - query_standardized)/model.lengthscales_, axis=1)
    furthest_old = old_cache[np.argsort(old_distances)[-2:]]
    for point in furthest_old:
        assert not np.any(np.all(np.isclose(model.X_cache_raw_, point), axis=1))
    print("Smoke test passed.")
    print("Update info:", info)
    print("q before update:", q_before)
    print("q after update: ", q_after)
    print("dq_ds before update:", dq_ds_before)
    print("dq_ds after update: ", dq_ds_after)
    print("dq_dT before update:", dq_dT_before)
    print("dq_dT after update: ", dq_dT_after)
    print("variance before:", variance_before)
    print("variance after: ", variance_after)
    print("Current cache:")
    print(model.X_cache_raw_)

if __name__ == "__main__":
    main()