"""
Base Matern-5/2 GP with moving online cache.
- Regularization allowed. 
- Observation noise learned once with WhiteKernel. 
- Analytic derivatives of the posterior mean through the matern kernel. 
- New black-box observations can update the posterior without reoptimizing kernel hyperparams.
- When cache is full, oldest observations are removed first from posterior mean vector.

***USER NOTES, PLEASE READ FOR BACKGROUND***

This adapted GP separates initial training from online posterior maintenance.

During initial "fit," it learns and then freezes standardization, matern kernel parameters, 
and WhiteKernel noise variance. Namely, we constuct the posterior system:
    A = K(X,X) + (sigma_n^2 + lambda_f + epsilon)I, 
then store its cholesky factor A = LL^{\top}, and compute:
    alpha = Ainv*y
with cho_solve. 

When update_posterior(s_new, T_new, q_new) is called, model does not redo hyperparam opti. 

Instead, we:
    1. standardize the new observations using original scaling.
    2. remove oldest cached observations when needed to preserve max_cache_size (so basically FIFO)
    3. append new covariance rows and columns
    4. update cholesky factor using block-cholesky algebra rather than fitting a new GP from scratch
    5. recomputes only the posterior coefficient vector alpha

Hence, this surrogate acts as a bounded moving local approximation: new bb info enters the posterior, 
while older cache entries are discarded from the posterior. 

User implementation:
- continue using this model exactly as before: 
    q, dq_ds, dq_dT, variance = model.evaluate(s_query, T_query, return_variance=True)
- when your external testing interface decides which uncertain points require further sampling, update
the existing model using: 
    model.update_posterior(s_new, T_new, q_new)
- subsequent calls to evaluate will automatically use the updated posterior; no new model object is needed,
and no kernel or noise hyperparameters are reoptimized during these posterior updates 

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
        self.max_cache_size = None if max_cache_size is None else int(max_cache_size)
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    def _kernel_parts(self, X, Y):
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
        self.training_system_ = K_train + (self.effective_diagonal_variance_ * np.eye(self.X_train_.shape[0]))
        self.training_cholesky_ = np.linalg.cholesky(self.training_system_)
        self._refresh_posterior()

    @staticmethod
    def _chol_rank_one_update_lower(L, x):
        L = np.asarray(L, dtype=float).copy()
        x = np.asarray(x, dtype=float).copy()
        
        for k in range(x.size):
            diagonal = L[k,k]
            r = np.hypot(diagonal, x[k])
            c = r/diagonal
            s = x[k] / diagonal
            L[k,k] = r

            if k + 1 < x.size:
                L[k+1:, k] = (L[k+1 :, k] + s * x[k+1 :]) / c
                x[k+1 :] = (c * x[k+1 :] - s * L[k+1 :, k])
        return L
    
    def _drop_oldest_point(self):
        if self.X_train_.shape[0] <= 1:
            raise ValueError("cannot remove the only cached observation")
        L22 = self.training_cholesky_[1:, 1:]
        l21 = self.training_cholesky_[1:, 0]
        try:
            new_cholesky = self._chol_rank_one_update_lower(L22, l21)
        except (FloatingPointError, np.linalg.LinAlgError):
            new_cholesky = None
        self.X_cache_raw_ = self.X_cache_raw_[1:]
        self.q_cache_raw_ = self.q_cache_raw_[1:]
        self.X_train_ = self.X_train_[1:]
        self.y_train_ = self.y_train_[1:]
        self.training_system_ = self.training_system_[1:, 1:]
        if (new_cholesky is None or np.any(~np.isfinite(new_cholesky))):
            self._rebuild_current_posterior()
        else:
            self.training_cholesky_ = new_cholesky
        
    def _append_block(self, X_new, y_new):
        n_old = self.X_train_.shape[0]
        n_new = X_new.shape[0]
        K_old_new = self._K(self.X_train_, X_new)
        K_new_new = self._K(X_new, X_new)
        K_new_new += (self.effective_diagonal_variance_ * np.eye(n_new))
        V = solve_triangular(self.training_cholesky_, 
                                K_old_new, 
                                lower=True,
                                check_finite=False)
        schur = K_new_new - V.T @ V
        schur = 0.5 * (schur + schur.T)
        try:
            new_block_cholesky = np.linalg.cholesky(schur)
        except np.linalg.LinAlgError:
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
        if self.max_cache_size is None:
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
        # supplied noise is retained only as a reference diagnostic
        self.supplied_noise_variance_standardized_ = (sigma / self.y_scale_) ** 2
        lengthscale = np.asarray(self.lengthscale, dtype=float)
        if lengthscale.size == 1:
            lengthscale = np.full(2, lengthscale.item())
        if lengthscale.size != 2 or np.any(lengthscale <= 0.0):
            raise ValueError("lengthscale must be positive scalar or length-2 array.")
        signal_kernel = ConstantKernel(
            self.kernel_variance,
            constant_value_bounds="fixed",
        ) * Matern(
            length_scale=lengthscale,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        white_kernel = WhiteKernel(
            noise_level=self.noise_variance,
            noise_level_bounds="fixed",
        )

        fitted_gp = GaussianProcessRegressor(
            kernel=signal_kernel + white_kernel,
            alpha=self.jitter,
            normalize_y=False,
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=0,
        )
        fitted_gp.fit(X, y)
        fitted_signal_kernel = fitted_gp.kernel_.k1
        fitted_white_kernel = fitted_gp.kernel_.k2
        self.variance_ = float(fitted_signal_kernel.k1.constant_value)
        self.lengthscales_ = np.asarray(fitted_signal_kernel.k2.length_scale, dtype=float)
        if self.lengthscales_.size == 1:
            self.lengthscales_ = np.full(2, float(self.lengthscales_.reshape(-1)[0]))
        # WhiteKernel.noise_level is a variance in standardized output units
        self.learned_noise_variance_ = float(fitted_white_kernel.noise_level)
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))
        self.learned_noise_variance_physical_ = float(self.learned_noise_variance_ * self.y_scale_**2)
        self.learned_noise_std_physical_ = float(np.sqrt(self.learned_noise_variance_physical_))
        self.gp_kernel_ = fitted_gp.kernel_
        self.log_marginal_likelihood_ = float(fitted_gp.log_marginal_likelihood_value_)
        if self.reg_function > 0.0 and self.reg_function >= self.learned_noise_variance_:
            self.reg_function *= self.learned_noise_variance_
        self.effective_diagonal_variance_ = (self.learned_noise_variance_ + self.reg_function + self.jitter)
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
            variance_standardized = self.variance_ - np.sum(K_query_train * solved.T, axis=1)
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
    
    def update_posterior(self, s_new, T_new, q_new):
        s, T, q = self._training_arrays(s_new, T_new, q_new)
        X_raw_new = np.column_stack([s,T])
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_q(q)
        n_requested = int(q.size)
        if n_requested >= self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            dropped = int(self.X_train_.shape[0])
            self.X_cache_raw_ = X_raw_new[keep].copy()
            self.q_cache_raw_ = q[keep].copy()
            self.X_train_ = X_new[keep].copy()
            self.y_train_ = y_new[keep].copy()
            self._rebuild_current_posterior()
            added = int(self.max_cache_size)
        else:
            overflow = max(0, self.X_train_.shape[0] + n_requested - self.max_cache_size)
            for _ in range(overflow):
                self._drop_oldest_point()
            self.X_cache_raw_ = np.vstack([self.X_cache_raw_, X_raw_new])
            self.q_cache_raw_ = np.concatenate([self.q_cache_raw_, q])
            self._append_block(X_new, y_new)
            self._refresh_posterior()
            dropped = int(overflow)
            added = n_requested
        self.posterior_updates_ += 1
        self.total_points_added_ += added
        self.total_points_dropped_ += dropped
        return {
            "n_added": added,
            "n_dropped": dropped,
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
        }

def main():
    "minimal smoke test"

    def true_flux(s, T):
        return -(1.0 + 0.1 * T**2 + 0.05 * s**2) * s
    
    # initial cache of eight observations
    s_train = np.linspace(-1.0, 1.0, 8)
    T_train = np.linspace(0.5, 2.0, 8)
    q_train = true_flux(s_train, T_train)

    model = GPFluxST(
        s_train, 
        T_train, 
        q_train,
        noise_std=0.01,
        learn_neg_flux=True,
        n_restarts_optimizer=0,
        max_cache_size=8,
    )

    s_query = np.array([-0.5, 0.0, 0.5])
    T_query = np.array([1.0, 1.25, 1.5])

    q_before, dq_ds_before, dq_dT_before, var_before = model.evaluate(s_query, 
                                                                      T_query, 
                                                                      return_variance=True)
    
    # add 2 new black box observations
    # cache is trivially already full, so two oldest observations are removed
    s_new = np.array([0.8, 1.1])
    T_new = np.array([1.7, 1.9])
    q_new = true_flux(s_new, T_new)
    update_info = model.update_posterior(s_new, T_new, q_new)

    q_after, dq_ds_after, dq_dT_after, var_after = model.evaluate(
        s_query,
        T_query,
        return_variance=True,
    )

    assert update_info["n_added"] == 2
    assert update_info["n_dropped"] == 2
    assert update_info["cache_size"] == 8
    assert model.X_cache_raw_.shape == (8, 2)
    assert np.all(np.isfinite(q_after))
    assert np.all(np.isfinite(dq_ds_after))
    assert np.all(np.isfinite(dq_dT_after))
    assert np.all(np.isfinite(var_after))

    print("Smoke test passed.")
    print("Update info:", update_info)
    print("q before update:", q_before)
    print("q after update: ", q_after)
    print("variance before:", var_before)
    print("variance after: ", var_after)


if __name__ == "__main__":
    main()

"""
SMOKE TEST RESULTS: 

Smoke test passed.
Update info: {'n_added': 2, 'n_dropped': 2, 'cache_size': 8, 'posterior_updates': 1}
q before update: [ 4.56794896e-01  1.94458365e-05 -5.22558621e-01]
q after update:  [ 5.41458379e-01 -7.18799437e-05 -6.17381596e-01]
variance before: [1.26778504e-02 3.21609391e-07 1.26778504e-02]
variance after:  [1.73247375e-04 4.03975807e-07 5.42127047e-06]

"""
