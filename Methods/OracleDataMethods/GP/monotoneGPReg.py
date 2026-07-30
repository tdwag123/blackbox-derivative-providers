"""
MUST DO:
* dynamic virtual updates, look more at Minka paper 
* look more at using EP evidence for efficient hyperparameter optimization
--> in fact, look at the Minka paper to understand interleaved hyperparameter updates
* consider a different type of regularization/penalty that penalizes a quadratic form 
wrt the latent vector

DONE: 
** for regularization, see tracy's email
use WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1)), 
set alpha=jitter in GPRegressor
--> then use learned noise level in the EP algorithm
--> set reg_function to be less than noise variance because shouldn't
override learned noise estimate
--> turn on optimizer restarts LBFGS opti is a local optimizer, try n_restarts_opti=5

"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import log_ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

class MonotoneGPFluxST: 
    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        *,
        noise_std=0.0,
        learn_neg_flux=True,
        n_virtual_per_axis=6,
        probit_nu=1e-3,
        ep_max_iter=60,
        ep_damping=0.5,
        ep_tol=1e-5,
        jitter=1e-8,
        n_restarts_optimizer=0,  
        reg_function=0.0, 
        reg_derivative=1e-2,
    ):
        self.learn_neg_flux = learn_neg_flux
        self.n_virtual_per_axis = n_virtual_per_axis
        self.probit_nu = probit_nu
        self.ep_max_iter = ep_max_iter
        self.ep_damping = ep_damping
        self.ep_tol = ep_tol
        self.jitter = jitter
        self.n_restarts_optimizer = n_restarts_optimizer
        self.reg_function = reg_function
        self.reg_derivative = reg_derivative
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    def _kernel_parts(self, X, Y):
        delta = X[:, None, :]- Y[None, :, :]
        r = np.sqrt(np.sum((delta/self.lengthscales_) ** 2, axis=2))
        return delta, r

    def _K(self, X, Y):
        _, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        return (self.variance_ * (1.0 + a * r + 5.0 * r**2 / 3.0) 
                * np.exp(-a * r))

    def _dK_dx(self, X, Y, dim):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        factor = -(5.0/3.0) * self.variance_ * (1.0 + a * r) * np.exp(-a * r)
        return factor * delta[:, :, dim] / self.lengthscales_[dim] ** 2

    def _dK_dy(self, X, Y, dim):
        return -self._dK_dx(X, Y, dim)

    def _d2K_dxdy(self, X, Y, dim_x, dim_y):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        lx = self.lengthscales_[dim_x]
        ly = self.lengthscales_[dim_y]
        diagonal = (5.0/3.0) * (1.0 + a * r) * (dim_x == dim_y) / lx**2
        outer = (25.0/3.0) * delta[:, :, dim_x] * delta[:, :, dim_y]/(lx**2 * ly**2)
        return self.variance_ * np.exp(-a * r) * (diagonal - outer)

    def _posterior(self, L, tau, eta, *, return_cholesky=False):
        n = L.shape[0]
        B = np.eye(n) + L.T @ (tau[:, None] * L)
        C = np.linalg.cholesky(B)
        mean_white = cho_solve((C, True), L.T @ eta, check_finite=False)
        mean = L @ mean_white
        solved = cho_solve((C, True), L.T, check_finite=False)
        variance = np.sum(L * solved.T, axis=1)
        alpha = eta - tau * mean
        if return_cholesky:
            return mean, variance, alpha, C
        return mean, variance, alpha

    def _run_ep(self, K, y, noise_variance):
        n = y.size
        m = K.shape[0] - n
        L = np.linalg.cholesky(K)
        noise_variance = np.asarray(noise_variance, dtype=float).reshape(-1)
        if noise_variance.size == 1:
            noise_variance = np.full(n, noise_variance.item())
        tau = np.zeros(n + m)
        eta = np.zeros(n + m)
        tau[:n] = 1.0/noise_variance
        eta[:n] = y/noise_variance

        for iteration in range(self.ep_max_iter):
            mean, variance, _ = self._posterior(L, tau, eta)
            old_tau = tau.copy()
            old_eta = eta.copy()

            for j in range(m):
                i = n + j
                cavity_precision = 1.0 / variance[i] - old_tau[i]
                if cavity_precision <= 1e-12:
                    continue
                cavity_variance = 1.0/cavity_precision
                cavity_mean = (mean[i]/variance[i] - old_eta[i]) * cavity_variance
                scale = np.sqrt(cavity_variance + self.probit_nu**2)
                z = cavity_mean/scale
                mills = np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))
                tilted_mean = cavity_mean + cavity_variance * mills / scale
                tilted_variance = cavity_variance * (1.0 - cavity_variance * mills * (mills + z)
                                                    / (cavity_variance + self.probit_nu**2))
                tilted_variance = max(tilted_variance, 1e-12)
                new_tau = 1.0/tilted_variance - cavity_precision
                if new_tau < 0.0:
                    new_tau = 0.0
                    new_eta = 0.0
                else:
                    new_eta = tilted_mean/tilted_variance - cavity_mean/cavity_variance
                tau[i] = (1.0 - self.ep_damping) * old_tau[i] + self.ep_damping * new_tau
                eta[i] = (1.0 - self.ep_damping) * old_eta[i] + self.ep_damping * new_eta
            tau_change = np.max(np.abs(tau-old_tau)/(1.0 + np.abs(old_tau)))
            eta_change = np.max(np.abs(eta - old_eta) / (1.0 + np.abs(old_eta)))
            change = max(tau_change, eta_change)
            if change < self.ep_tol:
                break
        self.ep_iterations_ = iteration + 1
        posterior_mean, posterior_variance, alpha, C = self._posterior(L, tau, eta, return_cholesky=True)
        self.ep_site_precision_ = tau.copy()
        self.ep_site_natural_parameter_ = eta.copy()
        self.ep_posterior_mean_ = posterior_mean
        self.ep_posterior_variance_ = posterior_variance
        self.ep_prior_cholesky_ = L
        self.ep_whitened_precision_cholesky_ = C
        return alpha
       

    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        s = np.asarray(s_train, dtype=float).reshape(-1)
        T = np.asarray(T_train, dtype=float).reshape(-1)
        q = np.asarray(q_train, dtype=float).reshape(-1)
        if not (s.size == T.size == q.size):
            raise ValueError("Training arrays must have the same length.")
        X_raw = np.column_stack([s, T])
        latent_raw = -q if self.learn_neg_flux else q
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = latent_raw.mean()
        self.y_scale_ = latent_raw.std()
        if self.y_scale_ == 0.0:
            self.y_scale_ = 1.0
        X = (X_raw - self.x_mean_) / self.x_scale_
        y = (latent_raw - self.y_mean_) / self.y_scale_
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma = np.full(y.size, sigma.item())
        if sigma.size != y.size:
            raise ValueError("noise_std must be scalar or match the training data.")
        if np.any(sigma < 0.0):
            raise ValueError("noise_std must be nonnegative.")
        self.supplied_noise_variance_standardized_ = (sigma/self.y_scale_)**2
        signal_kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=np.ones(2),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        sklearn_kernel = signal_kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10,1e1))
        gp = GaussianProcessRegressor(
            kernel=sklearn_kernel,
            alpha=self.jitter,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
        )
        gp.fit(X, y)
        fitted_signal_kernel = gp.kernel_.k1
        fitted_white_kernel = gp.kernel_.k2
        self.variance_ = float(fitted_signal_kernel.k1.constant_value)
        self.lengthscales_ = np.asarray(fitted_signal_kernel.k2.length_scale, dtype=float)
        self.learned_noise_variance_ = float(fitted_white_kernel.noise_level)
        self.learned_noise_std_ = np.sqrt(self.learned_noise_variance_)
        self.ep_observation_noise_variance_ = max(self.learned_noise_variance_, self.jitter)
        self.learned_noise_variance_physical_ = self.learned_noise_variance_ * self.y_scale_ ** 2
        self.learned_noise_std_physical_ = np.sqrt(self.learned_noise_variance_physical_)
        self.gp_kernel_ = gp.kernel_
        self.log_marginal_likelihood_ = float(gp.log_marginal_likelihood_value_)
        # reg function should not dominate learned observation noise
        if self.reg_function >= self.learned_noise_variance_:
            self.reg_function = self.reg_function * self.learned_noise_variance_
        s_axis = np.linspace(X[:, 0].min(), X[:, 0].max(), self.n_virtual_per_axis)
        T_axis = np.linspace(X[:, 1].min(), X[:, 1].max(), self.n_virtual_per_axis)
        S, TT = np.meshgrid(s_axis, T_axis, indexing="ij")
        Z = np.column_stack([S.ravel(), TT.ravel()])

        Kff = self._K(X, X)
        Kfg = self._dK_dy(X, Z, 0)
        Kgg = self._d2K_dxdy(Z, Z, 0, 0)
        K = np.block([[Kff, Kfg], [Kfg.T, Kgg]])
        K = 0.5 * (K + K.T)

        n_obs = X.shape[0]
        n_virt = Z.shape[0]
        tikhonov_diagonal = np.concatenate([np.full(n_obs, self.reg_function), 
                                            np.full(n_virt, self.reg_derivative)])
        K += np.diag(tikhonov_diagonal)
        K += self.jitter * np.eye(K.shape[0])
        ep_noise_variance = np.full(n_obs, self.ep_observation_noise_variance_)
        self.X_train_ = X
        self.X_virtual_ = Z
        self.ep_noise_variance_ = ep_noise_variance
        self.joint_condition_ = float(np.linalg.cond(K))
        self.alpha_ = self._run_ep(K, y, ep_noise_variance)
        self.alpha_norm_ = np.linalg.norm(self.alpha_) 
        return self
        
    def evaluate(self, s_q, T_q, return_variance=False):
        s, T = np.broadcast_arrays(np.asarray(s_q, dtype=float), np.asarray(T_q, dtype=float))
        shape = s.shape
        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X = (X_raw - self.x_mean_) / self.x_scale_
        value_covariance = np.hstack([self._K(X, self.X_train_), self._dK_dy(X, self.X_virtual_, 0)])
        latent = value_covariance @ self.alpha_
        gradient = np.empty((X.shape[0], 2))
        for dim in range(2):
            derivative_covariance = np.hstack([self._dK_dx(X, self.X_train_, dim),
                                                self._d2K_dxdy(X, self.X_virtual_, dim, 0)])
            gradient[:, dim] = derivative_covariance @ self.alpha_
        latent = self.y_mean_ + self.y_scale_ * latent
        gradient = self.y_scale_ * gradient / self.x_scale_
        sign = -1.0 if self.learn_neg_flux else 1.0
        q = sign * latent
        dq_ds = sign * gradient[:, 0]
        dq_dT = sign * gradient[:, 1]

        if return_variance:
            prior_projection = solve_triangular(self.ep_prior_cholesky_, 
                                                value_covariance.T, 
                                                lower=True, 
                                                check_finite=False)
            posterior_projection = solve_triangular(self.ep_whitened_precision_cholesky_, 
                                                    prior_projection, 
                                                    lower=True, 
                                                    check_finite=False)
            variance_standardized = (self.variance_ 
                                     - np.sum(prior_projection**2, axis=0)
                                     + np.sum(posterior_projection**2, axis = 0))
            variance = self.y_scale_**2 * np.maximum(variance_standardized, 0.0)
            return q.reshape(shape), dq_ds.reshape(shape), dq_dT.reshape(shape), variance.reshape(shape)

        return q.reshape(shape), dq_ds.reshape(shape), dq_dT.reshape(shape)