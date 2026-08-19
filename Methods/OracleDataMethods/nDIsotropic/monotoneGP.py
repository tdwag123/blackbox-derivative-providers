"""
Isotropic monotone Matérn-5/2 GP for thermal flux.

Physical background:
An isotropic heat-flux law is represented as q(s,T) = -kappa(rho,T) s, where rho = ||s||^2.
Only the scalar conductivity kappa is learned. Therefore isotropy is exact: q(Rs,T) = R q(s,T).
The flux derivatives follow from the chain rule; namely, 
    dq/ds = -kappa I - 2 kappa_rho s s^T,
    dq/dT = -kappa_T s.

Monotonicity constraints:
For the common nonlinear diffusion model, increasing |s| should not decrease conductivity, 
so EP imposes kappa_rho >= 0 at virtual points in the two-dimensional invariant domain (rho,T).
If kappa > 0 as well, then -dq/ds = kappa I + 2 kappa_rho s s^T is positive definite, giving the 
strong-monotonicity structure needed by the diffusion operator.

Hyperparameters are optimized once with an ordinary GP, then frozen. Online updates retain the 
EP derivative sites and evict old states furthest from the query in invariant (rho,T) distance.
"""

import numpy as np
from scipy.linalg import cho_solve, qr_delete, solve_triangular
from scipy.special import log_ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

class MonotoneGPFluxST:
    """Learn scalar conductivity kappa(rho, T) and return isotropic vector flux."""

    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        *,
        noise_std=0.0,
        learn_neg_flux=True,     # retained for interface compatibility
        n_virtual_per_axis=6,
        monotone_s_dims=None,    # retained; isotropic model instead constrains kappa_rho
        probit_nu=1e-3,
        ep_max_iter=10,
        online_ep_sweeps=1,
        ep_damping=0.5,
        ep_tol=1e-5,
        jitter=1e-8,
        n_restarts_optimizer=0,
        reg_function=0.0,
        reg_derivative=1e-2,
        kernel_variance=1.0,
        lengthscale=2.0,
        noise_variance=1e-2,
        max_cache_size=None,
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.n_virtual_per_axis = int(n_virtual_per_axis)
        self.monotone_s_dims = monotone_s_dims
        self.probit_nu = float(probit_nu)
        self.ep_max_iter = int(ep_max_iter)
        self.online_ep_sweeps = int(online_ep_sweeps)
        self.ep_damping = float(ep_damping)
        self.ep_tol = float(ep_tol)
        self.jitter = float(jitter)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.reg_function = float(reg_function)
        self.reg_derivative = float(reg_derivative)
        self.kernel_variance = float(kernel_variance)
        self.lengthscale = lengthscale
        self.noise_variance = float(noise_variance)
        self.max_cache_size = None if max_cache_size is None else int(max_cache_size)
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    @staticmethod
    def _state_arrays(s, T, q=None):
        s = np.asarray(s, dtype=float)
        if s.ndim == 1:
            s = s[:, None]
        T = np.asarray(T, dtype=float)
        if T.ndim == 1:
            T = T[:, None]
        if s.ndim != 2 or T.ndim != 2 or T.shape[1] != 1:
            raise ValueError("s must have shape (n,d) and T must have shape (n,1).")
        if s.shape[0] != T.shape[0]:
            raise ValueError("s and T must have the same number of rows.")
        if q is None:
            return s, T
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q[:, None]
        if q.shape != s.shape:
            raise ValueError("For isotropic heat flux, q must have the same shape as s.")
        return s, T, q

    @staticmethod
    def _conductivity_from_flux(s, q):
        rho = np.sum(s * s, axis=1)
        if np.any(rho <= 1e-14):
            raise ValueError("Training/update states must have ||s|| > 0 to infer kappa.")
        return -np.sum(q * s, axis=1) / rho

    @staticmethod
    def _invariant_features(s, T):
        return np.column_stack([
            np.sum(s * s, axis=1),
            T[:, 0],
        ])

    def _standardize_X(self, X_raw):
        return (X_raw - self.x_mean_) / self.x_scale_

    def _standardize_kappa(self, kappa):
        return (kappa - self.kappa_mean_) / self.kappa_scale_

    def _kernel_parts(self, X, Y):
        X = np.asarray(X, dtype=float).reshape(-1, 2)
        Y = np.asarray(Y, dtype=float).reshape(-1, 2)
        delta = X[:, None, :] - Y[None, :, :]
        r = np.sqrt(np.sum((delta / self.lengthscales_) ** 2, axis=2))
        return delta, r

    def _K(self, X, Y):
        _, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        return (
            self.variance_
            * (1.0 + a * r + 5.0 * r**2 / 3.0)
            * np.exp(-a * r)
        )

    def _dK_dx(self, X, Y, dim):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        factor = (
            -(5.0 / 3.0)
            * self.variance_
            * (1.0 + a * r)
            * np.exp(-a * r)
        )
        return factor * delta[:, :, dim] / self.lengthscales_[dim] ** 2

    def _dK_dy(self, X, Y, dim):
        return -self._dK_dx(X, Y, dim)

    def _d2K_dxdy(self, X, Y, dim_x, dim_y):
        delta, r = self._kernel_parts(X, Y)
        a = np.sqrt(5.0)
        lx = self.lengthscales_[dim_x]
        ly = self.lengthscales_[dim_y]
        same = float(dim_x == dim_y)

        diagonal = (
            (5.0 / 3.0)
            * (1.0 + a * r)
            * same
            / lx**2
        )
        outer = (
            (25.0 / 3.0)
            * delta[:, :, dim_x]
            * delta[:, :, dim_y]
            / (lx**2 * ly**2)
        )
        return (
            self.variance_
            * np.exp(-a * r)
            * (diagonal - outer)
        )

    @staticmethod
    def _posterior(L, tau, eta, *, return_cholesky=False):
        n = L.shape[0]
        B = np.eye(n) + L.T @ (tau[:, None] * L)
        C = np.linalg.cholesky(B)
        mean_white = cho_solve(
            (C, True),
            L.T @ eta,
            check_finite=False,
        )
        mean = L @ mean_white
        solved = cho_solve(
            (C, True),
            L.T,
            check_finite=False,
        )
        variance = np.sum(L * solved.T, axis=1)
        alpha = eta - tau * mean
        if return_cholesky:
            return mean, variance, alpha, C
        return mean, variance, alpha

    def _run_ep(
        self,
        derivative_tau=None,
        derivative_eta=None,
        max_iterations=None,
    ):
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        tau = np.zeros(m + n)
        eta = np.zeros(m + n)
        if derivative_tau is not None:
            tau[:m] = np.asarray(derivative_tau, dtype=float).reshape(m)
        if derivative_eta is not None:
            eta[:m] = np.asarray(derivative_eta, dtype=float).reshape(m)
        tau[m:] = 1.0 / self.ep_observation_noise_variance_
        eta[m:] = self.y_train_ / self.ep_observation_noise_variance_
        iterations = (
            self.ep_max_iter if max_iterations is None else int(max_iterations)
        )
        for iteration in range(iterations):
            mean, variance, _ = self._posterior(
                self.ep_prior_cholesky_,
                tau,
                eta,
            )
            old_tau = tau.copy()
            old_eta = eta.copy()
            for i in range(m):
                cavity_precision = 1.0 / variance[i] - old_tau[i]
                if cavity_precision <= 1e-12:
                    continue
                cavity_variance = 1.0 / cavity_precision
                cavity_mean = (
                    mean[i] / variance[i] - old_eta[i]
                ) * cavity_variance
                scale = np.sqrt(cavity_variance + self.probit_nu**2)
                z = cavity_mean / scale
                mills = np.exp(
                    -0.5 * z**2
                    - 0.5 * np.log(2.0 * np.pi)
                    - log_ndtr(z)
                )
                tilted_mean = (
                    cavity_mean
                    + cavity_variance * mills / scale
                )
                tilted_variance = cavity_variance * (
                    1.0
                    - cavity_variance
                    * mills
                    * (mills + z)
                    / (cavity_variance + self.probit_nu**2)
                )
                tilted_variance = max(tilted_variance, 1e-12)
                proposed_tau = (
                    1.0 / tilted_variance - cavity_precision
                )
                if proposed_tau < 0.0:
                    proposed_tau = 0.0
                    proposed_eta = 0.0
                else:
                    proposed_eta = (
                        tilted_mean / tilted_variance
                        - cavity_mean / cavity_variance
                    )
                tau[i] = (
                    (1.0 - self.ep_damping) * old_tau[i]
                    + self.ep_damping * proposed_tau
                )
                eta[i] = (
                    (1.0 - self.ep_damping) * old_eta[i]
                    + self.ep_damping * proposed_eta
                )
            tau_change = np.max(
                np.abs(tau[:m] - old_tau[:m])
                / (1.0 + np.abs(old_tau[:m]))
            )
            eta_change = np.max(
                np.abs(eta[:m] - old_eta[:m])
                / (1.0 + np.abs(old_eta[:m]))
            )
            if max(tau_change, eta_change) < self.ep_tol:
                break
        mean, variance, alpha, C = self._posterior(
            self.ep_prior_cholesky_,
            tau,
            eta,
            return_cholesky=True,
        )
        self.ep_iterations_ = iteration + 1
        self.ep_site_precision_ = tau
        self.ep_site_natural_parameter_ = eta
        self.ep_derivative_site_precision_ = tau[:m].copy()
        self.ep_derivative_site_natural_parameter_ = eta[:m].copy()
        self.ep_posterior_mean_ = mean
        self.ep_posterior_variance_ = variance
        self.ep_whitened_precision_cholesky_ = C
        self.alpha_ = alpha
        self.alpha_norm_ = float(np.linalg.norm(alpha))
        self.cache_size_ = int(n)

    def _make_initial_virtual_grid(self, X):
        rho_axis = np.linspace(
            X[:, 0].min(),
            X[:, 0].max(),
            self.n_virtual_per_axis,
        )
        T_axis = np.linspace(
            X[:, 1].min(),
            X[:, 1].max(),
            self.n_virtual_per_axis,
        )
        RR, TT = np.meshgrid(rho_axis, T_axis, indexing="ij")
        return np.column_stack([RR.ravel(), TT.ravel()])

    def _build_initial_joint_prior(self):
        # Virtual variables are g = d(kappa_std)/d(rho_std).
        Kgg = self._d2K_dxdy(
            self.X_virtual_,
            self.X_virtual_,
            0,
            0,
        )
        Kgf = self._dK_dx(
            self.X_virtual_,
            self.X_train_,
            0,
        )
        Kff = self._K(self.X_train_, self.X_train_)
        K_joint = np.block([
            [Kgg, Kgf],
            [Kgf.T, Kff],
        ])
        K_joint = 0.5 * (K_joint + K_joint.T)
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        diagonal = np.concatenate([
            np.full(m, self.reg_derivative + self.jitter),
            np.full(n, self.reg_function + self.jitter),
        ])
        self.ep_prior_cholesky_ = np.linalg.cholesky(
            K_joint + np.diag(diagonal)
        )

    @staticmethod
    def _delete_cholesky_index(L, index):
        R = L.T
        Q = np.eye(R.shape[0])
        _, R_reduced = qr_delete(
            Q,
            R,
            index,
            1,
            which="col",
            check_finite=False,
        )
        R_new = R_reduced[:-1, :]
        signs = np.sign(np.diag(R_new))
        signs[signs == 0.0] = 1.0
        return (signs[:, None] * R_new).T

    def _delete_function_indices(self, indices):
        m = self.X_virtual_.shape[0]
        for cache_index in sorted(
            np.asarray(indices, dtype=int),
            reverse=True,
        ):
            self.ep_prior_cholesky_ = self._delete_cholesky_index(
                self.ep_prior_cholesky_,
                m + cache_index,
            )
        keep = np.ones(self.X_train_.shape[0], dtype=bool)
        keep[np.asarray(indices, dtype=int)] = False
        self.s_cache_raw_ = self.s_cache_raw_[keep]
        self.T_cache_raw_ = self.T_cache_raw_[keep]
        self.q_cache_raw_ = self.q_cache_raw_[keep]
        self.X_cache_raw_ = self.X_cache_raw_[keep]
        self.X_train_ = self.X_train_[keep]
        self.y_train_ = self.y_train_[keep]

    def _append_function_block(self, X_new):
        covariance_old_new = np.vstack([
            self._dK_dx(self.X_virtual_, X_new, 0),
            self._K(self.X_train_, X_new),
        ])
        covariance_new_new = self._K(X_new, X_new)
        covariance_new_new += (
            self.reg_function + self.jitter
        ) * np.eye(X_new.shape[0])
        V = solve_triangular(
            self.ep_prior_cholesky_,
            covariance_old_new,
            lower=True,
            check_finite=False,
        )
        schur = covariance_new_new - V.T @ V
        schur = 0.5 * (schur + schur.T)
        new_block = np.linalg.cholesky(schur)
        old_size = self.ep_prior_cholesky_.shape[0]
        full = np.zeros(
            (old_size + X_new.shape[0], old_size + X_new.shape[0])
        )
        full[:old_size, :old_size] = self.ep_prior_cholesky_
        full[old_size:, :old_size] = V.T
        full[old_size:, old_size:] = new_block
        self.ep_prior_cholesky_ = full

    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        s, T, q = self._state_arrays(s_train, T_train, q_train)
        self.s_dim_ = int(s.shape[1])
        self.T_dim_ = 1
        self.output_dim_ = self.s_dim_
        kappa = self._conductivity_from_flux(s, q)
        if self.max_cache_size is None:
            self.max_cache_size = int(kappa.size)
        if self.max_cache_size < 1:
            raise ValueError("max_cache_size must be positive.")
        if kappa.size > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s, T, q, kappa = s[keep], T[keep], q[keep], kappa[keep]
        X_raw = self._invariant_features(s, T)
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.kappa_mean_ = float(kappa.mean())
        self.kappa_scale_ = float(kappa.std())
        if self.kappa_scale_ == 0.0:
            self.kappa_scale_ = 1.0
        X = self._standardize_X(X_raw)
        y = self._standardize_kappa(kappa)
        lengthscale = np.asarray(self.lengthscale, dtype=float).reshape(-1)
        if lengthscale.size == 1:
            lengthscale = np.full(2, lengthscale.item())
        if lengthscale.size != 2 or np.any(lengthscale <= 0.0):
            raise ValueError("lengthscale must be positive scalar or length-2 array.")
        # One ordinary-GP fit learns the covariance before EP.
        signal_kernel = ConstantKernel(
            self.kernel_variance,
            (1e-4, 1e4),
        ) * Matern(
            length_scale=lengthscale,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        white_kernel = WhiteKernel(
            noise_level=self.noise_variance,
            noise_level_bounds=(1e-10, 1e1),
        )
        gp = GaussianProcessRegressor(
            kernel=signal_kernel + white_kernel,
            alpha=self.jitter,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=0,
        )
        gp.fit(X, y)
        fitted_signal = gp.kernel_.k1
        fitted_white = gp.kernel_.k2
        self.variance_ = float(fitted_signal.k1.constant_value)
        self.lengthscales_ = np.asarray(
            fitted_signal.k2.length_scale,
            dtype=float,
        ).reshape(2)
        self.learned_noise_variance_ = float(fitted_white.noise_level)
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))
        self.ep_observation_noise_variance_ = max(
            self.learned_noise_variance_,
            self.jitter,
        )
        self.gp_kernel_ = gp.kernel_
        self.log_marginal_likelihood_ = float(
            gp.log_marginal_likelihood_value_
        )
        if (
            self.reg_function > 0.0
            and self.reg_function >= self.learned_noise_variance_
        ):
            self.reg_function = min(
                0.9 * self.learned_noise_variance_,
                self.reg_function * self.learned_noise_variance_,
            )
        self.s_cache_raw_ = s.copy()
        self.T_cache_raw_ = T.copy()
        self.q_cache_raw_ = q.copy()
        self.X_cache_raw_ = np.concatenate([s, T], axis=1)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.X_virtual_ = self._make_initial_virtual_grid(self.X_train_)
        self._build_initial_joint_prior()
        self._run_ep(max_iterations=self.ep_max_iter)
        return self

    def _query(self, s, T):
        s = np.asarray(s, dtype=float)
        if s.ndim == 1:
            if s.size != self.s_dim_:
                raise ValueError(f"s must have trailing dimension {self.s_dim_}.")
            shape = ()
            s = s.reshape(1, self.s_dim_)
        else:
            if s.shape[-1] != self.s_dim_:
                raise ValueError(f"s must have trailing dimension {self.s_dim_}.")
            shape = s.shape[:-1]
            s = s.reshape(-1, self.s_dim_)
        T = np.asarray(T, dtype=float)
        if T.ndim == 0:
            T = np.full((s.shape[0], 1), float(T))
        elif T.shape == shape:
            T = T.reshape(-1, 1)
        elif T.shape == shape + (1,):
            T = T.reshape(-1, 1)
        else:
            raise ValueError("T must be scalar or have the same leading shape as s.")
        X_raw = self._invariant_features(s, T)
        X = self._standardize_X(X_raw)
        return s, T, X_raw, X, shape

    def _ep_moments(self, covariance):
        prior_projection = solve_triangular(
            self.ep_prior_cholesky_,
            covariance.T,
            lower=True,
            check_finite=False,
        )
        posterior_projection = solve_triangular(
            self.ep_whitened_precision_cholesky_,
            prior_projection,
            lower=True,
            check_finite=False,
        )
        return prior_projection, posterior_projection

    def evaluate(self, s_q, T_q, return_variance=False):
        s, _, _, X, shape = self._query(s_q, T_q)
        value_cov = np.hstack([
            self._dK_dy(X, self.X_virtual_, 0),
            self._K(X, self.X_train_),
        ])
        rho_cov = np.hstack([
            self._d2K_dxdy(X, self.X_virtual_, 0, 0),
            self._dK_dx(X, self.X_train_, 0),
        ])
        T_cov = np.hstack([
            self._d2K_dxdy(X, self.X_virtual_, 1, 0),
            self._dK_dx(X, self.X_train_, 1),
        ])
        kappa_std = value_cov @ self.alpha_
        kappa_rho_std = rho_cov @ self.alpha_
        kappa_T_std = T_cov @ self.alpha_
        kappa = self.kappa_mean_ + self.kappa_scale_ * kappa_std
        kappa_rho = (
            self.kappa_scale_ * kappa_rho_std / self.x_scale_[0]
        )
        kappa_T = (
            self.kappa_scale_ * kappa_T_std / self.x_scale_[1]
        )
        q = -kappa[:, None] * s
        identity = np.eye(self.s_dim_)
        outer = s[:, :, None] * s[:, None, :]
        dq_ds = (
            -kappa[:, None, None] * identity
            - 2.0 * kappa_rho[:, None, None] * outer
        )
        dq_dT = -kappa_T[:, None] * s
        q = q.reshape(shape + (self.output_dim_,))
        dq_ds = dq_ds.reshape(
            shape + (self.output_dim_, self.s_dim_)
        )
        dq_dT = dq_dT.reshape(shape + (self.output_dim_,))
        if not return_variance:
            return q, dq_ds, dq_dT
        pv, qv = self._ep_moments(value_cov)
        pr, qr = self._ep_moments(rho_cov)
        pT, qT = self._ep_moments(T_cov)
        var_kappa_std = (
            self.variance_
            - np.sum(pv**2, axis=0)
            + np.sum(qv**2, axis=0)
        )
        prior_rho_var = (
            (5.0 / 3.0)
            * self.variance_
            / self.lengthscales_[0] ** 2
        )
        prior_T_var = (
            (5.0 / 3.0)
            * self.variance_
            / self.lengthscales_[1] ** 2
        )
        var_rho_std = (
            prior_rho_var
            - np.sum(pr**2, axis=0)
            + np.sum(qr**2, axis=0)
        )
        var_T_std = (
            prior_T_var
            - np.sum(pT**2, axis=0)
            + np.sum(qT**2, axis=0)
        )
        # prior Cov[kappa, kappa_rho] is zero for a stationary kernel.
        cov_value_rho_std = (
            -np.sum(pv * pr, axis=0)
            + np.sum(qv * qr, axis=0)
        )
        var_kappa = (
            self.kappa_scale_**2
            * np.maximum(var_kappa_std, 0.0)
        )
        var_kappa_rho = (
            self.kappa_scale_ / self.x_scale_[0]
        ) ** 2 * np.maximum(var_rho_std, 0.0)
        var_kappa_T = (
            self.kappa_scale_ / self.x_scale_[1]
        ) ** 2 * np.maximum(var_T_std, 0.0)
        cov_kappa_rho = (
            self.kappa_scale_**2 / self.x_scale_[0]
        ) * cov_value_rho_std
        var_q = s**2 * var_kappa[:, None]
        var_dq_dT = s**2 * var_kappa_T[:, None]
        delta = identity[None, :, :]
        a = delta
        b = 2.0 * outer
        var_dq_ds = (
            a**2 * var_kappa[:, None, None]
            + b**2 * var_kappa_rho[:, None, None]
            + 2.0 * a * b * cov_kappa_rho[:, None, None]
        )
        var_dq_ds = np.maximum(var_dq_ds, 0.0)
        return (
            q,
            dq_ds,
            dq_dT,
            var_q.reshape(shape + (self.output_dim_,)),
            var_dq_ds.reshape(
                shape + (self.output_dim_, self.s_dim_)
            ),
            var_dq_dT.reshape(shape + (self.output_dim_,)),
        )

    def update_posterior(
        self,
        s_new,
        T_new,
        q_new,
        *,
        s_query=None,
        T_query=None,
    ):
        """
        Update with distance-based eviction in invariant (rho,T) coordinates.
        EP derivative sites are retained and warm-started.
        """
        s, T, q = self._state_arrays(s_new, T_new, q_new)
        if s.shape[1] != self.s_dim_:
            raise ValueError("New s/q dimension does not match fitted model.")
        kappa = self._conductivity_from_flux(s, q)
        X_raw_new = self._invariant_features(s, T)
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_kappa(kappa)
        if s_query is None and T_query is None:
            X_reference_raw = X_raw_new.mean(axis=0, keepdims=True)
        elif s_query is None or T_query is None:
            raise ValueError("s_query and T_query must be supplied together.")
        else:
            _, _, X_reference_candidates, _, _ = self._query(s_query, T_query)
            X_reference_raw = X_reference_candidates.mean(axis=0, keepdims=True)
        X_reference = self._standardize_X(X_reference_raw)
        old_size = self.X_train_.shape[0]
        n_new = X_new.shape[0]
        capacity = self.max_cache_size
        derivative_tau = self.ep_derivative_site_precision_.copy()
        derivative_eta = self.ep_derivative_site_natural_parameter_.copy()
        old_physical = self.X_cache_raw_.copy()
        evicted_old_points = np.empty((0, self.s_dim_ + 1))
        rejected_new_points = np.empty((0, self.s_dim_ + 1))
        if n_new >= capacity:
            distances = np.linalg.norm(
                (X_new - X_reference) / self.lengthscales_,
                axis=1,
            )
            keep_new = np.argsort(distances)[:capacity]
            reject_new = np.setdiff1d(np.arange(n_new), keep_new)
            if old_size:
                self._delete_function_indices(np.arange(old_size))
            physical_new = np.concatenate([s, T], axis=1)
            evicted_old_points = old_physical
            rejected_new_points = physical_new[reject_new].copy()
            s = s[keep_new]
            T = T[keep_new]
            q = q[keep_new]
            X_new = X_new[keep_new]
            y_new = y_new[keep_new]
            added = capacity
            dropped = old_size
        else:
            overflow = max(0, old_size + n_new - capacity)
            if overflow:
                old_distances = np.linalg.norm(
                    (self.X_train_ - X_reference) / self.lengthscales_,
                    axis=1,
                )
                evict_old = np.argsort(old_distances)[-overflow:]
                evicted_old_points = old_physical[evict_old].copy()
                self._delete_function_indices(evict_old)
            added = n_new
            dropped = overflow
        self._append_function_block(X_new)
        self.s_cache_raw_ = np.vstack([self.s_cache_raw_, s])
        self.T_cache_raw_ = np.vstack([self.T_cache_raw_, T])
        self.q_cache_raw_ = np.vstack([self.q_cache_raw_, q])
        self.X_cache_raw_ = np.vstack([
            self.X_cache_raw_,
            np.concatenate([s, T], axis=1),
        ])
        self.X_train_ = np.vstack([self.X_train_, X_new])
        self.y_train_ = np.concatenate([self.y_train_, y_new])
        self._run_ep(
            derivative_tau=derivative_tau,
            derivative_eta=derivative_eta,
            max_iterations=self.online_ep_sweeps,
        )
        self.posterior_updates_ += 1
        self.total_points_added_ += int(added)
        self.total_points_dropped_ += int(dropped)
        return {
            "n_added": int(added),
            "n_dropped": int(dropped),
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
            "ep_refinement_sweeps": int(self.ep_iterations_),
            "evicted_old_points": evicted_old_points,
            "rejected_new_points": rejected_new_points,
        }
