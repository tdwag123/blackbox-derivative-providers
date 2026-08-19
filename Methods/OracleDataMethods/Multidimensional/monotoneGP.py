"""
Multidimensional monotone Matern-5/2 GP with online EP and a bounded, 
distance-based moving observation cache. 

Input dimensions remain general: 
    s : (n, d_s)
    T : (n, d_T)
    q : (n, d_q)

Each output component is modeled by an independent scalar latent GP. Components
share the same standardized input space, kernel hyperparameters, virtual point 
locations, and function-observation cache, but each component has its own EP sites
and posterior coefficients. 

By default: 
- if d_q == d_s, q_r is constrained monotonically wrd s_r;
- otherwise very q_r is constrained wrt s_0.

This mapping can be supplied explicitly through monotone_s_dims.

For a query batch with leading shape B, evaluate returns: 
    q             : B + (d_q,)
    dq/ds         : B + (d_q, d_s)
    dq/dT         : B + (d_q, d_T)
    var(q)        : B + (d_q,)
    var(dq/ds)    : B + (d_q, d_s)
    var(dq/dT)    : B + (d_q, d_T)

Online updates retain:
- the original input/output scaling,
- kernel hyperparameters and learned WhiteKernel variance,
- virtual derivative points,
- existing EP derivative sites.

When the bounded cache overflows, old function observations furthest from the
triggering query are removed first. EP is then warm-started for only online_ep_sweeps 
iterations.

"""

import itertools
import numpy as np
from scipy.linalg import cho_solve, qr_delete, solve_triangular
from scipy.special import log_ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class MonotoneGPError(RuntimeError):
    """Raised when monotone EP constraints cannot be fit reliably."""


class MonotoneGPFluxST:
    """Matern GP surrogate with EP-based monotonicity constraints."""

    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        *,
        noise_std=0.0,
        learn_neg_flux=True,
        n_virtual_per_axis=3, # reduced from 6 b/c curse of dimensionality
        monotone_s_dims=None,
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
        lengthscale=4.0,
        noise_variance=1.0e-2,
        optimize_hyperparameters=True,
        adaptive_virtual_refinement=True,
        monotonicity_check_points_per_axis=5,
        max_virtual_refinements=1,
        max_virtual_points_per_round=12,
        monotonicity_tolerance=1.0e-8,
        ep_min_damping=None,
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
        self.optimize_hyperparameters = bool(optimize_hyperparameters)
        self.adaptive_virtual_refinement = bool(adaptive_virtual_refinement)
        self.monotonicity_check_points_per_axis = int(monotonicity_check_points_per_axis)
        self.max_virtual_refinements = int(max_virtual_refinements)
        self.max_virtual_points_per_round = int(max_virtual_points_per_round)
        self.monotonicity_tolerance = float(monotonicity_tolerance)
        self.ep_min_damping = (
            max(float(ep_min_damping), 1.0e-4)
            if ep_min_damping is not None
            else max(self.ep_damping / 8.0, 1.0e-4)
        )
        self.max_cache_size = (None if max_cache_size is None else int(max_cache_size))
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.virtual_refinement_rounds_ = 0
        self.virtual_points_added_ = 0
        self.constraint_violation_fraction_ = np.nan
        self.constraint_worst_violation_ = np.nan
        self.ep_damping_used_ = []
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

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

    def _standardize_X(self, X_raw):
        return (X_raw - self.x_mean_) / self.x_scale_

    def _standardize_q(self, q_raw):
        # The monotonicity constraints are applied to the latent quantity.
        latent_raw = -q_raw if self.learn_neg_flux else q_raw
        return (latent_raw - self.y_mean_) / self.y_scale_

    def _kernel_parts(self, X, Y):
        # Kernel distances are measured after input standardization.
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

    @staticmethod
    def _posterior(L, tau, eta, *, return_cholesky=False):
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

    def _run_ep_output(
        self,
        output_index,
        derivative_tau=None,
        derivative_eta=None,
        max_iterations=None,
        damping=None,
    ):
        # EP works on a joint vector: derivative sites first, function data last.
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        tau = np.zeros(m + n)
        eta = np.zeros(m + n)
        if derivative_tau is not None:
            tau[:m] = np.asarray(derivative_tau, dtype=float).reshape(m)
        if derivative_eta is not None:
            eta[:m] = np.asarray(derivative_eta, dtype=float).reshape(m)
        tau[m:] = (1.0/self.ep_observation_noise_variance_)
        eta[m:] = (self.y_train_[:, output_index]/self.ep_observation_noise_variance_)
        iterations = self.ep_max_iter if max_iterations is None else int(max_iterations)
        L = self.ep_prior_cholesky_[output_index]
        ep_damping = self.ep_damping if damping is None else float(damping)
        if not 0.0 < ep_damping <= 1.0:
            raise ValueError("EP damping must be in (0, 1].")
        largest_change = np.inf
        for iteration in range(iterations):
            mean, variance, _ = self._posterior(L, tau, eta)
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
                raise MonotoneGPError("EP posterior produced nonfinite moments")
            old_tau = tau.copy()
            old_eta = eta.copy()
            for i in range(m):
                # Each derivative site enforces a probit soft constraint.
                cavity_precision = (1.0/variance[i] - old_tau[i])
                if not np.isfinite(cavity_precision) or cavity_precision <= 1e-12:
                    raise MonotoneGPError("EP cavity precision is nonpositive")
                cavity_variance = 1.0/cavity_precision
                cavity_mean = (mean[i] / variance[i] - old_eta[i]) * cavity_variance
                scale = np.sqrt(cavity_variance + self.probit_nu**2)
                z = cavity_mean/scale
                mills = np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))
                if not np.isfinite(mills):
                    raise MonotoneGPError("EP inverse Mills ratio is nonfinite")
                tilted_mean = cavity_mean + cavity_variance * mills / scale
                tilted_variance = cavity_variance * (1.0 - cavity_variance * mills * (mills + z) / (cavity_variance + self.probit_nu**2))
                if not np.isfinite(tilted_variance) or tilted_variance <= 1e-14:
                    raise MonotoneGPError("EP tilted variance collapsed")
                proposed_tau = 1.0/tilted_variance - cavity_precision
                negative_tolerance = 1e-10 * max(1.0, cavity_precision)
                if proposed_tau < -negative_tolerance:
                    raise MonotoneGPError("EP produced a negative site precision")
                if proposed_tau < 0.0:
                    proposed_tau = 0.0
                    proposed_eta = 0.0
                else:
                    proposed_eta = (tilted_mean / tilted_variance) - (cavity_mean / cavity_variance)
                if not np.isfinite(proposed_tau) or not np.isfinite(proposed_eta):
                    raise MonotoneGPError("EP produced nonfinite site parameters")
                tau[i] = (1.0 - ep_damping) * old_tau[i] + ep_damping * proposed_tau
                eta[i] = (1.0 - ep_damping) * old_eta[i] + ep_damping * proposed_eta
            tau_change = np.max(np.abs(tau[:m] - old_tau[:m]) / (1.0 + np.abs(old_tau[:m])))
            eta_change = np.max(np.abs(eta[:m] - old_eta[:m]) / (1.0 + np.abs(old_eta[:m])))
            largest_change = max(tau_change, eta_change)
            if largest_change < self.ep_tol:
                break
        if not np.isfinite(largest_change):
            raise MonotoneGPError("EP did not produce a finite convergence signal")
        mean, variance, alpha, C = self._posterior(L, tau, eta, return_cholesky=True)
        self.ep_iterations_[output_index] = iteration + 1
        self.ep_site_precision_[output_index] = tau
        self.ep_site_natural_parameter_[output_index] = eta
        self.ep_derivative_site_precision_[output_index] = tau[:m].copy()
        self.ep_derivative_site_natural_parameter_[output_index] = eta[:m].copy()
        self.ep_posterior_mean_[output_index] = mean
        self.ep_posterior_variance_[output_index] = variance
        self.ep_whitened_precision_cholesky_[output_index] = C
        self.alpha_[:, output_index] = alpha

    def _run_ep_once(self, derivative_tau=None, derivative_eta=None, max_iterations=None, damping=None):
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        joint_size = m + n
        self.ep_iterations_ = np.zeros(self.output_dim_, dtype=int)
        self.ep_site_precision_ = [None] * self.output_dim_
        self.ep_site_natural_parameter_ = [None] * self.output_dim_
        self.ep_derivative_site_precision_ = [None] * self.output_dim_
        self.ep_derivative_site_natural_parameter_ = [None] * self.output_dim_
        self.ep_posterior_mean_ = [None] * self.output_dim_
        self.ep_posterior_variance_ = [None] * self.output_dim_
        self.ep_whitened_precision_cholesky_ = [None] * self.output_dim_
        self.alpha_ = np.empty((joint_size, self.output_dim_), dtype=float)
        for output_index in range(self.output_dim_):
            tau_r = None if derivative_tau is None else derivative_tau[output_index]
            eta_r = None if derivative_eta is None else derivative_eta[output_index]
            self._run_ep_output(
                output_index,
                derivative_tau=tau_r,
                derivative_eta=eta_r,
                max_iterations=max_iterations,
                damping=damping,
            )
        self.alpha_norm_ = float(np.linalg.norm(self.alpha_))
        self.cache_size_ = int(n)

    def _run_ep(self, derivative_tau=None, derivative_eta=None, max_iterations=None):
        damping = float(self.ep_damping)
        failures = []
        while True:
            try:
                self._run_ep_once(
                    derivative_tau=derivative_tau,
                    derivative_eta=derivative_eta,
                    max_iterations=max_iterations,
                    damping=damping,
                )
                self.ep_damping_used_ = [damping] * self.output_dim_
                return
            except (MonotoneGPError, np.linalg.LinAlgError) as error:
                # Lower damping is slower but much more stable for small caches.
                failures.append(str(error))
                damping *= 0.5
                if damping < self.ep_min_damping:
                    raise MonotoneGPError(
                        "EP failed after adaptive damping retries: "
                        + " | ".join(failures)
                    ) from error
    
    def _make_initial_virtual_grid(self, X):
        axes = [np.linspace(X[:, dim].min(), X[:, dim].max(), self.n_virtual_per_axis) for dim in range(self.input_dim_)]
        return np.asarray(list(itertools.product(*axes)), dtype=float)

    def _make_constraint_check_grid(self, X):
        axes = [
            np.linspace(
                X[:, dim].min(),
                X[:, dim].max(),
                self.monotonicity_check_points_per_axis,
            )
            for dim in range(self.input_dim_)
        ]
        return np.asarray(list(itertools.product(*axes)), dtype=float)

    @staticmethod
    def _select_new_virtual_points(existing, candidates, severity, max_points, min_distance):
        if candidates.size == 0 or max_points <= 0:
            return np.empty((0, existing.shape[1]), dtype=float)
        order = np.argsort(-np.asarray(severity, dtype=float).reshape(-1))
        selected = []
        for index in order:
            point = np.asarray(candidates[index], dtype=float)
            if existing.size:
                existing_distance = np.min(np.linalg.norm(existing - point, axis=1))
                if existing_distance < min_distance:
                    continue
            if selected:
                selected_distance = np.min(np.linalg.norm(np.vstack(selected) - point, axis=1))
                if selected_distance < min_distance:
                    continue
            selected.append(point)
            if len(selected) >= max_points:
                break
        if not selected:
            return np.empty((0, existing.shape[1]), dtype=float)
        return np.vstack(selected)

    def _predict_standardized_state(self, X, X_virtual=None, alpha=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_virtual = self.X_virtual_ if X_virtual is None else np.asarray(X_virtual, dtype=float)
        alpha = self.alpha_ if alpha is None else np.asarray(alpha, dtype=float)
        latent = np.empty((X.shape[0], self.output_dim_), dtype=float)
        gradient = np.empty((X.shape[0], self.output_dim_, self.input_dim_), dtype=float)
        for output_index in range(self.output_dim_):
            monotone_dim = int(self.monotone_input_dims_[output_index])
            value_covariance = np.hstack([
                self._dK_dy(X, X_virtual, monotone_dim),
                self._K(X, self.X_train_),
            ])
            latent[:, output_index] = value_covariance @ alpha[:, output_index]
            for dim in range(self.input_dim_):
                derivative_covariance = np.hstack([
                    self._d2K_dxdy(X, X_virtual, dim, monotone_dim),
                    self._dK_dx(X, self.X_train_, dim),
                ])
                gradient[:, output_index, dim] = (
                    derivative_covariance @ alpha[:, output_index]
                )
        return latent, gradient

    def _scan_monotonicity_constraints(self):
        if self.monotonicity_check_points_per_axis < 2:
            return {
                "candidates": np.empty((0, self.input_dim_), dtype=float),
                "violations": np.zeros(0, dtype=bool),
                "severity": np.zeros(0, dtype=float),
                "fraction": 0.0,
                "worst": 0.0,
            }
        candidates = self._make_constraint_check_grid(self.X_train_)
        _, gradient = self._predict_standardized_state(candidates)
        constrained = np.empty((candidates.shape[0], self.output_dim_), dtype=float)
        for output_index in range(self.output_dim_):
            constrained[:, output_index] = gradient[
                :, output_index, int(self.monotone_input_dims_[output_index])
            ]
        scale = np.maximum(np.max(np.abs(constrained), axis=0), 1.0)
        tolerance = self.monotonicity_tolerance * scale
        violation_matrix = constrained < -tolerance[None, :]
        violations = np.any(violation_matrix, axis=1)
        severity = np.max(np.maximum(-constrained - tolerance[None, :], 0.0), axis=1)
        worst = float(np.max(np.maximum(-constrained, 0.0))) if constrained.size else 0.0
        return {
            "candidates": candidates,
            "violations": violations,
            "severity": severity,
            "fraction": float(np.mean(violations)) if violations.size else 0.0,
            "worst": worst,
        }

    def _fit_with_adaptive_virtual_refinement(self):
        self.X_virtual_ = self._make_initial_virtual_grid(self.X_train_)
        last_scan = {
            "fraction": np.nan,
            "worst": np.nan,
            "violations": np.zeros(0, dtype=bool),
            "candidates": np.empty((0, self.input_dim_), dtype=float),
            "severity": np.zeros(0, dtype=float),
        }
        rounds = self.max_virtual_refinements if self.adaptive_virtual_refinement else 0
        for refinement_round in range(rounds + 1):
            self._build_initial_joint_prior()
            self._run_ep(max_iterations=self.ep_max_iter)
            last_scan = self._scan_monotonicity_constraints()
            self.virtual_refinement_rounds_ = refinement_round
            self.constraint_violation_fraction_ = float(last_scan["fraction"])
            self.constraint_worst_violation_ = float(last_scan["worst"])
            if last_scan["fraction"] <= 0.0 or refinement_round == rounds:
                break
            # Add virtual points only where the fitted derivative violates the constraint.
            min_distance = 0.5 / max(self.monotonicity_check_points_per_axis - 1, 1)
            new_points = self._select_new_virtual_points(
                self.X_virtual_,
                last_scan["candidates"][last_scan["violations"]],
                last_scan["severity"][last_scan["violations"]],
                self.max_virtual_points_per_round,
                min_distance,
            )
            if new_points.shape[0] == 0:
                break
            self.X_virtual_ = np.vstack([self.X_virtual_, new_points])
            self.virtual_points_added_ += int(new_points.shape[0])
        return last_scan
    
    def _build_joint_prior_for_output(self, output_index):
        monotone_dim = int(self.monotone_input_dims_[output_index])
        # Joint prior over derivative virtual variables and function observations.
        Kgg = self._d2K_dxdy(self.X_virtual_, self.X_virtual_, monotone_dim, monotone_dim)
        Kgf = self._dK_dx(self.X_virtual_, self.X_train_, monotone_dim)
        Kff = self._K(self.X_train_, self.X_train_)
        K_joint = np.block([[Kgg, Kgf], [Kgf.T, Kff]])
        K_joint = 0.5 * (K_joint + K_joint.T)
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        diagonal = np.concatenate([np.full(m, self.reg_derivative + self.jitter),
                                   np.full(n, self.reg_function + self.jitter)])
        K_joint += np.diag(diagonal)
        return np.linalg.cholesky(K_joint) 
    
    def _build_initial_joint_prior(self):
        self.ep_prior_cholesky_ = [self._build_joint_prior_for_output(r) for r in range(self.output_dim_)]
    
    @staticmethod
    def _delete_cholesky_index(L, index):
        R = L.T
        Q = np.eye(R.shape[0])
        _, R_reduced = qr_delete(Q, R, index, 1, which="col", check_finite=False)
        R_new = R_reduced[:-1, :]
        signs = np.sign(np.diag(R_new))
        signs[signs == 0.0] = 1.0
        R_new = signs[:, None] * R_new
        return R_new.T
    
    def _delete_function_indices(self, indices):
        m = self.X_virtual_.shape[0]
        for cache_index in sorted(np.asarray(indices, dtype=int), reverse=True):
            joint_index = m + cache_index
            for output_index in range(self.output_dim_):
                self.ep_prior_cholesky_[output_index] = self._delete_cholesky_index(self.ep_prior_cholesky_[output_index], joint_index)
        keep = np.ones(self.X_train_.shape[0], dtype=bool)
        keep[np.asarray(indices, dtype=int)] = False
        self.X_cache_raw_ = self.X_cache_raw_[keep]
        self.q_cache_raw_ = self.q_cache_raw_[keep]
        self.X_train_ = self.X_train_[keep]
        self.y_train_ = self.y_train_[keep]
    
    def _append_function_block_output(self, output_index, X_new):
        monotone_dim = int(self.monotone_input_dims_[output_index])
        covariance_old_new = np.vstack([self._dK_dx(self.X_virtual_, X_new, monotone_dim),
                                        self._K(self.X_train_, X_new)])
        covariance_new_new = self._K(X_new, X_new)
        covariance_new_new += (self.reg_function + self.jitter) * np.eye(X_new.shape[0])
        L = self.ep_prior_cholesky_[output_index]
        V = solve_triangular(L, covariance_old_new, lower=True, check_finite=False)
        schur = covariance_new_new - V.T @ V
        schur = 0.5 * (schur + schur.T)
        new_block = np.linalg.cholesky(schur)
        old_size = L.shape[0]
        full = np.zeros((old_size + X_new.shape[0], old_size + X_new.shape[0]), dtype=float)
        full[:old_size, :old_size] = L
        full[old_size:, :old_size] = V.T
        full[old_size:, old_size:] = new_block
        self.ep_prior_cholesky_[output_index] = full
    
    def _append_function_block(self, X_new):
        for output_index in range(self.output_dim_):
            self._append_function_block_output(output_index, X_new)

    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        s, T, q = self._training_arrays(s_train, T_train, q_train)
        self.s_dim_ = int(s.shape[1])
        self.T_dim_ = int(T.shape[1])
        self.input_dim_ = self.s_dim_ + self.T_dim_
        self.output_dim_ = int(q.shape[1])
        if self.monotone_s_dims is None:
            if self.output_dim_ == self.s_dim_:
                monotone_s_dims = np.arange(self.output_dim_, dtype=int)
            else:
                monotone_s_dims = np.zeros(self.output_dim_, dtype=int)
        else:
            monotone_s_dims = np.asarray(self.monotone_s_dims, dtype=int).reshape(-1)
            if monotone_s_dims.size == 1:
                monotone_s_dims = np.full(self.output_dim_, monotone_s_dims.item(), dtype=int)
        self.monotone_s_dims = monotone_s_dims
        self.monotone_input_dims_ = self.monotone_s_dims.copy()
        sigma = np.asarray(noise_std, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(q.shape, float(sigma))
        elif sigma.shape == (q.shape[0],):
            sigma = np.repeat(sigma[:, None], self.output_dim_, axis=1)
        else:
            sigma = np.broadcast_to(sigma, q.shape).copy() 
        if self.max_cache_size is None:
            self.max_cache_size = int(q.shape[0])
        if q.shape[0] > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s = s[keep]
            T = T[keep]
            q = q[keep]
            sigma = sigma[keep]
        X_raw = np.concatenate([s, T], axis=1)
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
        signal_kernel = ConstantKernel(self.kernel_variance,constant_value_bounds=(1e-4, 1e4)) * Matern(
            length_scale=lengthscale, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        sklearn_kernel = signal_kernel + WhiteKernel(noise_level=self.noise_variance, noise_level_bounds=(1e-10, 1e1))
        gp = GaussianProcessRegressor(kernel=sklearn_kernel, alpha=self.jitter, normalize_y=False,
                                      n_restarts_optimizer=self.n_restarts_optimizer,
                                      optimizer="fmin_l_bfgs_b" if self.optimize_hyperparameters else None,
                                      random_state=0)
        gp.fit(X, y)
        # Hyperparameters come from an unconstrained GP fit; EP is applied after.
        fitted_signal_kernel = gp.kernel_.k1
        fitted_white_kernel = gp.kernel_.k2
        self.variance_ = float(fitted_signal_kernel.k1.constant_value)
        self.lengthscales_ = np.asarray(fitted_signal_kernel.k2.length_scale, dtype=float,).reshape(-1)
        self.learned_noise_variance_ = float(fitted_white_kernel.noise_level)
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))
        self.ep_observation_noise_variance_ = max(self.learned_noise_variance_, self.jitter)
        self.learned_noise_variance_physical_ = (self.learned_noise_variance_ * self.y_scale_**2)
        self.learned_noise_std_physical_ = np.sqrt(self.learned_noise_variance_physical_)
        self.gp_kernel_ = gp.kernel_
        self.log_marginal_likelihood_ = float(gp.log_marginal_likelihood_value_)
        if self.reg_function > 0.0 and self.reg_function >= self.learned_noise_variance_:
            self.reg_function = min(0.9*self.learned_noise_variance_, self.reg_function * self.learned_noise_variance_)
        self.X_cache_raw_ = X_raw.copy()
        self.q_cache_raw_ = q.copy()
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.virtual_refinement_rounds_ = 0
        self.virtual_points_added_ = 0
        self.constraint_violation_fraction_ = np.nan
        self.constraint_worst_violation_ = np.nan
        self._fit_with_adaptive_virtual_refinement()
        return self
        
    def evaluate(self, s_q, T_q, return_variance=False):
        X_raw, output_shape = self._query_matrix(s_q, T_q)
        X = self._standardize_X(X_raw)
        n_query = X.shape[0]
        latent_standardized = np.empty((n_query, self.output_dim_), dtype=float)
        gradient_standardized = np.empty((n_query, self.output_dim_, self.input_dim_), dtype=float)
        if return_variance:
            variance_q_standardized = np.empty((n_query, self.output_dim_), dtype=float)
            variance_gradient_standardized = np.empty((n_query, self.output_dim_, self.input_dim_), dtype=float)
        for output_index in range(self.output_dim_):
            monotone_dim = int(self.monotone_input_dims_[output_index])
            value_covariance = np.hstack([self._dK_dy(X, self.X_virtual_, monotone_dim),
                                          self._K(X, self.X_train_)])
            latent_standardized[:, output_index] = (value_covariance @ self.alpha_[:, output_index])
            for dim in range(self.input_dim_):
                derivative_covariance = np.hstack([self._d2K_dxdy(X, self.X_virtual_, dim, monotone_dim),
                                                   self._dK_dx(X, self.X_train_, dim)])
                gradient_standardized[:, output_index, dim] = (derivative_covariance @ self.alpha_[:, output_index])
                if return_variance:
                    prior_projection = (solve_triangular(self.ep_prior_cholesky_[output_index],
                                                         derivative_covariance.T,
                                                         lower=True,
                                                         check_finite=False))
                    posterior_projection = (solve_triangular(self.ep_whitened_precision_cholesky_[output_index],
                                                             prior_projection,
                                                             lower=True,
                                                             check_finite=False))
                    prior_derivative_variance = ((5.0 / 3.0) * self.variance_ / self.lengthscales_[dim] ** 2)
                    variance_gradient_standardized[:, output_index, dim] = (prior_derivative_variance - np.sum(prior_projection**2, axis=0) 
                                                                            + np.sum(posterior_projection**2, axis=0))
            if return_variance:
                prior_projection = solve_triangular(self.ep_prior_cholesky_[output_index],
                                                    value_covariance.T,
                                                    lower=True,
                                                    check_finite=False)
                posterior_projection = solve_triangular(self.ep_whitened_precision_cholesky_[output_index],
                                                        prior_projection,
                                                        lower=True,
                                                        check_finite=False)
                variance_q_standardized[:, output_index] = (self.variance_ - np.sum(prior_projection**2, axis=0)
                                                            + np.sum(posterior_projection**2, axis=0))
        latent_physical = (self.y_mean_[None, :] + self.y_scale_[None, :] * latent_standardized)
        gradient_physical = (self.y_scale_[None, :, None] * gradient_standardized / self.x_scale_[None, None, :])
        sign = (-1.0 if self.learn_neg_flux else 1.0)
        q = sign * latent_physical
        dq_ds = (sign * gradient_physical[:, :, :self.s_dim_])
        dq_dT = (sign * gradient_physical[:, :, self.s_dim_:])
        result = (self._format_values(q, output_shape), self._format_jacobian(dq_ds, output_shape, self.s_dim_),
                  self._format_jacobian(dq_dT, output_shape, self.T_dim_))
        if not return_variance:
            return result
        variance_q = (np.maximum(variance_q_standardized, 0.0) * self.y_scale_[None, :] ** 2)
        variance_gradient = (np.maximum(variance_gradient_standardized, 0.0) * self.y_scale_[None, :, None] ** 2
                             / self.x_scale_[None, None, :] ** 2)
        return result + (self._format_values(variance_q, output_shape), 
                         self._format_jacobian(variance_gradient[:, :, :self.s_dim_], output_shape, self.s_dim_),
                        self._format_jacobian(variance_gradient[:, :, self.s_dim_:], output_shape, self.T_dim_))
    
    def update_posterior(self, s_new, T_new, q_new, *, s_query=None, T_query=None):
        s, T, q = self._training_arrays(s_new, T_new, q_new)
        X_raw_new = np.concatenate([s, T], axis=1)
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_q(q)
        if s_query is None and T_query is None:
            X_reference_raw = X_raw_new.mean(axis=0, keepdims=True)
        else:
            candidates, _ = self._query_matrix(s_query, T_query)
            X_reference_raw = candidates.mean(axis=0, keepdims=True)
        X_reference = self._standardize_X(X_reference_raw)
        old_cache = self.X_cache_raw_.copy()
        old_size = int(self.X_train_.shape[0])
        n_new = int(X_new.shape[0])
        capacity = self.max_cache_size
        derivative_tau = [values.copy() for values in (self.ep_derivative_site_precision_)]
        derivative_eta = [values.copy() for values in (self.ep_derivative_site_natural_parameter_)]
        evicted_old_points = np.empty((0, self.input_dim_))
        rejected_new_points = np.empty((0, self.input_dim_))
        if n_new >= capacity:
            # If the update alone fills the cache, keep the new points nearest the query.
            new_distances = np.linalg.norm((X_new - X_reference) / self.lengthscales_, axis=1)
            keep_new = np.argsort(new_distances)[:capacity]
            reject_new = np.setdiff1d(np.arange(n_new), keep_new)
            if old_size:
                self._delete_function_indices(np.arange(old_size))
            evicted_old_points = old_cache
            rejected_new_points = (X_raw_new[reject_new].copy())
            X_raw_new = X_raw_new[keep_new]
            q = q[keep_new]
            X_new = X_new[keep_new]
            y_new = y_new[keep_new]
            n_added = capacity
            n_dropped = old_size
        else:
            overflow = max(0, old_size + n_new - capacity)
            if overflow:
                # Drop old function observations farthest from the current query.
                old_distances = np.linalg.norm((self.X_train_ - X_reference) / self.lengthscales_, axis=1)
                evict_old = np.argsort(old_distances)[-overflow:]
                evicted_old_points = (old_cache[evict_old].copy())
                self._delete_function_indices(evict_old)
            n_added = n_new
            n_dropped = overflow
        self._append_function_block(X_new)
        self.X_cache_raw_ = np.vstack([self.X_cache_raw_, X_raw_new])
        self.q_cache_raw_ = np.vstack([self.q_cache_raw_, q])
        self.X_train_ = np.vstack([self.X_train_, X_new])
        self.y_train_ = np.vstack([self.y_train_, y_new])
        self._run_ep(derivative_tau=derivative_tau, derivative_eta=derivative_eta, max_iterations=self.online_ep_sweeps)
        self.posterior_updates_ += 1
        self.total_points_added_ += int(n_added)
        self.total_points_dropped_ += int(n_dropped)
        return {
            "n_added": int(n_added),
            "n_dropped": int(n_dropped),
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
            "ep_refinement_sweeps": self.ep_iterations_.copy(),
            "evicted_old_points": evicted_old_points,
            "rejected_new_points": rejected_new_points,
        }


