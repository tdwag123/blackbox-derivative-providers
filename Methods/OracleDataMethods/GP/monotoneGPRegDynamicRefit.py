"""
regularized monotone matern-5/2 GP with a moving observation cache.
- online updates keep the initial scaling, kernel, noise variance, and virtual points fixed.
- joint prior is ordered as [virtual derivatives, function observations] so that function 
  observations can be removed/appended w/o rebuilding complete covariance.
- existing derivative EP sites are retained and only a few warm-states EP refinement sweeps 
  are run after each update.
"""

import numpy as np
from scipy.linalg import cho_solve, qr_delete, solve_triangular
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
        max_cache_size=None,
    ):
        self.learn_neg_flux = learn_neg_flux
        self.n_virtual_per_axis = n_virtual_per_axis
        self.probit_nu = probit_nu
        self.ep_max_iter = ep_max_iter
        self.online_ep_sweeps = online_ep_sweeps
        self.ep_damping = ep_damping
        self.ep_tol = ep_tol
        self.jitter = jitter
        self.n_restarts_optimizer = n_restarts_optimizer
        self.reg_function = reg_function
        self.reg_derivative = reg_derivative
        self.kernel_variance = float(kernel_variance)
        self.lengthscale = lengthscale
        self.noise_variance = float(noise_variance)
        self.max_cache_size = (None if max_cache_size is None else int(max_cache_size))
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    @staticmethod
    def _training_arrays(s, T, q):
        s = np.asarray(s, dtype=float).reshape(-1)
        T = np.asarray(T, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(T)) 
                and np.all(np.isfinite(q))):
            raise ValueError("Observations must be finite.")
        return s, T, q

    def _standardize_X(self, X_raw):
        return (X_raw - self.x_mean_) / self.x_scale_

    def _standardize_q(self, q_raw):
        latent_raw = -q_raw if self.learn_neg_flux else q_raw
        return (latent_raw - self.y_mean_) / self.y_scale_

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

    def _run_ep(self, derivative_tau=None, derivative_eta=None, max_iterations=None):
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        tau=np.zeros(m+n)
        eta=np.zeros(m+n)
        if derivative_tau is not None:
            tau[:m] = np.asarray(derivative_tau, dtype=float).reshape(m)
        if derivative_eta is not None:
            eta[:m] = np.asarray(derivative_eta, dtype=float).reshape(m)
        tau[m:] = 1.0 / self.ep_observation_noise_variance_
        eta[m:] = self.y_train_ / self.ep_observation_noise_variance_
        iterations = self.ep_max_iter if max_iterations is None else int(max_iterations)
        for iteration in range(iterations):
            mean, variance, _ = self._posterior(self.ep_prior_cholesky_, tau, eta)
            old_tau = tau.copy()
            old_eta = eta.copy()
            for i in range(m):
                cavity_precision = 1.0 / variance[i] - old_tau[i]
                if cavity_precision <= 1e-12:
                    continue
                cavity_variance = 1.0 / cavity_precision
                cavity_mean = (mean[i] / variance[i] - old_eta[i]) * cavity_variance
                scale = np.sqrt(cavity_variance + self.probit_nu**2)
                z = cavity_mean / scale
                mills = np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))
                tilted_mean = cavity_mean + cavity_variance * mills / scale
                tilted_variance = cavity_variance * (
                    1.0
                    - cavity_variance * mills * (mills + z)
                    / (cavity_variance + self.probit_nu**2)
                )
                tilted_variance = max(tilted_variance, 1e-12)
                proposed_tau = 1.0 / tilted_variance - cavity_precision
                if proposed_tau < 0.0:
                    proposed_tau = 0.0
                    proposed_eta = 0.0
                else:
                    proposed_eta = tilted_mean / tilted_variance - cavity_mean / cavity_variance
                tau[i] = (1.0 - self.ep_damping) * old_tau[i] + self.ep_damping * proposed_tau
                eta[i] = (1.0 - self.ep_damping) * old_eta[i] + self.ep_damping * proposed_eta
            tau_change = np.max(np.abs(tau[:m] - old_tau[:m]) / (1.0 + np.abs(old_tau[:m])))
            eta_change = np.max(np.abs(eta[:m] - old_eta[:m]) / (1.0 + np.abs(old_eta[:m])))
            if max(tau_change, eta_change) < self.ep_tol:
                break
        mean, variance, alpha, C = self._posterior(self.ep_prior_cholesky_, tau, eta, return_cholesky=True)
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
        s_axis = np.linspace(X[:, 0].min(), X[:, 0].max(), self.n_virtual_per_axis)
        T_axis = np.linspace(X[:, 1].min(), X[:, 1].max(), self.n_virtual_per_axis)
        S, TT = np.meshgrid(s_axis, T_axis, indexing="ij")
        return np.column_stack([S.ravel(), TT.ravel()])

    def _build_initial_joint_prior(self):
        Kgg = self._d2K_dxdy(self.X_virtual_, self.X_virtual_, 0, 0)
        Kgf = self._dK_dx(self.X_virtual_, self.X_train_, 0)
        Kff = self._K(self.X_train_, self.X_train_)
        K_joint = np.block([[Kgg, Kgf], [Kgf.T, Kff]])
        K_joint = 0.5 * (K_joint + K_joint.T)
        m = self.X_virtual_.shape[0]
        n = self.X_train_.shape[0]
        diagonal = np.concatenate([np.full(m, self.reg_derivative + self.jitter),
                                   np.full(n, self.reg_function + self.jitter)])
        K_joint += np.diag(diagonal)
        self.ep_prior_cholesky_ = np.linalg.cholesky(K_joint)

    def _delete_joint_index(self, index):
        """O(N^2) cholesky deletion using QR deletion on R=L.T"""
        R = self.ep_prior_cholesky_.T
        Q = np.eye(R.shape[0])
        _, R_reduced = qr_delete(Q, R, index, 1, which="col", check_finite=False)
        # qr_delete keeps one trailing zero row; discard it to recover the
        # square upper-triangular factor of the principal submatrix
        R_new = R_reduced[:-1, :]
        signs = np.sign(np.diag(R_new))
        signs[signs == 0.0] = 1.0
        R_new = signs[:, None] * R_new
        self.ep_prior_cholesky_ = R_new.T

    def _drop_oldest_function_observation(self):
        if self.X_train_.shape[0] <= 1:
            raise ValueError("Cannot remove the only cached observation.")
        m = self.X_virtual_.shape[0]
        self._delete_joint_index(m)
        self.X_cache_raw_ = self.X_cache_raw_[1:]
        self.q_cache_raw_ = self.q_cache_raw_[1:]
        self.X_train_ = self.X_train_[1:]
        self.y_train_ = self.y_train_[1:]

    def _append_function_block(self, X_new):
        n_new = X_new.shape[0]
        covariance_old_new = np.vstack([
            self._dK_dx(self.X_virtual_, X_new, 0),
            self._K(self.X_train_, X_new),
        ])
        covariance_new_new = self._K(X_new, X_new)
        covariance_new_new += (self.reg_function + self.jitter) * np.eye(n_new)
        V = solve_triangular(
            self.ep_prior_cholesky_, covariance_old_new, lower=True, check_finite=False
        )
        schur = covariance_new_new - V.T @ V
        schur = 0.5 * (schur + schur.T)
        new_block = np.linalg.cholesky(schur)
        old_size = self.ep_prior_cholesky_.shape[0]
        full = np.zeros((old_size + n_new, old_size + n_new))
        full[:old_size, :old_size] = self.ep_prior_cholesky_
        full[old_size:, :old_size] = V.T
        full[old_size:, old_size:] = new_block
        self.ep_prior_cholesky_ = full

    def fit(self, s_train, T_train, q_train, *, noise_std=0.0):
        s = np.asarray(s_train, dtype=float).reshape(-1)
        T = np.asarray(T_train, dtype=float).reshape(-1)
        q = np.asarray(q_train, dtype=float).reshape(-1)
        if not (s.size == T.size == q.size):
            raise ValueError("Training arrays must have the same length.")
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma = np.full(q.size, sigma.item())
        if sigma.size != q.size:
            raise ValueError("noise_std must be scalar or match the training data.")
        if np.any(sigma < 0.0):
            raise ValueError("noise_std must be nonnegative.")
        if self.max_cache_size is None:
            self.max_cache_size = int(q.size)
        if q.size > self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            s, T, q, sigma = s[keep], T[keep], q[keep], sigma[keep]
        X_raw = np.column_stack([s, T])
        latent_raw = -q if self.learn_neg_flux else q
        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = latent_raw.mean()
        self.y_scale_ = latent_raw.std()
        if self.y_scale_ == 0.0:
            self.y_scale_ = 1.0
        X = self._standardize_X(X_raw)
        y = self._standardize_q(q)
        self.supplied_noise_variance_standardized_ = (sigma/self.y_scale_)**2           
        lengthscale = np.asarray(self.lengthscale, dtype=float)
        signal_kernel = ConstantKernel(self.kernel_variance, constant_value_bounds="fixed") * Matern(
            length_scale=lengthscale,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        sklearn_kernel = signal_kernel + WhiteKernel(
            noise_level=self.noise_variance,
            noise_level_bounds="fixed",
        )
        gp = GaussianProcessRegressor(
            kernel=sklearn_kernel,
            alpha=self.jitter,
            normalize_y=False,
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=0,
        )
        gp.fit(X, y)
        fitted_signal_kernel = gp.kernel_.k1
        fitted_white_kernel = gp.kernel_.k2
        self.variance_ = float(fitted_signal_kernel.k1.constant_value)
        self.lengthscales_ = np.asarray(fitted_signal_kernel.k2.length_scale, dtype=float)
        if self.lengthscales_.size == 1:
            self.lengthscales_ = np.full(2, float(self.lengthscales_.reshape(-1)[0]))
        self.learned_noise_variance_ = float(fitted_white_kernel.noise_level)
        self.learned_noise_std_ = float(np.sqrt(self.learned_noise_variance_))
        self.ep_observation_noise_variance_ = max(self.learned_noise_variance_, self.jitter)
        self.learned_noise_variance_physical_ = float(self.learned_noise_variance_ * self.y_scale_ ** 2)
        self.learned_noise_std_physical_ = float(np.sqrt(self.learned_noise_variance_physical_))
        self.gp_kernel_ = gp.kernel_
        self.log_marginal_likelihood_ = float(gp.log_marginal_likelihood_value_)
        # reg function should not dominate learned observation noise
        if self.reg_function >= self.learned_noise_variance_:
            self.reg_function = self.reg_function * self.learned_noise_variance_
        self.X_cache_raw_ = X_raw.copy()
        self.q_cache_raw_ = q.copy()
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.X_virtual_ = self._make_initial_virtual_grid(self.X_train_)
        self._build_initial_joint_prior()
        self._run_ep(max_iterations=self.ep_max_iter)
        return self
        
    def evaluate(self, s_q, T_q, return_variance=False):
        s, T = np.broadcast_arrays(np.asarray(s_q, dtype=float), 
                                   np.asarray(T_q, dtype=float))
        shape = s.shape
        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X = self._standardize_X(X_raw)
        value_covariance = np.hstack([self._dK_dy(X, self.X_virtual_, 0),
                                      self._K(X, self.X_train_)])
        latent = value_covariance @ self.alpha_
        gradient = np.empty((X.shape[0], 2), dtype=float)
        for dim in range(2):
            derivative_covariance = np.hstack([self._d2K_dxdy(X, self.X_virtual_, dim, 0),
                                               self._dK_dx(X, self.X_train_, dim)])
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
    
    def update_posterior(self, s_new, T_new, q_new):
        s, T, q = self._training_arrays(s_new, T_new, q_new)
        X_raw_new = np.column_stack([s, T])
        X_new = self._standardize_X(X_raw_new)
        y_new = self._standardize_q(q)
        n_requested = int(q.size)
        old_cache_size = int(self.X_train_.shape[0])
        derivative_tau = self.ep_derivative_site_precision_.copy()
        derivative_eta = self.ep_derivative_site_natural_parameter_.copy()
        # keep only the newest max_cache_size incoming points if the batch is huge
        if n_requested >= self.max_cache_size:
            keep = slice(-self.max_cache_size, None)
            X_raw_new = X_raw_new[keep]
            q = q[keep]
            X_new = X_new[keep]
            y_new = y_new[keep]
            n_requested = self.max_cache_size
            # remove every old function variable from the joint factor
            m = self.X_virtual_.shape[0]
            for _ in range(old_cache_size):
                self._delete_joint_index(m)
            self.X_cache_raw_ = np.empty((0, 2))
            self.q_cache_raw_ = np.empty(0)
            self.X_train_ = np.empty((0, 2))
            self.y_train_ = np.empty(0)
            dropped = old_cache_size
        else:
            overflow = max(0, old_cache_size + n_requested - self.max_cache_size)
            for _ in range(overflow):
                self._drop_oldest_function_observation()
            dropped = int(overflow)
        # append covariance using the current cache before arrays are extended
        self._append_function_block(X_new)
        self.X_cache_raw_ = np.vstack([self.X_cache_raw_, X_raw_new])
        self.q_cache_raw_ = np.concatenate([self.q_cache_raw_, q])
        self.X_train_ = np.vstack([self.X_train_, X_new])
        self.y_train_ = np.concatenate([self.y_train_, y_new])
        self._run_ep(
            derivative_tau=derivative_tau,
            derivative_eta=derivative_eta,
            max_iterations=self.online_ep_sweeps,
        )
        self.posterior_updates_ += 1
        self.total_points_added_ += n_requested
        self.total_points_dropped_ += dropped
        return {
            "n_added": int(n_requested),
            "n_dropped": int(dropped),
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
            "ep_refinement_sweeps": int(self.ep_iterations_),
        }

def main():
    """minimal smoke test"""
    def flux(s, T):
        return -(1.0 + 0.1 * T**2 + 0.05 * s**2) * s

    s_train = np.linspace(-1.0, 1.0, 8)
    T_train = np.linspace(0.5, 2.0, 8)
    q_train = flux(s_train, T_train)

    model = MonotoneGPFluxST(
        s_train,
        T_train,
        q_train,
        noise_std=0.02,
        n_virtual_per_axis=4,
        ep_max_iter=8,
        online_ep_sweeps=2,
        max_cache_size=8,
        reg_derivative=1e-3,
    )

    s_query = np.array([-0.5, 0.0, 0.5])
    T_query = np.array([1.0, 1.25, 1.5])
    before = model.evaluate(s_query, T_query, return_variance=True)

    s_new = np.array([0.8, 1.1])
    T_new = np.array([1.7, 1.9])
    info = model.update_posterior(s_new, T_new, flux(s_new, T_new))
    after = model.evaluate(s_query, T_query, return_variance=True)

    assert info["n_added"] == 2
    assert info["n_dropped"] == 2
    assert info["cache_size"] == 8
    assert np.all(np.isfinite(after[0]))
    assert np.all(np.isfinite(after[1]))
    assert np.all(np.isfinite(after[2]))
    assert np.all(np.isfinite(after[3]))

    print("Smoke test passed.")
    print("Update info:", info)
    print("q before:", before[0])
    print("q after: ", after[0])
    print("dq_ds before:", before[1])
    print("dq_ds after: ", after[1])
    print("dq_dT before:", before[2])
    print("dq_dT after: ", after[2])
    print("variance before:", before[3])
    print("variance after: ", after[3])


if __name__ == "__main__":
    main()

"""
SMOKE TEST RESULTS: 

Smoke test passed.
Update info: {'n_added': 2, 'n_dropped': 2, 'cache_size': 8, 'posterior_updates': 1, 'ep_refinement_sweeps': 2}
q before: [ 0.51260846 -0.00068646 -0.57036407]
q after:  [ 5.24008323e-01 -2.43274885e-04 -5.88403847e-01]
dq_ds before: [-0.89292014 -0.86332135 -1.00690144]
dq_ds after:  [-0.91928633 -0.95047314 -1.13751462]
dq_dT before: [-0.27807609 -0.36913484 -0.5325848 ]
dq_dT after:  [-0.11634375 -0.27559855 -0.39275411]
variance before: [0.0087818  0.00435991 0.00917881]
variance after:  [0.00987725 0.00442573 0.00701209]

"""
