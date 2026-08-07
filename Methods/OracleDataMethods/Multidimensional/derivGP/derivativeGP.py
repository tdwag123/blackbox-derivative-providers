from pathlib import Path
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

class DerivativeGPFluxST:
    """
    Multidimensional derivative GP for q(s,T).

    Intended thermal use:
        s in R^3, T in R, q in R^3.

    Each q_r is one scalar latent GP over x=(s,T), conditioned jointly on all
    first partial derivatives.  Thus each output component remains integrable.

    evaluate(..., return_variance=True) returns
        q, dq_ds, dq_dT, var_q, var_dq_ds, var_dq_dT.
    """
    def __init__(
        self,
        X_derivative,
        derivatives,
        derivative_noise_covariance,
        anchor_s,
        anchor_T,
        anchor_q,
        *,
        kernel_nu=1.5,
        k_ref=None,
        reg_derivative=0.0,
        jitter=1e-10,
        n_restarts_optimizer=0,
        random_state=42,
        max_cache_size=None,
    ):
        self.kernel_nu = float(kernel_nu)
        if self.kernel_nu not in (1.5, 2.5):
            raise ValueError("kernel_nu must be 1.5 or 2.5")
        self.reg_derivative = float(reg_derivative)
        self.jitter = float(jitter)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = int(random_state)
        self.max_cache_size = None if max_cache_size is None else int(max_cache_size)
        self.posterior_updates_ = 0
        self.total_points_added_ = 0
        self.total_points_dropped_ = 0
        self.fit(
            X_derivative,
            derivatives,
            derivative_noise_covariance,
            anchor_s,
            anchor_T,
            anchor_q,
            k_ref,
        )

    @classmethod
    def from_npz(cls, path, **kwargs):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                np.asarray(data["X_derivative"], dtype=float),
                np.asarray(data["derivatives"], dtype=float),
                np.asarray(data["derivative_noise_covariance"], dtype=float),
                np.asarray(data["anchor_s"], dtype=float),
                np.asarray(data["anchor_T"], dtype=float),
                np.asarray(data["anchor_q"], dtype=float),
                **kwargs,
            )
        return model

    def _parts(self, X, Y, variance=None, lengthscales=None):
        if variance is None:
            variance = self.variance_
        if lengthscales is None:
            lengthscales = self.lengthscales_
        delta = X[:, None, :] - Y[None, :, :]
        r = np.sqrt(np.sum((delta / lengthscales) ** 2, axis=2))
        return delta, r, float(variance), np.asarray(lengthscales)

    def _K(self, X, Y, variance=None, lengthscales=None):
        _, r, variance, _ = self._parts(X, Y, variance, lengthscales)
        if self.kernel_nu == 1.5:
            a = np.sqrt(3.0)
            return variance * (1.0 + a * r) * np.exp(-a * r)
        a = np.sqrt(5.0)
        return variance * (1.0 + a * r + 5.0 * r**2 / 3.0) * np.exp(-a * r)

    def _dK_dx(self, X, Y, dim, variance=None, lengthscales=None):
        delta, r, variance, ell = self._parts(X, Y, variance, lengthscales)
        if self.kernel_nu == 1.5:
            a = np.sqrt(3.0)
            factor = -variance * a**2 * np.exp(-a * r)
        else:
            a = np.sqrt(5.0)
            factor = -(a**2 / 3.0) * variance * (1.0 + a * r) * np.exp(-a * r)
        return factor * delta[:, :, dim] / ell[dim] ** 2

    def _dK_dy(self, X, Y, dim, variance=None, lengthscales=None):
        return -self._dK_dx(X, Y, dim, variance, lengthscales)

    def _d2K_dxdy(self, X, Y, dim_x, dim_y, variance=None, lengthscales=None):
        delta, r, variance, ell = self._parts(X, Y, variance, lengthscales)
        lx, ly = ell[dim_x], ell[dim_y]
        same = float(dim_x == dim_y)
        if self.kernel_nu == 1.5:
            a = np.sqrt(3.0)
            diagonal = a**2 * same / lx**2
            outer = np.zeros_like(r)
            mask = r > 0.0
            outer[mask] = (
                a**3 * delta[:, :, dim_x][mask] * delta[:, :, dim_y][mask]
                / (lx**2 * ly**2 * r[mask])
            )
            return variance * np.exp(-a * r) * (diagonal - outer)
        a = np.sqrt(5.0)
        diagonal = (a**2 / 3.0) * (1.0 + a * r) * same / lx**2
        outer = (a**4 / 3.0) * delta[:, :, dim_x] * delta[:, :, dim_y] / (lx**2 * ly**2)
        return variance * np.exp(-a * r) * (diagonal - outer)

    def _joint_covariance(self, variance, lengthscales):
        return np.block([
            [self._d2K_dxdy(self.Z_, self.Z_, i, j, variance, lengthscales)
             for j in range(self.input_dim_)]
            for i in range(self.input_dim_)
        ])

    def _physics_derivative_mean(self, n):
        mean = np.zeros((n, self.output_dim_, self.input_dim_))
        for r in range(min(self.output_dim_, self.s_dim_)):
            mean[:, r, r] = -self.k_ref_
        return mean

    def _stack_derivatives(self, values):
        # [all dx0 centers, all dx1 centers, ...], one column per q component
        return values.transpose(2, 0, 1).reshape(self.input_dim_ * values.shape[0], self.output_dim_)

    def _objective(self, log_parameters):
        variance = np.exp(log_parameters[0])
        lengthscales = np.exp(log_parameters[1:])
        K = self._joint_covariance(variance, lengthscales)
        total = 0.0
        for r in range(self.output_dim_):
            A = K + self.noise_covariance_standardized_[r]
            A += (self.reg_derivative + self.jitter) * np.eye(A.shape[0])
            try:
                L = np.linalg.cholesky(0.5 * (A + A.T))
            except np.linalg.LinAlgError:
                return 1e100
            residual = self.derivative_residual_standardized_[:, r]
            alpha = cho_solve((L, True), residual)
            total += 0.5 * residual @ alpha + np.sum(np.log(np.diag(L)))
            total += 0.5 * A.shape[0] * np.log(2.0 * np.pi)
        return float(total)

    def fit(self, X_derivative, derivatives, derivative_noise_covariance,
            anchor_s, anchor_T, anchor_q, k_ref):
        X = np.asarray(X_derivative, dtype=float)
        D = np.asarray(derivatives, dtype=float)
        C = np.asarray(derivative_noise_covariance, dtype=float)
        if X.ndim != 2 or D.ndim != 3:
            raise ValueError("X_derivative must be (n,d_input) and derivatives must be (n,d_q,d_input)")
        if D.shape[0] != X.shape[0] or D.shape[2] != X.shape[1]:
            raise ValueError("derivative tensor dimensions do not match X")
        self.anchor_s_ = np.asarray(anchor_s, dtype=float).reshape(-1)
        self.anchor_T_ = np.asarray(anchor_T, dtype=float).reshape(-1)
        self.anchor_q_ = np.asarray(anchor_q, dtype=float).reshape(-1)
        self.s_dim_ = self.anchor_s_.size
        self.T_dim_ = self.anchor_T_.size
        self.input_dim_ = X.shape[1]
        self.output_dim_ = D.shape[1]
        if self.input_dim_ != self.s_dim_ + self.T_dim_:
            raise ValueError("X dimension must equal d_s + d_T")
        if self.anchor_q_.size != self.output_dim_:
            raise ValueError("anchor_q dimension must equal d_q")
        self.X_derivative_raw_ = X.copy()
        self.derivatives_raw_ = D.copy()
        self.derivative_noise_covariance_raw_ = C.copy()
        self.n_centers_ = X.shape[0]
        expected = (self.output_dim_, self.n_centers_ * self.input_dim_, self.n_centers_ * self.input_dim_)
        if C.shape != expected:
            raise ValueError(f"derivative_noise_covariance must have shape {expected}")
        if self.max_cache_size is None:
            self.max_cache_size = self.n_centers_
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.Z_ = (X - self.x_mean_) / self.x_scale_
        if k_ref is None:
            diag = np.concatenate([-D[:, r, r] for r in range(min(self.s_dim_, self.output_dim_))])
            self.k_ref_ = max(float(np.median(diag)), 1e-8)
        else:
            self.k_ref_ = float(k_ref)
        transformed = D * self.x_scale_[None, None, :]
        self.q_scale_ = np.sqrt(np.mean(transformed**2, axis=(0, 2)))
        self.q_scale_[self.q_scale_ == 0.0] = 1.0
        mean = self._physics_derivative_mean(self.n_centers_)
        standardized = (D - mean) * self.x_scale_[None, None, :] / self.q_scale_[None, :, None]
        self.derivative_residual_standardized_ = self._stack_derivatives(standardized)
        scale_by_dim = np.repeat(self.x_scale_, self.n_centers_)
        self.noise_covariance_standardized_ = np.empty_like(C)
        for r in range(self.output_dim_):
            scale = scale_by_dim / self.q_scale_[r]
            self.noise_covariance_standardized_[r] = scale[:, None] * C[r] * scale[None, :]
        bounds = [(np.log(1e-4), np.log(1e4))] + [(np.log(0.1), np.log(10.0))] * self.input_dim_
        initial_variance = max(float(np.var(self.derivative_residual_standardized_)), 1e-3)
        starts = [np.log(np.concatenate([[initial_variance], np.ones(self.input_dim_)]))]
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_restarts_optimizer):
            starts.append(np.array([rng.uniform(a, b) for a, b in bounds]))
        best = None
        for start in starts:
            result = minimize(self._objective, start, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
        self.variance_ = float(np.exp(best.x[0]))
        self.lengthscales_ = np.exp(best.x[1:])
        self.negative_log_marginal_likelihood_ = float(best.fun)
        self._rebuild_cached_posterior()
        return self

    def _rebuild_cached_posterior(self):
        self.n_centers_ = self.X_derivative_raw_.shape[0]
        self.Z_ = (self.X_derivative_raw_ - self.x_mean_) / self.x_scale_
        mean = self._physics_derivative_mean(self.n_centers_)
        standardized = (self.derivatives_raw_ - mean) * self.x_scale_[None, None, :] / self.q_scale_[None, :, None]
        self.derivative_residual_standardized_ = self._stack_derivatives(standardized)
        scale_by_dim = np.repeat(self.x_scale_, self.n_centers_)
        K = self._joint_covariance(self.variance_, self.lengthscales_)
        self.L_ = []
        self.alpha_ = np.empty((K.shape[0], self.output_dim_))
        self.noise_covariance_standardized_ = np.empty_like(self.derivative_noise_covariance_raw_)
        for r in range(self.output_dim_):
            scale = scale_by_dim / self.q_scale_[r]
            C = scale[:, None] * self.derivative_noise_covariance_raw_[r] * scale[None, :]
            self.noise_covariance_standardized_[r] = C
            A = K + C + (self.reg_derivative + self.jitter) * np.eye(K.shape[0])
            L = np.linalg.cholesky(0.5 * (A + A.T))
            self.L_.append(L)
            self.alpha_[:, r] = cho_solve((L, True), self.derivative_residual_standardized_[:, r])

        anchor = np.concatenate([self.anchor_s_, self.anchor_T_])[None, :]
        anchor = (anchor - self.x_mean_) / self.x_scale_
        K_anchor = np.hstack([self._dK_dy(anchor, self.Z_, j) for j in range(self.input_dim_)])
        self.anchor_residual_standardized_ = np.array([
            float((K_anchor @ self.alpha_[:, r]).item()) for r in range(self.output_dim_)
        ])
        self.cache_size_ = self.n_centers_
        self.alpha_norm_ = float(np.linalg.norm(self.alpha_))

    def _query(self, s, T):
        s = np.asarray(s, dtype=float)
        T = np.asarray(T, dtype=float)
        if self.s_dim_ == 1:
            if s.ndim == 0:
                shape = (); s = s.reshape(1, 1)
            elif s.ndim == 1:
                shape = s.shape; s = s[..., None]
            else:
                shape = s.shape[:-1]
        else:
            if s.shape[-1] != self.s_dim_:
                raise ValueError(f"s must have trailing dimension {self.s_dim_}")
            shape = s.shape[:-1]
        s = np.broadcast_to(s, shape + (self.s_dim_,))
        if self.T_dim_ == 1:
            if T.ndim == len(shape):
                T = np.broadcast_to(T, shape)[..., None]
            elif T.ndim == len(shape) + 1 and T.shape[-1] == 1:
                T = np.broadcast_to(T, shape + (1,))
            elif T.ndim == 0:
                T = np.broadcast_to(T, shape)[..., None]
            else:
                raise ValueError("T shape is incompatible with s")
        else:
            T = np.broadcast_to(T, shape + (self.T_dim_,))
        X_raw = np.concatenate([s.reshape(-1, self.s_dim_), T.reshape(-1, self.T_dim_)], axis=1)
        return X_raw, (X_raw - self.x_mean_) / self.x_scale_, shape

    def _query_derivative_covariance(self, X, query_dim):
        return np.hstack([
            self._d2K_dxdy(X, self.Z_, query_dim, data_dim)
            for data_dim in range(self.input_dim_)
        ])

    def predict_derivatives(self, s, T):
        _, X, shape = self._query(s, T)
        J = np.empty((X.shape[0], self.output_dim_, self.input_dim_))
        for dim in range(self.input_dim_):
            K_dim = self._query_derivative_covariance(X, dim)
            for r in range(self.output_dim_):
                baseline = -self.k_ref_ if dim < self.s_dim_ and r == dim else 0.0
                J[:, r, dim] = baseline + self.q_scale_[r] * (K_dim @ self.alpha_[:, r]) / self.x_scale_[dim]
        dq_ds = J[:, :, :self.s_dim_].reshape(shape + (self.output_dim_, self.s_dim_))
        dq_dT = J[:, :, self.s_dim_:]
        if self.T_dim_ == 1:
            dq_dT = dq_dT[:, :, 0].reshape(shape + (self.output_dim_,))
        else:
            dq_dT = dq_dT.reshape(shape + (self.output_dim_, self.T_dim_))
        return dq_ds, dq_dT

    def predict_flux(self, s, T):
        X_raw, X, shape = self._query(s, T)
        K_q = np.hstack([self._dK_dy(X, self.Z_, j) for j in range(self.input_dim_)])
        baseline = np.broadcast_to(self.anchor_q_[None, :], (X.shape[0], self.output_dim_)).copy()
        for r in range(min(self.output_dim_, self.s_dim_)):
            baseline[:, r] -= self.k_ref_ * (X_raw[:, r] - self.anchor_s_[r])
        q = np.empty_like(baseline)
        for r in range(self.output_dim_):
            residual = K_q @ self.alpha_[:, r] - self.anchor_residual_standardized_[r]
            q[:, r] = baseline[:, r] + self.q_scale_[r] * residual
        return q.reshape(shape + (self.output_dim_,))

    def predict_variance(self, s, T):
        _, X, shape = self._query(s, T)
        var_q = np.empty((X.shape[0], self.output_dim_))
        var_J = np.empty((X.shape[0], self.output_dim_, self.input_dim_))
        anchor = np.concatenate([self.anchor_s_, self.anchor_T_])[None, :]
        anchor = (anchor - self.x_mean_) / self.x_scale_
        K_q = np.hstack([self._dK_dy(X, self.Z_, j) for j in range(self.input_dim_)])
        K_anchor = np.hstack([self._dK_dy(anchor, self.Z_, j) for j in range(self.input_dim_)])
        K_delta = K_q - K_anchor
        prior_flux = 2.0 * self.variance_ - 2.0 * self._K(X, anchor).reshape(-1)
        prior_factor = 3.0 if self.kernel_nu == 1.5 else 5.0 / 3.0
        for r in range(self.output_dim_):
            solved_q = cho_solve((self.L_[r], True), K_delta.T)
            var_q_std = prior_flux - np.sum(K_delta * solved_q.T, axis=1)
            var_q[:, r] = self.q_scale_[r]**2 * np.maximum(var_q_std, 0.0)
            for dim in range(self.input_dim_):
                K_dim = self._query_derivative_covariance(X, dim)
                solved = cho_solve((self.L_[r], True), K_dim.T)
                prior = prior_factor * self.variance_ / self.lengthscales_[dim]**2
                var_std = prior - np.sum(K_dim * solved.T, axis=1)
                var_J[:, r, dim] = (self.q_scale_[r] / self.x_scale_[dim])**2 * np.maximum(var_std, 0.0)
        var_ds = var_J[:, :, :self.s_dim_].reshape(shape + (self.output_dim_, self.s_dim_))
        var_dT = var_J[:, :, self.s_dim_:]
        if self.T_dim_ == 1:
            var_dT = var_dT[:, :, 0].reshape(shape + (self.output_dim_,))
        else:
            var_dT = var_dT.reshape(shape + (self.output_dim_, self.T_dim_))
        return var_q.reshape(shape + (self.output_dim_,)), var_ds, var_dT

    @staticmethod
    def _combine_covariance(old_cov, new_cov, n_old, n_new, d_input):
        d_q = old_cov.shape[0]
        total = n_old + n_new
        out = np.zeros((d_q, d_input * total, d_input * total))
        for r in range(d_q):
            for i in range(d_input):
                for j in range(d_input):
                    oi = slice(i*n_old, (i+1)*n_old); oj = slice(j*n_old, (j+1)*n_old)
                    ni = slice(i*n_new, (i+1)*n_new); nj = slice(j*n_new, (j+1)*n_new)
                    ci_old = slice(i*total, i*total+n_old); cj_old = slice(j*total, j*total+n_old)
                    ci_new = slice(i*total+n_old, (i+1)*total); cj_new = slice(j*total+n_old, (j+1)*total)
                    out[r, ci_old, cj_old] = old_cov[r, oi, oj]
                    out[r, ci_new, cj_new] = new_cov[r, ni, nj]
        return out

    def update_posterior(self, field, *, s_query=None, T_query=None):
        close = False
        if isinstance(field, (str, Path)):
            data = np.load(Path(field), allow_pickle=False); close = True
        else:
            data = field
        try:
            X_new = np.asarray(data["X_derivative"], dtype=float)
            D_new = np.asarray(data["derivatives"], dtype=float)
            C_new = np.asarray(data["derivative_noise_covariance"], dtype=float)
            anchor_s_new = np.asarray(data["anchor_s"], dtype=float).reshape(-1)
            anchor_T_new = np.asarray(data["anchor_T"], dtype=float).reshape(-1)
            anchor_q_new = np.asarray(data["anchor_q"], dtype=float).reshape(-1)
        finally:
            if close:
                data.close()
        if X_new.shape[1] != self.input_dim_ or D_new.shape[1:] != (self.output_dim_, self.input_dim_):
            raise ValueError("new derivative field dimensions do not match model")
        n_old, n_new = self.X_derivative_raw_.shape[0], X_new.shape[0]
        total, capacity = n_old + n_new, self.max_cache_size
        X_all = np.vstack([self.X_derivative_raw_, X_new])
        D_all = np.concatenate([self.derivatives_raw_, D_new], axis=0)
        C_all = self._combine_covariance(self.derivative_noise_covariance_raw_, C_new, n_old, n_new, self.input_dim_)
        if s_query is None and T_query is None:
            query_raw = np.concatenate([anchor_s_new, anchor_T_new])
        elif s_query is None or T_query is None:
            raise ValueError("s_query and T_query must be supplied together")
        else:
            query_raw = np.concatenate([np.asarray(s_query, dtype=float).reshape(-1), np.asarray(T_query, dtype=float).reshape(-1)])
        query = (query_raw - self.x_mean_) / self.x_scale_
        if total <= capacity:
            keep = np.arange(total)
        elif n_new >= capacity:
            Xn = (X_new - self.x_mean_) / self.x_scale_
            keep_new = np.argsort(np.linalg.norm((Xn - query) / self.lengthscales_, axis=1))[:capacity]
            keep = n_old + keep_new
        else:
            old_slots = capacity - n_new
            Xo = (self.X_derivative_raw_ - self.x_mean_) / self.x_scale_
            keep_old = np.sort(np.argsort(np.linalg.norm((Xo - query) / self.lengthscales_, axis=1))[:old_slots])
            keep = np.concatenate([keep_old, np.arange(n_old, n_old+n_new)])
        kept_old = keep[keep < n_old]
        evicted = np.setdiff1d(np.arange(n_old), kept_old)
        cov_idx = np.concatenate([dim * total + keep for dim in range(self.input_dim_)])
        self.X_derivative_raw_ = X_all[keep].copy()
        self.derivatives_raw_ = D_all[keep].copy()
        self.derivative_noise_covariance_raw_ = C_all[:, cov_idx][:, :, cov_idx].copy()
        self.anchor_s_, self.anchor_T_, self.anchor_q_ = anchor_s_new, anchor_T_new, anchor_q_new
        self._rebuild_cached_posterior()
        self.posterior_updates_ += 1
        self.total_points_added_ += int(np.sum(keep >= n_old))
        self.total_points_dropped_ += int(evicted.size)
        return {
            "n_added": int(np.sum(keep >= n_old)),
            "n_dropped": int(evicted.size),
            "cache_size": int(self.cache_size_),
            "posterior_updates": int(self.posterior_updates_),
            "evicted_old_points": X_all[evicted].copy(),
        }

    def evaluate(self, s, T, return_variance=False):
        dq_ds, dq_dT = self.predict_derivatives(s, T)
        q = self.predict_flux(s, T)
        if not return_variance:
            return q, dq_ds, dq_dT
        var_q, var_dq_ds, var_dq_dT = self.predict_variance(s, T)
        return q, dq_ds, dq_dT, var_q, var_dq_ds, var_dq_dT
