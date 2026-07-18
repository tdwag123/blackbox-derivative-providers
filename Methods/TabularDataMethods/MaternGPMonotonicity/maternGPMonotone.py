# take 2 :D

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.linalg.lapack import dpocon
from scipy.special import log_ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class GPError(RuntimeError):
    """Raised when the GP fit or a numerical validation is unreliable."""


def _points(X, name, columns=2):
    value = np.asarray(X, dtype=float)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n, {columns})")
    if value.shape[0] == 0 or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain finite points")
    return value


def _condition_from_cholesky(cholesky, matrix_one_norm):
    reciprocal, info = dpocon(cholesky, float(matrix_one_norm), uplo="L")
    if info != 0 or not np.isfinite(reciprocal) or reciprocal <= 0.0:
        return np.inf
    return float(1.0 / reciprocal)


class Matern52:
    def __init__(self, variance, lengthscales):
        self.variance = float(variance)
        self.lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)

        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("kernel variance must be positive")

        if (
            self.lengthscales.size == 0
            or np.any(~np.isfinite(self.lengthscales))
            or np.any(self.lengthscales <= 0.0)
        ):
            raise ValueError("kernel lengthscales must be positive")

    def _delta_and_distance(self, X, Y):
        X = _points(X, "X", self.lengthscales.size)
        Y = _points(Y, "Y", self.lengthscales.size)

        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have same number of columns")

        if X.shape[1] != self.lengthscales.size:
            raise ValueError("input dimension doesn't match kernel")

        delta = X[:, None, :] - Y[None, :, :]
        scaled_delta = delta/self.lengthscales
        r = np.sqrt(np.sum(scaled_delta**2, axis=2))

        return delta, r

    def K(self, X, Y):
        _, r = self._delta_and_distance(X, Y)
        a = np.sqrt(5.0)

        return(self.variance*(1.0 + a*r + 5.0*r**2 / 3.0) * np.exp(-a*r))

    def dK_dx(self, X, Y, dim):
        if not 0 <= dim < self.lengthscales.size:
            raise ValueError("invalid derivative dimension")
        delta, r = self._delta_and_distance(X, Y)
        a = np.sqrt(5.0)
        lengthscale = self.lengthscales[dim]

        factor = (-(5.0/3.0)*self.variance * (1.0 + a * r) * np.exp(-a*r))

        return (factor * delta[:,:,dim]/lengthscale**2)

    def dK_dy(self, X, Y, dim):
        return -self.dK_dx(X, Y, dim)

    def d2K_dxdy(self, X, Y, dim_x, dim_y):
        dimensions = self.lengthscales.size
        if not (0 <= dim_x < dimensions and 0 <= dim_y < dimensions):
            raise ValueError("invalid derivative dimension")
        delta, r = self._delta_and_distance(X, Y)
        a = np.sqrt(5.0)
        lx = self.lengthscales[dim_x]
        ly = self.lengthscales[dim_y]

        same_dimension = float(dim_x == dim_y)

        diagonal_term = ((5.0/3.0) * (1.0 + a*r)*same_dimension/lx**2)
        outer_term = ((25.0/3.0)* delta[:,:,dim_x]*delta[:,:,dim_y]/(lx**2 * ly**2))

        return (self.variance * np.exp(-a*r) * (diagonal_term - outer_term))

def stable_cholesky(
    matrix,
    base_jitter=1e-12,
    max_attempts=12,
    max_relative_jitter=1e-3,
):
    matrix = np.asarray(matrix, dtype=float)
    if(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]):
        raise ValueError("cholesky input must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cholesky input entries must be finite")

    matrix = 0.5*(matrix+matrix.T)
    identity = np.eye(matrix.shape[0])

    if base_jitter <= 0.0 or max_attempts < 1 or max_relative_jitter <= 0.0:
        raise ValueError("invalid Cholesky jitter controls")

    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    max_jitter = max_relative_jitter * scale
    jitter = 0.0

    for _ in range(max_attempts):
        try:
            factor = np.linalg.cholesky(matrix + jitter * identity)
            return factor, jitter
        except np.linalg.LinAlgError:
            if jitter == 0.0:
                jitter = base_jitter * scale
            else:
                jitter *= 10.0
            if jitter > max_jitter:
                break

    raise GPError(
        "Cholesky failed within the allowed jitter; "
        f"final jitter={jitter:.3e}, maximum={max_jitter:.3e}"
    )

def posterior_from_sites(
    prior_cholesky,
    precision,
    eta,
    n_function,
    *,
    compute_alpha=False,
    max_relative_jitter=1e-3,
):
    precision = np.asarray(precision, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    L = np.asarray(prior_cholesky, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("prior_cholesky must be square")
    if not 0 < n_function < L.shape[0]:
        raise ValueError("n_function must split function and derivative sites")
    if precision.shape != (L.shape[0],) or eta.shape != (L.shape[0],):
        raise ValueError("site vectors must match the prior dimension")
    if (
        np.any(~np.isfinite(L))
        or np.any(~np.isfinite(precision))
        or np.any(~np.isfinite(eta))
        or np.any(precision < 0.0)
    ):
        raise ValueError("site system must be finite with nonnegative precision")

    weighted_L = (np.sqrt(precision)[:, None] * L)
    system = (np.eye(L.shape[0]) + weighted_L.T @ weighted_L)
    system_cholesky, posterior_jitter = stable_cholesky(
        system,
        max_relative_jitter=max_relative_jitter,
    )
    natural_white = L.T @ eta
    mean_white = cho_solve((system_cholesky, True), natural_white, check_finite=False)
    derivative_cholesky = L[n_function:, :]
    derivative_mean = derivative_cholesky @ mean_white
    derivative_white = solve_triangular(
        system_cholesky,
        derivative_cholesky.T,
        lower=True,
        check_finite=False,
    )
    derivative_variance = np.sum(derivative_white**2, axis=0)
    alpha = None
    if compute_alpha:
        alpha = solve_triangular(L.T, mean_white, lower=False, check_finite=False)

    return derivative_mean, derivative_variance, alpha, posterior_jitter

def make_regularization_precision(K_function, K_derivative, function_strength, derivative_strength):
    if (
        not np.isfinite(function_strength)
        or not np.isfinite(derivative_strength)
        or function_strength < 0.0
        or derivative_strength < 0.0
    ):
        raise ValueError("regularization strengths must be finite and nonnegative")
    function_variance = float(np.median(np.diag(K_function)))
    derivative_variance = float(np.median(np.diag(K_derivative)))
    if not np.isfinite(function_variance) or function_variance <= 0.0:
        raise GPError("function prior variance must be positive")
    if not np.isfinite(derivative_variance) or derivative_variance <= 0.0:
        raise GPError("derivative prior variance must be positive")
    function_precision = np.full(K_function.shape[0], function_strength/function_variance)
    derivative_precision = np.full(K_derivative.shape[0], derivative_strength/derivative_variance)
    return np.concatenate([function_precision, derivative_precision])

def _inverse_mills_terms(z):
    z = float(z)
    if not np.isfinite(z):
        raise GPError("inverse Mills ratio received a nonfinite value")
    if z < -10.0:
        inverse = -1.0 / z
        tail = inverse - 2.0 * inverse**3 + 10.0 * inverse**5 - 74.0 * inverse**7
        return -z + tail, tail
    ratio = float(np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z)))
    return ratio, ratio + z


def make_virtual_grid(X, n_per_axis):
    X = _points(X, "X")
    if n_per_axis < 2:
        raise ValueError("n_per_axis must be at least 2")
    s_axis = np.linspace(X[:,0].min(), X[:,0].max(), n_per_axis)
    T_axis = np.linspace(X[:,1].min(), X[:,1].max(), n_per_axis)
    s, TT = np.meshgrid(s_axis, T_axis, indexing="ij")
    return np.column_stack([s.ravel(), TT.ravel()])


def _select_new_virtual_points(
    existing,
    candidates,
    constrained_derivative,
    max_points,
    minimum_separation,
):
    existing = _points(existing, "existing virtual points")
    candidates = _points(candidates, "candidate virtual points")
    values = np.asarray(constrained_derivative, dtype=float).reshape(-1)
    if values.shape != (candidates.shape[0],) or np.any(~np.isfinite(values)):
        raise ValueError("candidate derivatives must be finite and match candidates")

    width = np.maximum(np.ptp(candidates, axis=0), 1.0)
    separation = max(float(minimum_separation), 100.0 * np.finfo(float).eps)
    selected = []
    for index in np.argsort(values):
        point = candidates[index]
        if np.min(np.linalg.norm((existing - point) / width, axis=1)) < separation:
            continue
        if selected:
            selected_array = np.asarray(selected)
            if np.min(np.linalg.norm((selected_array - point) / width, axis=1)) < separation:
                continue
        selected.append(point)
        if len(selected) >= max_points:
            break

    if not selected:
        return np.empty((0, candidates.shape[1]))
    return np.asarray(selected, dtype=float)

def run_ep(
        K_joint,
        y,
        noise_variance,
        regularization_precision,
        *,
        probit_nu,
        max_iterations,
        damping,
        tolerance,
        max_relative_jitter=1e-3,
):
    K_joint = np.asarray(K_joint, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    noise_variance = np.asarray(noise_variance, dtype=float).reshape(-1)
    regularization_precision = np.asarray(regularization_precision, dtype=float).reshape(-1)
    n_function = y.size
    if K_joint.ndim != 2 or K_joint.shape[0] != K_joint.shape[1]:
        raise ValueError("K_joint must be square")
    n_total = K_joint.shape[0]
    n_derivative = n_total - n_function

    if n_function == 0 or n_derivative <= 0:
        raise ValueError("EP requires function observations and derivative constraints")
    if noise_variance.shape != (n_function,):
        raise ValueError("expected one noise variance per observation")
    if regularization_precision.shape != (n_total,):
        raise ValueError("regularization precision must match K_joint")
    if (
        np.any(~np.isfinite(K_joint))
        or np.any(~np.isfinite(y))
        or np.any(~np.isfinite(noise_variance))
        or np.any(noise_variance <= 0.0)
        or np.any(~np.isfinite(regularization_precision))
        or np.any(regularization_precision < 0.0)
    ):
        raise ValueError("EP inputs must be finite with valid nonnegative precisions")
    if probit_nu <= 0.0 or max_iterations < 1 or not 0.0 < damping <= 1.0 or tolerance <= 0.0:
        raise ValueError("invalid EP controls")

    K_joint = 0.5 * (K_joint + K_joint.T)
    prior_cholesky, prior_jitter = stable_cholesky(
        K_joint,
        max_relative_jitter=max_relative_jitter,
    )
    shifted_joint = K_joint + prior_jitter * np.eye(n_total)
    joint_condition = _condition_from_cholesky(
        prior_cholesky,
        np.linalg.norm(shifted_joint, 1),
    )

    tau = np.zeros(n_total)
    eta = np.zeros(n_total)

    tau[:n_function] = 1.0/noise_variance
    eta[:n_function] = y/noise_variance

    converged = False
    posterior_jitter = 0.0
    largest_change = np.inf

    for iteration in range(1, max_iterations + 1):
        derivative_mean, derivative_variance, _, posterior_jitter = posterior_from_sites(
            prior_cholesky,
            tau + regularization_precision,
            eta,
            n_function,
            max_relative_jitter=max_relative_jitter,
        )
        next_tau = tau.copy()
        next_eta = eta.copy()
        largest_change = 0.0

        for j in range(n_derivative):
            index = n_function + j
            marginal_mean = float(derivative_mean[j])
            marginal_variance = float(derivative_variance[j])

            if(not np.isfinite(marginal_variance) or marginal_variance <= 0.0):
                raise RuntimeError("EP produced an invalid marginal variance")

            cavity_precision = (1.0/marginal_variance - tau[index])
            cavity_eta = (marginal_mean/marginal_variance - eta[index])

            if(not np.isfinite(cavity_precision) or cavity_precision <= 0.0):
                raise RuntimeError("EP produced a nonpositive cavity precision; try larger probit_nu, or smaller damping")

            cavity_variance = 1.0/cavity_precision
            cavity_mean = cavity_eta/cavity_precision
            denominator = np.sqrt(cavity_variance + probit_nu**2)
            z = cavity_mean/denominator
            ratio, ratio_plus_z = _inverse_mills_terms(z)

            tilted_mean = (cavity_mean + cavity_variance * ratio / denominator)
            correction = ratio * ratio_plus_z
            if not np.isfinite(correction) or not -1e-10 <= correction <= 1.0 + 1e-8:
                raise GPError("EP produced an invalid probit correction")
            correction = float(np.clip(correction, 0.0, 1.0))
            tilted_variance = cavity_variance * (
                1.0
                - cavity_variance * correction
                / (cavity_variance + probit_nu**2)
            )
            variance_floor = (1e-14 * max(1.0, cavity_variance))

            if not np.isfinite(tilted_variance) or tilted_variance <= variance_floor:
                raise GPError("EP tilted variance collapsed; try larger probit_nu or smaller damping")

            proposed_tau = (1.0/tilted_variance - cavity_precision)
            proposed_eta = tilted_mean/tilted_variance - cavity_eta
            negative_tolerance = (1e-10 * max(1.0, cavity_precision))

            if proposed_tau < -negative_tolerance:
                raise GPError("EP produced a negative site precision")

            if proposed_tau < 0.0:
                proposed_tau, proposed_eta = 0.0, 0.0
            else:
                proposed_tau = float(proposed_tau)
                proposed_eta = float(proposed_eta)
            if not np.isfinite(proposed_tau) or not np.isfinite(proposed_eta):
                raise GPError("EP produced nonfinite site parameters")

            updated_tau = (1.0-damping)*tau[index] + damping*proposed_tau
            updated_eta = (1.0-damping)*eta[index] + damping*proposed_eta

            tau_change = (abs(updated_tau - tau[index])/(1.0+abs(tau[index])))
            eta_change = (abs(updated_eta - eta[index])/(1.0+abs(eta[index])))

            largest_change = max(largest_change, tau_change, eta_change)

            next_tau[index] = updated_tau
            next_eta[index] = updated_eta

        tau = next_tau
        eta = next_eta

        if largest_change < tolerance:
            converged = True
            break

    if not converged:
        raise GPError(
            f"EP did not converge within {max_iterations} iterations; "
            f"last change={largest_change:.3e}"
        )

    _, _, alpha, posterior_jitter = posterior_from_sites(
        prior_cholesky,
        tau + regularization_precision,
        eta,
        n_function,
        compute_alpha=True,
        max_relative_jitter=max_relative_jitter,
    )

    return{"alpha": alpha,
           "iterations": iteration,
           "prior_jitter": prior_jitter,
           "posterior_jitter": posterior_jitter,
           "last_change": largest_change,
           "joint_condition_number": joint_condition,
          }

class MonotoneGPFluxST:
    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        *,
        noise_std,
        learn_neg_flux=True,
        lengthscale_bounds=(0.05, 100.0),
        n_virtual_per_axis=6,
        monotonicity_check_points_per_axis=25,
        max_virtual_refinements=3,
        max_virtual_points_per_round=16,
        probit_nu=1e-4,
        ep_max_iter=100,
        ep_damping=0.3,
        ep_tol = 1e-5,
        function_regularization=1e-5,
        derivative_regularization=1e-3,
        minimum_noise_variance=1e-8,
        n_restarts_optimizer=0,
        allow_extrapolation=False,
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.lengthscale_bounds = tuple(float(value) for value in lengthscale_bounds)
        self.n_virtual_per_axis = int(n_virtual_per_axis)
        self.monotonicity_check_points_per_axis = int(monotonicity_check_points_per_axis)
        self.max_virtual_refinements = int(max_virtual_refinements)
        self.max_virtual_points_per_round = int(max_virtual_points_per_round)
        self.probit_nu = float(probit_nu)
        self.ep_max_iter = int(ep_max_iter)
        self.ep_damping = float(ep_damping)
        self.ep_tolerance = float(ep_tol)
        self.function_regularization = float(function_regularization)
        self.derivative_regularization = float(derivative_regularization)
        self.minimum_noise_variance = float(minimum_noise_variance)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.allow_extrapolation = bool(allow_extrapolation)
        self.ep_tolerance = 1e-5
        self.ep_min_damping = self.ep_damping / 8.0
        self.max_relative_jitter = 1e-3
        self.condition_limit = 1e15
        self.condition_warning = self.condition_limit / 100.0
        self.monotonicity_tolerance = 1e-8
        self.random_state = 42
        self.fitted_ = False

        self._validate_configuration()
        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    def _validate_configuration(self):
        if self.n_virtual_per_axis < 2:
            raise ValueError("n_virtual_per_axis must be at least 2")
        if self.probit_nu <= 0.0 or self.ep_max_iter < 1:
            raise ValueError("EP scale and iteration limit must be positive")
        if not 0.0 < self.ep_damping <= 1.0:
            raise ValueError("ep_damping must be in (0, 1]")
        if self.ep_tolerance <= 0.0:
            raise ValueError("ep_tol must be positive")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer cannot be negative")
        if self.function_regularization < 0.0 or self.derivative_regularization < 0.0:
            raise ValueError("regularization strengths cannot be negative")
        if not np.isfinite(self.minimum_noise_variance) or self.minimum_noise_variance <= 0.0:
            raise ValueError("minimum_noise_variance must be positive")
        if len(self.lengthscale_bounds) != 2:
            raise ValueError("lengthscale_bounds must contain a lower and upper bound")
        lower, upper = self.lengthscale_bounds
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("lengthscale_bounds must satisfy 0 < lower < upper")
        if self.monotonicity_check_points_per_axis < 3:
            raise ValueError("monotonicity_check_points_per_axis must be at least 3")
        if self.max_virtual_refinements < 0 or self.max_virtual_points_per_round < 1:
            raise ValueError("adaptive virtual-point limits are invalid")

    @staticmethod
    def _prepare_training_arrays(s_train, T_train, q_train):
        s = np.asarray(s_train, dtype=float).reshape(-1)
        T = np.asarray(T_train, dtype=float).reshape(-1)
        q = np.asarray(q_train, dtype=float).reshape(-1)

        if not (s.size == T.size == q.size):
            raise ValueError("s_train, T_train, and q_train arrays must have the same size")
        if s.size < 8:
            raise ValueError("at least 8 training observations are required")
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(T)) or np.any(~np.isfinite(q)):
            raise ValueError("training arrays must be finite")

        return s, T, q

    def _prepare_noise(self, noise_std, n_samples, y_scale):
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma = np.full(n_samples, float(sigma[0]))
        if sigma.size != n_samples:
            raise ValueError("noise_std must be either a scalar or contain one training value per row")
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("noise_std must be finite and nonnegative")
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            raise ValueError("invalid output scale")

        standardized_variance = (sigma/y_scale)**2

        return np.maximum(standardized_variance, self.minimum_noise_variance)

    def _check_condition(self, name, value):
        if not np.isfinite(value) or value > self.condition_limit:
            raise GPError(f"{name} condition number is too large: {value:.3e}")
        if value > self.condition_warning:
            warnings.warn(f"{name} is ill-conditioned: {value:.3e}", RuntimeWarning)

    def _run_ep_with_retry(self, K_joint, y, noise_variance, regularization_precision):
        damping = self.ep_damping
        failures = []
        while True:
            try:
                result = run_ep(
                    K_joint,
                    y,
                    noise_variance,
                    regularization_precision,
                    probit_nu=self.probit_nu,
                    max_iterations=self.ep_max_iter,
                    damping=damping,
                    tolerance=self.ep_tolerance,
                    max_relative_jitter=self.max_relative_jitter,
                )
                result["damping"] = damping
                return result
            except (GPError, RuntimeError, np.linalg.LinAlgError) as error:
                failures.append(str(error))
                next_damping = 0.5 * damping
                if next_damping < self.ep_min_damping:
                    raise GPError(
                        "EP failed after adaptive damping retries: "
                        + " | ".join(failures)
                    ) from error
                damping = next_damping
                warnings.warn(
                    f"Retrying EP with damping={damping:.3g} after: {error}",
                    RuntimeWarning,
                )

    @staticmethod
    def _predict_from_state(kernel, X_query, X_train, X_virtual, alpha):
        X_query = _points(X_query, "X_query")
        value_covariance = np.hstack([
            kernel.K(X_query, X_train),
            kernel.dK_dy(X_query, X_virtual, dim=0),
        ])
        latent_mean = value_covariance @ alpha

        gradient = np.empty((X_query.shape[0], 2))
        for dim in range(2):
            derivative_covariance = np.hstack([
                kernel.dK_dx(X_query, X_train, dim=dim),
                kernel.d2K_dxdy(X_query, X_virtual, dim_x=dim, dim_y=0),
            ])
            gradient[:, dim] = derivative_covariance @ alpha
        return latent_mean, gradient

    def _fit_constraints(self, kernel, X, y, noise_variance, Z):
        K_function = kernel.K(X, X)
        K_function_derivative = kernel.dK_dy(X, Z, dim=0)
        K_derivative = kernel.d2K_dxdy(Z, Z, dim_x=0, dim_y=0)
        K_joint = np.block([
            [K_function, K_function_derivative],
            [K_function_derivative.T, K_derivative],
        ])
        K_joint = 0.5 * (K_joint + K_joint.T)
        regularization_precision = make_regularization_precision(
            K_function,
            K_derivative,
            self.function_regularization,
            self.derivative_regularization,
        )
        ep_result = self._run_ep_with_retry(
            K_joint,
            y,
            noise_variance,
            regularization_precision,
        )
        joint_condition = ep_result["joint_condition_number"]
        self._check_condition("joint covariance", joint_condition)
        return ep_result, joint_condition

    def _scan_constraints(self, kernel, X, Z, alpha):
        candidates = make_virtual_grid(X, self.monotonicity_check_points_per_axis)
        _, gradient = self._predict_from_state(kernel, candidates, X, Z, alpha)
        constrained_derivative = gradient[:, 0]
        scale = max(float(np.max(np.abs(constrained_derivative))), 1.0)
        tolerance = self.monotonicity_tolerance * scale
        violations = constrained_derivative < -tolerance
        return candidates, constrained_derivative, violations

    def fit(self, s_train, T_train, q_train, *, noise_std):
        s, T, q = self._prepare_training_arrays(s_train, T_train, q_train)
        X_raw = np.column_stack([s, T])
        coordinate_range = np.ptp(X_raw, axis=0)
        coordinate_resolution = 100.0 * np.finfo(float).eps * np.maximum(
            np.max(np.abs(X_raw), axis=0),
            1.0,
        )
        if np.any(coordinate_range <= coordinate_resolution):
            raise ValueError("both s and T must vary by a numerically resolvable amount")

        latent_raw = -q if self.learn_neg_flux else q
        output_range = float(np.ptp(latent_raw))
        output_resolution = 100.0 * np.finfo(float).eps * max(
            float(np.max(np.abs(latent_raw))),
            1.0,
        )
        if output_range <= output_resolution:
            raise ValueError("training flux has effectively zero variation")

        self.x_mean_ = X_raw.mean(axis=0)
        self.x_scale_ = X_raw.std(axis=0)
        self.y_mean_ = float(latent_raw.mean())
        self.y_scale_ = float(latent_raw.std())
        if np.any(~np.isfinite(self.x_scale_)) or np.any(self.x_scale_ <= 0.0):
            raise ValueError("invalid input standardization scale")
        if not np.isfinite(self.y_scale_) or self.y_scale_ <= 0.0:
            raise ValueError("invalid output standardization scale")

        X = (X_raw - self.x_mean_) / self.x_scale_
        y = (latent_raw - self.y_mean_) / self.y_scale_
        noise_variance = self._prepare_noise(noise_std, y.size, self.y_scale_)

        sklearn_kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=np.ones(2),
            length_scale_bounds=self.lengthscale_bounds,
            nu=2.5,
        )
        ordinary_gp = GaussianProcessRegressor(
            kernel=sklearn_kernel,
            alpha=noise_variance,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ordinary_gp.fit(X, y)
        optimizer_warnings = tuple(
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        )
        if optimizer_warnings:
            warnings.warn("GP optimizer: " + " | ".join(optimizer_warnings), RuntimeWarning)

        kernel = Matern52(
            variance=float(ordinary_gp.kernel_.k1.constant_value),
            lengthscales=np.asarray(ordinary_gp.kernel_.k2.length_scale, dtype=float),
        )
        lower, upper = self.lengthscale_bounds
        boundary_tolerance = 0.01
        near_lower = kernel.lengthscales <= lower * (1.0 + boundary_tolerance)
        near_upper = kernel.lengthscales >= upper * (1.0 - boundary_tolerance)
        dimension_names = np.array(["s", "temperature"])
        if np.any(near_lower):
            warnings.warn(
                "GP length scale reached its lower bound for "
                f"{dimension_names[near_lower].tolist()}; check noise and outliers",
                RuntimeWarning,
            )
        if np.any(near_upper):
            warnings.warn(
                "GP length scale reached its upper bound for "
                f"{dimension_names[near_upper].tolist()}",
                RuntimeWarning,
            )

        observation_covariance = kernel.K(X, X) + np.diag(noise_variance)
        observation_condition = _condition_from_cholesky(
            ordinary_gp.L_,
            np.linalg.norm(observation_covariance, 1),
        )
        self._check_condition("observation covariance", observation_condition)

        Z = make_virtual_grid(X, self.n_virtual_per_axis)
        constraint_satisfied = False

        for refinement_round in range(self.max_virtual_refinements + 1):
            ep_result, joint_condition = self._fit_constraints(
                kernel,
                X,
                y,
                noise_variance,
                Z,
            )
            candidates, constrained_derivative, violations = self._scan_constraints(
                kernel,
                X,
                Z,
                ep_result["alpha"],
            )
            violation_count = int(np.sum(violations))
            if violation_count == 0:
                constraint_satisfied = True
                break
            if refinement_round == self.max_virtual_refinements:
                break

            new_points = _select_new_virtual_points(
                Z,
                candidates[violations],
                constrained_derivative[violations],
                self.max_virtual_points_per_round,
                0.5 / (self.monotonicity_check_points_per_axis - 1),
            )
            if new_points.shape[0] == 0:
                break
            Z = np.vstack([Z, new_points])

        if not constraint_satisfied:
            raise GPError(
                "adaptive monotonicity refinement did not eliminate all violations; "
                f"remaining fraction={np.mean(violations):.3%}"
            )

        self.kernel_ = kernel
        self.X_train_ = X
        self.X_virtual_ = Z
        self.alpha_ = ep_result["alpha"]
        self.domain_min_ = X_raw.min(axis=0)
        self.domain_max_ = X_raw.max(axis=0)
        self.fit_diagnostics_ = {
            "kernel": str(ordinary_gp.kernel_),
            "physical_lengthscales": kernel.lengthscales * self.x_scale_,
            "observation_condition_number": observation_condition,
            "joint_condition_number": joint_condition,
            "ep_iterations": ep_result["iterations"],
            "prior_jitter": ep_result["prior_jitter"],
            "virtual_points": int(Z.shape[0]),
            "refinement_rounds": refinement_round,
            "violation_fraction": float(np.mean(violations)),
        }
        self.fitted_ = True
        return self

    def evaluate(self, s_q, T_q):
        if not self.fitted_:
            raise RuntimeError("The provider has not been fitted.")

        s = np.asarray(s_q, dtype=float)
        T = np.asarray(T_q, dtype=float)

        if s.ndim > 0 and T.ndim > 0 and s.shape != T.shape:
            raise ValueError("nonscalar s_q and T_q must have identical shapes.")

        s, T = np.broadcast_arrays(s, T)
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(T)):
            raise ValueError("query data contains NaN or infinity.")

        output_shape = s.shape
        X_raw = np.column_stack([s.ravel(), T.ravel()])

        domain_tolerance = 100.0 * np.finfo(float).eps * np.maximum(
            self.domain_max_ - self.domain_min_,
            1.0,
        )
        outside = np.any(
            (X_raw < self.domain_min_ - domain_tolerance)
            | (X_raw > self.domain_max_ + domain_tolerance),
            axis=1,
        )
        if np.any(outside):
            message = f"{outside.sum()} query points are outside the training rectangle"
            if not self.allow_extrapolation:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)

        X = (X_raw - self.x_mean_) / self.x_scale_
        latent_standardized, gradient_standardized = self._predict_from_state(
            self.kernel_,
            X,
            self.X_train_,
            self.X_virtual_,
            self.alpha_,
        )

        latent_physical = self.y_mean_ + self.y_scale_ * latent_standardized
        gradient_physical = (
                self.y_scale_ * gradient_standardized / self.x_scale_[None, :]
        )

        flux_sign = -1.0 if self.learn_neg_flux else 1.0
        q = flux_sign * latent_physical
        dq_ds = flux_sign * gradient_physical[:, 0]
        dq_dT = flux_sign * gradient_physical[:, 1]

        return (
            q.reshape(output_shape),
            dq_ds.reshape(output_shape),
            dq_dT.reshape(output_shape),
        )
