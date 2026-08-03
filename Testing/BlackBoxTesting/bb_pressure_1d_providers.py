import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from Data.BlackBoxOracle.tabularoracle import make_tabular_oracle
from Testing.BlackBoxTesting.bb_providers_tolerance import parse_method_spec

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
except (ImportError, OSError):
    GaussianProcessRegressor = None
    ConstantKernel = None
    Matern = None
    RBF = None
    WhiteKernel = None


STATE_DIM = 1


@dataclass
class Pressure1DOptions:
    s_bounds: tuple = (-5000.0, 0.0)
    T_bounds: tuple = (0.0, 3000.0)
    oracle_T: float = 1.0
    sample_radius: float = 0.20
    validation_radius_factor: float = 1.5
    initial_points_per_dim: int = 30
    max_points_per_dim: int = 30
    refill_points: int = 15
    max_refinements_per_eval: int = 0
    rng_seed: int = 0
    design: str = "random"
    include_center: bool = True
    oracle_key_decimals: int = 12
    mse_tolerance: float = 2.0e-1
    variance_tolerance: float = 2.5e-3
    min_mesh_radius_factor: float = 2.0
    mesh_spacing: float = 0.05

    @property
    def max_points(self):
        return self.max_points_per_dim * STATE_DIM

    @property
    def initial_points(self):
        return self.initial_points_per_dim * STATE_DIM

    @property
    def effective_sample_radius(self):
        min_radius = self.min_mesh_radius_factor * self.mesh_spacing
        return max(float(self.sample_radius), float(min_radius))

    @property
    def validation_radius(self):
        return self.validation_radius_factor * self.effective_sample_radius


class LocalPolynomialDerivativeProvider1D:
    def __init__(self, s_data, q_data, degree=3, ridge_strength=0.0):
        self.degree = int(degree)
        self.ridge_strength = float(ridge_strength)
        powers = np.arange(self.degree + 1)
        X = np.column_stack([np.asarray(s_data, dtype=float) ** p for p in powers])
        y = np.asarray(q_data, dtype=float).reshape(-1)

        if self.ridge_strength > 0.0:
            lhs = X.T @ X + self.ridge_strength * np.eye(X.shape[1])
            rhs = X.T @ y
            self.coef_ = np.linalg.solve(lhs, rhs)
        else:
            self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)

    def evaluate(self, s_q):
        s_q = np.asarray(s_q, dtype=float)
        shape = s_q.shape
        s = s_q.reshape(-1)
        q = np.zeros_like(s, dtype=float)
        dq_ds = np.zeros_like(s, dtype=float)

        for power, coef in enumerate(self.coef_):
            q += coef * s**power
            if power > 0:
                dq_ds += coef * power * s ** (power - 1)

        return q.reshape(shape), dq_ds.reshape(shape)


class GPDerivativeProvider1D:
    def __init__(self, s_data, q_data, kernel_kind="rbf", **kwargs):
        if GaussianProcessRegressor is None:
            raise ImportError("1D GP methods require scikit-learn GP dependencies")

        lengthscale = kwargs.get("lengthscale", 1.0)
        noise_variance = kwargs.get("noise_variance", 1.0e-2)
        kernel_variance = kwargs.get("kernel_variance", 1.0)

        if kernel_kind == "matern52":
            base_kernel = Matern(length_scale=lengthscale, nu=2.5)
        elif kernel_kind == "rbf":
            base_kernel = RBF(length_scale=lengthscale)
        else:
            raise ValueError(f"unknown 1D GP kernel kind: {kernel_kind}")

        kernel = ConstantKernel(kernel_variance) * base_kernel + WhiteKernel(
            noise_level=noise_variance,
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=kwargs.get("jitter", 1.0e-8),
            n_restarts_optimizer=kwargs.get("n_restarts_optimizer", 0),
            normalize_y=True,
        )
        self.model.fit(np.asarray(s_data, dtype=float).reshape(-1, 1), q_data)

    def evaluate(self, s_q, return_variance=False):
        s_q = np.asarray(s_q, dtype=float)
        shape = s_q.shape
        X = s_q.reshape(-1, 1)
        q, std = self.model.predict(X, return_std=True)
        dq_ds = self._finite_difference_derivative(s_q.reshape(-1)).reshape(shape)
        if return_variance:
            return q.reshape(shape), dq_ds, std.reshape(shape) ** 2
        return q.reshape(shape), dq_ds

    def _finite_difference_derivative(self, s):
        h = 1.0e-4 * np.maximum(1.0, np.abs(s))
        q_right = self.model.predict((s + h).reshape(-1, 1))
        q_left = self.model.predict((s - h).reshape(-1, 1))
        return (q_right - q_left) / (2.0 * h)


class OracleEvaluationCache1D:
    def __init__(self, oracle, options):
        self.oracle = oracle
        self.options = options
        self.active = OrderedDict()
        self.oracle_calls = 0
        self.cache_hits = 0
        self.prune_count = 0

        @lru_cache(maxsize=options.max_points)
        def cached_eval(s_key):
            self.oracle_calls += 1
            return float(self.oracle(float(s_key), float(self.options.oracle_T)))

        self._cached_eval = cached_eval

    def key(self, s):
        return round(float(s), self.options.oracle_key_decimals)

    def evaluate(self, s):
        key = self.key(s)
        before = self._cached_eval.cache_info()
        q = self._cached_eval(key)
        after = self._cached_eval.cache_info()
        if after.hits > before.hits:
            self.cache_hits += 1
        self.remember(key, q)
        return q

    def remember(self, key, q):
        if key in self.active:
            return
        self.active[key] = q
        while len(self.active) > self.options.max_points:
            self.active.popitem(last=False)
            self.prune_count += 1

    def arrays(self):
        if not self.active:
            return np.array([], dtype=float), np.array([], dtype=float)
        return (
            np.array(list(self.active.keys()), dtype=float),
            np.array(list(self.active.values()), dtype=float),
        )

    def info(self):
        info = self._cached_eval.cache_info()
        return {
            "active_cache_size": len(self.active),
            "oracle_lru_size": info.currsize,
            "oracle_lru_hits": info.hits,
            "oracle_lru_misses": info.misses,
            "oracle_calls": self.oracle_calls,
            "active_prunes": self.prune_count,
            "tracked_cache_hits": self.cache_hits,
        }


class AdaptivePressure1DProvider:
    def __init__(self, method_key, oracle, options=None, model_options=None):
        self.method_key = method_key
        self.oracle = oracle
        self.options = options or Pressure1DOptions()
        self.model_options = model_options or {}
        self.rng = np.random.default_rng(self.options.rng_seed)
        self.cache = OracleEvaluationCache1D(oracle, self.options)
        self.eval_count = 0
        self.refinement_count = 0
        self.surrogate_fit_count = 0
        self.failed_refinements = 0
        self.last_uncertainty = np.nan
        self.uncertainty_sum = 0.0
        self.uncertainty_count = 0
        self.last_status = "not_evaluated"
        self.current_surrogate = None
        self.center = 0.5 * (self.options.s_bounds[0] + self.options.s_bounds[1])
        self.scale = 0.5 * (self.options.s_bounds[1] - self.options.s_bounds[0])
        if self.scale == 0.0:
            self.scale = 1.0

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        q_values = np.empty_like(s_q, dtype=float)
        dq_ds_values = np.empty_like(s_q, dtype=float)
        dq_dT_values = np.zeros_like(s_q, dtype=float)
        for index, s in enumerate(s_q.ravel()):
            q, dq_ds = self._evaluate_one(float(s))
            q_values.ravel()[index] = q
            dq_ds_values.ravel()[index] = dq_ds
        return q_values, dq_ds_values, dq_dT_values

    def _evaluate_one(self, s):
        self.eval_count += 1
        s = self._clip_s(s)

        if len(self.cache.active) == 0:
            self._sample_neighborhood(s, self.options.initial_points)

        if self.current_surrogate is None:
            self.current_surrogate = self._fit_surrogate()

        result = self._surrogate_evaluate(self.current_surrogate, s)
        uncertainty = self._uncertainty(self.current_surrogate, s, result)
        self._record_uncertainty(uncertainty)

        if self._uncertainty_is_ok(uncertainty):
            self.last_status = "ok"
            return result[:2]

        for _ in range(self.options.max_refinements_per_eval + 1):
            self._sample_neighborhood(s, self.options.refill_points)
            self.refinement_count += 1
            self.current_surrogate = self._fit_surrogate()

            result = self._surrogate_evaluate(self.current_surrogate, s)
            uncertainty = self._uncertainty(self.current_surrogate, s, result)
            self._record_uncertainty(uncertainty)
            if self._uncertainty_is_ok(uncertainty):
                self.last_status = "refined_ok"
                return result[:2]

        self.failed_refinements += 1
        self.last_status = "max_refinements_uncertain"
        return result[:2]

    def _fit_surrogate(self):
        s_data, q_data = self.cache.arrays()
        if len(q_data) < max(STATE_DIM + 1, 4):
            raise RuntimeError("not enough blackbox samples to fit a local surrogate")
        s_scaled = self._to_scaled(s_data)
        self.surrogate_fit_count += 1

        if self.method_key in {"bb_poly", "bb_polynomial"}:
            return LocalPolynomialDerivativeProvider1D(
                s_scaled,
                q_data,
                degree=self.model_options.get("degree", 3),
                ridge_strength=self.model_options.get("ridge_strength", 0.0),
            )
        if self.method_key == "bb_gp":
            return GPDerivativeProvider1D(
                s_scaled,
                q_data,
                kernel_kind="rbf",
                **self.model_options,
            )
        if self.method_key == "bb_materngp":
            return GPDerivativeProvider1D(
                s_scaled,
                q_data,
                kernel_kind="matern52",
                **self.model_options,
            )

        raise ValueError(f"unknown pressure 1D blackbox method: {self.method_key}")

    def _surrogate_evaluate(self, surrogate, s):
        s_scaled = self._to_scaled(np.array([s]))
        if isinstance(surrogate, GPDerivativeProvider1D):
            q, dq_ds_hat, variance = surrogate.evaluate(s_scaled, return_variance=True)
        else:
            q, dq_ds_hat = surrogate.evaluate(s_scaled)
            variance = np.array([np.nan])
        dq_ds = float(dq_ds_hat[0] / self.scale)
        return float(q[0]), dq_ds, float(variance[0])

    def _uncertainty(self, surrogate, s, result):
        if isinstance(surrogate, GPDerivativeProvider1D):
            return max(float(result[2]), 0.0)

        s_data, q_data = self.cache.arrays()
        if len(q_data) == 0:
            return np.inf

        distances = np.abs(self._to_scaled(s_data) - self._to_scaled(np.array([s]))[0])
        local = distances <= self.options.validation_radius
        if np.count_nonzero(local) < max(4, STATE_DIM + 1):
            return np.inf

        q_pred, _ = surrogate.evaluate(self._to_scaled(s_data[local]))
        mse = np.mean((q_pred - q_data[local]) ** 2)
        q_scale = max(float(np.std(q_data[local])), 1.0)
        return float(mse / (q_scale**2))

    def _uncertainty_is_ok(self, uncertainty):
        if not np.isfinite(uncertainty):
            return False
        if self.method_key in {"bb_gp", "bb_materngp"}:
            return uncertainty <= self.options.variance_tolerance
        return uncertainty <= self.options.mse_tolerance

    def _record_uncertainty(self, uncertainty):
        self.last_uncertainty = uncertainty
        if np.isfinite(uncertainty):
            self.uncertainty_sum += float(uncertainty)
            self.uncertainty_count += 1

    def _sample_neighborhood(self, s, n_points):
        if n_points <= 0:
            return
        samples = []
        if self.options.include_center:
            samples.append(s)
        remaining = max(0, n_points - len(samples))
        if self.options.design in {"axis", "hybrid"}:
            samples.extend(self._structured_samples(s, remaining))
            remaining = max(0, n_points - len(samples))
        if remaining and self.options.design in {"random", "hybrid"}:
            samples.extend(self._random_samples(s, remaining))

        sampled = 0
        seen = set()
        attempts = 0
        while sampled < n_points and attempts < 20 * max(n_points, 1):
            attempts += 1
            if attempts <= len(samples):
                sample = samples[attempts - 1]
            else:
                sample = self._random_samples(s, 1)[0]
            sample = self._clip_s(sample)
            key = self.cache.key(sample)
            if key in seen:
                continue
            seen.add(key)
            self.cache.evaluate(sample)
            sampled += 1

        if sampled < max(4, STATE_DIM + 1):
            raise RuntimeError("could not build enough in-domain 1D samples")

    def _structured_samples(self, s, n_points):
        if n_points <= 0:
            return []
        radius = self.options.effective_sample_radius
        scaled_center = self._to_scaled(np.array([s]))[0]
        fractions = [1.0, 0.5, 0.25, 0.125]
        samples = []
        for fraction in fractions:
            for direction in (-1.0, 1.0):
                if len(samples) >= n_points:
                    return samples
                samples.append(self._from_scaled(scaled_center + direction * radius * fraction))
        return samples

    def _random_samples(self, s, n_points):
        radius = self.options.effective_sample_radius
        scaled_center = self._to_scaled(np.array([s]))[0]
        offsets = self.rng.uniform(-radius, radius, int(n_points))
        return [self._from_scaled(scaled_center + offset) for offset in offsets]

    def _clip_s(self, s):
        return float(np.clip(float(s), self.options.s_bounds[0], self.options.s_bounds[1]))

    def _to_scaled(self, s):
        return (np.asarray(s, dtype=float) - self.center) / self.scale

    def _from_scaled(self, s_scaled):
        return float(self.center + self.scale * float(s_scaled))

    def diagnostics(self):
        diag = self.cache.info()
        diag.update(
            {
                "bb_eval_count": self.eval_count,
                "bb_refinement_count": self.refinement_count,
                "bb_surrogate_fit_count": self.surrogate_fit_count,
                "bb_failed_refinements": self.failed_refinements,
                "bb_last_uncertainty": self.last_uncertainty,
                "bb_avg_uncertainty": (
                    self.uncertainty_sum / self.uncertainty_count
                    if self.uncertainty_count
                    else np.nan
                ),
                "bb_last_status": self.last_status,
                "bb_sample_radius": self.options.effective_sample_radius,
                "bb_validation_radius": self.options.validation_radius,
                "bb_cache_limit": self.options.max_points,
                "bb_state_dim": STATE_DIM,
                "bb_oracle_T": self.options.oracle_T,
            }
        )
        return diag


def build_provider(
    method,
    oracle_config,
    *,
    x_mesh=None,
    noisy=True,
    seed=0,
    provider_options=None,
):
    del noisy
    method_key, method_options = parse_method_spec(method)
    method_options = {**method_options, **(provider_options or {})}
    start = time.perf_counter()

    if x_mesh is None:
        mesh_spacing = method_options.get("mesh_spacing", 0.05)
    else:
        x_mesh = np.asarray(x_mesh, dtype=float)
        mesh_spacing = float(np.median(np.diff(x_mesh)))

    options = Pressure1DOptions(
        s_bounds=method_options.get("s_bounds", Pressure1DOptions.s_bounds),
        T_bounds=method_options.get("T_bounds", Pressure1DOptions.T_bounds),
        oracle_T=method_options.get("oracle_T", Pressure1DOptions.oracle_T),
        sample_radius=method_options.get("sample_radius", Pressure1DOptions.sample_radius),
        validation_radius_factor=method_options.get(
            "validation_radius_factor",
            Pressure1DOptions.validation_radius_factor,
        ),
        initial_points_per_dim=method_options.get(
            "initial_points_per_dim",
            Pressure1DOptions.initial_points_per_dim,
        ),
        max_points_per_dim=method_options.get(
            "max_points_per_dim",
            Pressure1DOptions.max_points_per_dim,
        ),
        refill_points=method_options.get("refill_points", Pressure1DOptions.refill_points),
        max_refinements_per_eval=method_options.get(
            "max_refinements_per_eval",
            Pressure1DOptions.max_refinements_per_eval,
        ),
        rng_seed=method_options.get("rng_seed", seed),
        design=method_options.get("design", Pressure1DOptions.design),
        mse_tolerance=method_options.get("mse_tolerance", Pressure1DOptions.mse_tolerance),
        variance_tolerance=method_options.get(
            "variance_tolerance",
            Pressure1DOptions.variance_tolerance,
        ),
        mesh_spacing=mesh_spacing,
    )
    oracle = make_tabular_oracle(oracle_config, noisy=True)
    provider = AdaptivePressure1DProvider(
        method_key=method_key,
        oracle=oracle,
        options=options,
        model_options=method_options,
    )

    def flux_law(s, T, xg):
        del xg
        q, dq_ds, dq_dT = provider.evaluate(np.array([s]), np.array([T]))
        return float(q[0]), float(dq_ds[0]), float(dq_dT[0])

    return {
        "method": method,
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "provider": provider,
        "h_s": options.effective_sample_radius * provider.scale,
        "h_T": np.nan,
        "oracle_config": oracle_config,
        "sample_radius": options.effective_sample_radius,
        "validation_radius": options.validation_radius,
        "pressure_1d": True,
    }
