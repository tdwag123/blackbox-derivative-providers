import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from Testing.BlackBoxTesting.bb_providers import (
    AdaptiveBBOptions,
    KernelDerivativeProviderST,
    RFFDerivativeProviderST,
    analytic_flux_for_config,
    make_diffusion_oracle,
    parse_method_spec,
)

STATE_DIM = 2


class MovingOracleCache:
    """One oracle cache, pruned geometrically around the current stencil center."""

    def __init__(self, oracle, options):
        self.oracle = oracle
        self.options = options
        self.samples = OrderedDict()
        self.oracle_calls = 0
        self.prune_count = 0

        @lru_cache(maxsize=options.cache_size)
        def cached_eval(s_key, T_key):
            self.oracle_calls += 1
            return float(self.oracle(float(s_key), float(T_key)))

        self._cached_eval = cached_eval

    def key(self, s, T):
        decimals = self.options.oracle_key_decimals
        return (round(float(s), decimals), round(float(T), decimals))

    def evaluate(self, s, T):
        key = self.key(s, T)
        q = self._cached_eval(*key)
        self.samples.setdefault(key, q)
        return q

    def prune_near(self, center, widths):
        if len(self.samples) <= self.options.cache_size:
            return
        items = list(self.samples.items())
        points = np.array([key for key, _ in items], dtype=float)
        distances = np.linalg.norm((points - center) / widths, axis=1)
        keep = set(np.argsort(distances)[: self.options.cache_size])
        self.samples = OrderedDict(
            (key, q) for idx, (key, q) in enumerate(items) if idx in keep
        )
        self.prune_count += len(items) - len(self.samples)

    def arrays(self):
        if not self.samples:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        points = np.array(list(self.samples.keys()), dtype=float)
        q = np.array(list(self.samples.values()), dtype=float)
        return points[:, 0], points[:, 1], q

    def info(self):
        info = self._cached_eval.cache_info()
        return {
            "oracle_cache_size": len(self.samples),
            "oracle_lru_hits": info.hits,
            "oracle_lru_misses": info.misses,
            "oracle_calls": self.oracle_calls,
            "oracle_cache_prunes": self.prune_count,
        }


class MovingLocalBlackBoxProvider:
    """One moving local cache and one moving local surrogate."""

    def __init__(self, method_key, oracle, options=None, model_options=None):
        self.method_key = method_key
        self.options = options or AdaptiveBBOptions()
        self.model_options = model_options or {}
        self.rng = np.random.default_rng(self.options.rng_seed)
        self.oracle_cache = MovingOracleCache(oracle, self.options)
        self.current_surrogate = None
        self.current_center = None
        self.eval_count = 0
        self.shift_count = 0
        self.surrogate_fit_count = 0
        self.last_status = "not_evaluated"
        self.shift_threshold = float(self.model_options.get("shift_threshold", 1.0))

        self.scale_center = np.array(
            [
                0.5 * (self.options.s_bounds[0] + self.options.s_bounds[1]),
                0.5 * (self.options.T_bounds[0] + self.options.T_bounds[1]),
            ],
            dtype=float,
        )
        self.scale = np.array(
            [
                0.5 * (self.options.s_bounds[1] - self.options.s_bounds[0]),
                0.5 * (self.options.T_bounds[1] - self.options.T_bounds[0]),
            ],
            dtype=float,
        )
        self.scale[self.scale == 0.0] = 1.0

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        q_values = np.empty_like(s_q, dtype=float)
        dq_ds_values = np.empty_like(s_q, dtype=float)
        dq_dT_values = np.empty_like(s_q, dtype=float)

        for index, (s, T) in enumerate(zip(s_q.ravel(), T_q.ravel())):
            q, dq_ds, dq_dT = self._evaluate_one(float(s), float(T))
            q_values.ravel()[index] = q
            dq_ds_values.ravel()[index] = dq_ds
            dq_dT_values.ravel()[index] = dq_dT

        return q_values, dq_ds_values, dq_dT_values

    def _evaluate_one(self, s, T):
        self.eval_count += 1
        point = self._clip_physical(np.array([s, T], dtype=float))

        if self.current_surrogate is None or self._region_distance(point, self.current_center) > self.shift_threshold:
            self._shift_to(point)

        self.last_status = "ok"
        return self._surrogate_evaluate(self.current_surrogate, point)

    def _shift_to(self, point):
        n_points = (
            self.options.initial_cache_samples
            if len(self.oracle_cache.samples) == 0
            else self.options.samples_per_region
        )
        self._sample_neighborhood(point, n_points)
        self.oracle_cache.prune_near(point, self._sample_widths())
        self.current_center = point.copy()
        self.current_surrogate = self._fit_surrogate()
        self.shift_count += 1

    def _fit_surrogate(self):
        s_data, T_data, q_data = self.oracle_cache.arrays()
        if len(q_data) < max(STATE_DIM + 1, 4):
            raise RuntimeError("not enough blackbox samples to fit a local surrogate")

        X_scaled = self._to_scaled(np.column_stack([s_data, T_data]))
        self.surrogate_fit_count += 1

        if self.method_key in {"bb_rbf", "bb_krr", "bb_rbf_krr"}:
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="gaussian",
                epsilon=self.model_options.get("epsilon", 0.3),
                ridge_strength=self.model_options.get("ridge_strength", 1.0e-3),
            )

        if self.method_key in {"bb_matern52_krr", "bb_matern_krr"}:
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="matern52",
                epsilon=self.model_options.get("epsilon", 0.3),
                ridge_strength=self.model_options.get("ridge_strength", 1.0e-3),
            )

        if self.method_key in {"bb_rff", "bb_ridge_rff"}:
            if RFFDerivativeProviderST is None:
                raise ImportError("bb_rff requires scikit-learn RFF dependencies")
            return RFFDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                regularization="ridge",
                n_components=self.model_options.get("n_components", 200),
                gamma=self.model_options.get("gamma", 0.5),
                alpha=self.model_options.get("alpha", 1.0e-1),
                random_state=self.model_options.get("rng_seed", 0),
            )

        raise ValueError(f"unknown moving-local blackbox method: {self.method_key}")

    def _surrogate_evaluate(self, surrogate, point):
        scaled = self._to_scaled(point.reshape(1, 2))
        q, dq_ds_hat, dq_dT_hat = surrogate.evaluate(
            np.array([scaled[0, 0]]),
            np.array([scaled[0, 1]]),
        )
        dq_ds = float(dq_ds_hat[0] / self.scale[0])
        dq_dT = float(dq_dT_hat[0] / self.scale[1])
        return float(q[0]), dq_ds, dq_dT

    def _sample_neighborhood(self, point, n_points):
        samples = [point.copy()] if self.options.include_center else []
        remaining = max(0, n_points - len(samples))
        if self.options.design in {"axis", "hybrid"}:
            samples.extend(self._structured_samples(point, remaining))
        if self.options.design in {"random", "hybrid"} and len(samples) < n_points:
            samples.extend(self._random_samples(point, n_points - len(samples)))

        sampled = 0
        seen = set()
        attempts = 0
        while sampled < n_points and attempts < 20 * max(n_points, 1):
            attempts += 1
            sample = (
                np.asarray(samples[attempts - 1], dtype=float)
                if attempts <= len(samples)
                else np.asarray(self._random_samples(point, 1)[0], dtype=float)
            )
            sample = self._fold_into_domain(sample)
            key = self.oracle_cache.key(sample[0], sample[1])
            if key in seen:
                continue
            seen.add(key)
            self.oracle_cache.evaluate(sample[0], sample[1])
            sampled += 1

    def _structured_samples(self, point, n_points):
        widths = self._sample_widths()
        directions = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2.0),
            np.array([1.0, -1.0]) / np.sqrt(2.0),
            np.array([-1.0, 1.0]) / np.sqrt(2.0),
            np.array([-1.0, -1.0]) / np.sqrt(2.0),
        ]
        fractions = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        samples = []
        for fraction in fractions:
            for direction in directions:
                if len(samples) >= n_points:
                    return samples
                samples.append(point + widths * fraction * direction)
        return samples

    def _random_samples(self, point, n_points):
        widths = self._sample_widths()
        samples = []
        for _ in range(n_points):
            direction = self.rng.normal(size=STATE_DIM)
            norm = np.linalg.norm(direction)
            direction = np.ones(STATE_DIM) / np.sqrt(STATE_DIM) if norm == 0.0 else direction / norm
            samples.append(point + widths * (self.rng.random() ** (1.0 / STATE_DIM)) * direction)
        return samples

    def _sample_widths(self):
        return np.array([self.options.sample_width_s, self.options.sample_width_T], dtype=float)

    def _region_distance(self, point, center):
        return float(np.linalg.norm((point - center) / self._sample_widths()))

    def _clip_physical(self, point):
        return np.array(
            [
                np.clip(point[0], self.options.s_bounds[0], self.options.s_bounds[1]),
                np.clip(point[1], self.options.T_bounds[0], self.options.T_bounds[1]),
            ],
            dtype=float,
        )

    def _fold_into_domain(self, point):
        point = np.asarray(point, dtype=float).copy()
        for dim, bounds in enumerate((self.options.s_bounds, self.options.T_bounds)):
            lower, upper = bounds
            if point[dim] < lower:
                point[dim] = lower + (lower - point[dim])
            if point[dim] > upper:
                point[dim] = upper - (point[dim] - upper)
            point[dim] = np.clip(point[dim], lower, upper)
        return point

    def _to_scaled(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.scale_center) / self.scale

    def diagnostics(self):
        diag = self.oracle_cache.info()
        diag.update(
            {
                "bb_eval_count": self.eval_count,
                "bb_refinement_count": self.shift_count,
                "bb_surrogate_fit_count": self.surrogate_fit_count,
                "bb_last_status": self.last_status,
                "bb_sample_width_s": self.options.sample_width_s,
                "bb_sample_width_T": self.options.sample_width_T,
                "bb_shift_threshold": self.shift_threshold,
                "bb_oracle_cache_size": len(self.oracle_cache.samples),
            }
        )
        return diag


def build_provider(method, oracle_config="nonlinear_high_noise", *, x_mesh=None, noisy=True, seed=0):
    method_key, method_options = parse_method_spec(method)
    start = time.perf_counter()

    if method_key == "analytic":
        return {
            "method": method,
            "flux": analytic_flux_for_config(oracle_config),
            "build_s": time.perf_counter() - start,
            "provider": None,
            "h_s": np.nan,
            "h_T": np.nan,
        }

    options = AdaptiveBBOptions(
        sample_width_s=method_options.get("sample_width_s", AdaptiveBBOptions.sample_width_s),
        sample_width_T=method_options.get("sample_width_t", AdaptiveBBOptions.sample_width_T),
        initial_cache_samples=method_options.get(
            "initial_cache_samples",
            AdaptiveBBOptions.initial_cache_samples,
        ),
        samples_per_region=method_options.get(
            "samples_per_region",
            AdaptiveBBOptions.samples_per_region,
        ),
        cache_size=method_options.get("cache_size", AdaptiveBBOptions.cache_size),
        rng_seed=method_options.get("rng_seed", seed),
        design=method_options.get("design", AdaptiveBBOptions.design),
    )

    oracle = make_diffusion_oracle(oracle_config, seed=seed, noisy=noisy)
    provider = MovingLocalBlackBoxProvider(
        method_key=method_key,
        oracle=oracle,
        options=options,
        model_options=method_options,
    )

    def flux_law(s, T, xg):
        q, dq_ds, dq_dT = provider.evaluate(np.array([s]), np.array([T]))
        return float(q[0]), float(dq_ds[0]), float(dq_dT[0])

    return {
        "method": f"moving_{method}",
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "provider": provider,
        "h_s": options.sample_width_s,
        "h_T": options.sample_width_T,
        "oracle_config": oracle_config,
        "sample_width_s": options.sample_width_s,
        "sample_width_T": options.sample_width_T,
        "initial_cache_samples": options.initial_cache_samples,
        "samples_per_region": options.samples_per_region,
        "cache_size": options.cache_size,
    }
