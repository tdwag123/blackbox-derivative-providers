import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Data.BlackBoxOracle.blackboxoracle import (  # noqa: E402
    ORACLE_CONFIGS,
    make_diffusion_oracle,
    physical_flux,
    physical_flux_derivatives,
)
from Methods.TabularDataMethods.KernelMethods import KernelDerivativeProviderST  # noqa: E402

try:
    from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
except (ImportError, OSError):
    KISSGPFluxST = None

try:
    from Methods.TabularDataMethods.RandomFeature.unconstrained_regularized_least_squares_buildup.RFF_general import (
        RFFDerivativeProviderST,
    )
except (ImportError, OSError):
    RFFDerivativeProviderST = None

try:
    from Methods.OracleDataMethods.monotoneGP import MonotoneGPFluxST
except (ImportError, OSError, AttributeError):
    MonotoneGPFluxST = None


STATE_DIM = 2


@dataclass
class AdaptiveBBOptions:
    # Physical state bounds used to keep random samples inside the oracle domain.
    # The provider scales these bounds to roughly [-1, 1]^2 before measuring
    # radii, so s and T do not dominate each other just because of units.
    s_bounds: tuple = (-6.0, 6.0)
    T_bounds: tuple = (0.0, 3.0)

    # Radius is measured in scaled (s, T) coordinates. The effective radius is
    # also forced to be at least 2 * FEM mesh spacing below.
    sample_radius: float = 0.25
    validation_radius_factor: float = 1.5

    # d = 2 for this 1D FEM problem because the constitutive state is (s, T).
    # Empty cache starts with 10d points; the active FIFO cache holds 30d points.
    initial_points_per_dim: int = 10
    max_points_per_dim: int = 30

    # When uncertainty is too high, add this many new oracle samples near the
    # current quadrature state, then refit a fresh local surrogate.
    refill_points: int = 6
    max_refinements_per_eval: int = 3
    rng_seed: int = 0

    # "hybrid" gives a small structured stencil first, then fills the rest with
    # random ball samples. Use "random" if you want purely random sampling.
    design: str = "random"
    include_center: bool = True

    # Exact floating-point equality is unreliable, so oracle inputs are rounded
    # before going through lru_cache.
    oracle_key_decimals: int = 12

    # GP-like methods use predictive variance. Non-GP methods use normalized
    # training MSE on cached points near the current query point.
    mse_tolerance: float = 2.5e-3
    variance_tolerance: float = 2.5e-3

    # Enforce your rule that the sampling ball is at least twice the FEM spacing.
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


def parse_method_spec(method):
    # Allows compact experiment strings like:
    # "bb_rbf+sample_radius=0.2+refill_points=4+ridge_strength=1e-4"
    text = str(method)
    if "+" not in text:
        return text.lower(), {}

    parts = text.split("+")
    method_key = parts[0].lower()
    options = {}

    for part in parts[1:]:
        for item in part.split(","):
            if not item.strip():
                continue
            if "=" in item:
                key, value = item.split("=", 1)
            elif ":" in item:
                key, value = item.split(":", 1)
            else:
                continue
            key = key.strip().lower()
            value = value.strip()
            try:
                if key in {
                    "sample_radius",
                    "validation_radius_factor",
                    "mse_tolerance",
                    "variance_tolerance",
                    "mesh_spacing",
                    "gamma",
                    "epsilon",
                    "ridge_strength",
                    "alpha",
                    "learning_rate",
                }:
                    options[key] = float(value)
                elif key in {
                    "refill_points",
                    "max_refinements_per_eval",
                    "rng_seed",
                    "n_components",
                    "training_iter",
                    "grid_size",
                    "n_virtual_per_axis",
                }:
                    options[key] = int(value)
                else:
                    options[key] = value
            except ValueError:
                options[key] = value

    return method_key, options


class OracleEvaluationCache:
    """
    Small active FIFO cache backed by functools.lru_cache for exact oracle calls.

    The active cache is what trains the local surrogate. The lru_cache avoids
    repeating exact expensive calls while this provider lives.
    """

    def __init__(self, oracle, options):
        self.oracle = oracle
        self.options = options

        # Active cache is the training set. OrderedDict gives FIFO pruning by
        # popping the oldest inserted point when we exceed 30d samples.
        self.active = OrderedDict()
        self.oracle_calls = 0
        self.cache_hits = 0
        self.prune_count = 0

        # This is the requested functools cache. It only prevents repeated exact
        # oracle calls while this provider object exists. The active FIFO cache
        # still decides what points train the local surrogate.
        @lru_cache(maxsize=options.max_points)
        def cached_eval(s_key, T_key):
            self.oracle_calls += 1
            return float(self.oracle(float(s_key), float(T_key)))

        self._cached_eval = cached_eval

    def key(self, s, T):
        decimals = self.options.oracle_key_decimals
        return (round(float(s), decimals), round(float(T), decimals))

    def evaluate(self, s, T):
        key = self.key(s, T)

        # Check cache_info before and after so diagnostics can report how often
        # an exact oracle call was avoided.
        before = self._cached_eval.cache_info()
        q = self._cached_eval(*key)
        after = self._cached_eval.cache_info()
        if after.hits > before.hits:
            self.cache_hits += 1
        self.remember(key, q)
        return q

    def remember(self, key, q):
        if key in self.active:
            return
        self.active[key] = q

        # FIFO prune: once the active training cache is full, discard the oldest
        # active point. This does not preserve old points for large-scale use.
        while len(self.active) > self.options.max_points:
            self.active.popitem(last=False)
            self.prune_count += 1

    def arrays(self):
        if not self.active:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        points = np.array(list(self.active.keys()), dtype=float)
        q = np.array(list(self.active.values()), dtype=float)
        return points[:, 0], points[:, 1], q

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


class AdaptiveBlackBoxProvider:
    """
    Stateful blackbox flux provider used by Newton.

    Each call receives one constitutive state (s, T), adapts the active cache if
    uncertainty is too high, trains a fresh surrogate on the active cache, and
    returns q plus surrogate derivatives. Nothing here uses true derivatives.
    """

    def __init__(self, method_key, oracle, options=None, model_options=None):
        self.method_key = method_key
        self.oracle = oracle
        self.options = options or AdaptiveBBOptions()
        self.model_options = model_options or {}
        self.rng = np.random.default_rng(self.options.rng_seed)
        self.cache = OracleEvaluationCache(oracle, self.options)
        self.eval_count = 0
        self.refinement_count = 0
        self.surrogate_fit_count = 0
        self.failed_refinements = 0
        self.last_uncertainty = np.nan
        self.last_status = "not_evaluated"

        # Scaling map for geometry. Sampling radii and cache-neighborhood tests
        # are performed in scaled coordinates, not raw physical units.
        self.center = np.array(
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
        # Provider classes are vector-shaped, but Newton usually calls the flux
        # law one scalar quadrature state at a time. This supports both.
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

        # Stay inside the oracle domain, especially for temperature.
        point = self._clip_physical(np.array([s, T], dtype=float))

        # First ever query: seed the active cache with 10d nearby samples.
        if len(self.cache.active) == 0:
            self._sample_neighborhood(point, self.options.initial_points)

        result = None
        for attempt in range(self.options.max_refinements_per_eval + 1):
            # Surrogate is deliberately retrained for this quadrature state.
            # There is no model cache; only oracle samples persist.
            surrogate = self._fit_surrogate()
            result = self._surrogate_evaluate(surrogate, point)
            uncertainty = self._uncertainty(surrogate, point, result)
            self.last_uncertainty = uncertainty

            if self._uncertainty_is_ok(uncertainty):
                self.last_status = "ok"
                return result[:3]

            # If uncertainty is bad, sample more near this point. If that pushes
            # active cache past 30d, OracleEvaluationCache prunes FIFO.
            if attempt == self.options.max_refinements_per_eval:
                self.failed_refinements += 1
                self.last_status = "max_refinements_uncertain"
                return result[:3]

            self._sample_neighborhood(point, self.options.refill_points)
            self.refinement_count += 1

        return result[:3]

    def _fit_surrogate(self):
        s_data, T_data, q_data = self.cache.arrays()
        if len(q_data) < max(STATE_DIM + 1, 4):
            raise RuntimeError("not enough blackbox samples to fit a local surrogate")

        # All local surrogate classes see scaled coordinates. Their derivative
        # outputs are converted back to physical derivatives in _surrogate_evaluate.
        X_scaled = self._to_scaled(np.column_stack([s_data, T_data]))
        self.surrogate_fit_count += 1

        # Kernel ridge / RBF is the simplest non-GP local surrogate.
        if self.method_key in {"bb_rbf", "bb_krr", "bb_rbf_krr"}:
            epsilon = self.model_options.get("epsilon", None)
            ridge = self.model_options.get("ridge_strength", 1.0e-4)
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="gaussian",
                epsilon=epsilon,
                ridge_strength=ridge,
            )

        # Same KRR machinery, but with a Matern 5/2 kernel.
        if self.method_key in {"bb_matern52_krr", "bb_matern_krr"}:
            epsilon = self.model_options.get("epsilon", None)
            ridge = self.model_options.get("ridge_strength", 1.0e-4)
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="matern52",
                epsilon=epsilon,
                ridge_strength=ridge,
            )

        # Random Fourier features surrogate. Still uses the same adaptive cache
        # policy; uncertainty is measured by local cached-point MSE.
        if self.method_key in {"bb_rff", "bb_ridge_rff"}:
            if RFFDerivativeProviderST is None:
                raise ImportError("bb_rff requires scikit-learn RFF dependencies")
            return RFFDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                regularization="ridge",
                n_components=self.model_options.get("n_components", 400),
                gamma=self.model_options.get("gamma", 1.0),
                alpha=self.model_options.get("alpha", 1.0e-5),
                random_state=self.model_options.get("rng_seed", 0),
            )

        # KISS-GP can return predictive variance, so this path uses variance as
        # the uncertainty trigger instead of MSE.
        if self.method_key in {"bb_kissgp", "bb_kiss-gp", "bb_gp"}:
            if KISSGPFluxST is None:
                raise ImportError("bb_kissgp requires torch/gpytorch dependencies")
            return KISSGPFluxST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                grid_size=self.model_options.get("grid_size", 16),
                training_iter=self.model_options.get("training_iter", 20),
                learning_rate=self.model_options.get("learning_rate", 0.08),
                ridge_strength=self.model_options.get("ridge_strength", 0.0),
            )

        # Hook for a monotone GP provider. This currently depends on the
        # importable state of Methods/OracleDataMethods/monotoneGP.py.
        if self.method_key in {"bb_monotonegp", "bb_materngpmonotone"}:
            if MonotoneGPFluxST is None or not hasattr(MonotoneGPFluxST, "evaluate"):
                raise ImportError("bb_monotonegp provider is not currently importable")
            return MonotoneGPFluxST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                noise_std=self.model_options.get("noise_std", 0.0),
                n_virtual_per_axis=self.model_options.get("n_virtual_per_axis", 6),
                n_restarts_optimizer=0,
            )

        raise ValueError(f"unknown blackbox method: {self.method_key}")

    def _surrogate_evaluate(self, surrogate, point):
        scaled = self._to_scaled(point.reshape(1, 2))
        if self._surrogate_has_variance(surrogate):
            q, dq_ds_hat, dq_dT_hat, variance = surrogate.evaluate(
                np.array([scaled[0, 0]]),
                np.array([scaled[0, 1]]),
                return_variance=True,
            )
        else:
            q, dq_ds_hat, dq_dT_hat = surrogate.evaluate(
                np.array([scaled[0, 0]]),
                np.array([scaled[0, 1]]),
            )
            variance = np.array([np.nan])

        # Chain rule: surrogate derivatives are with respect to scaled s,T.
        # Divide by physical scale to return dq/ds and dq/dT for Newton.
        dq_ds = float(dq_ds_hat[0] / self.scale[0])
        dq_dT = float(dq_dT_hat[0] / self.scale[1])
        return float(q[0]), dq_ds, dq_dT, float(variance[0])

    def _uncertainty(self, surrogate, point, result):
        # GP uncertainty: use predictive variance returned by the GP provider.
        if self._surrogate_has_variance(surrogate):
            return max(float(result[3]), 0.0)

        s_data, T_data, q_data = self.cache.arrays()
        if len(q_data) == 0:
            return np.inf

        X = np.column_stack([s_data, T_data])
        distances = np.linalg.norm(self._to_scaled(X) - self._to_scaled(point.reshape(1, 2)), axis=1)
        local = distances <= self.options.validation_radius

        # Do not accept a low training MSE from points far away from the query.
        # If the active cache has poor local coverage, force refinement.
        if np.count_nonzero(local) < max(4, STATE_DIM + 1):
            return np.inf

        # Non-GP uncertainty: normalized function-value MSE on nearby cached
        # training points. This is not a separate validation oracle call.
        X_scaled = self._to_scaled(X[local])
        q_pred, _, _ = surrogate.evaluate(X_scaled[:, 0], X_scaled[:, 1])
        mse = np.mean((q_pred - q_data[local]) ** 2)

        q_scale = max(float(np.std(q_data[local])), 1.0)
        return float(mse / (q_scale**2))

    def _uncertainty_is_ok(self, uncertainty):
        if not np.isfinite(uncertainty):
            return False
        if self.method_key in {"bb_kissgp", "bb_kiss-gp", "bb_gp", "bb_monotonegp", "bb_materngpmonotone"}:
            return uncertainty <= self.options.variance_tolerance
        return uncertainty <= self.options.mse_tolerance

    def _surrogate_has_variance(self, surrogate):
        return self.method_key in {"bb_kissgp", "bb_kiss-gp", "bb_gp"}

    def _sample_neighborhood(self, point, n_points):
        if n_points <= 0:
            return

        samples = []
        if self.options.include_center:
            samples.append(point.copy())

        remaining = max(0, n_points - len(samples))

        # Structured points give derivative-sensitive coverage around the query.
        if self.options.design in {"axis", "hybrid"}:
            structured = self._structured_ball_samples(point, remaining)
            samples.extend(structured)
            remaining = max(0, n_points - len(samples))

        # Random ball samples fill in less grid-like local coverage.
        if remaining and self.options.design in {"random", "hybrid"}:
            samples.extend(self._random_ball_samples(point, remaining))

        for sample in samples[:n_points]:
            sample = self._clip_physical(np.asarray(sample, dtype=float))
            self.cache.evaluate(sample[0], sample[1])

    def _structured_ball_samples(self, point, n_points):
        if n_points <= 0:
            return []

        radius = self.options.effective_sample_radius
        scaled_center = self._to_scaled(point.reshape(1, 2))[0]
        directions = [
            # Axis directions.
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            # Diagonal directions.
            np.array([1.0, 1.0]) / np.sqrt(2.0),
            np.array([1.0, -1.0]) / np.sqrt(2.0),
            np.array([-1.0, 1.0]) / np.sqrt(2.0),
            np.array([-1.0, -1.0]) / np.sqrt(2.0),
        ]
        fractions = [1.0, 0.5, 0.25]
        samples = []

        for fraction in fractions:
            for direction in directions:
                if len(samples) >= n_points:
                    return samples
                samples.append(self._from_scaled(scaled_center + radius * fraction * direction))

        return samples

    def _random_ball_samples(self, point, n_points):
        radius = self.options.effective_sample_radius
        scaled_center = self._to_scaled(point.reshape(1, 2))[0]
        samples = []
        for _ in range(n_points):
            direction = self.rng.normal(size=STATE_DIM)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                direction = np.ones(STATE_DIM) / np.sqrt(STATE_DIM)
            else:
                direction = direction / norm
            radial_fraction = self.rng.random() ** (1.0 / STATE_DIM)
            scaled = scaled_center + radius * radial_fraction * direction
            samples.append(self._from_scaled(scaled))
        return samples

    def _clip_physical(self, point):
        # This keeps samples in-domain. Right now the most important bound is T.
        return np.array(
            [
                np.clip(point[0], self.options.s_bounds[0], self.options.s_bounds[1]),
                np.clip(point[1], self.options.T_bounds[0], self.options.T_bounds[1]),
            ],
            dtype=float,
        )

    def _to_scaled(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.center) / self.scale

    def _from_scaled(self, scaled_point):
        scaled_point = np.asarray(scaled_point, dtype=float)
        return self.center + self.scale * scaled_point

    def diagnostics(self):
        # These are read after Newton. Reading diagnostics does not call the
        # oracle, train a surrogate, or mutate the experiment.
        diag = self.cache.info()
        diag.update(
            {
                "bb_eval_count": self.eval_count,
                "bb_refinement_count": self.refinement_count,
                "bb_surrogate_fit_count": self.surrogate_fit_count,
                "bb_failed_refinements": self.failed_refinements,
                "bb_last_uncertainty": self.last_uncertainty,
                "bb_last_status": self.last_status,
                "bb_sample_radius": self.options.effective_sample_radius,
                "bb_validation_radius": self.options.validation_radius,
                "bb_cache_limit": self.options.max_points,
            }
        )
        return diag


def analytic_flux_for_config(config):
    # Reference model for comparison only. The blackbox provider never receives
    # these derivatives as training labels.
    if config not in ORACLE_CONFIGS:
        raise ValueError(f"unknown oracle config: {config}")
    params = ORACLE_CONFIGS[config]

    def flux_law(s, T, xg):
        q = physical_flux(s, T, params["k_0"], params["alpha"], params["beta"])
        dq_ds, dq_dT = physical_flux_derivatives(
            s, T, params["k_0"], params["alpha"], params["beta"]
        )
        return float(q), float(dq_ds), float(dq_dT)

    return flux_law


def build_provider(method, oracle_config="nonlinear_high_noise", *, x_mesh=None, noisy=True, seed=0):
    # This mirrors the tabular build_provider pattern: return a dictionary with
    # a flux law callable and metadata for the comparison runner.
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

    if x_mesh is None:
        mesh_spacing = method_options.get("mesh_spacing", 0.05)
    else:
        x_mesh = np.asarray(x_mesh, dtype=float)
        mesh_spacing = float(np.median(np.diff(x_mesh)))

    # Mesh spacing sets the minimum allowed sampling radius.
    options = AdaptiveBBOptions(
        sample_radius=method_options.get("sample_radius", AdaptiveBBOptions.sample_radius),
        validation_radius_factor=method_options.get(
            "validation_radius_factor",
            AdaptiveBBOptions.validation_radius_factor,
        ),
        refill_points=method_options.get("refill_points", AdaptiveBBOptions.refill_points),
        max_refinements_per_eval=method_options.get(
            "max_refinements_per_eval",
            AdaptiveBBOptions.max_refinements_per_eval,
        ),
        rng_seed=method_options.get("rng_seed", seed),
        design=method_options.get("design", AdaptiveBBOptions.design),
        mse_tolerance=method_options.get("mse_tolerance", AdaptiveBBOptions.mse_tolerance),
        variance_tolerance=method_options.get(
            "variance_tolerance",
            AdaptiveBBOptions.variance_tolerance,
        ),
        mesh_spacing=mesh_spacing,
    )

    oracle = make_diffusion_oracle(oracle_config, seed=seed, noisy=noisy)
    provider = AdaptiveBlackBoxProvider(
        method_key=method_key,
        oracle=oracle,
        options=options,
        model_options=method_options,
    )

    def flux_law(s, T, xg):
        # Newton calls this. This is the only place the adaptive provider is
        # touched during the actual experiment.
        q, dq_ds, dq_dT = provider.evaluate(np.array([s]), np.array([T]))
        return float(q[0]), float(dq_ds[0]), float(dq_dT[0])

    return {
        "method": method,
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "provider": provider,
        "h_s": options.effective_sample_radius * provider.scale[0],
        "h_T": options.effective_sample_radius * provider.scale[1],
        "oracle_config": oracle_config,
        "sample_radius": options.effective_sample_radius,
        "validation_radius": options.validation_radius,
    }
