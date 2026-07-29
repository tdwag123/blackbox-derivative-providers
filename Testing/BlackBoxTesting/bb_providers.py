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

# ----------------------- import derivative providers ------------------------------------------------------------------
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
    from Methods.OracleDataMethods.GP.monotoneGPReg import MonotoneGPFluxST
except (ImportError, OSError, AttributeError):
    MonotoneGPFluxST = None
# -----------------------------------------------------------------------------------------------------------------------

STATE_DIM = 2


@dataclass # main purpose is storing data
class AdaptiveBBOptions: 
    # Physical state bounds used to keep random samples inside the oracle domain.
    # The provider scales these bounds to roughly [-1, 1]^2 before measuring
    # radii, so s and T do not dominate each other just because of units.
    s_bounds: tuple = (-6.0, 12.0)
    T_bounds: tuple = (0.0, 15.0)

    # Radius is measured in scaled (s, T) coordinates. The effective radius is
    # also forced to be at least 2 * FEM mesh spacing below.
    sample_radius: float = 0.25 
    # d = 2 for this 1D FEM problem because the constitutive state is (s, T).
    # Empty oracle cache starts full at 30d points.
    initial_points_per_dim: int = 30
    max_points_per_dim: int = 30

    # When a stencil is stale, add local oracle samples to the cache, then fit
    # the surrogate on the entire current cache.
    refill_points: int = 4
    stencil_refresh_fraction: float = 0.35
    max_refinements_per_eval: int = 0
    rng_seed: int = 0
    max_stencil_states: int = 128

    # "hybrid" gives a small structured stencil first, then fills the rest with
    # random ball samples. Use "random" if you want purely random sampling.
    # Options: "axis", "random", "hybrid"
    design: str = "axis"
    include_center: bool = True
    # Low reuse trains more, smaller regional stencils. High reuse shares each
    # fitted stencil over a wider state-space region, saving calls but risking
    # stale or overly broad derivatives.
    stencil_reuse_factor: float = 1.0

    # Exact floating-point equality is unreliable, so sample keys are rounded
    # before duplicate checks and lru_cache lookup.
    oracle_key_decimals: int = 12

    # GP-like methods use predictive variance. Non-GP methods use stencil
    # geometry only.
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
                    "variance_tolerance",
                    "mesh_spacing",
                    "gamma",
                    "epsilon",
                    "ridge_strength",
                    "alpha",
                    "learning_rate",
                    "stencil_reuse_factor",
                    "stencil_refresh_fraction",
                }:
                    options[key] = float(value)
                elif key in {
                    "refill_points",
                    "max_refinements_per_eval",
                    "initial_points_per_dim",
                    "max_points_per_dim",
                    "rng_seed",
                    "max_stencil_states",
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


class OracleCallCache:
    """Stores oracle evaluations; every surrogate fit uses this whole cache."""

    def __init__(self, oracle, options):
        self.oracle = oracle
        self.options = options
        self.samples = OrderedDict()
        self.oracle_calls = 0
        self.prune_count = 0

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
        q = self._cached_eval(*key)
        if key not in self.samples:
            self.samples[key] = q
            while len(self.samples) > self.options.max_points:
                self.samples.popitem(last=False)
                self.prune_count += 1
        return q

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


@dataclass
class StencilState:
    surrogate: object
    center: np.ndarray
    radius: float


class AdaptiveBlackBoxProvider:
    """
    Stateful blackbox flux provider used by Newton.

    Each call receives one constitutive state (s, T). Non-GP methods use a
    moving stencil: build/reuse a local surrogate while the query stays in the
    stencil region, and shift the stencil by adding only a few new in-domain
    samples when it moves. GP-like methods can still use predictive variance.
    Nothing here uses true derivatives.
    """

    def __init__(self, method_key, oracle, options=None, model_options=None):
        self.method_key = method_key
        self.oracle = oracle
        self.options = options or AdaptiveBBOptions()
        self.model_options = model_options or {}
        self.rng = np.random.default_rng(self.options.rng_seed)
        self.oracle_cache = OracleCallCache(oracle, self.options)
        self.eval_count = 0
        self.refinement_count = 0
        self.surrogate_fit_count = 0
        self.failed_refinements = 0
        self.last_uncertainty = np.nan
        self.last_status = "not_evaluated"
        self.current_surrogate = None
        self.current_stencil_center = None
        self.current_stencil_radius = None
        self.current_stencil_key = None
        self.stencil_states = OrderedDict()
        self.last_staleness_reason = "not_evaluated"

        # Scaling map for geometry. Sampling radii and neighborhood tests
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

        if not self._is_gp_method():
            return self._evaluate_one_stencil(point)

        # GP-like methods train from the whole oracle cache after adding local samples.
        if len(self.oracle_cache.samples) == 0:
            self._sample_neighborhood(point, self.options.initial_points)
        elif not self._has_local_coverage(point, self.options.effective_sample_radius):
            # GP-like methods still use local coverage because KISS-GP cannot
            # evaluate far outside its fitted interpolation grid.
            self._sample_neighborhood(point, self.options.refill_points)
            self.refinement_count += 1

        result = None
        for attempt in range(self.options.max_refinements_per_eval + 1):
            # GP-like methods still refit because variance is part of their
            # accept/refine decision. Non-GP methods use _evaluate_one_stencil.
            surrogate = self._fit_surrogate()
            result = self._surrogate_evaluate(surrogate, point)
            variance = self._gp_variance(result)
            self.last_uncertainty = variance

            if self._gp_variance_is_ok(variance):
                self.last_status = "ok"
                return result[:3]

            # If uncertainty is bad, resample locally and refit.
            if attempt == self.options.max_refinements_per_eval:
                self.failed_refinements += 1
                self.last_status = "max_refinements_uncertain"
                return result[:3]

            self._sample_neighborhood(point, self.options.refill_points)
            self.refinement_count += 1

        return result[:3]

    def _evaluate_one_stencil(self, point):
        self._load_best_stencil_state(point)
        if not self._stencil_is_fresh(point):
            self._refresh_stencil(point)
            self._save_stencil_state()

        result = self._surrogate_evaluate(self.current_surrogate, point)
        self.last_uncertainty = np.nan
        self.last_status = "stencil_ok"
        return result[:3]

    def _load_best_stencil_state(self, point):
        scaled = self._to_scaled(point.reshape(1, 2))[0]
        best_key = None
        best_distance = np.inf
        for key, state in self.stencil_states.items():
            distance = np.linalg.norm(scaled - state.center)
            if distance <= self.options.stencil_reuse_factor * state.radius and distance < best_distance:
                best_key = key
                best_distance = distance

        if best_key is None:
            self.current_stencil_key = None
            self.current_surrogate = None
            self.current_stencil_center = None
            self.current_stencil_radius = None
            return

        self.current_stencil_key = best_key
        state = self.stencil_states[best_key]
        self.stencil_states.move_to_end(best_key)
        self.current_surrogate = state.surrogate
        self.current_stencil_center = state.center
        self.current_stencil_radius = state.radius

    def _save_stencil_state(self):
        if self.current_stencil_key is None:
            self.current_stencil_key = ("region", len(self.stencil_states), self.eval_count)

        self.stencil_states[self.current_stencil_key] = StencilState(
            surrogate=self.current_surrogate,
            center=self.current_stencil_center.copy(),
            radius=self.current_stencil_radius,
        )
        self.stencil_states.move_to_end(self.current_stencil_key)
        while len(self.stencil_states) > self.options.max_stencil_states:
            self.stencil_states.popitem(last=False)

    def _stencil_is_fresh(self, point):
        if self.current_surrogate is None or self.current_stencil_center is None:
            self.last_staleness_reason = "no_surrogate"
            return False

        distance = np.linalg.norm(
            self._to_scaled(point.reshape(1, 2))[0] - self.current_stencil_center
        )
        reuse_radius = self.options.stencil_reuse_factor * self.current_stencil_radius
        if distance > reuse_radius:
            self.last_staleness_reason = "outside_reuse_radius"
            return False

        self.last_staleness_reason = "fresh"
        return True

    def _refresh_stencil(self, point):
        self.current_stencil_center = self._to_scaled(point.reshape(1, 2))[0]
        self.current_stencil_radius = self.options.effective_sample_radius
        points_to_sample = (
            self.options.initial_points
            if len(self.oracle_cache.samples) == 0
            else max(
                self.options.refill_points,
                int(round(self.options.max_points * self.options.stencil_refresh_fraction)),
            )
        )
        self._sample_neighborhood(point, points_to_sample)
        self.current_surrogate = self._fit_surrogate()
        self.refinement_count += 1

    def _fit_surrogate(self):
        s_data, T_data, q_data = self.oracle_cache.arrays()
        if len(q_data) < max(STATE_DIM + 1, 4):
            raise RuntimeError("not enough blackbox samples to fit a local surrogate")

        # All local surrogate classes see scaled coordinates. Their derivative
        # outputs are converted back to physical derivatives in _surrogate_evaluate.
        X_scaled = self._to_scaled(np.column_stack([s_data, T_data]))
        self.surrogate_fit_count += 1

        # Kernel ridge / RBF is the simplest non-GP local surrogate. With this
        # small oracle cache, the default ridge is deliberately not tiny; the
        # high-noise sweeps favored about 1e-3 over the older 1e-4 default.
        if self.method_key in {"bb_rbf", "bb_krr", "bb_rbf_krr"}:
            epsilon = self.model_options.get("epsilon", 0.3)
            ridge = self.model_options.get("ridge_strength", 1.0e-3)
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="gaussian",
                epsilon=epsilon,
                ridge_strength=ridge,
            )

        # Same KRR machinery, but with a Matern 5/2 kernel. The same low-sample
        # ridge default is used here; larger values oversmoothed in the sweeps.
        if self.method_key in {"bb_matern52_krr", "bb_matern_krr"}:
            epsilon = self.model_options.get("epsilon", 0.3)
            ridge = self.model_options.get("ridge_strength", 1.0e-3)
            return KernelDerivativeProviderST(
                X_scaled[:, 0],
                X_scaled[:, 1],
                q_data,
                function="matern52",
                epsilon=epsilon,
                ridge_strength=ridge,
            )

        # Random Fourier features surrogate. It uses the same regional stencil
        # policy as the KRR providers.
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

        # KISS-GP can return predictive variance, so this path uses variance as
        # the accept/refine signal.
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

    def _gp_variance(self, result):
        return max(float(result[3]), 0.0)

    def _gp_variance_is_ok(self, variance):
        if not np.isfinite(variance):
            return False
        return variance <= self.options.variance_tolerance

    def _surrogate_has_variance(self, surrogate):
        return self.method_key in {"bb_kissgp", "bb_kiss-gp", "bb_gp"}

    def _is_gp_method(self):
        return self.method_key in {
            "bb_kissgp",
            "bb_kiss-gp",
            "bb_gp",
            "bb_monotonegp",
            "bb_materngpmonotone",
        }

    def _has_local_coverage(self, point, radius):
        s_data, T_data, _ = self.oracle_cache.arrays()
        if len(s_data) < max(4, STATE_DIM + 1):
            return False

        X = np.column_stack([s_data, T_data])
        distances = np.linalg.norm(
            self._to_scaled(X) - self._to_scaled(point.reshape(1, 2)),
            axis=1,
        )
        return np.count_nonzero(distances <= radius) >= max(4, STATE_DIM + 1)

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

        sampled = 0
        seen = set()
        attempts = 0
        while sampled < n_points and attempts < 20 * max(n_points, 1):
            attempts += 1
            if attempts <= len(samples):
                sample = np.asarray(samples[attempts - 1], dtype=float)
            else:
                sample = np.asarray(self._random_ball_samples(point, 1)[0], dtype=float)

            sample = self._fold_into_domain(sample)

            key = self.oracle_cache.key(sample[0], sample[1])
            if key in seen:
                continue

            seen.add(key)
            self.oracle_cache.evaluate(sample[0], sample[1])
            sampled += 1

        if sampled < max(4, STATE_DIM + 1):
            raise RuntimeError(
                "could not build enough in-domain stencil samples near the query point"
            )

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
        fractions = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
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
        return (points - self.center) / self.scale

    def _from_scaled(self, scaled_point):
        scaled_point = np.asarray(scaled_point, dtype=float)
        return self.center + self.scale * scaled_point

    def diagnostics(self):
        # These are read after Newton. Reading diagnostics does not call the
        # oracle, train a surrogate, or mutate the experiment.
        diag = self.oracle_cache.info()
        diag.update(
            {
                "bb_eval_count": self.eval_count,
                "bb_refinement_count": self.refinement_count,
                "bb_surrogate_fit_count": self.surrogate_fit_count,
                "bb_failed_refinements": self.failed_refinements,
                "bb_last_uncertainty": self.last_uncertainty,
                "bb_last_status": self.last_status,
                "bb_sample_radius": self.options.effective_sample_radius,
                "bb_oracle_cache_size": len(self.oracle_cache.samples),
                "bb_stencil_reuse_factor": self.options.stencil_reuse_factor,
                "bb_stencil_refresh_fraction": self.options.stencil_refresh_fraction,
                "bb_last_staleness_reason": self.last_staleness_reason,
                "bb_stencil_state_count": len(self.stencil_states),
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
        initial_points_per_dim=method_options.get(
            "initial_points_per_dim",
            AdaptiveBBOptions.initial_points_per_dim,
        ),
        max_points_per_dim=method_options.get(
            "max_points_per_dim",
            AdaptiveBBOptions.max_points_per_dim,
        ),
        refill_points=method_options.get("refill_points", AdaptiveBBOptions.refill_points),
        max_refinements_per_eval=method_options.get(
            "max_refinements_per_eval",
            AdaptiveBBOptions.max_refinements_per_eval,
        ),
        rng_seed=method_options.get("rng_seed", seed),
        max_stencil_states=method_options.get(
            "max_stencil_states",
            AdaptiveBBOptions.max_stencil_states,
        ),
        design=method_options.get("design", AdaptiveBBOptions.design),
        variance_tolerance=method_options.get(
            "variance_tolerance",
            AdaptiveBBOptions.variance_tolerance,
        ),
        stencil_reuse_factor=method_options.get(
            "stencil_reuse_factor",
            AdaptiveBBOptions.stencil_reuse_factor,
        ),
        stencil_refresh_fraction=method_options.get(
            "stencil_refresh_fraction",
            AdaptiveBBOptions.stencil_refresh_fraction,
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
        q, dq_ds, dq_dT = provider.evaluate(
            np.array([s]),
            np.array([T]),
        )
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
        "stencil_reuse_factor": options.stencil_reuse_factor,
        "stencil_refresh_fraction": options.stencil_refresh_fraction,
        "max_stencil_states": options.max_stencil_states,
    }
