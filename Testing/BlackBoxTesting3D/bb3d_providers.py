"""Adaptive local black-box providers for 2D/3D diffusion flux oracles.

The physical oracle is conceptually only

    (grad_T, T) -> q.

These providers sample that oracle, fit a local vector-valued surrogate, and
return the fitted flux and fitted derivatives needed by Newton/FEM code:

    q, dq/dgrad_T, dq/dT.

The derivatives here are derivatives of the local fitted model, not extra
training labels from the oracle.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Data.BlackBoxOracle.bboracle3dDiffusion import (  # noqa: E402
    ORACLE_CONFIGS,
    make_diffusion_oracle,
    physical_flux,
    physical_flux_derivatives,
)


def _multiindices(n_features: int, degree: int) -> list[tuple[int, ...]]:
    powers: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        for exponent in product(range(total + 1), repeat=n_features):
            if sum(exponent) == total:
                powers.append(tuple(int(v) for v in exponent))
    return powers


@dataclass
class AdaptiveBB3DOptions:
    """Sampling and fitting options for the local 2D/3D diffusion provider."""

    grad_bounds: tuple[tuple[float, float], ...] = (
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
    )
    T_bounds: tuple[float, float] = (0.0, 3.0)
    sample_radius: float = 0.35
    validation_radius_factor: float = 1.5
    initial_points_per_dim: int = 12
    max_points_per_dim: int = 18
    refill_points: int = 12
    max_refinements_per_eval: int = 1
    rng_seed: int = 0
    design: str = "hybrid"
    include_center: bool = True
    oracle_key_decimals: int = 12
    active_prune_policy: str = "fifo"
    mse_tolerance: float = 5.0e-3
    min_mesh_radius_factor: float = 2.0
    mesh_spacing: float = 0.10

    def bounds_for_dim(self, dim: int) -> np.ndarray:
        return np.array([*self.grad_bounds[:dim], self.T_bounds], dtype=float)

    def max_points(self, state_dim: int) -> int:
        return self.max_points_per_dim * state_dim

    def initial_points(self, state_dim: int) -> int:
        return self.initial_points_per_dim * state_dim

    @property
    def effective_sample_radius(self) -> float:
        min_radius = self.min_mesh_radius_factor * self.mesh_spacing
        return max(float(self.sample_radius), float(min_radius))

    @property
    def validation_radius(self) -> float:
        return self.validation_radius_factor * self.effective_sample_radius


def parse_method_spec(method: str) -> tuple[str, dict]:
    """Parse compact specs like ``bb3d_poly+degree=3+sample_radius=0.5``."""
    text = str(method)
    if "+" not in text:
        return text.lower(), {}

    parts = text.split("+")
    method_key = parts[0].lower()
    options: dict[str, object] = {}
    for part in parts[1:]:
        if not part.strip():
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue

        key = key.strip().lower()
        value = value.strip()
        try:
            if key in {
                "sample_radius",
                "validation_radius_factor",
                "mse_tolerance",
                "mesh_spacing",
                "ridge_strength",
                "gamma",
            }:
                options[key] = float(value)
            elif key in {
                "degree",
                "initial_points_per_dim",
                "max_points_per_dim",
                "refill_points",
                "max_refinements_per_eval",
                "rng_seed",
            }:
                options[key] = int(value)
            elif key in {"include_center"}:
                options[key] = value.lower() in {"1", "true", "yes", "on"}
            else:
                options[key] = value
        except ValueError:
            options[key] = value
    return method_key, options


class OracleVectorCache:
    """Memoized cache of oracle samples keyed by physical state."""

    def __init__(self, oracle, dim: int, options: AdaptiveBB3DOptions) -> None:
        self.oracle = oracle
        self.dim = int(dim)
        self.options = options
        self.samples: OrderedDict[tuple[float, ...], np.ndarray] = OrderedDict()
        self.oracle_calls = 0
        self.prune_count = 0

        @lru_cache(maxsize=100_000)
        def cached_eval(*key: float) -> tuple[float, ...]:
            self.oracle_calls += 1
            grad = np.array(key[: self.dim], dtype=float)
            T = float(key[self.dim])
            return tuple(np.asarray(self.oracle(grad, T), dtype=float).reshape(self.dim))

        self._cached_eval = cached_eval

    def key(self, state: np.ndarray) -> tuple[float, ...]:
        decimals = self.options.oracle_key_decimals
        return tuple(round(float(v), decimals) for v in state)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        key = self.key(state)
        value = np.array(self._cached_eval(*key), dtype=float)
        self.samples.setdefault(key, value)
        return value

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            return np.empty((0, self.dim + 1)), np.empty((0, self.dim))
        return (
            np.array(list(self.samples.keys()), dtype=float),
            np.array(list(self.samples.values()), dtype=float),
        )

    def prune(self, center: np.ndarray, scale: np.ndarray, max_points: int, policy: str) -> None:
        if len(self.samples) <= max_points:
            return
        items = list(self.samples.items())
        if policy == "distance":
            points = np.array([key for key, _ in items], dtype=float)
            distances = np.linalg.norm((points - center) / scale, axis=1)
            keep = set(np.argsort(distances)[:max_points])
        else:
            keep = set(range(len(items) - max_points, len(items)))
        self.samples = OrderedDict(
            (key, q) for idx, (key, q) in enumerate(items) if idx in keep
        )
        self.prune_count += len(items) - len(self.samples)

    def info(self) -> dict:
        info = self._cached_eval.cache_info()
        return {
            "oracle_cache_size": len(self.samples),
            "oracle_lru_hits": info.hits,
            "oracle_lru_misses": info.misses,
            "oracle_calls": self.oracle_calls,
            "oracle_cache_prunes": self.prune_count,
        }


class PolynomialVectorProvider:
    """Local polynomial least-squares fit for vector flux with analytic derivatives."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, degree: int = 2, ridge_strength: float = 0.0):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.n_features = self.X.shape[1]
        self.degree = int(degree)
        self.ridge_strength = float(ridge_strength)
        self.powers = _multiindices(self.n_features, self.degree)

        design = self._design(self.X)
        if self.ridge_strength > 0.0:
            lhs = design.T @ design + self.ridge_strength * np.eye(design.shape[1])
            rhs = design.T @ self.Y
            self.coef_ = np.linalg.solve(lhs, rhs)
        else:
            self.coef_, *_ = np.linalg.lstsq(design, self.Y, rcond=None)

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        columns = []
        for power in self.powers:
            value = np.ones(X.shape[0], dtype=float)
            for axis, exponent in enumerate(power):
                if exponent:
                    value *= X[:, axis] ** exponent
            columns.append(value)
        return np.column_stack(columns)

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        q = self._design(X) @ self.coef_
        grad_q = np.zeros((X.shape[0], self.Y.shape[1], self.n_features), dtype=float)

        for term, power in enumerate(self.powers):
            coef = self.coef_[term]
            for diff_axis, exponent in enumerate(power):
                if exponent == 0:
                    continue
                value = np.full(X.shape[0], exponent, dtype=float)
                for axis, axis_power in enumerate(power):
                    reduced_power = axis_power - 1 if axis == diff_axis else axis_power
                    if reduced_power:
                        value *= X[:, axis] ** reduced_power
                grad_q[:, :, diff_axis] += value[:, np.newaxis] * coef
        return q, grad_q


class RBFVectorProvider:
    """Gaussian kernel ridge fit for vector flux with analytic derivatives."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0, ridge_strength: float = 1.0e-8):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.gamma = float(gamma)
        self.ridge_strength = float(ridge_strength)
        K = self._kernel(self.X, self.X)
        self.alpha_ = np.linalg.solve(K + self.ridge_strength * np.eye(K.shape[0]), self.Y)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        return np.exp(-self.gamma * np.sum(diff**2, axis=-1))

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K = self._kernel(X, self.X)
        q = K @ self.alpha_
        diff = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        dK = -2.0 * self.gamma * K[:, :, np.newaxis] * diff
        grad_q = np.einsum("nmf,md->ndf", dK, self.alpha_)
        return q, grad_q


class AdaptiveLocalFluxProvider:
    """Adaptive local vector surrogate around queried ``(grad_T, T)`` states."""

    def __init__(
        self,
        method_key: str,
        oracle,
        dim: int = 3,
        options: AdaptiveBB3DOptions | None = None,
        model_options: dict | None = None,
    ) -> None:
        self.method_key = method_key
        self.dim = int(dim)
        self.state_dim = self.dim + 1
        self.options = options or AdaptiveBB3DOptions()
        self.model_options = model_options or {}
        self.rng = np.random.default_rng(self.options.rng_seed)
        self.bounds = self.options.bounds_for_dim(self.dim)
        self.center = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        self.scale = 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.scale[self.scale == 0.0] = 1.0
        self.cache = OracleVectorCache(oracle, self.dim, self.options)
        self.surrogate = None

        self.eval_count = 0
        self.refinement_count = 0
        self.surrogate_fit_count = 0
        self.failed_refinements = 0
        self.last_uncertainty = np.nan
        self.uncertainty_sum = 0.0
        self.uncertainty_count = 0
        self.last_status = "not_evaluated"

    def evaluate(self, grad_T: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad = np.asarray(grad_T, dtype=float)
        temperature = np.asarray(T, dtype=float)
        if grad.shape == (self.dim,):
            q, A, b = self._evaluate_one(grad, float(temperature))
            return q, A, b

        if grad.shape[-1] != self.dim:
            raise ValueError(f"grad_T final axis must have length {self.dim}")
        flat_grad = grad.reshape(-1, self.dim)
        flat_T = np.broadcast_to(temperature, grad.shape[:-1]).reshape(-1)
        q = np.empty((flat_grad.shape[0], self.dim))
        A = np.empty((flat_grad.shape[0], self.dim, self.dim))
        b = np.empty((flat_grad.shape[0], self.dim))
        for i, (g_i, T_i) in enumerate(zip(flat_grad, flat_T)):
            q[i], A[i], b[i] = self._evaluate_one(g_i, float(T_i))
        return q.reshape(grad.shape), A.reshape(grad.shape[:-1] + (self.dim, self.dim)), b.reshape(grad.shape)

    def _evaluate_one(self, grad: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval_count += 1
        state = self._clip_state(np.r_[grad, T])

        if self.surrogate is None:
            self._sample_neighborhood(state, self.options.initial_points(self.state_dim))
            self._prune_cache_near(state)
            self._fit_surrogate()

        q, jac = self._surrogate_evaluate(state)
        uncertainty = self._validation_mse(state)
        refinements = 0
        while (
            uncertainty > self.options.mse_tolerance
            and refinements < self.options.max_refinements_per_eval
        ):
            self._sample_neighborhood(state, self.options.refill_points)
            self._prune_cache_near(state)
            self._fit_surrogate()
            q, jac = self._surrogate_evaluate(state)
            uncertainty = self._validation_mse(state)
            refinements += 1
            self.refinement_count += 1

        self._prune_cache_near(state)
        self.last_uncertainty = float(uncertainty)
        self.uncertainty_sum += float(uncertainty)
        self.uncertainty_count += 1
        self.last_status = "ok" if uncertainty <= self.options.mse_tolerance else "accepted_high_mse"
        A = jac[:, : self.dim]
        b = jac[:, self.dim]
        return q, A, b

    def _fit_surrogate(self) -> None:
        X_phys, Y = self.cache.arrays()
        if Y.shape[0] < max(self.state_dim + 1, 6):
            raise RuntimeError("not enough blackbox samples to fit local surrogate")
        X = self._to_scaled(X_phys)
        if self.method_key in {"bb3d_poly", "bb_poly", "poly"}:
            self.surrogate = PolynomialVectorProvider(
                X,
                Y,
                degree=int(self.model_options.get("degree", 2)),
                ridge_strength=float(self.model_options.get("ridge_strength", 1.0e-10)),
            )
        elif self.method_key in {"bb3d_rbf", "bb_rbf", "rbf"}:
            self.surrogate = RBFVectorProvider(
                X,
                Y,
                gamma=float(self.model_options.get("gamma", 1.0)),
                ridge_strength=float(self.model_options.get("ridge_strength", 1.0e-8)),
            )
        else:
            raise ValueError(f"unknown 3D blackbox method: {self.method_key}")
        self.surrogate_fit_count += 1

    def _surrogate_evaluate(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = self._to_scaled(state.reshape(1, self.state_dim))
        q, jac_scaled = self.surrogate.evaluate(scaled)
        jac_physical = jac_scaled[0] / self.scale[np.newaxis, :]
        return q[0], jac_physical

    def _validation_mse(self, state: np.ndarray) -> float:
        points = self._random_ball_samples(state, max(4, self.state_dim))
        errors = []
        for point in points:
            q_true = self.cache.evaluate(point)
            q_hat, _ = self._surrogate_evaluate(point)
            denom = np.maximum(1.0, np.abs(q_true))
            errors.append(np.mean(((q_hat - q_true) / denom) ** 2))
        return float(np.mean(errors)) if errors else 0.0

    def _sample_neighborhood(self, state: np.ndarray, n_points: int) -> None:
        samples = [state.copy()] if self.options.include_center else []
        if self.options.design in {"axis", "hybrid"}:
            samples.extend(self._axis_samples(state))
        if self.options.design in {"random", "hybrid"}:
            samples.extend(self._random_ball_samples(state, max(0, n_points - len(samples))))

        for sample in samples[:n_points]:
            self.cache.evaluate(self._clip_state(sample))

    def _prune_cache_near(self, state: np.ndarray) -> None:
        self.cache.prune(
            state,
            self.scale,
            self.options.max_points(self.state_dim),
            self.options.active_prune_policy,
        )

    def _axis_samples(self, state: np.ndarray) -> list[np.ndarray]:
        radius = self.options.effective_sample_radius
        scaled = self._to_scaled(state.reshape(1, self.state_dim))[0]
        samples = []
        for axis in range(self.state_dim):
            direction = np.zeros(self.state_dim)
            direction[axis] = 1.0
            samples.append(self._from_scaled(scaled + radius * direction))
            samples.append(self._from_scaled(scaled - radius * direction))
        return samples

    def _random_ball_samples(self, state: np.ndarray, n_points: int) -> list[np.ndarray]:
        radius = self.options.validation_radius
        scaled_center = self._to_scaled(state.reshape(1, self.state_dim))[0]
        samples = []
        for _ in range(n_points):
            direction = self.rng.normal(size=self.state_dim)
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                direction = np.ones(self.state_dim) / np.sqrt(self.state_dim)
            else:
                direction = direction / norm
            radial_fraction = self.rng.random() ** (1.0 / self.state_dim)
            samples.append(self._from_scaled(scaled_center + radius * radial_fraction * direction))
        return samples

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        return np.clip(state, self.bounds[:, 0], self.bounds[:, 1])

    def _to_scaled(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.center) / self.scale

    def _from_scaled(self, X: np.ndarray) -> np.ndarray:
        return self.center + self.scale * np.asarray(X, dtype=float)

    def diagnostics(self) -> dict:
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
                "bb_cache_limit": self.options.max_points(self.state_dim),
            }
        )
        return diag


def analytic_flux_for_config(config: str, dim: int = 3):
    """Reference flux law returning exact q, A=dq/dg, and b=dq/dT."""
    if config not in ORACLE_CONFIGS:
        raise ValueError(f"unknown oracle config: {config}")
    params = ORACLE_CONFIGS[config]

    def flux_law(grad_T, T, pt=None):
        q = physical_flux(grad_T, T, params["k_0"], params["alpha"], params["beta"])
        A, b = physical_flux_derivatives(
            grad_T, T, params["k_0"], params["alpha"], params["beta"]
        )
        return np.asarray(q, dtype=float).reshape(dim), np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(dim)

    return flux_law


def build_provider(
    method,
    oracle_config: str = "nonlinear_high_noise",
    *,
    dim: int = 3,
    mesh_spacing: float = 0.10,
    noisy: bool = True,
    seed: int = 0,
    provider_options: dict | None = None,
) -> dict:
    """Return a comparison-ready flux-law dictionary."""
    method_key, method_options = parse_method_spec(method)
    method_options = {**method_options, **(provider_options or {})}
    start = time.perf_counter()

    if dim not in (2, 3):
        raise ValueError(f"dim must be either 2 or 3, got {dim}.")

    if method_key == "analytic":
        return {
            "method": method,
            "flux": analytic_flux_for_config(oracle_config, dim=dim),
            "build_s": time.perf_counter() - start,
            "provider": None,
            "oracle_config": oracle_config,
            "dim": dim,
        }

    grad_bounds = method_options.get("grad_bounds", AdaptiveBB3DOptions.grad_bounds)
    if len(grad_bounds) < dim:
        raise ValueError(f"grad_bounds must provide at least {dim} bounds")

    options = AdaptiveBB3DOptions(
        grad_bounds=tuple(tuple(v) for v in grad_bounds),
        T_bounds=method_options.get("T_bounds", AdaptiveBB3DOptions.T_bounds),
        sample_radius=method_options.get("sample_radius", AdaptiveBB3DOptions.sample_radius),
        validation_radius_factor=method_options.get(
            "validation_radius_factor",
            AdaptiveBB3DOptions.validation_radius_factor,
        ),
        initial_points_per_dim=method_options.get(
            "initial_points_per_dim",
            AdaptiveBB3DOptions.initial_points_per_dim,
        ),
        max_points_per_dim=method_options.get(
            "max_points_per_dim",
            AdaptiveBB3DOptions.max_points_per_dim,
        ),
        refill_points=method_options.get("refill_points", AdaptiveBB3DOptions.refill_points),
        max_refinements_per_eval=method_options.get(
            "max_refinements_per_eval",
            AdaptiveBB3DOptions.max_refinements_per_eval,
        ),
        rng_seed=method_options.get("rng_seed", seed),
        design=method_options.get("design", AdaptiveBB3DOptions.design),
        include_center=method_options.get("include_center", AdaptiveBB3DOptions.include_center),
        active_prune_policy=method_options.get(
            "active_prune_policy",
            AdaptiveBB3DOptions.active_prune_policy,
        ),
        mse_tolerance=method_options.get("mse_tolerance", AdaptiveBB3DOptions.mse_tolerance),
        mesh_spacing=method_options.get("mesh_spacing", mesh_spacing),
    )
    oracle = make_diffusion_oracle(oracle_config, dim=dim, seed=seed, noisy=noisy)
    provider = AdaptiveLocalFluxProvider(
        method_key=method_key,
        oracle=oracle,
        dim=dim,
        options=options,
        model_options=method_options,
    )

    def flux_law(grad_T, T, pt=None):
        return provider.evaluate(grad_T, T)

    return {
        "method": method,
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "provider": provider,
        "oracle_config": oracle_config,
        "dim": dim,
        "sample_radius": options.effective_sample_radius,
        "validation_radius": options.validation_radius,
    }
