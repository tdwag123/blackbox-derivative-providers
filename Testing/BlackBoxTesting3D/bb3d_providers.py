"""Adaptive local black-box providers for 2D/3D diffusion flux oracles.

The physical oracle is conceptually only

    (grad_T, T) -> q.

These providers sample that oracle, fit a local vector-valued surrogate, and
return the fitted flux and fitted derivatives needed by Newton/FEM code:

    q, dq/dgrad_T, dq/dT.

The derivatives here are derivatives of the local fitted model, not extra
training labels from the oracle.

How to read this file
---------------------
The provider has four layers:

1. ``OracleVectorCache`` calls the expensive/black-box oracle and remembers
   samples. A sample point is the physical state ``[grad_T..., T]``.
2. ``PolynomialVectorProvider``, ``RBFVectorProvider``, and
   ``BaseGPVectorProvider`` are local surrogate models. They fit only flux
   values ``q`` and then differentiate the fitted model.
3. ``AdaptiveLocalFluxProvider`` decides where to sample near a Newton/FEM
   query point, when to refit, and when to add more nearby points.
4. ``build_provider`` is the public factory used by comparison scripts.

For 3D diffusion, the state and output are

    state = [Tx, Ty, Tz, T]
    q     = [qx, qy, qz]

and the FEM/Newton-facing return value is

    q, A, b

where ``A = dq/dgrad_T`` has shape ``(3, 3)`` and ``b = dq/dT`` has shape
``(3,)``.
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

try:
    from Methods.OracleDataMethods.Multidimensional.baseGP import GPFluxST as BaseGPFluxST
except (ImportError, OSError):
    BaseGPFluxST = None


def _multiindices(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """List polynomial powers up to a total degree."""
    powers: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        for exponent in product(range(total + 1), repeat=n_features):
            if sum(exponent) == total:
                powers.append(tuple(int(v) for v in exponent))
    return powers


@dataclass
class AdaptiveBB3DOptions:
    """Sampling and fitting options for the local 2D/3D diffusion provider."""

    # Bounds define the physical box where local samples are allowed to live.
    # For dim=2 the first two gradient bounds are used; for dim=3 all three are.
    grad_bounds: tuple[tuple[float, float], ...] = (
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
    )
    T_bounds: tuple[float, float] = (-5.0, 17.0)

    # Radii are measured after scaling each constitutive coordinate to
    # comparable size. This is a radius in (grad_T, T) state space, not a
    # physical mesh radius in x/y/z space.
    sample_radius: float = 0.75
    validation_radius_factor: float = 1.5

    # Point counts scale with state_dim = dim + 1 because the state is
    # [gradient components..., temperature]. In 3D, state_dim = 4, so the
    # default active cache holds 30 * 4 = 120 oracle samples.
    initial_points_per_dim: int = 30
    max_points_per_dim: int = 30
    refill_points: int = 12
    max_refinements_per_eval: int = 1
    max_total_refinements: int = 8
    rng_seed: int = 0
    design: str = "hybrid"
    include_center: bool = True
    oracle_key_decimals: int = 12
    active_prune_policy: str = "fifo"
    mse_tolerance: float = 5.0e-3
    # Kept as an option for experiments, but the default is zero because FEM
    # mesh spacing has different units than the scaled constitutive state.
    min_mesh_radius_factor: float = 0.0
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
                "noise_std",
                "jitter",
                "reg_function",
                "kernel_variance",
                "lengthscale",
                "noise_variance",
            }:
                options[key] = float(value)
            elif key in {
                "degree",
                "initial_points_per_dim",
                "max_points_per_dim",
                "refill_points",
                "max_refinements_per_eval",
                "max_total_refinements",
                "rng_seed",
            }:
                options[key] = int(value)
            elif key in {"include_center", "learn_neg_flux", "optimize_hyperparameters"}:
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
        self.clipped_sample_count = 0

        @lru_cache(maxsize=100_000)
        def cached_eval(*key: float) -> tuple[float, ...]:
            # This is the only place the black-box oracle is actually called.
            # The oracle receives grad_T and T separately and returns q only.
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
        """Keep the active cache bounded so the surrogate stays local."""
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
            "oracle_clipped_samples": self.clipped_sample_count,
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

        # One design matrix fits every flux component at once:
        # design @ coef_ approximates [qx, qy, qz].
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
        """Return fitted q and its Jacobian with respect to the scaled state."""
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
        """Return fitted q and its Jacobian with respect to the scaled state."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K = self._kernel(X, self.X)
        q = K @ self.alpha_
        diff = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        dK = -2.0 * self.gamma * K[:, :, np.newaxis] * diff
        grad_q = np.einsum("nmf,md->ndf", dK, self.alpha_)
        return q, grad_q


class BaseGPVectorProvider:
    """Adapter for Methods.OracleDataMethods.Multidimensional.baseGP.GPFluxST."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        dim: int,
        *,
        noise_std: float = 0.0,
        learn_neg_flux: bool = True,
        jitter: float = 1.0e-8,
        n_restarts_optimizer: int = 0,
        reg_function: float = 0.0,
        kernel_variance: float = 1.0,
        lengthscale: float | np.ndarray = 2.0,
        noise_variance: float = 1.0e-2,
        optimize_hyperparameters: bool = False,
        max_cache_size: int | None = None,
    ) -> None:
        if BaseGPFluxST is None:
            raise ImportError(
                "bb3d_basegp requires Methods.OracleDataMethods.Multidimensional.baseGP"
            )
        self.dim = int(dim)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # The multidimensional GP uses the older naming convention:
        #   s = gradient-like input, shape (n, dim)
        #   T = temperature input, shape (n, 1)
        #   q = vector flux, shape (n, dim)
        self.model = BaseGPFluxST(
            X[:, : self.dim],
            X[:, self.dim : self.dim + 1],
            Y,
            noise_std=noise_std,
            learn_neg_flux=learn_neg_flux,
            jitter=jitter,
            n_restarts_optimizer=n_restarts_optimizer,
            reg_function=reg_function,
            kernel_variance=kernel_variance,
            lengthscale=lengthscale,
            noise_variance=noise_variance,
            optimize_hyperparameters=optimize_hyperparameters,
            max_cache_size=max_cache_size,
        )

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return GP q and concatenate dq/dgrad_T with dq/dT."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        q, dq_ds, dq_dT = self.model.evaluate(
            X[:, : self.dim],
            X[:, self.dim : self.dim + 1],
        )
        q = np.asarray(q, dtype=float).reshape(X.shape[0], self.dim)
        dq_ds = np.asarray(dq_ds, dtype=float).reshape(X.shape[0], self.dim, self.dim)
        dq_dT = np.asarray(dq_dT, dtype=float).reshape(X.shape[0], self.dim, 1)
        return q, np.concatenate([dq_ds, dq_dT], axis=2)

    def uncertainty(self, X: np.ndarray) -> float:
        """Return a scalar posterior-variance signal without oracle calls."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        _, _, _, var_q, var_dq_ds, var_dq_dT = self.model.evaluate(
            X[:, : self.dim],
            X[:, self.dim : self.dim + 1],
            return_variance=True,
        )
        var_q = np.asarray(var_q, dtype=float)
        var_dq_ds = np.asarray(var_dq_ds, dtype=float)
        var_dq_dT = np.asarray(var_dq_dT, dtype=float)
        return float(np.mean(var_q) + 0.01 * (np.mean(var_dq_ds) + np.mean(var_dq_dT)))


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
        self.uncertainty_source = "not_evaluated"
        self.out_of_bounds_query_count = 0

    def evaluate(self, grad_T: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the local learned flux law at one point or a batch of points."""
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
        """Sample/refit if needed, then return q, dq/dgrad_T, and dq/dT."""
        self.eval_count += 1
        state = np.asarray(np.r_[grad, T], dtype=float)
        if self._outside_bounds(state):
            self.out_of_bounds_query_count += 1

        if self.surrogate is None:
            # First query: build the first local cloud around this FEM state.
            self._sample_neighborhood(state, self.options.initial_points(self.state_dim))
            self._prune_cache_near(state)
            self._fit_surrogate()

        q, jac = self._surrogate_evaluate(state)
        uncertainty = self._uncertainty(state)

        # If the local fit is poor, add nearby oracle samples and refit.
        # The oracle still supplies only q values; derivatives come from the fit.
        refinements = 0
        while (
            uncertainty > self.options.mse_tolerance
            and refinements < self.options.max_refinements_per_eval
            and self.refinement_count < self.options.max_total_refinements
        ):
            self._sample_neighborhood(state, self.options.refill_points)
            self._prune_cache_near(state)
            self._fit_surrogate()
            q, jac = self._surrogate_evaluate(state)
            uncertainty = self._uncertainty(state)
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
        """Fit the selected local model to cached oracle samples."""
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
        elif self.method_key in {"bb3d_basegp", "bb_basegp", "basegp"}:
            self.surrogate = BaseGPVectorProvider(
                X,
                Y,
                self.dim,
                noise_std=float(self.model_options.get("noise_std", 0.0)),
                learn_neg_flux=bool(self.model_options.get("learn_neg_flux", True)),
                jitter=float(self.model_options.get("jitter", 1.0e-8)),
                n_restarts_optimizer=int(
                    self.model_options.get("n_restarts_optimizer", 0)
                ),
                reg_function=float(self.model_options.get("reg_function", 0.0)),
                kernel_variance=float(self.model_options.get("kernel_variance", 1.0)),
                lengthscale=self.model_options.get("lengthscale", 2.0),
                noise_variance=float(self.model_options.get("noise_variance", 1.0e-2)),
                optimize_hyperparameters=bool(
                    self.model_options.get("optimize_hyperparameters", False)
                ),
                max_cache_size=self.options.max_points(self.state_dim),
            )
        else:
            raise ValueError(f"unknown 3D blackbox method: {self.method_key}")
        self.surrogate_fit_count += 1

    def _surrogate_evaluate(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate in scaled coordinates and convert derivatives back to physical units."""
        scaled = self._to_scaled(state.reshape(1, self.state_dim))
        q, jac_scaled = self.surrogate.evaluate(scaled)
        jac_physical = jac_scaled[0] / self.scale[np.newaxis, :]
        return q[0], jac_physical

    def _validation_mse(self, state: np.ndarray) -> float:
        """Estimate local fit error using nearby q-only oracle checks."""
        points = self._random_ball_samples(state, max(4, self.state_dim))
        errors = []
        for point in points:
            sample_point = self._clip_sample_state(point)
            q_true = self.cache.evaluate(sample_point)
            q_hat, _ = self._surrogate_evaluate(sample_point)
            denom = np.maximum(1.0, np.abs(q_true))
            errors.append(np.mean(((q_hat - q_true) / denom) ** 2))
        return float(np.mean(errors)) if errors else 0.0

    def _uncertainty(self, state: np.ndarray) -> float:
        """Use GP posterior variance when available; otherwise validate by oracle calls."""
        if isinstance(self.surrogate, BaseGPVectorProvider):
            scaled = self._to_scaled(state.reshape(1, self.state_dim))
            self.uncertainty_source = "basegp_posterior_variance"
            return self.surrogate.uncertainty(scaled)
        self.uncertainty_source = "local_oracle_validation_mse"
        return self._validation_mse(state)

    def _sample_neighborhood(self, state: np.ndarray, n_points: int) -> None:
        """Add structured/random local samples around the current FEM state."""
        samples = [state.copy()] if self.options.include_center else []
        if self.options.design in {"axis", "hybrid"}:
            samples.extend(self._axis_samples(state))
        if self.options.design in {"random", "hybrid"}:
            samples.extend(self._random_ball_samples(state, max(0, n_points - len(samples))))

        for sample in samples[:n_points]:
            self.cache.evaluate(self._clip_sample_state(sample))

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

    def _outside_bounds(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=float)
        return bool(np.any((state < self.bounds[:, 0]) | (state > self.bounds[:, 1])))

    def _clip_sample_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        clipped = np.clip(state, self.bounds[:, 0], self.bounds[:, 1])
        if not np.allclose(clipped, state):
            self.cache.clipped_sample_count += 1
        return clipped

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
                "bb_max_total_refinements": self.options.max_total_refinements,
                "bb_surrogate_fit_count": self.surrogate_fit_count,
                "bb_failed_refinements": self.failed_refinements,
                "bb_last_uncertainty": self.last_uncertainty,
                "bb_avg_uncertainty": (
                    self.uncertainty_sum / self.uncertainty_count
                    if self.uncertainty_count
                    else np.nan
                ),
                "bb_uncertainty_source": self.uncertainty_source,
                "bb_last_status": self.last_status,
                "bb_out_of_bounds_query_count": self.out_of_bounds_query_count,
                "bb_clipped_eval_count": self.out_of_bounds_query_count,
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
    """Return a comparison-ready flux-law dictionary.

    Normal comparison code calls ``model["flux"](grad_T, T, pt)``. The returned
    function has the Newton/FEM interface ``q, A, b`` even though the underlying
    black-box oracle itself returns only ``q``.
    """
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
        max_total_refinements=method_options.get(
            "max_total_refinements",
            AdaptiveBB3DOptions.max_total_refinements,
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
