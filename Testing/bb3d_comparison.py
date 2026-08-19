"""Comparison runner for 2D/3D black-box nonlinear diffusion flux providers.

This file answers one question:

    If we replace the exact flux law with a local learned provider, can the
    ND Newton/FEM solver still solve a small heat-diffusion problem?

The provider file builds a flux law with interface

    flux_law(grad_T, T, pt) -> q, A, b.

This comparison file puts that flux law inside ``Basic.newton_nd.NM`` and
records solver diagnostics. This mirrors the 1D comparison pattern:

1. Choose a boundary-value problem.
2. Solve it with the exact analytic oracle flux on the FEM mesh.
3. Solve it again with a learned black-box provider.
4. Compare the learned-provider FEM solution to the analytic-oracle FEM
   reference solution.

The boundary data below is deliberately non-affine. It gives the solver a
nontrivial problem without pretending to be a closed-form exact solution.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "Basic"))

from Basic.newton_nd import NM  # noqa: E402
from Data.bboracle3dDiffusion import ORACLE_CONFIGS  # noqa: E402
from Testing.bb3d_providers import build_provider  # noqa: E402


PHYSICS_TOL = 1.0e-10
DEFAULT_BOUNDARY_OFFSET = 1.0
DEFAULT_BOUNDARY_GRADIENT_3D = np.array([0.8, -0.4, 0.3], dtype=float)


def default_domain_bounds(dim: int):
    """Default physical FEM domain: [0, 14]^dim."""
    return [(0.0, 14.0)] * dim


class TimedFluxLaw:
    """Record timing and basic physical diagnostics for flux-law calls."""

    def __init__(self, flux_law):
        self.flux_law = flux_law
        self.calls = 0
        self.elapsed_s = 0.0
        self.q_dot_grad_values = []
        self.max_A_eig_values = []
        self.T_values = []
        self.grad_norm_values = []
        self.grad_component_values = []

    def __call__(self, grad_T, T, pt):
        # Newton/FEM calls this at quadrature points. We time only the flux-law
        # call, not assembly or linear solves.
        start = time.perf_counter()
        q, A, b = self.flux_law(grad_T, T, pt)
        self.elapsed_s += time.perf_counter() - start
        self.calls += 1
        grad = np.asarray(grad_T, dtype=float)
        self.T_values.append(float(T))
        self.grad_norm_values.append(float(np.linalg.norm(grad)))
        self.grad_component_values.append(grad.copy())

        # For heat flux q = -a grad(T), the quantity q dot grad(T) should be
        # nonpositive. Positive values are a sign of entropy/physics violation.
        self.q_dot_grad_values.append(float(np.dot(q, grad)))

        # The exact dq/dgrad_T is negative semidefinite for this model. We track
        # positive eigenvalues as a monotonicity diagnostic for learned fits.
        sym_A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
        self.max_A_eig_values.append(float(np.linalg.eigvalsh(sym_A).max()))
        return q, A, b

    def physical_correctness(self) -> dict:
        q_dot_grad = np.asarray(self.q_dot_grad_values, dtype=float)
        max_A_eig = np.asarray(self.max_A_eig_values, dtype=float)
        T_values = np.asarray(self.T_values, dtype=float)
        grad_norms = np.asarray(self.grad_norm_values, dtype=float)
        grad_components = np.asarray(self.grad_component_values, dtype=float)
        if q_dot_grad.size == 0:
            return {
                "physics_eval_source": "newton_flux_calls",
                "physics_tol": PHYSICS_TOL,
                "heat_entropy_violation_%": np.nan,
                "worst_heat_entropy_violation": np.nan,
                "A_monotonicity_violation_%": np.nan,
                "worst_A_monotonicity_violation": np.nan,
                "visited_T_min": np.nan,
                "visited_T_max": np.nan,
                "visited_grad_norm_min": np.nan,
                "visited_grad_norm_max": np.nan,
            }
        diagnostics = {
            "physics_eval_source": "newton_flux_calls",
            "physics_tol": PHYSICS_TOL,
            "heat_entropy_violation_%": 100.0 * np.mean(q_dot_grad > PHYSICS_TOL),
            "worst_heat_entropy_violation": np.nanmax(np.maximum(0.0, q_dot_grad)),
            "A_monotonicity_violation_%": 100.0 * np.mean(max_A_eig > PHYSICS_TOL),
            "worst_A_monotonicity_violation": np.nanmax(np.maximum(0.0, max_A_eig)),
            "visited_T_min": float(np.nanmin(T_values)),
            "visited_T_max": float(np.nanmax(T_values)),
            "visited_grad_norm_min": float(np.nanmin(grad_norms)),
            "visited_grad_norm_max": float(np.nanmax(grad_norms)),
        }
        for axis in range(grad_components.shape[1]):
            diagnostics[f"visited_grad_{axis}_min"] = float(np.nanmin(grad_components[:, axis]))
            diagnostics[f"visited_grad_{axis}_max"] = float(np.nanmax(grad_components[:, axis]))
        return diagnostics


def make_spatial_source(domain_bounds):
    """Create a smooth nonconstant source term independent of T."""
    bounds = np.asarray(domain_bounds, dtype=float)
    lower = bounds[:, 0]
    length = bounds[:, 1] - bounds[:, 0]

    def spatial_source(T, pt):
        x = (np.asarray(pt, dtype=float) - lower) / length
        value = 1.0
        value += 0.35 * np.sin(np.pi * x[0])
        if x.size >= 2:
            value += 0.20 * np.cos(2.0 * np.pi * x[1])
        if x.size == 3:
            value += 0.15 * np.sin(np.pi * (x[0] + x[2]))
        return float(value)

    return spatial_source


def dsource_dT(T, pt):
    """Derivative of the source term; the current source is independent of T."""
    return 0.0


def make_curved_boundary_temperature(domain_bounds, offset: float, gradient):
    """Create a smooth non-affine Dirichlet boundary temperature."""
    bounds = np.asarray(domain_bounds, dtype=float)
    lower = bounds[:, 0]
    length = bounds[:, 1] - bounds[:, 0]
    gradient = np.asarray(gradient, dtype=float)

    def boundary_temperature(pt):
        pt_array = np.asarray(pt, dtype=float)
        x = (pt_array - lower[: pt_array.size]) / length[: pt_array.size]
        value = offset + np.dot(gradient[: pt_array.size], pt_array)
        value += 0.75 * np.sin(np.pi * x[0])
        if pt_array.size >= 2:
            value += 0.45 * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
        if pt_array.size == 3:
            value += 0.35 * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])
        return float(value)

    return boundary_temperature


def make_affine_boundary_temperature(offset: float, gradient):
    """Create T_boundary(x) = offset + gradient dot x."""
    gradient = np.asarray(gradient, dtype=float)

    def boundary_temperature(pt):
        pt_array = np.asarray(pt, dtype=float)
        return float(offset + np.dot(gradient[: pt_array.size], pt_array))

    return boundary_temperature


def boundary_conditions_for_dim(dim: int, boundary_temperature):
    """Dirichlet temperature on every outer side."""
    sides = ["xmin", "xmax"]
    if dim >= 2:
        sides.extend(["ymin", "ymax"])
    if dim == 3:
        sides.extend(["zmin", "zmax"])
    return {side: ("dirichlet", boundary_temperature) for side in sides}


def relative_error(solution, reference):
    """Euclidean relative error."""
    return float(np.linalg.norm(solution - reference) / np.linalg.norm(reference))


def make_experiment_dir(output_dir, exp_name):
    """Create a fresh result directory without overwriting old runs."""
    safe_exp_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in exp_name)
    exp_dir = output_dir / safe_exp_name
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
        return exp_dir
    counter = 1
    while True:
        numbered = output_dir / f"{safe_exp_name}_{counter}"
        if not numbered.exists():
            numbered.mkdir(parents=True)
            return numbered
        counter += 1


def default_boundary_points(dim: int, n_per_axis: int = 5, domain_bounds=None):
    """Build a uniform tensor-product grid on a rectangular box."""
    if domain_bounds is None:
        domain_bounds = default_domain_bounds(dim)
    if len(domain_bounds) != dim:
        raise ValueError(f"domain_bounds must have length dim={dim}")

    boundary_points = []
    for lower, upper in domain_bounds:
        if upper <= lower:
            raise ValueError("each domain upper bound must exceed lower bound")
        boundary_points.append(np.linspace(float(lower), float(upper), n_per_axis))
    return boundary_points


def interpolate_reference_to_points(reference_points, reference_values, query_points):
    """Interpolate a tensor-grid reference solution onto query points."""
    shape = tuple(len(axis) for axis in reference_points)
    interpolator = RegularGridInterpolator(
        reference_points,
        np.asarray(reference_values, dtype=float).reshape(shape),
        bounds_error=False,
        fill_value=None,
    )
    return interpolator(query_points)


def mesh_points(boundary_points):
    """Return all tensor-grid node coordinates in the same flat order as newton_nd."""
    meshes = np.meshgrid(*boundary_points, indexing="ij")
    return np.column_stack([axis.reshape(-1) for axis in meshes])


def newton(
    model,
    reference_model,
    boundary_points,
    boundary_conditions,
    source_fn,
    reference_boundary_points=None,
):
    """Run one FEM/Newton solve and return one CSV-ready result row."""
    timed_flux_law = TimedFluxLaw(model["flux"])
    t0 = time.perf_counter()

    try:
        # First solve with the exact analytic flux. This is the apples-to-apples
        # FEM reference on the same mesh.
        U_ref, _, _ = NM(
            boundary_points,
            reference_model["flux"],
            source_fn,
            dsource_dT,
            boundary_conditions,
            tol=1.0e-11,
            maxiter=30,
            verbose=False,
        )

        # Then solve with the experiment model. This model may be analytic or a
        # black-box provider that samples the oracle and fits local surrogates.
        U, residual_history, num_iterations = NM(
            boundary_points,
            timed_flux_law,
            source_fn,
            dsource_dT,
            boundary_conditions,
            tol=1.0e-8,
            maxiter=30,
            verbose=False,
        )
        FEM_sol_err = relative_error(U, U_ref)
        if reference_boundary_points is None:
            true_sol_err = np.nan
        else:
            U_fine_ref, _, _ = NM(
                reference_boundary_points,
                reference_model["flux"],
                source_fn,
                dsource_dT,
                boundary_conditions,
                tol=1.0e-10,
                maxiter=40,
                verbose=False,
            )
            U_fine_on_coarse = interpolate_reference_to_points(
                reference_boundary_points,
                U_fine_ref,
                mesh_points(boundary_points),
            )
            true_sol_err = relative_error(U, U_fine_on_coarse)
        status = "ok"
    except Exception as exc:
        residual_history = [np.nan]
        num_iterations = np.nan
        FEM_sol_err = np.nan
        true_sol_err = np.nan
        status = f"failed: {type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    provider = model.get("provider")

    # Diagnostics are read-only counters/state from the provider. Reading them
    # does not call the oracle or refit a surrogate.
    diagnostics = provider.diagnostics() if provider is not None else {}
    row = {
        "status": status,
        "build_s": model["build_s"],
        "newton_steps": num_iterations,
        "flux_calls": timed_flux_law.calls,
        "final_residual": residual_history[-1],
        "FEM_sol_err": FEM_sol_err,
        "true_sol_err": true_sol_err,
        "solve_total_s": elapsed,
        "flux_eval_s": timed_flux_law.elapsed_s,
        "nonflux_s": elapsed - timed_flux_law.elapsed_s,
        "avg_flux_eval_us": (
            1.0e6 * timed_flux_law.elapsed_s / timed_flux_law.calls
            if timed_flux_law.calls
            else np.nan
        ),
    }
    row.update(timed_flux_law.physical_correctness())
    row.update(diagnostics)
    oracle_calls = row.get("oracle_calls", 0)
    row["flux_calls_per_oracle_call"] = row["flux_calls"] / oracle_calls if oracle_calls else np.nan
    return row


def comparison(
    exp_name,
    methods,
    oracle_configs,
    *,
    dim: int = 3,
    n_per_axis: int = 16,
    reference_n_per_axis: int | None = None,
    domain_bounds=None,
    boundary_offset: float = DEFAULT_BOUNDARY_OFFSET,
    boundary_gradient=None,
    noisy: bool = True,
    seed: int = 0,
    provider_options: dict | None = None,
):
    """Run a 2D/3D FEM comparison and save one CSV per oracle config.

    Parameters
    ----------
    exp_name:
        Name of the result folder under ``Results``.
    methods:
        Method strings such as ``"analytic"``, ``"bb3d_basegp+lengthscale=1.0"``,
        ``"bb3d_monotonegp"``, or ``"bb3d_rff_basegp"``.
    oracle_configs:
        Names from ``ORACLE_CONFIGS``.
    dim:
        Either 2 or 3.
    n_per_axis:
        Number of FEM nodes per coordinate direction.
    reference_n_per_axis:
        Optional finer analytic-oracle mesh for a stricter reference error.
    domain_bounds:
        Optional list of ``(lower, upper)`` bounds, one per coordinate direction.
        Defaults to ``[0, 14]^dim``.
    boundary_offset, boundary_gradient:
        Parameters for the linear part of the curved Dirichlet boundary data.
    noisy:
        Whether black-box providers see noisy oracle flux values.

    Returns
    -------
    list[pathlib.Path]
        CSV files written under ``Results/<exp_name>/``.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be either 2 or 3, got {dim}.")

    output_dir = ROOT / "Results"
    exp_dir = make_experiment_dir(output_dir, exp_name)
    print(f"\nSaving 3D blackbox results to: {exp_dir}")
    result_paths = []

    if boundary_gradient is None:
        boundary_gradient = DEFAULT_BOUNDARY_GRADIENT_3D[:dim]
    effective_domain_bounds = domain_bounds if domain_bounds is not None else default_domain_bounds(dim)
    boundary_temperature = make_curved_boundary_temperature(
        effective_domain_bounds,
        boundary_offset,
        boundary_gradient,
    )
    source_fn = make_spatial_source(effective_domain_bounds)

    boundary_points = default_boundary_points(
        dim,
        n_per_axis=n_per_axis,
        domain_bounds=effective_domain_bounds,
    )
    mesh_spacing = float(np.median(np.diff(boundary_points[0])))
    reference_boundary_points = (
        default_boundary_points(
            dim,
            n_per_axis=reference_n_per_axis,
            domain_bounds=effective_domain_bounds,
        )
        if reference_n_per_axis is not None
        else None
    )

    # The current comparison problem is fully Dirichlet. The side-based format
    # matches the current API of Basic.newton_nd.NM.
    boundary_conditions = boundary_conditions_for_dim(dim, boundary_temperature)

    for oracle_config in oracle_configs:
        if oracle_config not in ORACLE_CONFIGS:
            raise ValueError(f"unknown oracle config: {oracle_config}")
        params = ORACLE_CONFIGS[oracle_config]
        dataset_results = []

        # The analytic model is used only for reference error. It is not passed
        # into the black-box provider as derivative training data.
        reference_model = build_provider(
            "analytic",
            oracle_config,
            dim=dim,
            mesh_spacing=mesh_spacing,
            noisy=False,
            seed=seed,
        )

        print(f"\n=== {dim}D blackbox diffusion oracle: {oracle_config} ===")
        print("k0={k_0}, alpha={alpha}, beta={beta}, sigma={sigma}".format(**params))

        for method in methods:
            print(f"\n--- {method} ---")
            try:
                print("Building...")
                # Build the flux law. For black-box methods, this is where the
                # provider object is created, but it will sample lazily when
                # Newton first asks for a flux value.
                model = build_provider(
                    method,
                    oracle_config,
                    dim=dim,
                    mesh_spacing=mesh_spacing,
                    noisy=noisy,
                    seed=seed,
                    provider_options=provider_options,
                )
                row = {
                    "experiment": exp_name,
                    "oracle_config": oracle_config,
                    "method": model["method"],
                    "dim": dim,
                    "n_per_axis": n_per_axis,
                    "reference_n_per_axis": reference_n_per_axis,
                    "domain_bounds": str(effective_domain_bounds),
                    "boundary_offset": boundary_offset,
                    "boundary_gradient": str(np.asarray(boundary_gradient, dtype=float)),
                    "noisy_oracle": noisy,
                    "seed": seed,
                }
                print("Starting FEM/NM...")
                row.update(
                    newton(
                        model,
                        reference_model,
                        boundary_points,
                        boundary_conditions,
                        source_fn,
                        reference_boundary_points=reference_boundary_points,
                    )
                )
                print("done")
            except Exception as exc:
                print(":(")
                row = {
                    "experiment": exp_name,
                    "oracle_config": oracle_config,
                    "method": method,
                    "dim": dim,
                    "status": f"failed: {type(exc).__name__}",
                    "error": str(exc),
                }
            dataset_results.append(row)

        result_df = pd.DataFrame(dataset_results)
        result_path = exp_dir / f"{oracle_config}_{dim}d.csv"
        result_df.to_csv(result_path, index=False)
        result_paths.append(result_path)
        print(f"\nSaved CSV results: {result_path}")

    return result_paths


if __name__ == "__main__":
    comparison(
        "bb3d_smoke",
        methods=[
            "bb3d_basegp+max_refinements_per_eval=0",
            "bb3d_monotonegp+max_refinements_per_eval=0",
            "bb3d_rff_basegp+max_refinements_per_eval=0",
        ],
        oracle_configs=["nonlinear_no_noise"],
        dim=3,
        noisy=False,
        seed=0,
    )
