"""Comparison runner for 2D/3D black-box nonlinear diffusion flux providers."""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "Basic"))

from Basic.newton_nd import NM, global_id, grid  # noqa: E402
from Data.BlackBoxOracle.bboracle3dDiffusion import ORACLE_CONFIGS  # noqa: E402
from Testing.BlackBoxTesting3D.bb3d_providers import build_provider  # noqa: E402


PHYSICS_TOL = 1.0e-10


class TimedFluxLaw:
    """Record timing and basic physical diagnostics for flux-law calls."""

    def __init__(self, flux_law):
        self.flux_law = flux_law
        self.calls = 0
        self.elapsed_s = 0.0
        self.q_dot_grad_values = []
        self.max_A_eig_values = []

    def __call__(self, grad_T, T, pt):
        start = time.perf_counter()
        q, A, b = self.flux_law(grad_T, T, pt)
        self.elapsed_s += time.perf_counter() - start
        self.calls += 1
        grad = np.asarray(grad_T, dtype=float)
        self.q_dot_grad_values.append(float(np.dot(q, grad)))
        sym_A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
        self.max_A_eig_values.append(float(np.linalg.eigvalsh(sym_A).max()))
        return q, A, b

    def physical_correctness(self) -> dict:
        q_dot_grad = np.asarray(self.q_dot_grad_values, dtype=float)
        max_A_eig = np.asarray(self.max_A_eig_values, dtype=float)
        if q_dot_grad.size == 0:
            return {
                "physics_eval_source": "newton_flux_calls",
                "physics_tol": PHYSICS_TOL,
                "heat_entropy_violation_%": np.nan,
                "worst_heat_entropy_violation": np.nan,
                "A_monotonicity_violation_%": np.nan,
                "worst_A_monotonicity_violation": np.nan,
            }
        return {
            "physics_eval_source": "newton_flux_calls",
            "physics_tol": PHYSICS_TOL,
            "heat_entropy_violation_%": 100.0 * np.mean(q_dot_grad > PHYSICS_TOL),
            "worst_heat_entropy_violation": np.nanmax(np.maximum(0.0, q_dot_grad)),
            "A_monotonicity_violation_%": 100.0 * np.mean(max_A_eig > PHYSICS_TOL),
            "worst_A_monotonicity_violation": np.nanmax(np.maximum(0.0, max_A_eig)),
        }


def zero_source(T, pt):
    return 0.0


def zero_dsource_dT(T, pt):
    return 0.0


def affine_temperature(pt):
    pt = np.asarray(pt, dtype=float)
    coeff = np.array([0.4, -0.2, 0.15], dtype=float)[: pt.size]
    return float(1.0 + np.dot(coeff, pt))


def boundary_dirichlet_nodes(boundary_points):
    grid_vars = grid(boundary_points)
    dirichlet_nodes = {}
    for idx in grid_vars["all_node_indices"]:
        if any(idx[d] == 0 or idx[d] == grid_vars["n_nodes_per_axis"][d] - 1 for d in range(grid_vars["dim"])):
            pt = np.array([grid_vars["coords"][d][idx[d]] for d in range(grid_vars["dim"])])
            dirichlet_nodes[idx] = affine_temperature(pt)
    return dirichlet_nodes


def exact_solution_on_grid(boundary_points):
    grid_vars = grid(boundary_points)
    values = np.zeros(grid_vars["n_nodes"])
    for idx in grid_vars["all_node_indices"]:
        pt = np.array([grid_vars["coords"][d][idx[d]] for d in range(grid_vars["dim"])])
        values[global_id(idx, grid_vars)] = affine_temperature(pt)
    return values


def relative_error(solution, reference):
    return float(np.linalg.norm(solution - reference) / np.linalg.norm(reference))


def make_experiment_dir(output_dir, exp_name):
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


def default_boundary_points(dim: int, n_per_axis: int = 5):
    coords = np.linspace(0.0, 1.0, n_per_axis)
    return [coords.copy() for _ in range(dim)]


def newton(model, reference_model, boundary_points, dirichlet_nodes, exact_solution):
    timed_flux_law = TimedFluxLaw(model["flux"])
    t0 = time.perf_counter()

    try:
        U_ref, _, _ = NM(
            boundary_points,
            reference_model["flux"],
            zero_source,
            zero_dsource_dT,
            dirichlet_nodes,
            tol=1.0e-11,
            maxiter=30,
            verbose=False,
        )
        U, residual_history, num_iterations = NM(
            boundary_points,
            timed_flux_law,
            zero_source,
            zero_dsource_dT,
            dirichlet_nodes,
            tol=1.0e-8,
            maxiter=30,
            verbose=False,
        )
        FEM_sol_err = relative_error(U, U_ref)
        true_sol_err = relative_error(U, exact_solution)
        status = "ok"
    except Exception as exc:
        residual_history = [np.nan]
        num_iterations = np.nan
        FEM_sol_err = np.nan
        true_sol_err = np.nan
        status = f"failed: {type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    provider = model.get("provider")
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
    n_per_axis: int = 5,
    noisy: bool = True,
    seed: int = 0,
    provider_options: dict | None = None,
):
    """Run a small 2D/3D FEM comparison and save one CSV per oracle config."""
    if dim not in (2, 3):
        raise ValueError(f"dim must be either 2 or 3, got {dim}.")

    output_dir = ROOT / "Results"
    exp_dir = make_experiment_dir(output_dir, exp_name)
    print(f"\nSaving 3D blackbox results to: {exp_dir}")
    result_paths = []

    boundary_points = default_boundary_points(dim, n_per_axis=n_per_axis)
    mesh_spacing = float(np.median(np.diff(boundary_points[0])))
    dirichlet_nodes = boundary_dirichlet_nodes(boundary_points)
    exact_solution = exact_solution_on_grid(boundary_points)

    for oracle_config in oracle_configs:
        if oracle_config not in ORACLE_CONFIGS:
            raise ValueError(f"unknown oracle config: {oracle_config}")
        params = ORACLE_CONFIGS[oracle_config]
        dataset_results = []
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
                    "noisy_oracle": noisy,
                    "seed": seed,
                }
                print("Starting FEM/NM...")
                row.update(newton(model, reference_model, boundary_points, dirichlet_nodes, exact_solution))
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
            "bb3d_poly+degree=2+max_refinements_per_eval=0",
            "bb3d_rbf+gamma=1.0+max_refinements_per_eval=0",
        ],
        oracle_configs=["nonlinear_no_noise"],
        dim=3,
        n_per_axis=4,
        noisy=False,
        seed=0,
    )
