import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Error columns:
# - FEM_sol_err compares the experiment solve to the coarse analytic-flux FEM solve.
# - true_sol_err compares to a non-surrogate reference on the same coarse nodes.
#   Linear configs use the closed-form solution; nonlinear configs use a much
#   finer analytic-flux FEM solve sampled back onto the coarse mesh.

# Allow this script to be run directly while still importing repo modules.
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Basic.newton_1d_fire_documentation import NM  # noqa: E402
from Data.BlackBoxOracle.blackboxoracle import ORACLE_CONFIGS  # noqa: E402
from Testing.BlackBoxTesting.bb_moving_local_providers import (  # noqa: E402
    build_provider as build_moving_provider,
)
from Testing.BlackBoxTesting.bb_providers_stencil import build_provider  # noqa: E402
from Testing.BlackBoxTesting.bb_providers_tolerance import (  # noqa: E402
    build_provider as build_tolerance_provider,
)

PHYSICS_TOL = 1.0e-10
FINE_TRUE_NODES = 201


def build_test_provider(
    method,
    oracle_config,
    *,
    x_mesh=None,
    noisy=True,
    seed=0,
    provider_options=None,
):
    method_text = str(method)
    if method_text.lower().startswith("moving_"):
        return build_moving_provider(
            method_text[len("moving_") :],
            oracle_config,
            x_mesh=x_mesh,
            noisy=noisy,
            seed=seed,
        )
    if method_text.lower().startswith("tolerance_"):
        return build_tolerance_provider(
            method_text[len("tolerance_") :],
            oracle_config,
            x_mesh=x_mesh,
            noisy=noisy,
            seed=seed,
            provider_options=provider_options,
        )
    return build_provider(
        method,
        oracle_config,
        x_mesh=x_mesh,
        noisy=noisy,
        seed=seed,
    )


class TimedFluxLaw:
    """Wrap a flux law so Newton timing/call-count diagnostics are recorded."""

    def __init__(self, flux_law):
        self.flux_law = flux_law
        self.calls = 0
        self.elapsed_s = 0.0
        self.entropy_values = []
        self.monotonicity_values = []

    def __call__(self, s, T, xg):
        start = time.perf_counter()
        result = self.flux_law(s, T, xg)
        self.elapsed_s += time.perf_counter() - start
        self.calls += 1
        q, dq_ds, _ = result
        self.entropy_values.append(float(q) * float(s))
        self.monotonicity_values.append(float(dq_ds))
        return result

    def physical_correctness(self):
        entropy = np.asarray(self.entropy_values, dtype=float)
        monotonicity = np.asarray(self.monotonicity_values, dtype=float)
        if entropy.size == 0:
            return {
                "physics_eval_source": "newton_flux_calls",
                "physics_tol": PHYSICS_TOL,
                "entropy_violation_%": np.nan,
                "worst_entropy_violation": np.nan,
                "monotonicity_violation_%": np.nan,
                "worst_monotonicity_violation": np.nan,
            }
        return {
            "physics_eval_source": "newton_flux_calls",
            "physics_tol": PHYSICS_TOL,
            "entropy_violation_%": 100.0 * np.mean(entropy > PHYSICS_TOL),
            "worst_entropy_violation": np.nanmax(np.maximum(0.0, entropy)),
            "monotonicity_violation_%": 100.0 * np.mean(monotonicity > PHYSICS_TOL),
            "worst_monotonicity_violation": np.nanmax(np.maximum(0.0, monotonicity)),
        }


def source(T, xg):
    # Same constant source term used in the tabular comparison.
    return 1.0


def is_csv_oracle_config(oracle_config):
    return Path(str(oracle_config)).suffix.lower() == ".csv"


def oracle_result_name(oracle_config):
    if is_csv_oracle_config(oracle_config):
        return Path(str(oracle_config)).stem
    return str(oracle_config)


def relative_error(solution, reference):
    return float(np.linalg.norm(solution - reference) / np.linalg.norm(reference))


def linear_true_solution(x_mesh, oracle_config):
    params = ORACLE_CONFIGS[oracle_config]
    if params["alpha"] != 0.0 or params["beta"] != 0.0:
        return None

    k0 = params["k_0"]
    return 1.5 * x_mesh + x_mesh * (1.0 - x_mesh) / (2.0 * k0)


def true_solution_on_mesh(x_mesh, oracle_config, reference_model):
    linear_solution = linear_true_solution(x_mesh, oracle_config)
    if linear_solution is not None:
        return linear_solution

    x_fine = np.linspace(float(x_mesh[0]), float(x_mesh[-1]), FINE_TRUE_NODES)
    U_fine, _, _ = NM(
        x_fine,
        reference_model["flux"],
        source,
        T_dirichlet_left=0.0,
        T_dirichlet_right=1.5,
        tol=1e-10,
        maxiter=60,
        verbose=False,
    )
    return np.interp(x_mesh, x_fine, U_fine)


def newton(model, reference_model, x_mesh, true_solution=None, pressure=False):
    # This function is the actual experiment. The adaptive blackbox provider is
    # only called through timed_flux_law inside NM below.
    timed_flux_law = TimedFluxLaw(model["flux"])
    t0 = time.perf_counter()

    if pressure:
        reference_model = None

    # Reference solve uses the analytic flux for the same oracle configuration.
    # This does not mutate the blackbox provider.
    if reference_model is not None:
        U_ref, _, _ = NM(
            x_mesh,
            reference_model["flux"],
            source,
            T_dirichlet_left=0.0,
            T_dirichlet_right=1.5,
            tol=1e-10,
            maxiter=40,
            verbose=False,
        )
    else:
        U_ref = None

    if pressure:
        T_dirichlet_left = 3000.0
        T_dirichlet_right = 0.0
        nsource = lambda T, xg: 0.1
    else:
        T_dirichlet_left = 0.0 
        T_dirichlet_right = 1.5
        nsource = source

    try:
        # Newton calls model["flux"], which may sample the oracle and fit local
        # surrogates. No other metric calls it.
        U, residual_history, num_iterations = NM(
            x_mesh,
            timed_flux_law,
            nsource,
            T_dirichlet_left=T_dirichlet_left,
            T_dirichlet_right=T_dirichlet_right,
            tol=1e-8,
            maxiter=40,
            verbose=False,
        )
        FEM_sol_err = relative_error(U, U_ref) if U_ref is not None else np.nan
        true_sol_err = (
            relative_error(U, true_solution)
            if true_solution is not None
            else np.nan
        )
        status = "ok"
    except Exception as exc:
        residual_history = [np.nan]
        num_iterations = np.nan
        FEM_sol_err = np.nan
        true_sol_err = np.nan
        status = f"failed: {type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    provider = model.get("provider")

    # Diagnostics are read-only counters/state. This does not run new flux calls.
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
    row["flux_calls_per_oracle_call"] = (
        row["flux_calls"] / oracle_calls
        if oracle_calls
        else np.nan
    )
    return row

def make_experiment_dir(output_dir, exp_name):
    # Avoid overwriting previous runs by appending _1, _2, ... if needed.
    safe_exp_name = "".join(
        c if c.isalnum() or c in {"-", "_"} else "_"
        for c in exp_name
    )

    exp_dir = output_dir / safe_exp_name
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
        return exp_dir

    counter = 1
    while True:
        numbered_exp_dir = output_dir / f"{safe_exp_name}_{counter}"
        if not numbered_exp_dir.exists():
            numbered_exp_dir.mkdir(parents=True)
            return numbered_exp_dir
        counter += 1


def comparison(exp_name, methods, oracle_configs, noisy=True, seed=0, pressure=False):
    # Main blackbox comparison runner. It intentionally does NOT run a separate
    # 500-point clean accuracy pass, because that would call another adaptive
    # provider and create misleading results.
    output_dir = ROOT / "Results"
    exp_dir = make_experiment_dir(output_dir, exp_name)
    print(f"\nSaving blackbox results to: {exp_dir}")
    result_paths = []

    x_mesh = np.linspace(0.0, 1.0, 21)
    provider_options = None
    if pressure:
        provider_options = {
            "s_bounds": (-5000.0, 0.0),
            "T_bounds": (1.0, 1.0),
        }

    for oracle_config in oracle_configs:
        is_csv = is_csv_oracle_config(oracle_config)
        oracle_name = oracle_result_name(oracle_config)
        if is_csv and not Path(str(oracle_config)).is_file():
            raise FileNotFoundError(f"tabular oracle CSV not found: {oracle_config}")
        if not is_csv and oracle_config not in ORACLE_CONFIGS:
            raise ValueError(f"unknown oracle config: {oracle_config}")

        params = None if is_csv else ORACLE_CONFIGS[oracle_config]
        dataset_results = []

        # Analytic reference is used only for reference solution error.
        if not is_csv:
            reference_model = build_provider(
                "Analytic",
                oracle_config,
                x_mesh=x_mesh,
                noisy=False,
            )
        else:
            reference_model = None
        true_solution = (
            true_solution_on_mesh(x_mesh, oracle_config, reference_model)
            if reference_model is not None and not pressure
            else None
        )

        print(f"\n=== Blackbox oracle: {oracle_config} ===")
        if params is not None:
            print(
                "k0={k_0}, alpha={alpha}, beta={beta}, sigma={sigma}".format(**params)
            )
        else:
            print(f"tabular CSV: {Path(str(oracle_config)).name}")

        for method in methods:
            print(f"\n--- {method} ---")
            try:
                print("Building...")
                model = build_test_provider(
                    method,
                    oracle_config,
                    x_mesh=x_mesh,
                    noisy=noisy,
                    seed=seed,
                    provider_options=provider_options,
                )
                row = {
                    "experiment": exp_name,
                    "oracle_config": oracle_name,
                    "method": model["method"],
                    "noisy_oracle": noisy,
                    "seed": seed,
                }
                print("Starting FEM/NM...")

                # This is the only place in the comparison loop where the
                # experiment provider is evaluated.
                row.update(
                    newton(
                        model,
                        reference_model,
                        x_mesh,
                        true_solution,
                        pressure=pressure,
                    )
                )
                print("done")
            except Exception as exc:
                print(":(")
                row = {
                    "experiment": exp_name,
                    "oracle_config": oracle_name,
                    "method": method,
                    "status": f"failed: {type(exc).__name__}",
                    "error": str(exc),
                }
            dataset_results.append(row)

        result_df = pd.DataFrame(dataset_results)
        result_path = exp_dir / f"{oracle_name}.csv"
        result_df.to_csv(result_path, index=False)
        result_paths.append(result_path)
        print(f"\nSaved CSV results: {result_path}")

    return result_paths


if __name__ == "__main__":
    # exp_name = "tolerance_smoke"
    # methods = [
    #     "tolerance_bb_rbf",
    #     "tolerance_bb_matern52_krr",
    #     "tolerance_bb_rff",
    #     # "bb_kissgp",
    #     # "bb_monotonegp",
    # ]
    # oracle_configs = [
    #     "nonlinear_high_noise",
    # ]

    # exp_name = "oracle_test"
    # methods = ['tolerance_bb_rbf']
    # oracle_configs = [ROOT / 'Data/NoisyDeterministicOracles/datasets/nonlinear_high_noise.csv']
    # comparison(exp_name, methods, oracle_configs, noisy=True, seed=0)

    exp_name = "pressure_finalver"
    methods = ['tolerance_bb_rbf', 'tolerance_bb_basegp', 'tolerance_bb_poly']
    oracle_configs = [ROOT / "Data/PressureDataset/pressure_filtered_5.csv"]
    comparison(exp_name, methods, oracle_configs, noisy=True, pressure=True)
