import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_test_provider(method, oracle_config, *, x_mesh=None, noisy=True, seed=0):
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

    def __call__(self, s, T, xg):
        start = time.perf_counter()
        result = self.flux_law(s, T, xg)
        self.elapsed_s += time.perf_counter() - start
        self.calls += 1
        return result


def source(T, xg):
    # Same constant source term used in the tabular comparison.
    return 1.0


def is_csv_oracle_config(oracle_config):
    return Path(str(oracle_config)).suffix.lower() == ".csv"


def oracle_result_name(oracle_config):
    if is_csv_oracle_config(oracle_config):
        return Path(str(oracle_config)).stem
    return str(oracle_config)


def newton(model, reference_model, x_mesh):
    # This function is the actual experiment. The adaptive blackbox provider is
    # only called through timed_flux_law inside NM below.
    timed_flux_law = TimedFluxLaw(model["flux"])
    t0 = time.perf_counter()

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

    try:
        # Newton calls model["flux"], which may sample the oracle and fit local
        # surrogates. No other metric calls it.
        U, residual_history, num_iterations = NM(
            x_mesh,
            timed_flux_law,
            source,
            T_dirichlet_left=0.0,
            T_dirichlet_right=1.5,
            tol=1e-8,
            maxiter=40,
            verbose=False,
        )
        rel_err = (
            np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
            if U_ref is not None
            else np.nan
        )
        status = "ok"
    except Exception as exc:
        residual_history = [np.nan]
        num_iterations = np.nan
        rel_err = np.nan
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
        "rel_solution_err": rel_err,
        "solve_total_s": elapsed,
        "flux_eval_s": timed_flux_law.elapsed_s,
        "nonflux_s": elapsed - timed_flux_law.elapsed_s,
        "avg_flux_eval_us": (
            1.0e6 * timed_flux_law.elapsed_s / timed_flux_law.calls
            if timed_flux_law.calls
            else np.nan
        ),
    }
    row.update(diagnostics)
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


def comparison(exp_name, methods, oracle_configs, noisy=True, seed=0):
    # Main blackbox comparison runner. It intentionally does NOT run a separate
    # 500-point clean accuracy pass, because that would call another adaptive
    # provider and create misleading results.
    output_dir = ROOT / "Results"
    exp_dir = make_experiment_dir(output_dir, exp_name)
    print(f"\nSaving blackbox results to: {exp_dir}")
    result_paths = []

    x_mesh = np.linspace(0.0, 1.0, 21)

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
                row.update(newton(model, reference_model, x_mesh))
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

    exp_name = "dataset_test"
    methods = ['tolerance_bb_rbf']
    oracle_configs = [ROOT / 'Data/NoisyDeterministicOracles/datasets/nonlinear_high_noise.csv']
    comparison(exp_name, methods, oracle_configs, noisy=True, seed=0)
