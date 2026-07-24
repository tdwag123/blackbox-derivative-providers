import sys
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Basic.newton_1d_fire_documentation import NM
from Methods.AnalyticReference import a_true, b_true, q_true
from Testing.Deprecated.data import CHOP_GRID_S, CHOP_GRID_T, MAX_ROWS_PER_CELL, grid_chop_dataframe
from Testing.TabularTesting.providers import build_provider

# =================== accuracy evaluation functions =========================================

CLEAN_EVAL_POINTS = 500
NOISY_TEST_POINTS = 500
PHYSICS_TOL = 1.0e-10

def evaluate_flux_law_on_points(flux_law, s_values, T_values):
    q_pred = np.zeros_like(s_values, dtype=float)
    a_pred = np.zeros_like(s_values, dtype=float)
    b_pred = np.zeros_like(s_values, dtype=float)

    for i, (s, T) in enumerate(zip(s_values, T_values)):
        try:
            q_pred[i], a_pred[i], b_pred[i] = flux_law(s, T, 0.0)
        except Exception:
            q_pred[i] = np.nan
            a_pred[i] = np.nan
            b_pred[i] = np.nan

    return q_pred, a_pred, b_pred

def make_clean_eval_set(df, n_points=CLEAN_EVAL_POINTS, seed=23):
    rng = np.random.default_rng(seed)
    s_values = rng.uniform(float(df["s"].min()), float(df["s"].max()), n_points)
    T_values = rng.uniform(float(df["T"].min()), float(df["T"].max()), n_points)
    return s_values, T_values

def rmse(predicted, reference):
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))

def noise_normalized_rmse(predicted, reference, noise_std):
    if np.any(noise_std <= 0.0):
        return np.nan
    return rmse(predicted / noise_std, reference / noise_std)

def finite_difference_dq_ds(flux_law, s_values, T_values, h, s_min, s_max):
    a_fd = np.zeros_like(s_values, dtype=float)

    for i, (s, T) in enumerate(zip(s_values, T_values)):
        s_left = max(s_min, s - h)
        s_right = min(s_max, s + h)

        if s_right == s_left:
            a_fd[i] = np.nan
            continue

        try:
            q_right, _, _ = flux_law(s_right, T, 0.0)
            q_left, _, _ = flux_law(s_left, T, 0.0)
        except Exception:
            a_fd[i] = np.nan
            continue

        a_fd[i] = (q_right - q_left) / (s_right - s_left)

    return a_fd

def accuracy_correctness(model, df, training_df):
    flux_law = model["flux"]
    h_s = model["h_s"]
    h_T = model["h_T"]

    sigma = float(df["sigma"].iloc[0])
    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])

    noisy_test = df[~df["_row_id"].isin(training_df["_row_id"])]
    noisy_test = noisy_test.sample(
        n=min(NOISY_TEST_POINTS, len(noisy_test)),
        random_state=12,
    )

    s_noisy = noisy_test["s"].to_numpy()
    T_noisy = noisy_test["T"].to_numpy()
    q_obs_noisy = noisy_test["q_noisy"].to_numpy()
    q_true_noisy = noisy_test["q_true"].to_numpy()
    noise_std_noisy = sigma * np.maximum(1.0, np.abs(q_true_noisy))

    s_clean, T_clean = make_clean_eval_set(df)

    q_clean = q_true(s_clean, T_clean, k0, alpha, beta)
    a_clean = a_true(s_clean, T_clean, k0, alpha, beta)
    b_clean = b_true(s_clean, T_clean, k0, alpha, beta)

    noise_std_clean = sigma * np.maximum(1.0, np.abs(q_clean))
    dq_ds_noise_floor = np.sqrt(
        np.mean((noise_std_clean / (np.sqrt(2.0) * h_s)) ** 2)
    )
    dq_dT_noise_floor = np.sqrt(
        np.mean((noise_std_clean / (np.sqrt(2.0) * h_T)) ** 2)
    )

    q_pred_clean, a_pred_clean, b_pred_clean = evaluate_flux_law_on_points(flux_law, s_clean, T_clean)
    q_pred_noisy, _, _ = evaluate_flux_law_on_points(flux_law, s_noisy, T_noisy,)

    if not np.all(np.isfinite(a_pred_clean)):
        a_pred_clean = finite_difference_dq_ds(flux_law, s_clean, T_clean, h_s, float(df["s"].min()),float(df["s"].max()),)

    noisy_q_rmse = rmse(q_pred_noisy, q_obs_noisy)
    clean_q_rmse = rmse(q_pred_clean, q_clean)
    clean_dq_ds_rmse = rmse(a_pred_clean, a_clean)
    clean_dq_dT_rmse = rmse(b_pred_clean, b_clean)
    noisy_q_noise_units = noise_normalized_rmse(q_pred_noisy, q_obs_noisy, noise_std_noisy,)

    entropy_values = np.maximum(0.0, q_pred_clean * s_clean)
    deriv_values = np.maximum(0.0, a_pred_clean)

    return {
        "test_obs_q_RMSE": noisy_q_rmse,
        "clean_q_RMSE": clean_q_rmse,
        "clean_dq_ds_RMSE": clean_dq_ds_rmse,
        "clean_dq_dT_RMSE": clean_dq_dT_rmse,
        "test_obs_q_RMSE/noise": noisy_q_noise_units,
        "clean_dq_ds_RMSE/noise": clean_dq_ds_rmse / dq_ds_noise_floor if dq_ds_noise_floor else np.nan,
        "clean_dq_dT_RMSE/noise": clean_dq_dT_rmse / dq_dT_noise_floor if dq_dT_noise_floor else np.nan,
        "entropy_violation_%": 100.0 * np.mean(q_pred_clean * s_clean > PHYSICS_TOL),
        "worst_entropy_violation": np.nanmax(entropy_values),
        "deriv_violation_%": 100.0 * np.mean(a_pred_clean > PHYSICS_TOL),
        "worst_deriv_violation": np.nanmax(deriv_values),
    }

# ==================================================================================

# ===================== newton iteration functions ==================================

class TimedFluxLaw:
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

def newton(model, reference_model):
    def source(T, xg):
        return 1.0

    x_mesh = np.linspace(0.0, 1.0, 21)
    TL = 0.0
    TR = 1.5

    U_ref, _, _ = NM(x_mesh,
        reference_model["flux"],
        source,
        T_dirichlet_left=TL,
        T_dirichlet_right=TR,
        tol=1e-10,
        maxiter=40,
        verbose=False,
    )

    timed_flux_law = TimedFluxLaw(model["flux"])
    t0 = time.perf_counter()
    try:
        U, residual_history, num_iterations = NM(
            x_mesh,
            timed_flux_law,
            source,
            T_dirichlet_left=TL,
            T_dirichlet_right=TR,
            tol=1e-8,
            maxiter=40,
            verbose=False,
        )

        err = np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
        status = "ok"

    except Exception as exc:
        residual_history = [np.nan]
        num_iterations = np.nan
        err = np.nan
        status = f"failed: {type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0

    return {
        "status": status,
        "build_s": model["build_s"],
        "newton_steps": num_iterations,
        "flux_calls": timed_flux_law.calls,
        "final_residual": residual_history[-1],
        "rel_solution_err": err,
        "solve_total_s": elapsed,
        "flux_eval_s": timed_flux_law.elapsed_s,
        "nonflux_s": elapsed - timed_flux_law.elapsed_s,
        "avg_flux_eval_us": (
            1.0e6 * timed_flux_law.elapsed_s / timed_flux_law.calls
            if timed_flux_law.calls
            else np.nan
        ),
    }

# ==========================================================================

# ===== comparison =============================================================================

def make_experiment_dir(output_dir, exp_name):
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

def comparison(exp_name, methods, datasets):
    '''This is the main function!!!'''
    output_dir = ROOT / "Results"
    exp_dir = make_experiment_dir(output_dir, exp_name)
    print(f"\nSaving results to: {exp_dir}")
    result_paths = []

    for dataset_path in datasets:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.stem
        df = pd.read_csv(dataset_path)
        df["_row_id"] = np.arange(len(df))
        training_df = grid_chop_dataframe(df)

        k0 = float(df["k_0"].iloc[0])
        alpha = float(df["alpha"].iloc[0])
        beta = float(df["beta"].iloc[0])
        sigma = float(df["sigma"].iloc[0])
        dataset_results = []

        print(f"\n=== Dataset: {dataset_name} ===")
        print(f"k0={k0}, alpha={alpha}, beta={beta}, sigma={sigma}")
        print(
            f"training rows after {CHOP_GRID_S}x{CHOP_GRID_T} chop: "
            f"{len(training_df)} / {len(df)}"
        )

        reference_model = build_provider("Analytic", df, training_df)

        for method in methods:
            print(f"\n--- {method} ---")
            try:
                print("Building...")
                model = build_provider(method, df, training_df)
                row = {
                    "experiment": exp_name,
                    "dataset": dataset_name,
                    "method": model["method"],
                    "original_rows": len(df),
                    "training_rows": len(training_df),
                    "chop_grid_s": CHOP_GRID_S,
                    "chop_grid_T": CHOP_GRID_T,
                    "max_rows_per_cell": MAX_ROWS_PER_CELL,
                    "regularization_type": model.get("regularization_type", "none"),
                    "regularization_strength": model.get("regularization_strength", 0.0),
                    "model_regularization_type": model.get("model_regularization_type", "none"),
                    "model_regularization_strength": model.get("model_regularization_strength", 0.0),
                }
                print("Starting accuracy evaluation..")
                row.update(accuracy_correctness(model, df, training_df))
                print("Starting FEM/NM..")
                row.update(newton(model, reference_model)) # do i need reference model everywhere?
                print("done :P")
            except Exception as exc:
                print(":(")
                row = {
                    "experiment": exp_name,
                    "dataset": dataset_name,
                    "method": method,
                    "status": f"failed: {type(exc).__name__}",
                    "error": str(exc),
                }
            dataset_results.append(row)

        result_df = pd.DataFrame(dataset_results)
        result_path = exp_dir / f"{dataset_name}.csv"
        result_df.to_csv(result_path, index=False)
        result_paths.append(result_path)
        print(f"\nSaved CSV results: {result_path}")
    
    return result_paths


if __name__ == "__main__":
    '''
    All methods: (probably should organize this list ALSO pchip not working)
    'Analytic',
    'FiniteDiff',
    'CubicSpline',
    'PCHIP',
    'SavGol',
    'Smooth+PCHIP',
    'RBF',
    'KISS-GP',
    'RidgeRFF'
    'PenaltyRFF'
    'LinRegRFF'
    'RFF'
    'RFF_flexible'
    'RFF_constrained'
    'MaternGPMonotone'
    'MonotoneGPKPEP'

    Datasets:
    'nonlinear_no_noise.csv',
    'nonlinear_low_noise.csv',
    'nonlinear_high_noise.csv',
    'linear_no_noise.csv',
    'linear_medium_noise.csv',
    '''

    exp_name = "monotoneGPKPEPtest"

    regularization_options = [
        "",
        "+reg=laplacian:0.01",
        "+reg=laplacian:0.1",
        "+reg=gradient:0.03",
        "+reg=gradient:0.3",
    ]
    regularized_methods = [
        #"FiniteDiff",
        #"rff",
        # "rff_flexible+alpha:1e-6, p=2, n_components=2000",
        # "rff_flexible+alpha:1e-6, p=4, n_components=2000",
        # "rff_constrained, alpha=1e-6, p=2, n_components=2000, osqp",
        # "rff_constrained, alpha=1e-6, p=4, n_components=2000, osqp",
        #"cubicspline",
        #"pchip",
        #"smooth+pchip",
        #"rbf",
        #"FDMatern52",
        #"FDIntegratedEpanechnikov",
        #"savgol",
    ]
    dependency_check_methods = [
        #"kissgp",
        #"mlp",
        # "materngpmonotone_unregularized",
        # "materngpmonotone_regularized",
        "monotonegpkpep",
    ]
    methods = [
        method + regularization
        for method in regularized_methods
        for regularization in regularization_options
    ] + dependency_check_methods

    dataset_dir = ROOT / "Data" / "NoisyDeterministicOracles" / "datasets"
    dataset_paths = [
        dataset_dir / "linear_medium_noise.csv",
        dataset_dir / "linear_no_noise.csv",
        dataset_dir / "nonlinear_high_noise.csv",
        dataset_dir / "nonlinear_low_noise.csv",
        dataset_dir / "nonlinear_no_noise.csv",
    ]

    comparison(exp_name, methods, dataset_paths)
