"""
Testing setup for preliminary GP troubleshoots. 
- includes regularization sweeps, plots reconstructed flux and derivatives

NEED TO DO: 
* update so can test different function/deriv reg strengths

DONE: 
* update so compatible with regularized versions of code 

"""
import numpy as np
import matplotlib.pyplot as plt

import csv
from pathlib import Path
import time

from monotoneGPReg import MonotoneGPFluxST 

# EXPERIMENT CONTROLS
seed = 42
n_train  = 50
true_noise_std = 0.20 # learned noise level should match this 
n_virt = 50

# func reg def as fraction of standardized noise var learned by WhiteKernel
function_regularization_fractions = [0.0]
derivative_regularization_strengths = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 0.0]

OUTPUT_DIR = Path(__file__).resolve().parent/"GPDerivRegSweepResults(n_train=50, n_virt=50)"

# TRUE FLUX AND DERIVATIVES
def exact_flux(s, T):
    k0, alpha, beta = 1.25, 0.18, 0.12
    return -(k0 * (1.0 + alpha * T**2) + beta * s**2) * s


def exact_dq_ds(s, T):
    k0, alpha, beta = 1.25, 0.18, 0.12
    return -(k0 * (1.0 + alpha * T**2) + 3.0 * beta * s**2)


def exact_dq_dT(s, T):
    k0, alpha = 1.25, 0.18
    return -2.0 * k0 * alpha * T * s

# HELPERS
def rmse(prediction, truth):
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def diagnostics(model):
    return float(model.joint_condition_), float(model.alpha_norm_)

def common_model_arguments():
    return{
        "noise_std": true_noise_std,
        "learn_neg_flux": True,
        "n_virtual_per_axis": n_virt,
        "probit_nu": 1e-3, 
        "ep_max_iter": 60,
        "ep_damping": 0.5, 
        "ep_tol": 1e-5,
        "n_restarts_optimizer": 5,
    }

def make_training_data():
    rng = np.random.default_rng(seed)
    s_train = rng.uniform(-1.5, 1.5, n_train)
    T_train = rng.uniform(0.5, 2.5, n_train)

    q_true = exact_flux(s_train, T_train)
    q_train = q_true + rng.normal(0.0, true_noise_std, n_train)
    return s_train, T_train, q_train

def case_key(function_fraction, derivative_strength):
    return float(function_fraction), float(derivative_strength)


def case_type(function_strength, derivative_strength):
    if function_strength == 0.0 and derivative_strength == 0.0:
        return "unregularized"
    if function_strength == 0.0:
        return "derivative-only"
    if derivative_strength == 0.0:
        return "function-only"
    return "joint"


def case_label(function_fraction, function_strength, derivative_strength):
    kind = case_type(function_strength, derivative_strength)

    if kind == "unregularized":
        return "unregularized"
    if kind == "derivative-only":
        return f"derivative-only: lambda_d={derivative_strength:.0e}"
    if kind == "function-only":
        return (
            f"function-only: lambda_f={function_strength:.0e} "
            f"({function_fraction:.0e} noise)"
        )
    return (
        f"lambda_f={function_strength:.0e} "
        f"({function_fraction:.0e} noise), "
        f"lambda_d={derivative_strength:.0e}"
    )


def plot_case_label(function_fraction, derivative_strength):
    if function_fraction == 0.0:
        return f"f=0, d={derivative_strength:.0e}"
    return f"f/noise={function_fraction:.0e}, d={derivative_strength:.0e}"


def build_regularization_cases(learned_noise_variance):
    learned_noise_variance = float(learned_noise_variance)
    if not np.isfinite(learned_noise_variance) or learned_noise_variance <= 0.0:
        raise ValueError("learned_noise_variance must be positive and finite")

    cases = []

    for fraction in function_regularization_fractions:
        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(
                "function regularization fractions must satisfy 0 <= fraction < 1"
            )

        function_strength = fraction * learned_noise_variance

        for derivative_strength in derivative_regularization_strengths:
            derivative_strength = float(derivative_strength)
            if not np.isfinite(derivative_strength) or derivative_strength < 0.0:
                raise ValueError(
                    "derivative regularization strengths must be finite and nonnegative"
                )

            if function_strength == 0.0 and derivative_strength == 0.0:
                continue

            cases.append(
                {
                    "function_fraction": fraction,
                    "function_strength": float(function_strength),
                    "derivative_strength": derivative_strength,
                }
            )

    return cases

# MODEL FITTING
def fit_model(s_train, T_train, q_train, reg_function, reg_derivative):
    start = time.perf_counter()
    model = MonotoneGPFluxST(
        s_train, 
        T_train, 
        q_train, 
        reg_function=float(reg_function),
        reg_derivative=float(reg_derivative),
        **common_model_arguments(),
    )
    fit_time = time.perf_counter() - start
    return model, fit_time

def run_regularization_sweep(s_train, T_train, q_train, learned_noise_variance):
    models = {}
    fit_times = {}
    cases = build_regularization_cases(learned_noise_variance)
    for case in cases:
        function_fraction = case["function_fraction"]
        function_strength = case["function_strength"]
        derivative_strength = case["derivative_strength"]
        key = case_key(function_fraction, derivative_strength)

        print(
            "Fitting model with "
            f"lambda_f={function_strength:.3e} "
            f"({function_fraction:.3e} * learned noise variance), "
            f"lambda_d={derivative_strength:.3e}"
        )

        model, fit_time = fit_model(
            s_train,
            T_train,
            q_train,
            reg_function=function_strength,
            reg_derivative=derivative_strength,
        )

        models[key] = model
        fit_times[key] = fit_time

    return models, fit_times, cases

# MODEL EVALUATION
def evaluate_model(name, model, fit_time, S, TT, function_fraction, function_strength, derivative_strength):
    q_pred, dq_ds_pred, dq_dT_pred = model.evaluate(S, TT)
    condition_number, alpha_norm = diagnostics(model)
    positive_violation = np.maximum(dq_ds_pred, 0.0)
    return {
        "name": name,
        "case_type": case_type(function_strength, derivative_strength),
        "function_regularization_fraction_of_noise": float(function_fraction),
        "function_regularization_strength": float(function_strength),
        "derivative_regularization_strength": float(derivative_strength),
        "q_rmse": rmse(q_pred, exact_flux(S, TT)),
        "dq_ds_rmse": rmse(dq_ds_pred, exact_dq_ds(S, TT)),
        "dq_dT_rmse": rmse(dq_dT_pred, exact_dq_dT(S, TT)),
        "violation_percent": 100.0 * float(np.mean(dq_ds_pred > 0.0)),
        "worst_violation": float(np.max(positive_violation)),
        "condition_number": condition_number,
        "alpha_norm": alpha_norm,
        "learned_noise_variance_standardized": float(model.learned_noise_variance_),
        "learned_noise_std_physical": float(model.learned_noise_std_physical_),
        "fit_seconds": float(fit_time),
    }

def evaluate_all_models(unregularized_model, unregularized_fit_time, regularized_models, regularized_fit_times, cases, S, TT):
    results = [evaluate_model(name="unregularized", 
                              model=unregularized_model, 
                              fit_time=unregularized_fit_time,
                              S=S,
                              TT=TT,
                              regularization_strength=0.0, 
                              regularization_fraction=0.0)]
    for case in cases:
        function_fraction = case["function_fraction"]
        function_strength = case["function_strength"]
        derivative_strength = case["derivative_strength"]
        key = case_key(function_fraction, derivative_strength)

        results.append(
            evaluate_model(
                name=case_label(
                    function_fraction,
                    function_strength,
                    derivative_strength,
                ),
                model=regularized_models[key],
                fit_time=regularized_fit_times[key],
                S=S,
                TT=TT,
                function_fraction=function_fraction,
                function_strength=function_strength,
                derivative_strength=derivative_strength,
            )
        )

    return results

# REPORTING
def print_noise_summary(model):
    print("\nNoise summary")
    print(f"True physical noise std:          {true_noise_std:.6g}")
    print(f"Learned physical noise std:       {model.learned_noise_std_physical_:.6g}")
    print(f"Learned standardized noise var:   {model.learned_noise_variance_:.6e}")
    print(f"Optimized kernel: {model.gp_kernel_}")

def print_report(results):
    header = (
        f"{'model':<42}"
        f"{'lambda_f':>12}"
        f"{'f/noise':>11}"
        f"{'lambda_d':>12}"
        f"{'q RMSE':>12}"
        f"{'dq/ds RMSE':>14}"
        f"{'dq/dT RMSE':>14}"
        f"{'viol. %':>10}"
        f"{'worst viol.':>14}"
        f"{'condition no.':>16}"
        f"{'alpha norm':>15}"
        f"{'fit s':>10}"
    )

    print("\n" + header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['name']:<42}"
            f"{result['function_regularization_strength']:>12.3e}"
            f"{result['function_regularization_fraction_of_noise']:>11.3e}"
            f"{result['derivative_regularization_strength']:>12.3e}"
            f"{result['q_rmse']:>12.5e}"
            f"{result['dq_ds_rmse']:>14.5g}"
            f"{result['dq_dT_rmse']:>14.5g}"
            f"{result['violation_percent']:>10.3f}"
            f"{result['worst_violation']:>14.5e}"
            f"{result['condition_number']:>16.5e}"
            f"{result['alpha_norm']:>15.5e}"
            f"{result['fit_seconds']:>10.3f}"
        )


def save_report(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "regularization_component_sweep_metrics.csv"

    fieldnames = [
        "name",
        "case_type",
        "function_regularization_fraction_of_noise",
        "function_regularization_strength",
        "derivative_regularization_strength",
        "q_rmse",
        "dq_ds_rmse",
        "dq_dT_rmse",
        "violation_percent",
        "worst_violation",
        "condition_number",
        "alpha_norm",
        "learned_noise_variance_standardized",
        "learned_noise_std_physical",
        "fit_seconds",
    ]

    with output_path.open(mode="w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved metrics to:\n{output_path}")


def print_best_models(results):
    regularized_results = [
        result for result in results if result["case_type"] != "unregularized"
    ]

    if not regularized_results:
        print("\nNo nonzero regularization cases were requested.")
        return

    best_metrics = [
        ("Flux reconstruction", "q_rmse"),
        ("dq/ds reconstruction", "dq_ds_rmse"),
        ("dq/dT reconstruction", "dq_dT_rmse"),
    ]

    print("\nBest regularized models")

    for title, metric in best_metrics:
        best = min(regularized_results, key=lambda result: result[metric])
        print(
            f"{title}: "
            f"lambda_f={best['function_regularization_strength']:.3e}, "
            f"f/noise={best['function_regularization_fraction_of_noise']:.3e}, "
            f"lambda_d={best['derivative_regularization_strength']:.3e} "
            f"({metric}={best[metric]:.6g})"
        )

# RECONSTRUCTION PLOTS
def make_reconstruction_plots(
    unregularized_model,
    regularized_models,
    cases,
    s_train,
    T_train,
    q_train,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s_line = np.linspace(s_train.min(), s_train.max(), 300)
    T_slices = np.quantile(T_train, [0.2, 0.5, 0.8])

    fig, axes = plt.subplots(
        1,
        len(T_slices),
        figsize=(16, 4.8),
        sharey=True,
    )

    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(
            s_line,
            exact_flux(s_line, T_line),
            linestyle="--",
            linewidth=2.5,
            label="Exact",
        )

        q_unreg, _, _ = unregularized_model.evaluate(s_line, T_line)
        ax.plot(s_line, q_unreg, linewidth=2.0, label="Unregularized")

        for case in cases:
            function_fraction = case["function_fraction"]
            derivative_strength = case["derivative_strength"]
            key = case_key(function_fraction, derivative_strength)
            model = regularized_models[key]

            q_reg, _, _ = model.evaluate(s_line, T_line)
            ax.plot(
                s_line,
                q_reg,
                linewidth=1.2,
                alpha=0.75,
                label=plot_case_label(function_fraction, derivative_strength),
            )


        near = np.abs(T_train - temperature) < 0.12
        ax.scatter(
            s_train[near],
            q_train[near],
            marker="x",
            alpha=0.6,
            label="Nearby observations",
        )
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")

    axes[0].set_ylabel("q(s, T)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.suptitle("Flux-law reconstruction across regularization strengths")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "flux_reconstruction_all_strengths.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(T_slices),
        figsize=(16, 4.8),
        sharey=True,
    )

    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(
            s_line,
            exact_dq_ds(s_line, T_line),
            linestyle="--",
            linewidth=2.5,
            label="Exact",
        )

        _, dq_ds_unreg, _ = unregularized_model.evaluate(s_line, T_line)
        ax.plot(
            s_line,
            dq_ds_unreg,
            linewidth=2.0,
            label="Unregularized",
        )

        for case in cases:
            function_fraction = case["function_fraction"]
            derivative_strength = case["derivative_strength"]
            key = case_key(function_fraction, derivative_strength)
            model = regularized_models[key]

            _, dq_ds_reg, _ = model.evaluate(s_line, T_line)
            ax.plot(
                s_line,
                dq_ds_reg,
                linewidth=1.2,
                alpha=0.75,
                label=plot_case_label(function_fraction, derivative_strength),
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")

    axes[0].set_ylabel("dq/ds")
    axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.suptitle("Physical derivative reconstruction: dq/ds should be nonpositive")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "dq_ds_reconstruction_all_strengths.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

# REG SWEEP PLOTS
def _derivative_axis_linthresh(results):
    positive = [
        result["derivative_regularization_strength"]
        for result in results
        if result["derivative_regularization_strength"] > 0.0
    ]
    return min(positive) / 10.0 if positive else 1e-12


def make_metric_sweep_plot(
    results,
    metric,
    ylabel,
    title,
    filename,
    *,
    log_y=False,
):
    """
    Plot a metric against lambda_d, with one curve for each function level.

    This representation remains valid for a derivative-only sweep
    (function fraction = 0) and for a full two-dimensional regularization grid.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = next(
        result for result in results if result["case_type"] == "unregularized"
    )
    regularized_results = [
        result for result in results if result["case_type"] != "unregularized"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    function_fractions = sorted(
        {
            result["function_regularization_fraction_of_noise"]
            for result in regularized_results
        }
    )

    for fraction in function_fractions:
        group = [
            result
            for result in regularized_results
            if result["function_regularization_fraction_of_noise"] == fraction
        ]
        group.sort(key=lambda result: result["derivative_regularization_strength"])

        x = np.asarray(
            [result["derivative_regularization_strength"] for result in group],
            dtype=float,
        )
        y = np.asarray([result[metric] for result in group], dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if log_y:
            finite &= y > 0.0

        if np.any(finite):
            ax.plot(
                x[finite],
                y[finite],
                marker="o",
                label=f"lambda_f/noise={fraction:.0e}",
            )

    ax.axhline(
        baseline[metric],
        linestyle="--",
        label="zero-regularization baseline",
    )
    ax.set_xscale("symlog", linthresh=_derivative_axis_linthresh(results))

    if log_y and baseline[metric] > 0.0:
        ax.set_yscale("log")

    ax.set_xlabel("Derivative regularization strength, lambda_d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_sweep_plots(results):
    make_metric_sweep_plot(
        results,
        "q_rmse",
        "q RMSE",
        "Flux RMSE across function/derivative regularization",
        "q_rmse_component_sweep.png",
    )
    make_metric_sweep_plot(
        results,
        "dq_ds_rmse",
        "dq/ds RMSE",
        "Monotone-derivative RMSE across regularization",
        "dq_ds_rmse_component_sweep.png",
    )
    make_metric_sweep_plot(
        results,
        "dq_dT_rmse",
        "dq/dT RMSE",
        "Temperature-derivative RMSE across regularization",
        "dq_dT_rmse_component_sweep.png",
    )
    make_metric_sweep_plot(
        results,
        "violation_percent",
        "Physical violation percentage",
        "Positive dq/ds violations across regularization",
        "violation_percent_component_sweep.png",
    )
    make_metric_sweep_plot(
        results,
        "worst_violation",
        "Largest positive dq/ds",
        "Worst physical monotonicity violation",
        "worst_violation_component_sweep.png",
    )
    make_metric_sweep_plot(
        results,
        "alpha_norm",
        "||alpha||_2",
        "Alpha norm across function/derivative regularization",
        "alpha_norm_component_sweep.png",
        log_y=True,
    )
    make_metric_sweep_plot(
        results,
        "condition_number",
        "Condition number",
        "Joint condition number across regularization",
        "condition_number_component_sweep.png",
        log_y=True,
    )
    make_metric_sweep_plot(
        results,
        "fit_seconds",
        "Fit time (seconds)",
        "Fit time across function/derivative regularization",
        "fit_time_component_sweep.png",
    )

# MAIN EXPERIMENT
def main():
    s_train, T_train, q_train = make_training_data()

    print("Fitting zero-regularization learned-noise reference")
    unregularized_model, unregularized_fit_time = fit_model(
        s_train,
        T_train,
        q_train,
        reg_function=0.0,
        reg_derivative=0.0,
    )

    learned_noise_variance = float(unregularized_model.learned_noise_variance_)
    print_noise_summary(unregularized_model)

    regularized_models, regularized_fit_times, cases = run_regularization_sweep(
        s_train,
        T_train,
        q_train,
        learned_noise_variance,
    )

    s_test = np.linspace(s_train.min(), s_train.max(), 70)
    T_test = np.linspace(T_train.min(), T_train.max(), 70)
    S, TT = np.meshgrid(s_test, T_test, indexing="ij")

    results = evaluate_all_models(
        unregularized_model,
        unregularized_fit_time,
        regularized_models,
        regularized_fit_times,
        cases,
        S,
        TT,
    )

    print_report(results)
    print_best_models(results)
    save_report(results)

    make_reconstruction_plots(
        unregularized_model=unregularized_model,
        regularized_models=regularized_models,
        cases=cases,
        s_train=s_train,
        T_train=T_train,
        q_train=q_train,
    )
    make_sweep_plots(results)

    return unregularized_model, regularized_models, results


if __name__ == "__main__":
    main()