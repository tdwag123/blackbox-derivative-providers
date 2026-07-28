"""
Testing setup for preliminary GP troubleshoots. 
- includes regularization sweeps, plots reconstructed flux and derivatives

"""
import numpy as np
import matplotlib.pyplot as plt

import csv
from pathlib import Path
import time

from monotoneGP import MonotoneGPFluxST as UnregMonotoneGPFluxST
from monotoneGPReg import MonotoneGPFluxST as RegMonotoneGPFluxST

# EXPERIMENT CONTROLS
seed = 42
n_train  = 15
noise_std = 0.20 # learned noise level ~ here 
n_virt = 10

regularization_strengths = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

OUTPUT_DIR = Path(__file__).resolve().parent/"GPComparisonResults"

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
        "noise_std": noise_std,
        "learn_neg_flux": True,
        "n_virtual_per_axis": n_virt,
        "probit_nu": 1e-3, 
        "ep_max_iter": 60,
        "ep_damping": 0.5, 
        "ep_tol": 1e-5,
        "n_restarts_optimizer": 0,
    }

# DATA AND MODEL FITTING
def make_training_data():
    rng = np.random.default_rng(seed)
    s_train = rng.uniform(-1.5, 1.5, n_train)
    T_train = rng.uniform(0.5, 2.5, n_train)

    q_true = exact_flux(s_train, T_train)
    q_train = q_true + rng.normal(0.0, noise_std, n_train)
    return s_train, T_train, q_train


def fit_unregularized_model(s_train, T_train, q_train):
    start = time.perf_counter()
    model = UnregMonotoneGPFluxST(
        s_train, 
        T_train,
        q_train,
        **common_model_arguments(),
    )
    fit_time = time.perf_counter() - start
    return model, fit_time

def fit_regularized_model(s_train, T_train, q_train, strength):
    start = time.perf_counter()
    model = RegMonotoneGPFluxST(
        s_train, 
        T_train, 
        q_train, 
        reg_function=strength,
        reg_derivative=strength,
        **common_model_arguments(),
    )
    fit_time = time.perf_counter() - start
    return model, fit_time

def run_regularization_sweep(s_train, T_train, q_train):
    models = {}
    fit_times = {}
    for strength in regularization_strengths:
        print(f"Fitting regularized model with strength = {strength:.1e}")
        model, fit_time = fit_regularized_model(s_train, T_train, q_train, strength)
        models[strength] = model
        fit_times[strength] = fit_time
    return models, fit_times

# MODEL EVALUATION
def evaluate_model(name, model, fit_time, S, TT, regularization_strength):
    q_pred, dq_ds_pred, dq_dT_pred = model.evaluate(S, TT)
    condition_number, alpha_norm = diagnostics(model)
    return{
        "name": name,
        "regularization_strength": regularization_strength,
        "q_rmse": rmse(q_pred, exact_flux(S,TT)),
        "dq_ds_rmse": rmse(dq_ds_pred, exact_dq_ds(S,TT)),
        "dq_dT_rmse": rmse(dq_dT_pred, exact_dq_dT(S, TT)),
        "violation_percent": 100.0 * float(np.mean(dq_ds_pred > 0.0)),
        "worst_dq_ds": float(np.max(dq_ds_pred)), 
        "condition_number": condition_number, 
        "alpha_norm": alpha_norm,
        "fit_seconds": fit_time,
    }

def evaluate_all_models(unregularized_model, unregularized_fit_time, regularized_models, regularized_fit_times, S, TT):
    results = [evaluate_model(name="unregularized", 
                              model=unregularized_model, 
                              fit_time=unregularized_fit_time,
                              S=S,
                              TT=TT,
                              regularization_strength=0.0)]
    for strength in regularization_strengths:
        results.append(evaluate_model(name=f"reg={strength:.0e}",
                                      model=regularized_models[strength],
                                      fit_time=regularized_fit_times[strength],
                                      S=S,
                                      TT=TT,
                                      regularization_strength=strength))
    return results

# REPORTING
def print_report(results):
    header = (f"{'model':<16}" 
              f"{'strength':>12}" 
              f"{'q RMSE':>12}" 
              f"{'dq/ds RMSE':>14}" 
              f"{'dq/dT RMSE':>14}" 
              f"{'viol. %':>11}" 
              f"{'condition no.':>16}" 
              f"{'alpha norm':>15}" 
              f"{'fit s':>10}")
    print("\n" + header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['name']:<16}"
            f"{result['regularization_strength']:>12.1e}"
            f"{result['q_rmse']:>12.5e}"
            f"{result['dq_ds_rmse']:>14.5g}"
            f"{result['dq_dT_rmse']:>14.5g}"
            f"{result['violation_percent']:>11.3f}"
            f"{result['condition_number']:>16.5e}"
            f"{result['alpha_norm']:>15.5e}"
            f"{result['fit_seconds']:>10.3f}"
        )

def save_report(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR/"regularization_sweep_metrics.csv"
    fieldnames = ["name",
                  "regularization_strength", 
                  "q_rmse",
                  "dq_ds_rmse",
                  "dq_dT_rmse",
                  "violation_percent",
                  "worst_dq_ds",
                  "condition_number",
                  "alpha_norm",
                  "fit_seconds"]
    with output_path.open(mode="w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved metrics to: \n{output_path}")

def print_best_models(results):
    regularized_results = [result for result in results if result["regularization_strength"] > 0.0]
    best_q = min(regularized_results, key = lambda result: result["q_rmse"])
    best_dq_ds = min(regularized_results, key=lambda result: result["dq_ds_rmse"])
    best_dq_dT = min(regularized_results, key=lambda result: result["dq_dT_rmse"])
    print("\nBest regularized models")
    print(f"Flux reconstruction: {best_q['regularization_strength']:.1e} (RMSE = {best_q['q_rmse']:.6g})") 
    print(f"dq/ds reconstruction: {best_dq_ds['regularization_strength']:.1e} (RMSE = {best_dq_ds['dq_ds_rmse']:.6g})") 
    print(f"dq/dT reconstruction: {best_dq_dT['regularization_strength']:.1e} (RMSE = {best_dq_dT['dq_dT_rmse']:.6g})")

# RECONSTRUCTION PLOTS
def make_reconstruction_plots(
        unregularized_model,
        regularized_models,
        s_train,
        T_train,
        q_train,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s_line = np.linspace(s_train.min(), s_train.max(), 300)
    T_slices = np.quantile(T_train, [0.2, 0.5, 0.8])

    fig, axes = plt.subplots(1, len(T_slices), figsize=(16, 4.8), sharey=True)
    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(s_line, exact_flux(s_line, T_line), linestyle="--", linewidth=2.5, label="Exact")
        q_unreg, _, _ = unregularized_model.evaluate(s_line, T_line)
        ax.plot(s_line, q_unreg, linewidth=2.0, label="unregularized")
        for strength, model in regularized_models.items():
            q_reg, _, _ = model.evaluate(s_line, T_line)
            ax.plot(s_line, q_reg, linewidth=1.2, alpha=0.75, label=f"reg {strength:.0e}")
        near = np.abs(T_train - temperature) < 0.12
        ax.scatter(s_train[near], q_train[near], marker="x", alpha=0.6, label="Nearby observations")
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")
    axes[0].set_ylabel("q(s, T)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.suptitle("Flux-law reconstruction across regularization strengths")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"flux_reconstruction_all_strengths.png", dpi=200, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(T_slices), figsize=(16, 4.8), sharey=True)
    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(s_line, exact_dq_ds(s_line, T_line), linestyle="--", linewidth=2.5, label="Exact")
        _, dq_ds_unreg, _ = unregularized_model.evaluate(s_line, T_line)
        ax.plot(s_line, dq_ds_unreg, linewidth=2.0, label="unregularized")
        for strength, model in regularized_models.items():
            _, dq_ds_reg, _ = model.evaluate(s_line, T_line)
            ax.plot(s_line, dq_ds_reg, linewidth=1.2, alpha=0.75, label=f"reg {strength:.0e}")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")
    axes[0].set_ylabel("dq/ds")
    axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    fig.suptitle("Derivative reconstruction across regularization strengths")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"dq_ds_reconstruction_all_strengths.png", dpi=200, bbox_inches="tight")

# REG SWEEP PLOTS
def make_sweep_plots(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unregularized_result=next(result for result in results if result["regularization_strength"] == 0.0)
    regularized_results = [result for result in results if result["regularization_strength"] > 0.0]
    strengths = np.asarray([result["regularization_strength"] for result in regularized_results], dtype=float)
    
    plt.figure(figsize=(8,5))
    plt.semilogx(strengths, [result["q_rmse"] for result in regularized_results], marker = "o", label = "q RMSE")
    plt.semilogx(strengths, [result["dq_ds_rmse"] for result in regularized_results], marker = "o", label = "dq/ds RMSE")
    plt.semilogx(strengths, [result["dq_dT_rmse"] for result in regularized_results], marker = "o", label = "dq/dT RMSE")
    plt.axhline(unregularized_result["q_rmse"], linestyle="--", label="unregularized q RMSE")
    plt.axhline(unregularized_result["dq_ds_rmse"], linestyle="--", label="unregularized dq/ds RMSE")
    plt.axhline(unregularized_result["dq_dT_rmse"], linestyle="--", label="unregularized dq/dT RMSE")
    plt.xlabel("Regularization strength")
    plt.ylabel("RMSE")
    plt.title("Accuracy versus regularization strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"regularization_rmse_sweep.png", dpi=200, bbox_inches="tight")

    plt.figure(figsize=(8,5))
    plt.semilogx(strengths, [result["alpha_norm"] for result in regularized_results], marker = "o", label = "regularized alpha norm")
    plt.axhline(unregularized_result["alpha_norm"], linestyle="--", label="unregularized alpha norm")
    plt.xlabel("Regularization strength")
    plt.ylabel("||alpha||_2")
    plt.title("alpha norm versus regularization strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"regularization_alpha_norm_sweep.png", dpi=200, bbox_inches="tight")

    condition_numbers = np.asarray([result["condition_number"] for result in regularized_results], dtype=float) 
    finite_condition_mask = (np.isfinite(condition_numbers) & (condition_numbers > 0.0)) 
    if np.any(finite_condition_mask): 
        plt.figure(figsize=(8, 5)) 
        plt.loglog(strengths[finite_condition_mask], 
                   condition_numbers[finite_condition_mask], 
                   marker="o", 
                   label="Regularized condition number") 
        unregularized_condition = (unregularized_result["condition_number"]) 
        if (np.isfinite(unregularized_condition) and unregularized_condition > 0.0 ): 
            plt.axhline(unregularized_condition, linestyle="--", label="Unregularized condition number") 
        plt.xlabel("Regularization strength") 
        plt.ylabel("Condition number") 
        plt.title("condition number versus regularization strength") 
        plt.legend() 
        plt.tight_layout() 
        plt.savefig( OUTPUT_DIR / "regularization_condition_sweep.png", dpi=200, bbox_inches="tight")

# MAIN EXPERIMENT
def main():
    s_train, T_train, q_train = make_training_data()
    print("Fitting unregularized model")
    unregularized_model, unregularized_fit_time = fit_unregularized_model(s_train, T_train, q_train)
    regularized_models, regularized_fit_times = run_regularization_sweep(s_train, T_train, q_train)
    s_test = np.linspace(s_train.min(), s_train.max(), 70)
    T_test = np.linspace(T_train.min(), T_train.max(), 70)
    S, TT = np.meshgrid(s_test, T_test, indexing="ij")
    results = evaluate_all_models(unregularized_model, unregularized_fit_time, regularized_models, regularized_fit_times, S, TT)
    print_report(results)
    print_best_models(results)
    save_report(results)
    make_reconstruction_plots(unregularized_model=unregularized_model,
                              regularized_models=regularized_models,
                              s_train=s_train,
                              T_train=T_train,
                              q_train=q_train)
    make_sweep_plots(results)
    return unregularized_model, regularized_models, results

if __name__ == "__main__":
    main()