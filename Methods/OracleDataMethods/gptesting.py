"""
Testing setup for preliminary GP troubleshoots. 
"""
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import time

from monotoneGP import MonotoneGPFluxST as UnregMonotoneGPFluxST
from monotoneGPReg import MonotoneGPFluxST as RegMonotoneGPFluxST

# EXPERIMENT CONTROLS
seed = 42
n_train  = 20
noise_std = 0.15
n_virt = 20

function_regularization = 1e-3
derivative_regularization = 1e-2

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
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def diagnostics(model):
    """Read the diagnostic attributes added to both providers."""
    joint_condition = getattr(
        model,
        "joint_condition",
        getattr(model, "joint_condition", np.nan),
    )
    alpha_norm = getattr(
        model,
        "alpha_norm",
        np.linalg.norm(model.alpha_),
    )
    return float(joint_condition), float(alpha_norm)

# DATA AND MODEL FITTING
def make_training_data():
    rng = np.random.default_rng(seed)
    s_train = rng.uniform(-1.5, 1.5, n_train)
    T_train = rng.uniform(0.5, 2.5, n_train)

    q_true = exact_flux(s_train, T_train)
    q_train = q_true + rng.normal(0.0, noise_std, n_train)
    return s_train, T_train, q_train


def fit_models(s_train, T_train, q_train):
    common = dict(
        noise_std=noise_std,
        learn_neg_flux=True,
        n_virtual_per_axis=n_virt,
        probit_nu=1e-2,
        ep_max_iter=100,
        ep_damping=0.3,
        ep_tol=1e-6,
        n_restarts_optimizer=0,
    )

    start = time.perf_counter()
    unregularized = UnregMonotoneGPFluxST(
        s_train,
        T_train,
        q_train,
        **common,
    )
    unreg_time = time.perf_counter() - start

    start = time.perf_counter()
    regularized = RegMonotoneGPFluxST(
        s_train,
        T_train,
        q_train,
        function_regularization=function_regularization,
        derivative_regularization=derivative_regularization,
        **common,
    )
    reg_time = time.perf_counter() - start

    return unregularized, regularized, unreg_time, reg_time

# REPORT METRICS
def evaluate_model(name, model, fit_time, S, TT):
    q_pred, dq_ds_pred, dq_dT_pred = model.evaluate(S, TT)

    joint_condition, alpha_norm = diagnostics(model)

    return {
        "name": name,
        "q_rmse": rmse(q_pred, exact_flux(S, TT)),
        "dq_ds_rmse": rmse(dq_ds_pred, exact_dq_ds(S, TT)),
        "dq_dT_rmse": rmse(dq_dT_pred, exact_dq_dT(S, TT)),
        "violation_percent": 100.0 * float(np.mean(dq_ds_pred > 0.0)),
        "joint_condition": joint_condition,
        "alpha_norm": alpha_norm,
        "fit_seconds": fit_time,
    }


def print_report(results):
    header = (
        f"{'model':<16}{'q RMSE':>12}{'dq/ds RMSE':>14}{'dq/dT RMSE':>14}"
        f"{'viol. %':>11}{'joint cond.':>16}{'alpha norm':>15}{'fit s':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['name']:<16}"
            f"{result['q_rmse']:>12.5g}"
            f"{result['dq_ds_rmse']:>14.5g}"
            f"{result['dq_dT_rmse']:>14.5g}"
            f"{result['violation_percent']:>11.3f}"
            f"{result['joint_condition']:>16.5e}"
            f"{result['alpha_norm']:>15.5e}"
            f"{result['fit_seconds']:>10.3f}"
        )


# PLOTS
def make_plots(models, s_train, T_train, q_train):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s_line = np.linspace(s_train.min(), s_train.max(), 300)
    T_slices = np.quantile(T_train, [0.2, 0.5, 0.8])

    fig, axes = plt.subplots(1, len(T_slices), figsize=(15, 4.5), sharey=True)
    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(s_line, exact_flux(s_line, T_line), "--", label="Exact")

        for label, model in models.items():
            q_pred, _, _ = model.evaluate(s_line, T_line)
            ax.plot(s_line, q_pred, label=label)

        near = np.abs(T_train - temperature) < 0.12
        ax.scatter(s_train[near], q_train[near], marker="x", alpha=0.6)
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")

    axes[0].set_ylabel("q(s, T)")
    axes[-1].legend()
    fig.suptitle("Flux-law reconstruction")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "flux_reconstruction.png", dpi=200)

    fig, axes = plt.subplots(1, len(T_slices), figsize=(15, 4.5), sharey=True)
    for ax, temperature in zip(axes, T_slices):
        T_line = np.full_like(s_line, temperature)
        ax.plot(s_line, exact_dq_ds(s_line, T_line), "--", label="Exact")

        for label, model in models.items():
            _, dq_ds_pred, _ = model.evaluate(s_line, T_line)
            ax.plot(s_line, dq_ds_pred, label=label)

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"T = {temperature:.2f}")
        ax.set_xlabel("s")

    axes[0].set_ylabel("dq/ds")
    axes[-1].legend()
    fig.suptitle("Derivative reconstruction")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dq_ds_reconstruction.png", dpi=200)

    plt.show()


# MAIN EXPERIMENT
def main():
    s_train, T_train, q_train = make_training_data()

    unregularized, regularized, unreg_time, reg_time = fit_models(
        s_train,
        T_train,
        q_train,
    )

    s_test = np.linspace(s_train.min(), s_train.max(), 70)
    T_test = np.linspace(T_train.min(), T_train.max(), 70)
    S, TT = np.meshgrid(s_test, T_test, indexing="ij")

    results = [
        evaluate_model("unregularized", unregularized, unreg_time, S, TT),
        evaluate_model("regularized", regularized, reg_time, S, TT),
    ]
    print_report(results)

    make_plots(
        {
            "Unregularized": unregularized,
            "Regularized": regularized,
        },
        s_train,
        T_train,
        q_train,
    )

    return unregularized, regularized, results


if __name__ == "__main__":
    main()
