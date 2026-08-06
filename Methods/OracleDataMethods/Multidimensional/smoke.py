"""Generic multidimensional GP-provider smoke-test infrastructure."""

from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baseGP import GPFluxST
from monotoneGP import MonotoneGPFluxST

### ADD MORE METHODS HERE ###

def exact_flux(s, T):
    s = np.asarray(s, dtype=float)
    T = np.asarray(T, dtype=float)
    q0 = (
        -(1.0 + 0.10 * T[..., 0] ** 2 + 0.05 * s[..., 0] ** 2)
        * s[..., 0]
        + 0.20 * s[..., 1] * T[..., 1]
    )
    q1 = (
        -(0.8 + 0.08 * T[..., 1] ** 2 + 0.04 * s[..., 1] ** 2)
        * s[..., 1]
        + 0.15 * s[..., 0] * T[..., 0]
    )
    return np.stack([q0, q1], axis=-1)

def exact_dq_ds(s, T):
    s = np.asarray(s, dtype=float)
    T = np.asarray(T, dtype=float)
    J = np.empty(s.shape[:-1] + (2, 2), dtype=float)
    J[..., 0, 0] = -(1.0 + 0.10 * T[..., 0] ** 2 + 0.15 * s[..., 0] ** 2)
    J[..., 0, 1] = 0.20 * T[..., 1]
    J[..., 1, 0] = 0.15 * T[..., 0]
    J[..., 1, 1] = -(0.8 + 0.08 * T[..., 1] ** 2 + 0.12 * s[..., 1] ** 2)
    return J

def exact_dq_dT(s, T):
    s = np.asarray(s, dtype=float)
    T = np.asarray(T, dtype=float)
    J = np.empty(s.shape[:-1] + (2, 2), dtype=float)
    J[..., 0, 0] = -0.20 * T[..., 0] * s[..., 0]
    J[..., 0, 1] = 0.20 * s[..., 1]
    J[..., 1, 0] = 0.15 * s[..., 0]
    J[..., 1, 1] = -0.16 * T[..., 1] * s[..., 1]
    return J

def rmse(prediction, truth):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))

def evaluate_provider(name, model, s_test, T_test):
    start = time.perf_counter()
    q, dq_ds, dq_dT, var_q, var_dq_ds, var_dq_dT = model.evaluate(
        s_test, T_test, return_variance=True
    )
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "q_rmse": rmse(q, exact_flux(s_test, T_test)),
        "dq_ds_rmse": rmse(dq_ds, exact_dq_ds(s_test, T_test)),
        "dq_dT_rmse": rmse(dq_dT, exact_dq_dT(s_test, T_test)),
        "mean_q_variance": float(np.mean(var_q)),
        "mean_dq_ds_variance": float(np.mean(var_dq_ds)),
        "mean_dq_dT_variance": float(np.mean(var_dq_dT)),
        "evaluate_seconds": elapsed,
    }

def main():
    rng = np.random.default_rng(42)
    n_train = 18
    s_train = rng.uniform([-1.5, -1.0], [1.5, 1.0], size=(n_train, 2))
    T_train = rng.uniform([0.5, 0.8], [2.5, 2.0], size=(n_train, 2))
    q_train = exact_flux(s_train, T_train)
    q_train += rng.normal(0.0, 0.02, size=q_train.shape)

    fit_start = time.perf_counter()
    base = GPFluxST(s_train, T_train, q_train, noise_std=0.015, 
                    n_restarts_optimizer=0,max_cache_size=n_train)
    base_fit_seconds = time.perf_counter() - fit_start

    fit_start = time.perf_counter()
    monotone = MonotoneGPFluxST(s_train, T_train, q_train, noise_std=0.015,
                                n_virtual_per_axis=2, monotone_s_dims=[0, 1],
                                ep_max_iter=5, online_ep_sweeps=1, max_cache_size=n_train)
    monotone_fit_seconds = time.perf_counter() - fit_start

    ### ADD MORE PROVIDERS HERE WITH SAME EVALUATE/UPDATE CONTRACT ###
    
    providers = {
        "baseGP" : base,
        "monotoneGP" : monotone,
    }

    fit_times = {
        "baseGP" : base_fit_seconds,
        "monotoneGP" : monotone_fit_seconds
    }

    s_test = rng.uniform([0.75, 0.30], [1.35, 0.85], size=(25, 2))
    T_test = rng.uniform([1.45, 1.10], [2.10, 1.75], size=(25, 2))

    print("\nPre-update metrics")
    print("\n------------------")
    pre_update = {}
    for name, model in providers.items():
        metrics = evaluate_provider(name, model, s_test, T_test)
        metrics["fit_seconds"] = fit_times[name]
        pre_update[name] = metrics
        print(metrics)

    s_new = rng.uniform([0.90, 0.40], [1.30, 0.80], size=(4, 2))
    T_new = rng.uniform([1.55, 1.20], [2.00, 1.65], size=(4, 2))
    q_new = exact_flux(s_new, T_new)
    query_s = np.mean(s_test, axis=0)
    query_T = np.mean(T_test, axis=0)

    print("\nDistance-based posterior updates")
    print("\n--------------------------------")
    for name, model in providers.items():
        start = time.perf_counter()
        info = model.update_posterior(
            s_new,
            T_new,
            q_new,
            s_query=query_s,
            T_query=query_T,
        )
        update_seconds = time.perf_counter() - start
        print(f"\n{name}")
        print("\nEvicted old points:")
        print(info["evicted_old_points"])
        print("Update seconds:", update_seconds)

    print("\nPost-update metrics")
    print("\n-------------------")
    for name, model in providers.items():
        metrics = evaluate_provider(name, model, s_test, T_test)
        print(metrics)
        print(
            "\nVariance change:",
            {
                "q_mean": (metrics["mean_q_variance"] - pre_update[name]["mean_q_variance"]),
                "dq_ds_mean": (metrics["mean_dq_ds_variance"] - pre_update[name]["mean_dq_ds_variance"]),
                "dq_dT_mean": (metrics["mean_dq_dT_variance"] - pre_update[name]["mean_dq_dT_variance"]),
            },
        )
        print("\n")
        assert model.cache_size_ == n_train
        assert np.all(np.isfinite(model.evaluate(s_test, T_test, return_variance=True)[-1]))
    print("Parallel smoke test passed.")


if __name__ == "__main__":
    main()