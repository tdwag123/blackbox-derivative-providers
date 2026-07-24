from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


def phase_weight(T, T_c, delta):
    z = (T - T_c) / delta
    return 0.5 * (1.0 + np.tanh(z))


def phase_weight_derivative(T, T_c, delta):
    z = (T - T_c) / delta
    return 0.5 * (1.0 - np.tanh(z) ** 2) / delta


def phase_conductivity(s, T, k_0, alpha, beta):
    return k_0 * (1.0 + alpha * T**2) + beta * s**2


def physical_flux(s, T, params):
    T_c = params["transition_temperature"]
    delta = params["transition_width"]

    w = phase_weight(T, T_c, delta)
    k_1 = phase_conductivity(
        s,
        T,
        params["phase_1"]["k_0"],
        params["phase_1"]["alpha"],
        params["phase_1"]["beta"],
    )
    k_2 = phase_conductivity(
        s,
        T,
        params["phase_2"]["k_0"],
        params["phase_2"]["alpha"],
        params["phase_2"]["beta"],
    )
    k_effective = (1.0 - w) * k_1 + w * k_2

    return -k_effective * s


def physical_flux_derivatives(s, T, params):
    T_c = params["transition_temperature"]
    delta = params["transition_width"]

    phase_1 = params["phase_1"]
    phase_2 = params["phase_2"]

    w = phase_weight(T, T_c, delta)
    w_T = phase_weight_derivative(T, T_c, delta)

    k_1 = phase_conductivity(
        s,
        T,
        phase_1["k_0"],
        phase_1["alpha"],
        phase_1["beta"],
    )
    k_2 = phase_conductivity(
        s,
        T,
        phase_2["k_0"],
        phase_2["alpha"],
        phase_2["beta"],
    )
    k_effective = (1.0 - w) * k_1 + w * k_2

    dk_1_ds = 2.0 * phase_1["beta"] * s
    dk_2_ds = 2.0 * phase_2["beta"] * s
    dk_effective_ds = (1.0 - w) * dk_1_ds + w * dk_2_ds

    dk_1_dT = 2.0 * phase_1["k_0"] * phase_1["alpha"] * T
    dk_2_dT = 2.0 * phase_2["k_0"] * phase_2["alpha"] * T
    dk_effective_dT = (
        (1.0 - w) * dk_1_dT
        + w * dk_2_dT
        + w_T * (k_2 - k_1)
    )

    a_true = -k_effective - s * dk_effective_ds
    b_true = -s * dk_effective_dT
    return a_true, b_true


def make_dataset(params, n_samples, rng):
    s = rng.uniform(-3.0, 3.0, n_samples)
    T = rng.uniform(0.0, 3.0, n_samples)
    x = rng.uniform(0.0, 1.0, n_samples)

    phase_1 = params["phase_1"]
    sigma = params["sigma"]

    q_true = physical_flux(s, T, params)
    a_true, b_true = physical_flux_derivatives(s, T, params)

    noise = sigma * np.maximum(1.0, np.abs(q_true)) * rng.standard_normal(n_samples)
    q_noisy = q_true + noise

    data = np.column_stack(
        [
            s,
            T,
            x,
            np.full(n_samples, phase_1["k_0"]),
            np.full(n_samples, phase_1["alpha"]),
            np.full(n_samples, phase_1["beta"]),
            np.full(n_samples, sigma),
            q_true,
            q_noisy,
            a_true,
            b_true,
        ]
    )

    header = "s,T,x,k_0,alpha,beta,sigma,q_true,q_noisy,a_true,b_true"
    return data, header


def save_dataset(output_dir, name, data, header):
    output_path = output_dir / f"{name}.csv"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    return output_path


def save_metadata(output_dir, name, params):
    output_path = output_dir / f"{name}_metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
        f.write("\n")
    return output_path


def plot_overview(image_path, name, data):
    s = data[:, 0]
    T = data[:, 1]
    q_true = data[:, 7]
    q_noisy = data[:, 8]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter = axes[0].scatter(
        s,
        T,
        c=q_noisy,
        s=8,
        cmap="coolwarm",
        alpha=0.75,
    )
    axes[0].set_title(f"{name}: samples colored by noisy flux")
    axes[0].set_xlabel("s = dT/dx")
    axes[0].set_ylabel("T")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], label="q_noisy")

    T_slice = 1.5
    slice_width = 0.08
    near_slice = np.abs(T - T_slice) < slice_width
    order = np.argsort(s[near_slice])

    axes[1].plot(
        s[near_slice][order],
        q_true[near_slice][order],
        "k.",
        markersize=3,
        alpha=0.8,
        label="q_true near slice",
    )
    axes[1].plot(
        s[near_slice][order],
        q_noisy[near_slice][order],
        "r.",
        markersize=3,
        alpha=0.45,
        label="q_noisy near slice",
    )
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].axvline(0.0, color="k", linewidth=0.8)
    axes[1].set_title(f"{name}: slice near T = {T_slice}")
    axes[1].set_xlabel("s = dT/dx")
    axes[1].set_ylabel("q")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def plot_temperature_slices(image_path, name, data):
    T_slices = [0.5, 1.5, 2.5]
    slice_width = 0.08

    s = data[:, 0]
    T = data[:, 1]
    q_true = data[:, 7]
    q_noisy = data[:, 8]

    fig, axes = plt.subplots(1, len(T_slices), figsize=(15, 4), sharex=True)

    for col, T_slice in enumerate(T_slices):
        ax = axes[col]
        near_slice = np.abs(T - T_slice) < slice_width
        order = np.argsort(s[near_slice])

        ax.plot(
            s[near_slice][order],
            q_true[near_slice][order],
            "k.",
            markersize=3,
            alpha=0.8,
            label="q_true",
        )
        ax.plot(
            s[near_slice][order],
            q_noisy[near_slice][order],
            "r.",
            markersize=3,
            alpha=0.45,
            label="q_noisy",
        )
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_title(f"{name}\nT near {T_slice}")
        ax.set_xlabel("s = dT/dx")
        ax.set_ylabel("q")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def main():
    seed = 7
    rng = np.random.default_rng(seed)
    n_samples = 3000
    name = "nonlinear_phase_transition_medium_noise"

    params = {
        "phase_1": {
            "k_0": 0.7,
            "alpha": 0.15,
            "beta": 0.04,
        },
        "phase_2": {
            "k_0": 1.4,
            "alpha": 0.45,
            "beta": 0.16,
        },
        "transition_temperature": 1.5,
        "transition_width": 0.08,
        "sigma": 0.05,
        "sampling_ranges": {
            "s": [-3.0, 3.0],
            "T": [0.0, 3.0],
            "x": [0.0, 1.0],
        },
        "random_seed": seed,
        "n_samples": n_samples,
    }

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "NoisyDeterministicOracles" / "datasets"
    image_dir = repo_root / "Images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    data, header = make_dataset(params, n_samples, rng)

    dataset_path = save_dataset(output_dir, name, data, header)
    print(f"Saved {dataset_path}")

    metadata_path = save_metadata(output_dir, name, params)
    print(f"Saved {metadata_path}")

    overview_path = image_dir / f"{name}_overview.png"
    plot_overview(overview_path, name, data)
    print(f"Saved {overview_path}")

    slices_path = image_dir / f"{name}_temperature_slices.png"
    plot_temperature_slices(slices_path, name, data)
    print(f"Saved {slices_path}")


if __name__ == "__main__":
    main()
