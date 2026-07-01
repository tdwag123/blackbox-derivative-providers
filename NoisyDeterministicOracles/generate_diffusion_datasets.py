from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def physical_flux(s, T, k_0, alpha, beta):
    return -(k_0 * (1.0 + alpha * T**2) + beta * s**2) * s


def physical_flux_derivatives(s, T, k_0, alpha, beta):
    a_true = -k_0 * (1.0 + alpha * T**2) - 3.0 * beta * s**2
    b_true = -2.0 * k_0 * alpha * T * s
    return a_true, b_true


def make_dataset(name, params, n_samples, rng):
    s = rng.uniform(-3.0, 3.0, n_samples)
    T = rng.uniform(0.0, 3.0, n_samples)
    x = rng.uniform(0.0, 1.0, n_samples)

    k_0 = params["k_0"]
    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]

    q_true = physical_flux(s, T, k_0, alpha, beta)
    a_true, b_true = physical_flux_derivatives(s, T, k_0, alpha, beta)

    noise = sigma * np.maximum(1.0, np.abs(q_true)) * rng.standard_normal(n_samples)
    q_noisy = q_true + noise

    data = np.column_stack(
        [
            s,
            T,
            x,
            np.full(n_samples, k_0),
            np.full(n_samples, alpha),
            np.full(n_samples, beta),
            np.full(n_samples, sigma),
            q_true,
            q_noisy,
            a_true,
            b_true,
        ]
    )

    header = "s,T,x,k_0,alpha,beta,sigma,q_true,q_noisy,a_true,b_true"
    return name, data, header


def save_dataset(output_dir, name, data, header):
    output_path = output_dir / f"{name}.csv"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    return output_path


def plot_overview(image_path, datasets):
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 12))

    for row, (name, data, _) in enumerate(datasets):
        s = data[:, 0]
        T = data[:, 1]
        q_true = data[:, 7]
        q_noisy = data[:, 8]

        scatter = axes[row, 0].scatter(
            s,
            T,
            c=q_noisy,
            s=8,
            cmap="coolwarm",
            alpha=0.75,
        )
        axes[row, 0].set_title(f"{name}: samples colored by noisy flux")
        axes[row, 0].set_xlabel("s = dT/dx")
        axes[row, 0].set_ylabel("T")
        axes[row, 0].grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=axes[row, 0], label="q_noisy")

        T_slice = 1.5
        slice_width = 0.08
        near_slice = np.abs(T - T_slice) < slice_width
        order = np.argsort(s[near_slice])

        axes[row, 1].plot(
            s[near_slice][order],
            q_true[near_slice][order],
            "k.",
            markersize=3,
            alpha=0.8,
            label="q_true near slice",
        )
        axes[row, 1].plot(
            s[near_slice][order],
            q_noisy[near_slice][order],
            "r.",
            markersize=3,
            alpha=0.45,
            label="q_noisy near slice",
        )
        axes[row, 1].axhline(0.0, color="k", linewidth=0.8)
        axes[row, 1].axvline(0.0, color="k", linewidth=0.8)
        axes[row, 1].set_title(f"{name}: slice near T = {T_slice}")
        axes[row, 1].set_xlabel("s = dT/dx")
        axes[row, 1].set_ylabel("q")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(image_path, dpi=150)


def plot_temperature_slices(image_path, datasets):
    T_slices = [0.5, 1.5, 2.5]
    slice_width = 0.08

    fig, axes = plt.subplots(
        len(datasets),
        len(T_slices),
        figsize=(15, 11),
        sharex=True,
    )

    for row, (name, data, _) in enumerate(datasets):
        s = data[:, 0]
        T = data[:, 1]
        q_true = data[:, 7]
        q_noisy = data[:, 8]

        for col, T_slice in enumerate(T_slices):
            ax = axes[row, col]
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
            if row == 0 and col == 0:
                ax.legend()

    fig.tight_layout()
    fig.savefig(image_path, dpi=150)


def main():
    rng = np.random.default_rng(7)
    n_samples = 3000

    configs = [
        (
            "nonlinear_high_noise",
            {"k_0": 1.0, "alpha": 0.5, "beta": 0.2, "sigma": 0.10},
        ),
        (
            "nonlinear_low_noise",
            {"k_0": 0.7, "alpha": 0.2, "beta": 0.05, "sigma": 0.02},
        ),
        (
            "linear_medium_noise",
            {"k_0": 1.5, "alpha": 0.0, "beta": 0.0, "sigma": 0.05},
        ),
    ]

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "NoisyDeterministicOracles" / "datasets"
    image_dir = repo_root / "Images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name, params in configs:
        dataset = make_dataset(name, params, n_samples, rng)
        datasets.append(dataset)
        path = save_dataset(output_dir, *dataset)
        print(f"Saved {path}")

    image_path = image_dir / "diffusion_datasets_overview.png"
    plot_overview(image_path, datasets)
    print(f"Saved {image_path}")

    slices_path = image_dir / "diffusion_datasets_temperature_slices.png"
    plot_temperature_slices(slices_path, datasets)
    print(f"Saved {slices_path}")


if __name__ == "__main__":
    main()
