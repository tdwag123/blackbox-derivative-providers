import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.signal import savgol_filter, convolve

def make_noisy_sine_data(n=200, noise_std=0.1,seed=0):
    """
    Makes some noisy sine data
    """
    rng = np.random.default_rng(seed)
    x= np.linspace(0, 4*np.pi, n)
    y_true = np.sin(x)
    dy_true = np.cos(x)
    noise = rng.normal(0, noise_std, size=n)
    y_noisy = y_true + noise
    return x, y_noisy, y_true, dy_true

def finite_difference_uniform(x, y):
    """
    Computes FD with even spacing h
    """
    h=x[1]-x[0]
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2])/(2*h)
    dy[0] = (y[1]-y[0])/h
    dy[-1] = (y[-1]-y[-2])/h
    return dy

def smooth_with_kernel(y, kernel):
    """
    Smooths y with a kernel using convolution
    """
    kernel = np.asarray(kernel, dtype=float)
    kernel = kernel / np.sum(kernel)  # Normalize the kernel
    return convolve(y, kernel, mode='same')

def rmse(a,b):
    return np.sqrt(np.mean((a-b)**2))


def candidate_kernels():
    return {
        "box5": np.ones(5),
        "box7": np.ones(7),
        "box9": np.ones(9),
        "binomial5": np.array([1, 4, 6, 4, 1]),
        "binomial7": np.array([1, 6, 15, 20, 15, 6, 1]),
        "binomial9": np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]),
        "triangular5": np.array([1, 2, 3, 2, 1]),
        "triangular7": np.array([1, 2, 3, 4, 3, 2, 1]),
        "triangular9": np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
        "triangular11": np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]),
    }


def run_experiment(
        n=200,
        noise_std=0.1,
        seed=12
):
    x, y_noisy, y_true, dy_true = make_noisy_sine_data(n=n, noise_std=noise_std, seed=seed)

    start = perf_counter()
    dy_fd_raw = finite_difference_uniform(x, y_noisy)
    raw_fd_time_s = perf_counter() - start
    kernels = candidate_kernels()
    kernel_results = []

    for name, kernel in kernels.items():
        start = perf_counter()
        dy_fd_then_smooth = smooth_with_kernel(dy_fd_raw, kernel)
        fd_then_smooth_time_s = raw_fd_time_s + perf_counter() - start

        start = perf_counter()
        y_smooth = smooth_with_kernel(y_noisy, kernel)
        dy_smooth_then_fd = finite_difference_uniform(x, y_smooth)
        smooth_then_fd_time_s = perf_counter() - start
        kernel_results.append(
            {
                "kernel": name,
                "width": len(kernel),
                "kernel_weights": kernel / np.sum(kernel),
                "fd_then_smooth_rmse": rmse(dy_fd_then_smooth, dy_true),
                "smooth_then_fd_rmse": rmse(dy_smooth_then_fd, dy_true),
                "fd_then_smooth_time_us": 1.0e6 * fd_then_smooth_time_s,
                "smooth_then_fd_time_us": 1.0e6 * smooth_then_fd_time_s,
                "dy_fd_then_smooth": dy_fd_then_smooth,
                "dy_smooth_then_fd": dy_smooth_then_fd,
                "y_smooth": y_smooth,
            }
        )

    best_fd_then_smooth = min(kernel_results, key=lambda row: row["fd_then_smooth_rmse"])
    best_smooth_then_fd = min(kernel_results, key=lambda row: row["smooth_then_fd_rmse"])

    # ------------------------------------------------------------
    # Method C:
    # Savitzky-Golay derivative
    # ------------------------------------------------------------
    h = x[1] - x[0]

    # window_length must be odd
    window_length = 15
    polyorder = 3

    start = perf_counter()
    dy_savgol = savgol_filter(
        y_noisy,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=h,
        mode="interp",
    )
    savgol_time_us = 1.0e6 * (perf_counter() - start)

    # ------------------------------------------------------------
    # Print errors
    # ------------------------------------------------------------
    print("Derivative RMSE against true cos(x)")
    print("-----------------------------------")
    print(f"Raw finite difference:          {rmse(dy_fd_raw, dy_true):.6f}  {1.0e6 * raw_fd_time_s:.2f} us")
    print(f"Savitzky-Golay derivative:      {rmse(dy_savgol, dy_true):.6f}  {savgol_time_us:.2f} us")
    print()
    print("Convolution kernel sweep")
    print("------------------------")
    print("kernel        width  FD-then-smooth   time us   smooth-then-FD   time us")
    for row in sorted(kernel_results, key=lambda item: item["fd_then_smooth_rmse"]):
        print(
            f"{row['kernel']:<13} {row['width']:>5}  "
            f"{row['fd_then_smooth_rmse']:>14.6f} "
            f"{row['fd_then_smooth_time_us']:>9.2f}   "
            f"{row['smooth_then_fd_rmse']:>14.6f} "
            f"{row['smooth_then_fd_time_us']:>9.2f}"
        )
    print()
    print(
        "Best FD then smooth: "
        f"{best_fd_then_smooth['kernel']} "
        f"RMSE={best_fd_then_smooth['fd_then_smooth_rmse']:.6f}"
    )
    print(
        "Best smooth then FD: "
        f"{best_smooth_then_fd['kernel']} "
        f"RMSE={best_smooth_then_fd['smooth_then_fd_rmse']:.6f}"
    )

    # ------------------------------------------------------------
    # Plot noisy data
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(x, y_true, label="true sin(x)", linewidth=2)
    plt.scatter(x, y_noisy, s=12, alpha=0.6, label="noisy samples")
    plt.plot(
        x,
        best_smooth_then_fd["y_smooth"],
        label=f"smoothed noisy data ({best_smooth_then_fd['kernel']})",
        linewidth=2,
    )
    plt.title("Noisy sine data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noisy_sine_data.png")

    # ------------------------------------------------------------
    # Plot derivative estimates
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(x, dy_true, label="true derivative cos(x)", linewidth=3)
    plt.plot(x, dy_fd_raw, label="raw finite difference", alpha=0.5)
    plt.plot(
        x,
        best_fd_then_smooth["dy_fd_then_smooth"],
        label=f"FD then smooth ({best_fd_then_smooth['kernel']})",
        linewidth=2,
    )
    plt.plot(
        x,
        best_smooth_then_fd["dy_smooth_then_fd"],
        label=f"smooth then FD ({best_smooth_then_fd['kernel']})",
        linewidth=2,
    )
    plt.plot(x, dy_savgol, label="Savitzky-Golay derivative", linewidth=2)
    plt.title("Derivative estimates")
    plt.xlabel("x")
    plt.ylabel("dy/dx")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("derivative_estimates.png")
    plt.savefig("derivative_estimates_revised.png")

    return {
        "x": x,
        "y_noisy": y_noisy,
        "y_true": y_true,
        "dy_true": dy_true,
        "dy_fd_raw": dy_fd_raw,
        "kernel_results": kernel_results,
        "best_fd_then_smooth": best_fd_then_smooth,
        "best_smooth_then_fd": best_smooth_then_fd,
        "dy_savgol": dy_savgol,
        "raw_fd_time_s": raw_fd_time_s,
        "savgol_time_us": savgol_time_us,
    }


if __name__ == "__main__":
    results = run_experiment()
