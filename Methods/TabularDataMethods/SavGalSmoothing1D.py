import numpy as np
import matplotlib.pyplot as plt

def savitsky_galoy1D(x_vals, y_vals, stencil_size, poly_order):
    """
    Do Savitsky-Galoy smoothing on 1D data
    """
    N = len(x_vals)
    W = stencil_size
    P = poly_order
    half = W // 2

    new_y = np.zeros(N)
    for i in range(N):
        if i < half or i >= N - half:
            new_y[i] = y_vals[i]
            continue
    
        x_window = x_vals[i - half:i + half + 1]
        y_window = y_vals[i - half:i + half + 1]

        x_center = x_vals[i]
        x_shifted = x_window - x_center

        A = np.zeros((W, P + 1))
        for row in range(W):
            for col in range(P + 1):
                A[row, col] = x_shifted[row] ** col
        
        beta, _, _, _ = np.linalg.lstsq(A, y_window, rcond=None)

        new_y[i] = beta[0]

    return new_y

# --- Test usage ---
if __name__ == "__main__":
    # 1. Create synthetic data: a smooth sine wave with added noise
    np.random.seed(42)  # for reproducible results
    x = np.linspace(0, 4 * np.pi, 200)          # 200 points from 0 to 4π
    y_true = np.sin(x) + 0.5 * np.cos(2 * x)    # clean signal
    noise = 0.3 * np.random.randn(len(x))       # Gaussian noise
    y_noisy = y_true + noise                    # noisy measurements

    # 2. Apply the Savitzky–Golay filter
    window = 21          # odd number, larger = smoother
    order = 3            # cubic polynomial fits well for smooth curves
    y_smoothed = savitsky_galoy1D(x, y_noisy, window, order)

    # 3. Print some numerical comparisons (first 10 points)
    print("First 10 points:")
    print("Index  |  Noisy  |  Smoothed  |  True")
    for i in range(10):
        print(f"{i:5d}  | {y_noisy[i]:7.4f} | {y_smoothed[i]:9.4f} | {y_true[i]:6.4f}")

    # 4. Compute and print overall improvement (noise reduction)
    noise_std = np.std(y_noisy - y_true)
    residual_std = np.std(y_smoothed - y_true)
    print(f"\nNoise standard deviation (raw):  {noise_std:.4f}")
    print(f"Noise standard deviation (filtered): {residual_std:.4f}")
    print(f"Noise reduced by: {100 * (1 - residual_std/noise_std):.1f}%")

    # 5. Plot the results
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true, 'k--', label='True signal', linewidth=1.5)
    plt.plot(x, y_noisy, 'b.', label='Noisy data', markersize=3, alpha=0.5)
    plt.plot(x, y_smoothed, 'r-', label='Savitzky–Golay filtered', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Savitzky–Golay test (window={window}, order={order})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot instead of showing (works in all environments)
    plt.savefig('savitzky_golay_test.png', dpi=150)
    print("\nPlot saved as 'savitzky_golay_test.png'")

    # If you're in an interactive environment, uncomment the next line:
    # plt.show()