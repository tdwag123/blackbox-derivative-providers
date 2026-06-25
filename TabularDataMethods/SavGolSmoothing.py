import numpy as np

# NOTE THIS DOES NOT NORMALIZE THE DATA. PLEASE NORMALIZE BEFORE USING

def quadratic_basis(eta):
    """
    Add a comment later
    """
    eta = np.asarray(eta, dtype=float)
    p = len(eta)

    phi = [1.0]
    for i in range(p):
        phi.append(eta[i])
    
    for i in range(p):
        phi.append(eta[i] ** 2)
    
    for i in range(p):
        for j in range(i + 1, p):
            phi.append(eta[i] * eta[j])
    
    return np.array(phi)

def quadratic_savgol_smoothing(z_data, q_data, K, h):
    """
    For example
    z_i = [g_x, T]
    or
    z_i = [g_x, g_y, g_z, T, m_1, ..., m_n]

    q_data array of data w shape (N,)

    K: # of neighbors
    """
    z_data = np.asarray(z_data, dtype=float)
    q_data = np.asarray(q_data, dtype=float)

    N = z_data.shape[0]
    q_smooth = np.zeros(N)

    for i in range (N):
        z0= z_data[i]

        distances = np.linalg.norm(z_data - z0, axis=1)

        neighbor_indices = np.argpartition(distances, K - 1)[:K]

        Phi_rows = []
        q_near = []
        weights = []

        for j in neighbor_indices:
            eta = z_data[j] - z0
            distance = np.linalg.norm(eta)
            weight = np.exp(-(distance / h) ** 2)

            phi = quadratic_basis(eta)

            Phi_rows.append(phi)
            q_near.append(q_data[j])
            weights.append(weight)
        
        Phi = np.array(Phi_rows)
        q_near = np.array(q_near)
        weights = np.array(weights)

        sqrt_weights = np.sqrt(weights)

        Phi_weighted = Phi * sqrt_weights[:, None]
        q_weighted = q_near * sqrt_weights
        beta, _, _, _ = np.linalg.lstsq(Phi_weighted, q_weighted, rcond=None)

        q_smooth[i] = beta[0]

    return q_smooth



if __name__ == "__main__":
    np.random.seed(0)

    N = 100

    gx = np.linspace(-2, 2, N)
    T = np.linspace(0, 5, N)

    z_data = np.column_stack([gx, T])

    q_true = 2.0 + 0.5 * gx - 0.2 * T + 0.1 * gx**2
    q_noisy = q_true + 0.2 * np.random.randn(N)

    q_smooth = quadratic_savgol_smoothing(
        z_data=z_data,
        q_data=q_noisy,
        K=15,
        h=1.0,
    )

    print("First 10 values:")
    print("index | noisy | smooth | true")

    for i in range(10):
        print(f"{i:5d} | {q_noisy[i]:7.3f} | {q_smooth[i]:7.3f} | {q_true[i]:7.3f}")
