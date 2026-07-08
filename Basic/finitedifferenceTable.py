import numpy as np


def centered_flux_derivatives(gx_q, T_q, phi_h, h):
    """
    Compute centered finite-difference derivatives of q = phi_h(gx, T).
    """
    gx_q = np.asarray(gx_q, dtype=float)
    T_q = np.asarray(T_q, dtype=float)

    if gx_q.shape != T_q.shape:
        raise ValueError("gx_q and T_q must have the same shape")
    if h <= 0:
        raise ValueError("h must be positive")

    N_q = len(gx_q)
    a_q = np.zeros(N_q)
    b_q = np.zeros(N_q)

    for g in range(N_q):
        gx_g = gx_q[g]
        T_g = T_q[g]

        h_gx = h * max(1.0, abs(gx_g))
        h_T = h * max(1.0, abs(T_g))

        q_gx_plus = phi_h(gx_g + h_gx, T_g)
        q_gx_minus = phi_h(gx_g - h_gx, T_g)
        a_q[g] = (q_gx_plus - q_gx_minus) / (2 * h_gx)

        q_T_plus = phi_h(gx_g, T_g + h_T)
        q_T_minus = phi_h(gx_g, T_g - h_T)
        b_q[g] = (q_T_plus - q_T_minus) / (2 * h_T)

    return a_q, b_q


def example_problem():
    def phi_h(gx, T):
        return -0.5 * gx + 0.2 * T**2

    gx_q = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    T_q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    h = 1.0e-6

    a_q, b_q = centered_flux_derivatives(gx_q, T_q, phi_h, eta)

    print("gx_q:")
    print(gx_q)
    print("\nT_q:")
    print(T_q)
    print("\ndq/dgx:")
    print(a_q)
    print("\ndq/dT:")
    print(b_q)


if __name__ == "__main__":
    example_problem()
