def q_true(s, T, k0, alpha, beta):
    return -(k0 * (1.0 + alpha * T**2) + beta * s**2) * s


def a_true(s, T, k0, alpha, beta):
    return -k0 * (1.0 + alpha * T**2) - 3.0 * beta * s**2


def b_true(s, T, k0, alpha, beta):
    return -2.0 * k0 * alpha * T * s