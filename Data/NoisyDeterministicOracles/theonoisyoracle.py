import numpy as np


def deterministic_oracle(s, T, x, m):
    """
    Return a deterministic flux law that is purposefully noisy.
    s: gradient dT/dx
    T: temperature
    x: spatial coord
    m: param vector [k_0,alpha,beta,epsilon,a_s,a_t,a_x]
    Output: q
    """
    if len(m) != 7:
        raise ValueError("m must have length 7: [k_0, alpha, beta, epsilon, a_s, a_t, a_x]")

    k_0 = m[0]
    alpha = m[1]
    beta = m[2]
    epsilon = m[3] # noise amplitude
    a_s = m[4] # noise scale in s
    a_t = m[5] # noise scale in T
    a_x = m[6] # noise scale in x

    q_true = -k_0 * (1 + alpha * T * T) + beta * s * s * s

    z = a_s * s + a_t * T + a_x * x

    #change these as you wish :>
    c = 12.9898
    d = 43758.5453

    h = np.sin(c * z) * d
    h = h - np.floor(h)

    n_tilde = 2 * h - 1
    delta = epsilon * np.abs(q_true) * n_tilde

    q = q_true + delta

    return q


def oracle_eval(s, T, x, m):
    return deterministic_oracle(s, T, x, m)


def example_problem():
    """
    Problem with the following params:
    k_0     = 1.0
    alpha   = 5.0
    beta    = 0.1
    epsilon = 0.05
    a_s     = 3.1
    a_T     = 1.7
    a_x     = 2.3
    """
    m = np.array([1.0, 5.0, 0.1, 0.05, 3.1, 1.7, 2.3])

    s = -0.25
    T = 2.0
    x = 0.4

    q1 = deterministic_oracle(s, T, x, m)
    q2 = deterministic_oracle(s, T, x, m)

    print("q:")
    print(q1)
    print("\nRepeated call with same inputs:")
    print(q2)
    print("\nDifference:")
    print(q2 - q1)


if __name__ == "__main__":
    example_problem()
