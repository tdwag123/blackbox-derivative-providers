import numpy as np


def thomas_solve(lower, diagonal, upper, rhs):
    """
    Solve a tridiagonal linear system with the Thomas algorithm.
    """
    lower = np.asarray(lower, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    n = len(diagonal)
    gamma = np.zeros(n, dtype=float)
    rho = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    gamma[0]= upper[0] / diagonal[0]
    rho[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denom = diagonal[i] - lower[i - 1] * gamma[i - 1]
        gamma[i] = upper[i] / denom if i < n - 1 else 0.0
        rho[i] = (rhs[i] - lower[i - 1] * rho[i - 1]) / denom

    # Back substitution
    x[-1] = rho[-1]
    for i in range(n - 2, -1, -1):
        x[i] = rho[i] - gamma[i] * x[i + 1]

    return x


def tridiagonal_diagonals(matrix):
    """Return lower, diagonal, upper arrays from a square tridiagonal matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    return np.diag(matrix, -1), np.diag(matrix), np.diag(matrix, 1)


def example_problem():
    # Example: start from a hand-written tridiagonal matrix, extract its
    # three diagonals, then solve with the Thomas algorithm.
    matrix = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.3, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 4.0],
        ]
    )
    rhs = np.array([5.0, 6.0, 10.6, 23.0])

    lower, diagonal, upper = tridiagonal_diagonals(matrix)
    solution = thomas_solve(lower, diagonal, upper, rhs)
    numpy_solution = np.linalg.solve(matrix, rhs)
    residual = matrix @ solution - rhs

    print("Matrix A:")
    print(matrix)
    print("\nExtracted lower diagonal:")
    print(lower)
    print("\nExtracted main diagonal:")
    print(diagonal)
    print("\nExtracted upper diagonal:")
    print(upper)
    print("\nRight-hand side b:")
    print(rhs)
    print("\nThomas solution:")
    print(solution)
    print("\nNumPy solution:")
    print(numpy_solution)
    print("\nResidual A @ solution - b:")
    print(residual)


if __name__ == "__main__":
    example_problem()
