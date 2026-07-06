import numpy as np

def GLquadrature_twoPoint(f, a, b):
    '''two-point GL quadrature over [a,b]'''
    
    # rmk: the xi nodes in [-1,1] and weights are standard results for two point quadrature
    nodes = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    weights = [1.0, 1.0]

    # we have the xi - x(xi) mapping as follows: x(xi) = (a+b)/2 + [(b-a)/2]*xi
    jacobian = (b - a) / 2.0
    midpoint = (a + b) / 2.0

    # we compute (*) = w_1 * J_1 * f(xi_1) = w_2 * J_2 * f(xi_2)
    # here, J_1 = J_2 = jacobian; w_1 = w_2 = 1; xi_2 = -xi_1 = 1/sqrt(3)
    
    integral = None

    for node, weight in zip(nodes, weights):
        transformed_x = midpoint + jacobian * node
        value = np.asarray(f(transformed_x), dtype=float)

        if integral is None:
            integral = weight * value
        else:
            integral = integral + weight * value

    result = jacobian * integral

    # resturn a float if the result is scalar
    if result.shape == ():
        return float(result)

    return result


def constructTridiag(lower, diag, upper, e: int, Ke):
    '''Given element e over [x_e, x_e+1] with associated Ke stiffness matrix, construct the global tridiagonal matrix'''
    diag[e] += Ke[0,0]
    upper[e] += Ke[0,1]
    lower[e] += Ke[1,0]
    diag[e+1] += Ke[1,1]


def thomas_solve(lower, diagonal, upper, rhs):
    '''Solve a tridiagonal linear system with the Thomas algorithm.'''
    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    n = len(diagonal)

    # error check
    if n == 0:
        return np.array([], dtype = float)

    if n == 1: 
        if abs(diagonal[0]) < 1.0e-14:
            raise ZeroDivisionError("Zero pivot in 1x1 tridiagonal solve")
        return np.array([rhs[0]/diagonal[0]], dtype=float)
        
    gamma = np.zeros(n, dtype=float)
    rho = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    # error check
    if abs(diagonal[0]) < 1.0e-14:
        raise ZeroDivisionError("Zero pivot at row 0.")

    gamma[0]= upper[0] / diagonal[0]
    rho[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        
        denom = diagonal[i] - lower[i - 1] * gamma[i - 1]

        # error check 
        if abs(denom) < 1.0e-14:
            raise ZeroDivisionError(f"Zero pivot at row {i}.")
            
        gamma[i] = upper[i] / denom if i < n - 1 else 0.0
        rho[i] = (rhs[i] - lower[i - 1] * rho[i - 1]) / denom

    # Back substitution
    x[-1] = rho[-1]
    for i in range(n - 2, -1, -1):
        x[i] = rho[i] - gamma[i] * x[i + 1]

    return x


def tridiag_block(lower, diag, upper, start, end):
    '''extracts tridiagonal block when we are looking at Dirichlet BC'''
    # recall: with Dirichlet BC, don's solve for boundary values
    # if T(0) = TL, T(L) = TR, U0 = TL, UN = TR
    diag_f = diag[start:end].copy()
    lower_f = lower[start:end - 1].copy()
    upper_f = upper[start:end - 1].copy()
    return lower_f, diag_f, upper_f