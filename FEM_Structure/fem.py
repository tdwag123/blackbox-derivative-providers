import numpy as np

'''
Ideally, FEM infrastructure should not change depending on physics, i.e. between heat, 
diffusion, elasticity, nonlinear elasticity, etc. It should only know how to
integrate, assemble, solve. The physics supplies the constitutive law and source terms.
'''

# n = number of elements
# n + 1 = number of nodes

# -------------------------------------------------------------------------------------------------
# Create a uniform mesh with n elements on the interval I = [start, end] and define 
# the corresponding space of continuous piecewise linear functions V_{h,0}.
# ----------------------------------------------------------------------------------------------------
def create_1D_mesh(n: int, start: float, end: float):
    if n <= 0:
        raise ValueError("n should be positive")
    if end <= start:
        raise ValueError("end should be greater than start")

    nodes = np.linspace(start, end, n + 1) # generates an array of evenly spaced numbers over range I
    elements = np.array([[i, i + 1] for i in range(n)], dtype=int) # this array has data type int

    return nodes, elements

# -------------------------------------------------------------------------------------------------
# Note: An n-point Gauss-Legendre quadrature can perfectly integrate any polynomial of degree < 2n.
# Using 2-point Gauss-Legendre quadrature, integrate a function f(x) from a to b.
# ----------------------------------------------------------------------------------------------------
def get_gl2_values(dim: int):
    """
    Returns quadrature nodes and weight values for any positive dimension using Gauss tensor rule.
    For standard interval [-1,1], the two integration pts (nodes) and their corresponding weights are
    Nodes: x_1 = -1/sqrt(3), x_2 = 1/sqrt(3);     Weights: w_1 = 1.0, w_2 = 1.0.
    """

    if dim < 1:
        raise ValueError("dim must be positive")
    
    # computes the sample points and weights required to perform 2-point GL quadrature
    x1D, w1D = np.polynomial.legendre.leggauss(2)
    
    # x1D = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]) 
    # w1D = np.array([1.0, 1.0])

    x1D = x1D.reshape(-1,1)
    quad_pts = x1D # reshapes into 1 column
    weights = w1D

    if dim == 1:
        return quad_pts, weights
    
    two_ones = np.ones((2, 1))
    for k in range(2,dim+1):
        old_quad_pts = quad_pts
        old_weights = weights

        k_ones = np.ones((2 ** (k-1), 1))

        quad_pts = np.hstack((np.kron(x1D, k_ones), np.kron(two_ones, old_quad_pts)))
        weights = np.kron(old_weights, w1D)
    
    return quad_pts, weights


def gl2_quadrature_integration(f, a, b, dim: int):
    """
    Integrate f(x) over [a_1, b_1] x [a_2, b_2] x ... x [a_dim, b_dim] using 2-pt Gauss-Legendre quadrature in
    each coordinate direction (2 points in each direction) by integrating f over the reference domain [-1, 1]^dim.
    """

    quad_pts, weights = get_gl2_values(dim)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != (dim,):
        raise ValueError("a should have length dim")
    if b.shape != (dim,):
        raise ValueError("b should have length dim")
    if np.any(b <= a):
        raise ValueError("each entry of b must > a")

    # we have the xi - x(xi) mapping as follows: x(xi) = (a+b)/2 + [(b-a)/2]*xi
    jacobian = (b - a) / 2.0
    midpoint = (b + a) / 2.0

    # we compute (*) = w_1 * J_1 * f(xi_1) = w_2 * J_2 * f(xi_2)
    # here, J_1 = J_2 = jacobian; w_1 = w_2 = 1; xi_2 = -xi_1 = 1/sqrt(3)
    
    integral = None

    for node, weight in zip(quad_pts, weights):
        transformed_x = midpoint + jacobian * node
        value = np.asarray(f(transformed_x), dtype=float)

        if integral is None:
            integral = weight * value
        else:
            integral = integral + weight * value

    result = np.prod(jacobian) * integral

    # return a float if the result is scalar
    if result.shape == ():
        return float(result)

    return result


def shape_functions(x, nodes, n):
    return 0


def assemble_jacobian():
    return 0


def assemble_residual():
    return 0


def apply_boundary_conditions():
    return 0


def newton_solve():
    return 0
