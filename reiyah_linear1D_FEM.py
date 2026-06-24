import numpy as np
import matplotlib.pyplot as plt


# 1) -------------------------------------------------------------------------------------------------
# Create a mesh with n elements on the interval I and define the corresponding space of continuous
# piecewise linear functions V_{h,0}.
# ----------------------------------------------------------------------------------------------------

def create_mesh(n: int, start: float, end: float):
    # Create a uniform mesh with n elements on the interval I = [start, end]
    nodes = np.linspace(start, end, n + 1) # generates an array of evenly spaced numbers over range I

    return nodes


def quadrature():
    return 0


def shape_functions(x, nodes, n):
    return 0


def assemble_Jacobian():
    return 0


def assemble_residual():
    return 0


def boundary_conditions():
    return 0


def newton_solve():
    return 0







# 2) -------------------------------------------------------------------------------------------------
# Solve the linear system Ae = b, where A is the stiffness matrix and b is the load vector. 
# Use appropriate boundary conditions for the problem.
# ----------------------------------------------------------------------------------------------------

def construct_stiffness_matrix(n:int, nodes, kappa, a):
    A = np.zeros((n + 1, n + 1))  # stiffness matrix

    for i in range(n):
        h = nodes[i+1] - nodes[i]  # element length
        xmid = (nodes[i] + nodes[i+1]) / 2  # midpoint of the element
        amid = a(xmid)  # value of a(x) at the midpoint
        A[i, i] += amid / h
        A[i, i+1] -= amid / h
        A[i+1, i] -= amid / h
        A[i+1, i+1] += amid / h

    A[0,0] += kappa[0]
    A[n, n] += kappa[1]

    return A


def construct_load_vector(n:int, nodes, f):
    b = np.zeros(n + 1)  # load vector

    for i in range(n):
        h = nodes[i+1] - nodes[i]  # element length
        xmid = (nodes[i] + nodes[i+1]) / 2  # midpoint of the element
        fmid = f(xmid)  # value of f(x) at the midpoint
        b[i] += fmid * h / 2
        b[i+1] += fmid * h / 2

    return b


def linear_solve(start, end, n:int, kappa, a, f, g):
    nodes = create_mesh(n, start, end)
    A = construct_stiffness_matrix(n, nodes, kappa, a)
    b = construct_load_vector(n, nodes, f)

    # Apply boundary conditions
    b[0] += kappa[0] * g[0]  # u(0) = g[0]
    b[n] += kappa[1] * g[1]  # u(L) = g[1]

    # Solve the linear system Ae = b
    e = np.linalg.solve(A, b)

    return e, nodes


def example_problem():
    start = 2.0
    end = 8.0
    n = int((end - start)/ 0.1);  # number of elements
    kappa = [10**6, 0];  # boundary condition coefficients
    g = [-1.0, 0.0];  # boundary condition values

    
    def a(x):
        return 0.1 * (5 - 0.6 * x)  # conductivity function a(x)  [thermal conductivity times area]
    
    def f(x):
        return 0.03 * (x - 6)**4  # heat source function f(x)

    e, nodes = linear_solve(start, end, n, kappa, a, f, g)
    plt.plot(nodes, e, label='Numerical Solution')
    plt.show()

    

if __name__ == "__main__":
    example_problem()
