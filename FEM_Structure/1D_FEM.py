import numpy as np


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




    

