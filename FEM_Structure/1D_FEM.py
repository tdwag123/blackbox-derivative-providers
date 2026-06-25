import numpy as np

'''
Ideally, FEM infrastructure should not change depending on physics, i.e. between heat, 
diffusion, elasticity, nonlinear elasticity, etc. It should only know how to
integrate, assemble, solve. The physics supplies the constitutive law and source terms.
'''

# -------------------------------------------------------------------------------------------------
# Create a uniform mesh with n elements on the interval I = [start, end] and define 
# the corresponding space of continuous piecewise linear functions V_{h,0}.
# ----------------------------------------------------------------------------------------------------
def create_mesh(n: int, start: float, end: float):
    nodes = np.linspace(start, end, n + 1) # generates an array of evenly spaced numbers over range I

    elements = np.array([
        [i, i + 1]
        for i in range(n)
    ], dtype=int) # this array has data type int

    return nodes, elements


def gauss_quadrature():
    return 0


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
