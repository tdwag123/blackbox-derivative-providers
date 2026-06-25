import reiyah_linear1D_FEM as fem
import matplotlib.pyplot as plt


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

    e, nodes = fem.linear_solve(start, end, n, kappa, a, f, g)
    plt.plot(nodes, e, label='Numerical Solution')
    plt.show()