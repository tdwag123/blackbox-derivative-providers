import numpy as np 
from newton_nd import NM
from darcy import darcy_anisotropic, darcy_isotropic

'''
Test bench for n-dimensional FEM/Newton's method.
'''

def dim1test():
    x = np.linspace(0.0, 1.0, 11)
    boundary_points = [x]

    flux_law = lambda gradT, T, pt: (-gradT, -np.eye(1), np.zeros(1))
    source_fn = lambda T, pt: 1.0
    dsource_dT = lambda T, pt: 0.0

    boundary_conditions = {
        "xmin": ("dirichlet", 0.0),
        "xmax": ("dirichlet", 0.0),
    }

    U, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True)
    U_exact = 0.5 * x * (1.0 - x)

    print(f"U: \n{U}")
    print(f"U_exact: \n{U_exact}")

    print(f"error norm: {np.linalg.norm(U - U_exact)}")


def dim2test():
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 5)
    boundary_points = [x, y]

    exact = lambda pt: 1.0 + 2.0 * pt[0] - 3.0 * pt[1]

    flux_law = lambda gradT, T, pt: (-gradT, -np.eye(2), np.zeros(2))
    source_fn = lambda T, pt: 0.0
    dsource_dT = lambda T, pt: 0.0

    boundary_conditions = {
        "xmin": ("dirichlet", exact),
        "xmax": ("dirichlet", exact),
        "ymin": ("dirichlet", exact),
        "ymax": ("dirichlet", exact),
    }

    U, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True, return_grid=True)

    U_exact = np.array([exact((x[i], y[j])) for i in range(len(x)) for j in range(len(y))])

    print(f"U grid: \n{U}")
    print(f"U_exact: \n{U_exact}")
    print("error norm:", np.linalg.norm(U.reshape(-1) - U_exact))


def darcy2dtest():
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 9)
    boundary_points = [x, y]

    P_left = 1.0
    P_right = 0.0
    Lx = x[-1] - x[0]

    def exact(pt):
        return P_left + (P_right - P_left) * (pt[0] - x[0]) / Lx

    flux_law = darcy_isotropic
    source_fn = lambda P, pt: 0.0
    dsource_dT = lambda P, pt: 0.0

    boundary_conditions = {
        "xmin": ("dirichlet", P_left),
        "xmax": ("dirichlet", P_right),
        "ymin": ("neumann", 0.0),
        "ymax": ("neumann", 0.0),
    }

    P, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True,)

    P_exact = np.array([exact((x[i], y[j])) for i in range(len(x)) for j in range(len(y))])

    print(f"P:\n{P}")
    print(f"P_exact:\n{P_exact}")

    print("error norm:", np.linalg.norm(P - P_exact))


def darcy_anisotropic_2dtest():
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(0.0, 1.0, 6)
    boundary_points = [x, y]

    amp = 0.2

    def exact(pt):
        x_pt, y_pt = pt
        return (
            1.0
            - 0.8 * x_pt
            + 0.3 * y_pt
            + amp * np.sin(np.pi * x_pt) * np.sin(np.pi * y_pt)
        )

    def grad_exact(pt):
        x_pt, y_pt = pt
        return np.array([
            -0.8 + amp * np.pi * np.cos(np.pi * x_pt) * np.sin(np.pi * y_pt),
            0.3 + amp * np.pi * np.sin(np.pi * x_pt) * np.cos(np.pi * y_pt),
        ])

    def hess_exact(pt):
        x_pt, y_pt = pt
        diagonal = -amp * np.pi**2 * np.sin(np.pi * x_pt) * np.sin(np.pi * y_pt)
        off_diagonal = amp * np.pi**2 * np.cos(np.pi * x_pt) * np.cos(np.pi * y_pt)
        return np.array([
            [diagonal, off_diagonal],
            [off_diagonal, diagonal],
        ])

    flux_law = darcy_anisotropic

    def source_fn(P, pt):
        _, du_dgradp, _ = flux_law(grad_exact(pt), exact(pt), pt)
        return float(np.sum(du_dgradp * hess_exact(pt)))

    dsource_dT = lambda P, pt: 0.0

    boundary_conditions = {
        "xmin": ("dirichlet", exact),
        "xmax": ("dirichlet", exact),
        "ymin": ("dirichlet", exact),
        "ymax": ("dirichlet", exact),
    }

    P, history, iters = NM(
        boundary_points,
        flux_law,
        source_fn,
        dsource_dT,
        boundary_conditions,
        verbose=True,
    )
    P_exact = np.array([exact((x[i], y[j])) for i in range(len(x)) for j in range(len(y))])

    print(f"P:\n{P}")
    print(f"P_exact:\n{P_exact}")
    print("error norm:", np.linalg.norm(P - P_exact))
    print("iterations:", iters)


def nonlinear_flux(alpha=0.25, beta=0.10):
    def flux_law(gradT, T, pt):
        gradT = np.asarray(gradT, dtype=float)
        dim = len(gradT)
        k = 1.0 + alpha * T * T + beta * np.dot(gradT, gradT)

        q = -k * gradT
        dq_dgrad = -k * np.eye(dim) - 2.0 * beta * np.outer(gradT, gradT)
        dq_dT = -2.0 * alpha * T * gradT

        return q, dq_dgrad, dq_dT

    return flux_law


def mixed_bc_1dtest():
    x = np.linspace(0.0, 1.0, 11)
    boundary_points = [x]

    alpha = 0.25
    beta = 0.10
    flux_law = nonlinear_flux(alpha=alpha, beta=beta)
    source_fn = lambda T, pt: -2.0 - 24.0 * beta * T - 10.0 * alpha * T * T
    dsource_dT = lambda T, pt: -24.0 * beta - 20.0 * alpha * T

    boundary_conditions = {
        "xmin": ("dirichlet", 0.0),
        "xmax": ("neumann", -(2.0 + 2.0 * alpha + 8.0 * beta)),
    }

    U, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True)
    U_exact = x * x

    print(f"U:\n{U}")
    print(f"U_exact:\n{U_exact}")
    print("error norm:", np.linalg.norm(U - U_exact))


def mixed_bc_2dtest():
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 5)
    boundary_points = [x, y]

    exact = lambda pt: pt[0] * pt[0]

    alpha = 0.25
    beta = 0.10
    flux_law = nonlinear_flux(alpha=alpha, beta=beta)
    source_fn = lambda T, pt: -2.0 - 24.0 * beta * T - 10.0 * alpha * T * T
    dsource_dT = lambda T, pt: -24.0 * beta - 20.0 * alpha * T

    boundary_conditions = {
        "xmin": ("dirichlet", 0.0),
        "xmax": ("dirichlet", 1.0),
        "ymin": ("neumann", 0.0),
        "ymax": ("neumann", 0.0),
    }

    U, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True)
    U_exact = np.array([exact((x[i], y[j])) for i in range(len(x)) for j in range(len(y))])

    print(f"U:\n{U}")
    print(f"U_exact:\n{U_exact}")
    print("error norm:", np.linalg.norm(U - U_exact))


def mixed_bc_3dtest():
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 4)
    z = np.linspace(0.0, 1.0, 3)
    boundary_points = [x, y, z]

    exact = lambda pt: pt[0] * pt[0]

    alpha = 0.25
    beta = 0.10
    flux_law = nonlinear_flux(alpha=alpha, beta=beta)
    source_fn = lambda T, pt: -2.0 - 24.0 * beta * T - 10.0 * alpha * T * T
    dsource_dT = lambda T, pt: -24.0 * beta - 20.0 * alpha * T

    boundary_conditions = {
        "xmin": ("dirichlet", 0.0),
        "xmax": ("dirichlet", 1.0),
        "ymin": ("neumann", 0.0),
        "ymax": ("neumann", 0.0),
        "zmin": ("neumann", 0.0),
        "zmax": ("neumann", 0.0),
    }

    U, history, iters = NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, verbose=True)
    U_exact = np.array([
        exact((x[i], y[j], z[k]))
        for i in range(len(x))
        for j in range(len(y))
        for k in range(len(z))
    ])

    print(f"U:\n{U}")
    print(f"U_exact:\n{U_exact}")
    print("error norm:", np.linalg.norm(U - U_exact))
