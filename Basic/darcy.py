import numpy as np
from scipy.optimize import fsolve

def darcy_isotropic(grad_psi, psi, pt):
    '''
    (Inverted) isotropic Darcy flow (B=I, K=kI)
    '''

    k = 1.0
    mu = 1.0
    rho = 1.0
    beta = 1.0

    # physical values (groundwater)
    # k    = 1e-11      # m^2
    # mu   = 1e-3       # Pa s
    # rho  = 1000.0     # kg/m^3
    # beta = 1e4        # 1/m

    a = mu / k
    b = rho * beta

    g = np.asarray(grad_psi, dtype=float)
    r = np.linalg.norm(g)

    # zero-gradient limit
    if r == 0.0:
        u = np.zeros_like(g)
        u_g = -(k / mu) * np.eye(len(g))
        return u, u_g, np.zeros(len(g))

    sqrt_term = np.sqrt(a * a + 4 * b * r)

    # stable evaluation of f(r)
    f = (2 * r) / (a + sqrt_term)

    # stable derivative f'(r)
    fp = a / (sqrt_term * (a + sqrt_term))

    # velocity
    u = -(f / r) * g

    # jacobian du/d(grad_psi)
    h = f / r
    hp = (fp * r - f) / (r * r)
    I = np.eye(len(g))
    u_g = -h * I - (hp / r) * np.outer(g, g)

    return u, u_g, np.zeros(len(g))


def darcy_anisotropic(grad_p, p, pt):
    '''
    Anisotropic nonlinear flow with negligible velocity.
    '''

    grad_p = np.asarray(grad_p, dtype=float).reshape(-1)

    # basic inputs and material parameters
    dim = len(grad_p)
    mu = 1.0 # dynamic viscosity
    rho = 1.0 # fluid density
    # R = np.eye(dim) # rotation matrix
    # k = np.ones(dim) # permeability in principal directions
    # beta = np.ones(dim) # Forchheimer coefficients in principal directions

    if dim == 2:
        # 2d anisotropic setup
        k = np.array([1.0, 0.1])
        beta = np.array([1.0, 5.0])
        theta = np.deg2rad(30.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        k = np.array([1.0, 0.1])
        beta = np.array([1.0, 5.0])
    elif dim == 3:
        R = np.eye(3)
        k = np.array([1.0, 0.1, 0.01])
        beta = np.array([1.0, 3.0, 10.0])
    else:
        return

    # constuct relevant tensors
    K = R @ np.diag(k) @ R.T # permeability tensor
    # B = R @ np.diag(beta) @ R.T # anisotropic inertial-resistance tensor
    C = R @ np.diag(beta ** (2.0/3.0)) @ R.T # B^{2/3}
    K_inv = R @ np.diag(1.0 / k) @ R.T

    # Darcy-Forchheimer equation + grad_p
    def residual(u):
        Cu = C @ u 
        s = np.sqrt(max(float(u @ Cu), 0.0))
        return mu * (K_inv @ u) + rho * s * Cu + grad_p

    u = fsolve(residual, np.ones(dim))

    # implicit differentiation gives derivative wrt g=grad_p
    # H(u, g) = mu K^{-1}u + rho sqrt(u^T C u) C u + g = 0
    if np.linalg.norm(u) == 0.0:
        du_dgradp = -(1.0 / mu) * K # linear limit
    else:
        Cu = C @ u
        s = np.sqrt(float(u @ Cu))
        dHdu = mu * K_inv + rho * (s * C + np.outer(Cu, Cu) / s)
        du_dgradp = -np.linalg.solve(dHdu, np.eye(dim))

    return u, du_dgradp, np.zeros(dim)
    

def darcyvis(flux_law):
    import matplotlib.pyplot as plt 

    g1 = np.linspace(-5.0, 5.0, 21)
    g2 = np.linspace(-5.0, 5.0, 21)
    G1, G2 = np.meshgrid(g1, g2)

    Q1 = np.zeros_like(G1)
    Q2 = np.zeros_like(G2)

    T = 1.0
    pt = np.array([0.0, 0.0])

    for i in range(G1.shape[0]):
        for j in range(G1.shape[1]):
            gradT = np.array([G1[i, j], G2[i, j]])
            q, _, _ = flux_law(gradT, T, pt)
            Q1[i, j] = q[0]
            Q2[i, j] = q[1]

    plt.figure()
    plt.quiver(G1, G2, Q1, Q2)
    plt.xlabel("grad_psi[0]")
    plt.ylabel("grad_psi[1]")
    plt.title("Darcy flux (u) vector field")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # darcyvis(darcy_isotropic)
    darcyvis(darcy_anisotropic)
