from __future__ import annotations
import numpy as np
import math 

"""
FE residual form as computed over an element [x_e, x_{e+1}]:
    R_i  = - int N_i' phi(T_h', T_h, x; m) dx   - int N_i r(T_h, x; m) dx   + (R_i)_{gamma_q} (*)

Note: (R_i)_{gamma_q} = Neumann boundary terms. If Dirichlet boundary cond.s only, this term vanishes.

We want a method to solve for the U vector using Newton's method:
    (1) - Use quadrature to compute integrals in the closed form residual given in (*)
    (2) - Construct residual and tangent element by element to be used in Newton 
    (3) - Implement Newton's method to solve for temperature 
"""

# -------------------------------------------------------------------------------------------------
# Note: An n-point Gauss-Legendre quadrature can perfectly integrate any polynomial of degree < 2n.
# Using 2-point Gauss-Legendre quadrature, integrate a function f(x) over [a,b].
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

    # Compute (*) = w_1 * J_1 * f(xi_1) = w_2 * J_2 * f(xi_2)
    # Here, J_1 = J_2 = jacobian; w_1 = w_2 = 1; xi_2 = -xi_1 = 1/sqrt(3)
    
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


# -------------------------------------------------------------------------------------------------
# Given element e over [x_e, x_e+1] with associated stiffness matrix Ke,
# construct the global tridiagonal matrix
# ----------------------------------------------------------------------------------------------------
def construct_tridiag(lower, diag, upper, e: int, Ke):
    diag[e] += Ke[0,0]
    upper[e] += Ke[0,1]
    lower[e] += Ke[1,0]
    diag[e+1] += Ke[1,1]


# -------------------------------------------------------------------------------------------------
# 1D + linear elements --> tangent matrix R_T will be tridiagonal in Newton's method
#
# Thomas algorithm is fast method for solving a tridiagonal system of linear equations Ax = d
# where A has nonzero values only on the main diagonal, diag above it, and diag below it.
# -------------------------------------------------------------------------------------------------
def thomas_solve(lower, diagonal, upper, rhs):
    """
    Solve a tridiagonal linear system with the Thomas algorithm.
    """

    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    n = len(diagonal)

    # error check ----------------------------------------------------------
    if n == 0:
        return np.array([], dtype = float)
    
    if abs(diagonal[0]) < 1.0e-14:
        raise ZeroDivisionError("Zero pivot at row 0.")
    # ----------------------------------------------------------------------

    if n == 1: 
        return np.array([rhs[0]/diagonal[0]], dtype=float)
        
    gamma = np.zeros(n, dtype=float)
    rho = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

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


# Extracts tridiagonal block for Dirichlet boundary conditions
# recall: with Dirichlet BC, don's solve for boundary values
# if T(0) = TL, T(L) = TR, U0 = TL, UN = TR
def tridiag_block(lower, diag, upper, start, end):
    diag_f = diag[start:end].copy()
    lower_f = lower[start:end - 1].copy()
    upper_f = upper[start:end - 1].copy()
    return lower_f, diag_f, upper_f


# -------------------------------------------------------------------------------------------------
# RESIDUAL/TANGENT CONSTRUCTION ELEMENT BY ELEMENT
#
# PDE setup
#   q'(x) = r(T,x)
#   q = phi(s,T,x), where s = T'(x)
#
#   Weak residual
#       R_i = -int N_i' phi(T_h', T_h, x)dx   -int N_i r(T_h, x)dx +    Neumann boundary terms.
#
#   Tangent/Jacobian
#       J_ij    = dR_i/dU_j
#               = - int N_i' [phi_s N_j' + phi_T N_j] dx    - int N_i r_T N_j dx.
# -------------------------------------------------------------------------------------------------

def residAndTan(x, U, fluxLaw, r, rT = None, q_BCL = None, q_BCR = None):
    """
    Input: 
        x : vector containing position coordinates
        U : vector containing nodal temperature values (unknown, to be solved in NM for T estimate)
        fluxLaw: analytic constitive map, nonlinear putting q = phi(T, T', x; m)
            callable; returns (q, phi_T', phi_T)
        r: heat source
            callable; returns r(T, xg) returns source value
        rT: dr/dT; if T-independent, None
        q_BCL: left boundary condition (Neumann)
        q_BCR: right boundary condition (Neumann)
    Remark: should be no Dirichlet boundaries
    (FACT CHECK): If Dirichlet boundaries here, simply preallocate U_0, U_L = TL, TR
    """
    if rT is None:
        rT = lambda T, xg: 0.0

    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)
    
    n_nodes = len(x)

    # Preallocate residual vector and lower, diagonal, and upper diagonal entries
    R = np.zeros(n_nodes) 

    # tridiag storage setup for the newton tangent J
    lower = np.zeros(n_nodes-1)
    diag = np.zeros(n_nodes)
    upper = np.zeros(n_nodes-1)

    # compute local nodal temperatures and derivatives of local hat functions
    # rmk: we work with linear elements, so these will be constant 
    
    for e in range(n_nodes - 1):
        xl = x[e]
        xr = x[e + 1]
        h = xr - xl

        if h <= 0.0:
            raise ValueError("mesh nodes need to be strictly increasing")
        
        # compute local nodal temperatures for this element
        Ue = np.array([U[e], U[e + 1]])

        # derivatives of local linear hat functions
        dNdx = np.array([-1.0 / h, 1.0 / h])

        # return local linear hat functions evaluated at xg
        def N_j(xg):
            return np.array([(xr - xg)/h, (xg - xl)/h])
            
        # returns 2-vector with element contribution to residual integrand
        def residual_integrand(xg):
            N = N_j(xg)
        
            # temperature at quadrature points
            Tg = N @ Ue

            # deriv of temperature at quadrature points; will be constant for linear elements 
            Tg_prime = dNdx @ Ue

            # constitutive flux law and heat source evaluated at quadrature point 
            qg, phi_s, phi_T = fluxLaw(Tg_prime, Tg, xg)
            rg = r(Tg, xg)

            # residual integrand
            return -dNdx * qg - N * rg
        
        # returns 2x2 matrix representing element contribution to NM tangent at quadrature point
        def tangent_integrand(xg):
            N = N_j(xg)

            Tg = N @ Ue
            Tg_prime = dNdx @ Ue

            qg, phi_s, phi_T = fluxLaw(Tg_prime, Tg, xg)
            rTg = rT(Tg, xg)

            # Flux tangent component:
            #   -N_i' [phi_s N_j' + phi_T N_j]
            K_flux = -np.outer(dNdx, phi_s * dNdx + phi_T * N)

            # Source tangent component:
            #   -N_i r_T N_j
            K_source = -rTg * np.outer(N, N)

            return K_flux + K_source

        # use 2-pt GL quadrature to compute integrals
        Re = gl2_quadrature_integration(residual_integrand, xl, xr, 1)
        Ke = gl2_quadrature_integration(tangent_integrand, xl, xr, 1)

        # compile local element contributions into global triagonal form
        R[e:e + 2] += Re
        construct_tridiag(lower, diag, upper, e, Ke)

    # neumann boundary terms - qbar means prescribed outward flux q*n (simple in 1D case)
    # at x=0, n=-1, so qbar_left = -q(0)
    # at x=L, n=+1, so qbar_right = q(L)
    # !!!! do not add on Dirichlet boundaries !!!!
    
    if q_BCL is not None:
        R[0] += q_BCL

    if q_BCR is not None:
        R[-1] += q_BCR

    return R, lower, diag, upper

# -------------------------------------------------------------------------------------------------
# NEWTON SOLVER FOR NODAL TEMPERATURE VECTOR U
# -------------------------------------------------------------------------------------------------

"""
some things to keep in mind on boundary conditions
(1) Dirichlet temperatures (TL, TR); Put None if Neumann
(2) Neumann fluxes (q_BCL, q_BCR); Put None if Dirichlet
(3) Dirichlet-Neumann possible with Robin-Type BC
"""
def NM(x, fluxLaw, r, TL, TR, rT=None, U0=None, q_BCL=None, q_BCR=None, tol=1e-10, maxiter=30, verbose=True, line_search=True):
    
    x = np.asarray(x, dtype=float)
    n_nodes = len(x)

    left_dirich = TL is not None
    right_dirich = TR is not None

    if left_dirich and q_BCL is not None:
        raise ValueError("Left BC cannot be Dirichlet and Neumann")
    if right_dirich and q_BCR is not None:
        raise ValueError("Right BC cannot be Dirichlet and Neumann")

    if not left_dirich and not right_dirich:
        raise ValueError("1D problem is singular ie we need both Dirichlet BC; provide the missing Dirichlet BC, or add Robin penalty")

    # construct initial estimate; check if U0 given or not
    # rmk: we guarantee here that one need not provide an initial estimate for newton solver to work
    # can execute based on given boundary conditions 
    
    if U0 is None:

        # case 1 - choose values between prescribed dirichlet BC
        if left_dirich and right_dirich:
            U = np.linspace(TL, TR, n_nodes)
            
        # case 2 - put all as left dirichlet BC if given
        elif left_dirich:
            U = np.full(n_nodes, TL, dtype=float)

        # case 3 - put all as right dirichlet BC if given
        elif right_dirich:
            U = np.full(n_nodes, TR, dtype=float)
            
        # case 4 - put all as zero
        else:
            U = np.zeros(n_nodes, dtype=float)
    else:
        U = np.asarray(U0, dtype=float).copy()

    # prefix dirichlet BC if applicable
    if left_dirich:
        U[0] = TL
    if right_dirich:
        U[-1] = TR

    # account for dirichlet reduction of tridiagonal matrix into a submatrix 
    start = 1 if left_dirich else 0
    end = n_nodes - 1 if right_dirich else n_nodes

    log = []

    for iteration in range(maxiter):
        
        R, lower, diag, upper = residAndTan(x, U, fluxLaw, r, rT=rT, q_BCL=q_BCL, q_BCR=q_BCR)

        # construct the effective residual in the case of dirichlet BC
        R_eff = R[start:end]
        norm_R = np.linalg.norm(R_eff, ord=2) # here we use the 2-norm
        log.append(norm_R)
        
        if verbose:
            print(f"Newton {iteration:2d}: ||R_eff||_2 = {norm_R:.3e}")
            
        if norm_R < tol:
            return U, log

        lower_eff, diag_eff, upper_eff = tridiag_block(lower, diag, upper, start, end)

        # newton step
        dU_eff = thomas_solve(lower_eff, diag_eff, upper_eff, -R_eff)
        alpha = 1.0
        if line_search:
            accepted = False
            while alpha > 1.0e-12:
                U_trial = U.copy()
                U_trial[start:end] += alpha * dU_eff

                if left_dirich:
                    U_trial[0] = TL
                if right_dirich:
                    U_trial[-1] = TR

                R_trial, _, _, _ = residAndTan(x, U_trial, fluxLaw, r, rT=rT, q_BCL=q_BCL, q_BCR=q_BCR)
                
                norm_trial = np.linalg.norm(R_trial[start:end], ord=2)
                
                if norm_trial < norm_R:
                    accepted = True
                    break

                alpha *= 0.5
                
            if not accepted:
                raise RuntimeError("line search failed to minimize residual")
        
        else:
            U_trial = U.copy()
            U_trial[start:end] += dU_eff

            # impose dirichlet BC again if applicable
            
            if left_dirich:
                U_trial[0] = TL
            if right_dirich:
                U_trial[-1] = TR

        # update U 
        U = U_trial
        
    raise RuntimeError("NM didn't converge within desired number maxiter")


## NONLINEAR FLUX LAW DUMMY CASE ##
# rmk: here we have an smooth, analytic constitutive law that is differentiable by hand


x = np.linspace(0.0, 1.0, 21) # mesh

def source(T, xg): # source
    return 1.0

def fluxLinSanityCheck(T_prime, T, xg):
    
    q = -T_prime # we put k = 1 in q = -kT', the fourier law
    phi_s = -1.0
    phi_T = 0.0
    
    return q, phi_s, phi_T

# since q = -T', q' = r = 1, we have -T'' = 1. use Dirichlet BC T(0) = 0, T(1) = 0. 
U_exact = lambda x: 0.5*x*(1.0-x)

U_sanity, log_sanity = NM(x, fluxLinSanityCheck, source, TL = 0.0, TR = 0.0, verbose = True)
print("\nLinear Sanity check solution U:\n")
for i in range(len(U_sanity)):
    print(U_sanity[i])
print("\nError between NM U and exact U (2-norm of U_sanity - U_true):\n")
U_true = U_exact(x)
error_vec = U_sanity - U_true
error = np.linalg.norm(error_vec, ord=2)
print(error)
print("\n")

def fluxNonlinExample(T_prime, T, xg):
    
    alpha = 0.25
    beta = 0.05

    q = -((1.0 + alpha * T * T) + beta * T_prime * T_prime) * T_prime
    phi_s = -(1.0 + alpha * T * T) - 3.0 * beta * T_prime * T_prime
    phi_T = -2.0 * alpha * T * T_prime

    return q, phi_s, phi_T

# using Dirichlet BC TL = TR = 0.0

U_test, log_test = NM(x, fluxNonlinExample, source, TL = 0.0, TR = 0.0, verbose = True)
print("\nNonlinear solution U:\n")
for i in range(len(U_test)):
    print(U_test[i])
