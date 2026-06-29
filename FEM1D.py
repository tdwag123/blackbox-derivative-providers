import numpy as np
from utils import *

def residAndTan(x, U, fluxLaw, r, rT = None, q_BCL = None, q_BCR = None):
    '''
    Construct residual and tangent.

    PDE setup

        q'(x) = r(T,x)
        q     = phi(s,T,x), where s = T'(x)

    Weak residual

        R_i = - int N_i' phi(T_h', T_h, x) dx
              - int N_i  r(T_h, x) dx
              + Neumann boundary terms.

    Tangent/Jacobian

        J_ij = dR_i/dU_j
             = - int N_i' [phi_s N_j' + phi_T N_j] dx
               - int N_i r_T N_j dx.

    ALSO returns number of flux law evaluations

input: 
    x vector containing position coordinates
    U vector containing nodal temperature values (unknown, to be solved in NM for T estimate)
    fluxLaw: analytic constitive map, nonlinear putting q = phi(T, T', x; m)
        callable; returns (q, phi_T', phi_T)
    r: heat source
        callable; returns r(T, xg) returns source value
    rT: dr/dT; if T-independent, None
    q_BCL: left boundary condition (Neumann)
    q_BCR: right boundary condition (Neumann)
    rmk: should be no Dirichlet boundaries
    '''

    if rT is None:
        rT = lambda T, xg: 0.0

    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)
    
    n_nodes = len(x)

    # we preallocate the residual vector and the lower, diagonal, and upper diagonal entries
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
            
        # returns 2-vector with element contribution to the residual integrand
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
        Re = GLquadrature_twoPoint(residual_integrand, xl, xr)
        Ke = GLquadrature_twoPoint(tangent_integrand, xl, xr)

        # compile local element contributions into global triagonal form
        R[e:e + 2] += Re
        constructTridiag(lower, diag, upper, e, Ke)

    # neumann boundary terms - qbar means prescribed outward flux q*n (simple in 1D case)
    # at x=0, n=-1, so qbar_left = -q(0)
    # at x=L, n=+1, so qbar_right = q(L)
    # !!!! do not add on Dirichlet boundaries !!!!
    
    if q_BCL is not None:
        R[0] += q_BCL

    if q_BCR is not None:
        R[-1] += q_BCR

    return R, lower, diag, upper

def NM(x, fluxLaw, r, TL, TR, rT=None, U0=None, q_BCL=None, q_BCR=None, tol=1e-10, maxiter=30, verbose=True, line_search=True):
    '''
    NEWTON SOLVER FOR NODAL TEMPERATURE VECTOR U

    some things to keep in mind on boundary conditions
    (1) Dirichlet temperatures (TL, TR); Put None if Neumann
    (2) Neumann fluxes (q_BCL, q_BCR); Put None if Dirichlet
    (3) Dirichlet-Neumann possible with Robin-Type BC
    '''
    
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
    num_iter = 0

    for iteration in range(maxiter):
        num_iter += 1
        
        R, lower, diag, upper = residAndTan(x, U, fluxLaw, r, rT=rT, q_BCL=q_BCL, q_BCR=q_BCR)

        # construct the effective residual in the case of dirichlet BC
        R_eff = R[start:end]
        norm_R = np.linalg.norm(R_eff, ord=2) # here we use the 2-norm
        log.append(norm_R)
        
        if verbose:
            print(f"Newton {iteration:2d}: ||R_eff||_2 = {norm_R:.3e}")
            
        if norm_R < tol:
            return U, log, num_iter

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
                raise RuntimeError("line search failed to minimize  residual")
        
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