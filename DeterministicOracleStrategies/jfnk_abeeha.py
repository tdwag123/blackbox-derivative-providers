#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:31:48 2026

@author: abeehamirza

REMARKS: 
- can be incorporated within an outer Newton-Raphson loop
- ILU preconditioner on the Newton linear system 
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, spilu

# construct JF linear map using first order Taylor approx 

def jacobianApprox(R, u, epsilon = 1e-8):
    
    """ 
    inputs (at a Newton iterate k)
        R: callable vector-valued resid function
        u: current delta T
        epsilon: tunable perturbation
    
    outputs 
        J: matrix-free approx to the Jacobian
    
    """
    
    R_u = R(u)
    n = len(u)
    
    # approximate matrix-vec product Jv
    def matvec(v):
        R_uplusev = R(u + epsilon * v) # u plus ev
        return (R_uplusev - R_u)/epsilon
    
    # construct the JF lin map
    J = LinearOperator(shape=(n,n), matvec = matvec, rmatvec = None)
    
    return J

# solve the JF Newton-Krylov lin system with GMRES 

def solveJFSystem(R, u_0, b, tol=1e-10, precon=True):
    
    """
    inputs (at a Newton iterate k)
        R: callable vector-valued residual function
        u_0: initial guess for eval point
        b: RHS; here, will be -R(u_0)
        tol: tolerance
    
    outputs
        x: converged solution to lin system 
        status: 0 for convergence
        
    """
    
    J = jacobianApprox(R, u_0)
    
    x, status = gmres(J, b, rtol=tol, atol=0.0)
    
    return x, status


if __name__ == "__main__":
    
    # test a simple nonlinear residual
    
    def dummyCase(x):
        return np.array([x[0]**2 + x[1]**2 - 1.0, 
                         x[0] - x[1]**2])
    
    x_0 = np.array([-1.5,0.5])
    
    rhs = -dummyCase(x_0)
    
    solution, status = solveJFSystem(R=dummyCase, u_0=x_0, b=rhs)
    
    print("solution: ", solution)
    print("status: ", status)
 
    
