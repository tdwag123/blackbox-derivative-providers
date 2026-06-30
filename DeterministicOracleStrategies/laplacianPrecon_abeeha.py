import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import LinearOperator, factorized


# some remarks:
    
"""
    
    This is a high-level construction of the discrete Laplacian given some 
    inputs (namely, a residual r and a known number of nodal temps in the FE 
    charac.of temp) and can be implemented downstream within the kth Newton 
    iterate to construct a Laplacian preconditioner to be implemented in GMRES. 
    
    Should be tested within a broader Newton loop to ensure it works and speeds 
    up GMRES convergence vs. without preconditioning. 
    
"""

"""
build 1D second-difference matrix d2/d(x2) over n interior unknowns
remark: unknowns here are nodal temp values

inputs:
    - n corresp to number unknowns in 1D
    - dx corresp to stepsize in finite difference approx to lap.
"""

def secondDifference1D(n, dx):
    
    e = np.ones(n)
    D = diags(diagonals=[e[:-1],-2.0*e,e[:-1]], offsets=[-1, 0, 1], 
              shape=(n,n))/(dx**2)
    
    return D

"""
build the dD discrete laplacian P using kronecker sums of 
1D second-difference matrices

inputs:
    - unknowns: tuple[int] corresp. to number unknowns in each of the 
                d directions
    - stepsizes: tuple[float] corresp to stepsizes in each direction

"""
def buildLap(unknowns, stepsizes):
    
    # D_global contains 1D second-difference matrices for each of d dimensions
    
    D_global = [secondDifference1D(n,dx) for n,dx in zip(unknowns,stepsizes)]
    
    # compute P as a Kronecker sum
    
    P = D_global[0]
    
    for D in D_global[1:]:
        P = kronsum(P, D)

    return P

"""
define and return matrix-vector action as a linear operator for preconditioner 
M = Pinv

"""

def M(P):
    
    solverP = factorized(P) # factorized works with an LU decomp
    n = P.shape[0]
    
    def applyM(r):
        return solverP(r)
    
    M = LinearOperator(shape=(n,n), matvec=applyM, dtype=float)
    
    return M

    
    
