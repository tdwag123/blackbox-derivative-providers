"""
Performs five complementary checks: 
1. fits a matern-5/2 GP flux provider with learned WhiteKernel noise.
2. measures dq/ds accuracy and verifies provider derivative against a centered 
   finite difference of its own reconstructed flux.
3. builds a one dimensional nonlinear flux-divergence residual with a known exact
   discrete solution. 
4. compares the analytic assembled jacobian against a centered finite difference 
   jacobian of the complete residual.
5. runs newton solves using exact/GP fluxes, analytic/finite-difference jacobians, 
   and direct/GMRES linear solvers 
"""

import csv
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu

from monotoneGPReg import MonotoneGPFluxST

# EXPERIMENT CONTROLS
seed = 42
n_train = 50
true_noise_std = 20
s_train_range = (-1.5, 1.5)
t_train_range = (0.5, 2.5)

reg_function_values = 0.0 # can add more to compare regularization effects on newton stability
jitter = 1e-8
n_restarts_optimizer = 5

n_elements = 20
max_newton_iterations = 15
newton_residual_tolerance = 1e-8

fd_jacobian_relative_step = float(np.cbrt(np.finfo(float).eps))
fd_flux_relative_step = 1e-4
line_search_c1 = 1e-4
line_search_reduction = 0.5
line_search_min_step = 2.0**-14
gmres_rtol = 1e-10
gmres_maxiter = 200
gmres_restart = 40
ilu_drop_tol = 1e-4
ilu_fill_factor = 10.0
output_dir = Path(__file__).resolve().parent/"GPJacobianStabilityResults"