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
reg_function_values = 0.0
jitter = 1e-8
