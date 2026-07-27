"""
METHOD OUTLINE----------------------------------------------------------
MaternHalfInteger1D: evaluates k, dk/dx, d2k/dx2
KernelPacketAxis: builds local packet coefficients and support intervals
                  evaluates packet values/derivative matrices
ProductKernelPacketBasis: tensorizes the s and T packet axes
                          evaluates sparse value/derivative matrices
MonotonePacketKernelRegressor: builds weighted sparse QP
                               solves for packet coefficients
MonotonePacketKernelFluxST: standardizes physical input units and outputs
                            exposes evaluate(s, T)
"""

import warnings
from itertools import product
from math import factorial
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.sparse import bmat, block_diag, csc_matrix, coo_matrix, diags, hstack, kron
from scipy.sparse.linalg import splu

def half_integer_order(nu):
    nu = float(nu)
    p = int(round(nu - 0.5))
    return p

def _matern_coefficients(p):
    coefficients = np.empty(p+1, dtype=float)
    scale = factorial(p)/factorial(2*p)
    for degree in range(p+1):
        coefficients[degree] = (
            scale
            * (2.0**degree)
            * factorial(2*p - degree)
            / (factorial(p-degree) * factorial(degree))
        )
    return coefficients

class MaternHalfInteger1D:
    def __init__(self, nu=2.5, lengthscale=1.0):
        self.nu = float(nu)
        self.lengthscale = float(lengthscale)
        self.p = half_integer_order(self.nu)
        self.packet_degree = 2 * self.p + 3
        self.decay_rate = np.sqrt(2.0 * self.nu) / self.lengthscale
        self.coefficients = _matern_coefficients(self.p)
        self.first_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        self.second_coefficients = np.polynomial.polynomial.polyder(
            self.first_coefficients
        )

    def covariance(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.decay_rate * np.abs(x[:, None] - y[None, :])
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        return np.exp(-z) * polynomial

    def covariance_derivative(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        delta = x[:, None] - y[None, :]
        z = self.decay_rate * np.abs(delta)
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, self.first_coefficients)
        result = (
            self.decay_rate
            * np.sign(delta)
            * np.exp(-z)
            * (first - polynomial)
        )
        result[delta == 0.0] = 0.0
        return result

    def covariance_second_derivative(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.decay_rate * np.abs(x[:, None] - y[None, :])
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, self.first_coefficients)
        second = np.polynomial.polynomial.polyval(z, self.second_coefficients)
        return (
            self.decay_rate**2
            * np.exp(-z)
            * (second - 2.0 * first + polynomial)
        )