"""
Looking at scikit-learn's RBFSampler, we construct MaternSampler that follows the same format
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_approximation.py
"""

import numpy as np


class MaternSampler:
    """
    If we draw frequencies from a Student t's distribution, we are approximating a Matern kernel.
    """
    def __init__(
        self,
        match_gaussian_variance,
        df, # controls tail heaviness
        n_components=400, 
        gamma=0.1,
        random_state=None
    ):
        
        
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if df <= 0:
            raise ValueError("df must be positive.")
        if match_gaussian_variance and df <= 2:
            raise ValueError(
                "df must be greater than 2 when matching Gaussian variance."
            )
        

        

