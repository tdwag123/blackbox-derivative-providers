"""
Looking at scikit-learn's RBFSampler, we construct MaternSampler that follows the same format
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_approximation.py
"""

import numpy as np


class MaternSampler:
    """
    Approximate a Matern kernel feature map using random Fourier features that draw frequencies from Student t's distribution.

    Parameters -------------------------------------------------------
        n_components: number of random Fourier features. int, default=400
            Number of Monte Carlo samples per original feature.
            Equals the dimensionality of the computed feature space.

        gamma: 'scale' or float, default=1.0
            Parameter of RBF kernel: exp(-gamma * x^2).
            If ``gamma='scale'`` is passed then it uses
            1 / (n_features * X.var()) as value of gamma.

        random_state: int, RandomState instance or None, default=None
            Pseudo-random number generator to control the generation of the random
            weights and random offset when fitting the training data.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
    """
    def __init__(
        self,
        match_gaussian_variance, 
        df, # controls tail heaviness
        n_components=400, 
        gamma=0.1,
        random_state=None
        ):

        self.n_components = n_components
        self.gamma = gamma
        
        
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
        

        

