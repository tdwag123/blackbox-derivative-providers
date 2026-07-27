"""
Looking at scikit-learn's RBFSampler, we construct MaternSampler that follows the same format
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_approximation.py
"""

from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions

from sklearn.utils import _align_api_if_sparse, check_random_state

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    validate_data,
)



class MaternSampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
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

    Attributes --------------------------------------------------------
        random_offset_ : ndarray of shape (n_components,), dtype={np.float64, np.float32}
            Random offset used to compute the projection in the `n_components`
            dimensions of the feature space.

        random_weights_ : ndarray of shape (n_features, n_components), dtype={np.float64, np.float32}
            Random projection directions drawn from the Fourier transform
            of the RBF kernel. 
    """

    _parameter_constraints: dict = {
        "gamma": [
            StrOptions({"scale"}),
            Interval(Real, 0.0, None, closed="left"),
        ],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, *, gamma=0.1, n_components=400, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.
    
        Samples random projection according to n_features.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
    
        y : array-like, shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
    
        Returns
        -------
        self : object
        Returns the instance itself.
        """

        X = validate_data(self, X, accept_sparse="csr")
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        sparse = sp.issparse(X)
        if self.gamma == "scale":
            # var = E[X^2] - E[X]^2 if sparse
            X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
            self._gamma = 1.0 / (n_features * X_var) if X_var != 0 else 1.0
        else:
            self._gamma = self.gamma
        self.random_weights_ = (2.0 * self._gamma) ** 0.5 * random_state.normal(
            size=(n_features, self.n_components)
        )
    
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)
    
        if X.dtype == np.float32:
            # Setting the data type of the fitted attribute will ensure the
            # output data type during `transform`.
            self.random_weights_ = self.random_weights_.astype(X.dtype, copy=False)
            self.random_offset_ = self.random_offset_.astype(X.dtype, copy=False)
        self._n_features_out = self.n_components
        return self
    
    def transform(self, X):
        """Apply the approximate feature map to X.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
    
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """

        check_is_fitted(self)
    
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= (2.0 / self.n_components) ** 0.5
        return projection
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags
        

        

