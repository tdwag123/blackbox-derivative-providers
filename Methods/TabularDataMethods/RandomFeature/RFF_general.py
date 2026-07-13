"""
Random Fourier Features general framework for toggling regularization.

Regularization options:
    - No regularization (linear regression). WARNING: RFF with no regularization can overfit!

    Regularized least squares
    - Tikhonov regularization (ridge regression)
    - Frequency-weighted Tikhonov regularization accounting for noise

none: min_beta ||Phi beta - y||^2
ridge: min_beta ||Phi beta - y||^2 + alpha sum beta_k^2
frequency_weighted: min_beta ||Phi beta - y||^2 + alpha sum ||omega_k||^2 beta_k^2
"""

import numpy as np
from sklearn.kernel_approximation import RBFSampler


class RFFDerivativeProviderST():
    def __init__(self, s_data, T_data, q_data, **kwargs):
        X = np.column_stack([s_data, T_data])
        self.model = RFFModel(**kwargs)
        self.model.fit(X, q_data)

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)

        Xq = np.column_stack([s_q.ravel(), T_q.ravel()])

        q = self.model.predict(Xq)
        dq = self.model.predict_dq_dX(Xq)

        q = q.reshape(s_q.shape)
        dq_ds = dq[:, 0].reshape(s_q.shape)  # dq/ds
        dq_dT = dq[:, 1].reshape(s_q.shape)  # dq/dT

        return q, dq_ds, dq_dT


class RFFModel():
    def __init__(
            self, 
            regularization, 
            n_components=400, gamma=0.1, random_state=None, 
            alpha=1e-6
        ):
        """
            regularization: 'none' / 'ridge' / 'weighted'

            n_components: number of random Fourier features, controls # of cosine basis functions. 
                if large, can fit more complicated fns, slower, may overfit. if small, faster but may underfit.

            gamma: scale of RBF kernel approximation in RBFSampler
                if large, cosines wiggle quickly + model can learn sharper/local changes + more risk of fitting noise
                if small, w_k values usually smaller + model learns smoother, broader trends

            random_state: optionally reproducible random features.
                if None, each run samples different random features. if 0, each run uses same features (good for debugging)

            alpha: regularization strength.
                if larger, smoother model + less overfitting + more possible underfitting. 
                if smaller, more flexible model but more risk of fitting noise.
        """

        if regularization not in ['none', 'ridge', 'weighted']:
            raise ValueError(f"Regularization cannot be '{regularization}'."
            "RFFModel only supports 'none', 'ridge', and 'weighted'."
        )
        self.regularization = regularization

        self.n_components = n_components
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.regression_model = None
        
        self.feature_map = RBFSampler(
            n_components=n_components, 
            gamma=gamma, 
            random_state=random_state
        )


    def _regression_model(regularization):
        match regularization:
            case 'none':
                return LinearRegression(fit_intercept=True, copy_X=True, tol=0.0001)
            case 'ridge':
                return Ridge(alpha=self.alpha, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver="auto")
            case 'weighted':
                return 0
            case _:
                raise ValueError(f"RFFModel only supports 'none', 'ridge', and 'weighted' regularizations.")
        






