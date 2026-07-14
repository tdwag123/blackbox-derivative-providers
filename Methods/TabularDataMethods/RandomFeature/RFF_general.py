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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


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
    def __init__(self, regularization, 
            n_components=400, gamma=0.1, random_state=None, 
            alpha=1e-6, fit_intercept=True, copy_X=True,
            tol=0.0001, max_iter=None, solver="auto"
        ):
        """
            regularization: 'none' / 'ridge' / 'frequency_weighted'

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

            fit_intercept: whether the model learns a constant offset term.
                if True, q_hat = Phi*beta + b. if False, q_hat = Phi*beta.
                default is True because target q may not be centered around 0.

            copy_X: whether scikit-learn can copy feature matrix before fitting

            tol: tolerance for solver, controls when iterative solver decides it has converged.
                if small, more precise solve but possibly slower. if large, less precise but faster.

            max_iter: max # of iterations for iterative ridge solvers. if None, then sklearn can choose.

            solver: which numerical method is used in Ridge. if "auto", sklearn can choose.
                other possible options: "svd", "cholesky", "lsqr", "sag"/"saga"
        """

        if regularization not in ['none', 'ridge', 'frequency_weighted']:
            raise ValueError(f"Regularization cannot be '{regularization}'. "
            "RFFModel only supports 'none', 'ridge', and 'frequency_weighted'."
        )
        self.regularization = regularization

        self.n_components = n_components
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.regression_model = None
        
        self.feature_map = RBFSampler(
            n_components=n_components, 
            gamma=gamma, 
            random_state=random_state
        )

        if regularization == 'none':
            self.regression_model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X)
        elif regularization == 'ridge':
            self.regression_model = Ridge(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver)

        
    def fit(self, X, y):
        y = np.asarray(y)   # either of shape (N,) or (N,1)
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(f"y has shape {y.shape}, which looks like {y.shape[1]} targets. "
            "RFFModel.predict_dq_dX only supports single-output regression. "
            "Fit a separate RFFModel per output column instead."
        )
        y = y.ravel()   # turns into (N,)

        Phi = self.feature_map.fit_transform(X, y)

        # Fit ridge regression model onto mapped feature matrix (the transformed version of X)
        if self.regularization == 'frequency_weighted':
            """
            Solving min_beta ||Phi*beta - y||^2 + alpha * (sum ||omega_k||^2 beta_k^2).
            If we take derivative wrt beta, we get 

            Idea:   1) Get each random feature's frequency.
                    2) Measure how high-frequency it is.
                    3) Create penalty matrix that penalizes high-frequency feature coefficients more.
            """
            W = self.feature_map.random_weights_    # stores each feature's random frequency vector (each column is 1 random freq. vector)

            weights = np.linalg.norm(W, axis=0) ** 2    # computes squared length of each frequency vector
            D = np.diag(weights)

            if self.fit_intercept:
                Phi_mean = Phi.mean(axis=0)
                y_mean = y.mean()
            else:
                Phi_mean = np.zeros(Phi.shape[1])
                y_mean = 0.0

            Phi_centered = Phi - Phi_mean
            y_centered = y - y_mean

            self.coef_ = np.linalg.solve(
                Phi_centered.T @ Phi_centered + self.alpha * D,
                Phi_centered.T @ y_centered
            )
            self.intercept_ = y_mean - Phi_mean @ self.coef_
        else:
            self.regression_model.fit(Phi, y)
            self.coef_ = self.regression_model.coef_
            self.intercept_ = self.regression_model.intercept_

        return self # feature_map is now fitted, model is now trained
    

    def predict(self, X):
        """
        Based on reg_model formed by fit, predicts value for specific point.
        """

        if self.coef_ is None:
            raise RuntimeError("RFFModel must be fit before prediction.")
        
        new_X = self.feature_map.transform(X) # apply approximate feature map to input
        return new_X @ self.coef_ + self.intercept_ # predict using linear model
    

    def predict_dq_dX(self, X):

        if self.coef_ is None:
            raise RuntimeError("RFFModel must be fit before prediction.")
        
        W = self.feature_map.random_weights_ # shape (n_features, n_components)
        b = self.feature_map.random_offset_ # shape n_components, )

        beta = self.coef_

        sin_input = X @ W + b
        scalar = (-1) * np.sqrt(2/self.n_components)
        scalar_sine = scalar * np.sin(sin_input)
        weighted = scalar_sine * beta
        dq_dX = weighted @ W.T
        
        return dq_dX


    
        






