"""
Random Fourier Features general framework for toggling regularization.
REGULARIZED LEAST-SQUARES PROBLEM SUBJECT TO MONOTONICITY CONSTRAINTS.
NOTE: WORK IN PROGRESS. CURRENTLY MONOTONICITY ONLY ENFORCED AT TRAINING POINTS.

Note: constrained least squares problem can be equivalently formulated as quadratic programming problem.
    We will solve this problem as a QP.

Conceptually our optimization becomes
        min_c ||Ac - y||_2^2 + alpha sum ||omega_k||_2^p |c_k|^2
        subject to:
            dq/ds >= 0 at chosen points
            dq/dT >= 0 at chosen points

    =>  min_c (1/2) c.T Q c + (-2A.Ty).Tc   where Q = 2(A.TA+alphaD)
        subject to:
            -Bc <= 0 where B is the derivative coefficient matrix of shape (M*n_input_features, n_components)
            note: M is the number of points at which we enforce monotonicity

Regularization options:
    - No regularization (linear regression); alpha = 0. WARNING: RFF with no regularization can overfit!
    - Regularization=True; alpha != 0.  
        ---- Tikhonov regularization ----
        If p=0, then ridge regression.

        ---- frequency-weighted Tikhonov regularization accounting for noise ----
        If p=1, penalty grows linearly with frequency length.
        If p=2, penalty grows quadratically with frequency length. 
        If p>2, high-frequency features are punished very strongly.
"""

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from qpsolvers import solve_qp

class ConstrainedRFFDerivativeProviderST():
    def __init__(self, s_data, T_data, q_data, **kwargs):
        X = np.column_stack([s_data, T_data])
        self.model = ConstrainedRFFModel(**kwargs)
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


"""
none: min_c ||Ac - y||^2
ridge: min_c ||Ac - y||^2 + alpha sum |c_k|^2
frequency_weighted: min_c ||Ac - y||_2^2 + alpha sum ||omega_k||_2^p |c_k|^2

=> min_c (1/2) c.T [2(A.T A + alpha D)] c + (-2 A.T y).T c
"""
class ConstrainedRFFModel():
    def __init__(self, alpha=1e-6, freq_weight=2, n_components=400, 
            gamma=0.1, random_state=None, fit_intercept=True
        ):
        """
            alpha: regularization strength.
                if larger, smoother model + less overfitting + more possible underfitting. 
                if smaller, more flexible model but more risk of fitting noise.
                if 0, no regularization (=> LINEAR REGRESSION), regardless of p.

            freq_weight: frequency weighting, p.
                if p=0, no frequency weighting (=> RIDGE REGRESSION).
                if p=1: penalty grows linearly with frequency length.
                if p=2: penalty grows quadratically with frequency length.
                if p>2: high-frequency features are punished very strongly.

            n_components: number of random Fourier features, controls # of cosine basis functions. 
                if large, can fit more complicated fns, slower, may overfit. if small, faster but may underfit.

            gamma: scale of RBF kernel approximation in RBFSampler.
                if large, cosines wiggle quickly + model can learn sharper/local changes + more risk of fitting noise.
                if small, w_k values usually smaller + model learns smoother, broader trends.

            random_state: optionally reproducible random features.
                if None, each run samples different random features. if 0, each run uses same features (good for debugging).

            fit_intercept: whether the model learns a constant offset term.
                if True, q_hat = A*c + b. if False, q_hat = A*c.
                default is True because target q may not be centered around 0.
        """

        if freq_weight < 0:
            raise ValueError(f"p={freq_weight}. Frequency weighting must be nonnegative for frequency-weighted regularization."
        )
        self.p = freq_weight
        self.alpha = alpha

        self.n_components = n_components
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
        self.feature_map = RBFSampler(
            n_components=n_components, 
            gamma=gamma, 
            random_state=random_state
        )

        
    def fit(self, X, y):
        """
            X: input data, has shape (n_samples, n_input_features).
                e.g.) for s,T case, X has shape (n_samples, 2)

            y: target/output data we want to learn, usually has shape (n_samples,) or (n_samples, 1).
                e.g.) for flux law, y is q_data

            Idea: X[i] = [s_i, T_i]; y[i] = q_i. Model is learning (s,T) -> q.
        """
        y = np.asarray(y)   # converts input into numpy array of either of shape (N,) or (N,1)

        # if y.ndim=0, scalar. if 1, vector. if 2, matrix. etc. y.shape[1] returns # of columns in y.
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(f"y has shape {y.shape}, which looks like {y.shape[1]} targets. "
            "ConstrainedRFFModel.predict_dq_dX only supports single-output regression. "
            "Fit a separate ConstrainedRFFModel per output column instead."
        )

        y = y.ravel()   # turns into (N,)

        # computes random Fourier features to approximate an RBF (Gaussian) kernel
        A = self.feature_map.fit_transform(X) 

        # now, fit regression model onto mapped feature matrix A
        # 1) Get each random feature's frequency.
        # 2) Measure how high-frequency it is.
        # 3) Create penalty matrix that penalizes high-frequency feature coefficients more.

        """
        Recall: min_c (1/2) c.T [2(A.T A + alpha D)] c + (-2 A.T y).T c
            taking the derivative wrt c: 2(A.T*A + alpha*D)c + (-2A.T *y)
            setting to zero: (A.T*A + alpha*D)c = A.T*y => c = (A.T*A + alpha*D)^{-1}(A.T*y)
        """

        omega = self.feature_map.random_weights_    # stores each feature's random frequency vector (each column is 1 random freq. vector)
        frequency = np.linalg.norm(omega, axis=0) ** self.p # by setting axis=0, we compute the norm of each COLUMN
        D = np.diag(frequency)

        # if q_hat = A*c + b where b != 0， c is coef_ and b is intercept_
        if self.fit_intercept: # we want to learn both c and b
            A_mean = A.mean(axis=0) # computes the average of each COLUMN in A
            y_mean = y.mean() # computes the average of y
            # basically, we have computed the average feature values and average target value
        else: # b is forced to be zero
            A_mean = np.zeros(A.shape[1])
            y_mean = 0.0

        # if fit_intercept, the following describes how far each value is from the average.
        #   remove avg level first, fit coeffs to leftover variation, then add avg level back through the intercept
        # if not, the following say the same as A and y respectively
        A_centered = A - A_mean 
        y_centered = y - y_mean

        """
        We need to build the derivative coefficient matrix B for phi(x) to enforce monotonicity constraints.

        First, X_constraint = the points on which we want to enforce these constraints on.
            For now, we set X_constraint = X (enforcing on training points only). Need to change this later.

        in RBFSampler, mapping function is phi(x):
            phi(x) = sqrt(2/n_components) cos(W^T x + offset)  where W is random_weights_ and offset is random_offset_
        A is made by applying this mapping function to every training/input point.

        regression model predicts q_hat = A*c + b where c = coef_ and b = intercept_ 
        this means that q_hat_i = phi(x_i)^T c + b  => q_hat(x) = phi(x)^T c + b
        => grad_x(q_hat) = c^T * grad_x(phi(x))
        => grad_x(q_hat) = c^T * -sqrt(2/n_components) sin(W^T x + offset) * W^T
        """

        X_constraint = X[:100]
        W = self.feature_map.random_weights_ # shape (n_features, n_components)
        offset = self.feature_map.random_offset_ # shape n_components, )

        sin_input = X_constraint @ W + offset
        base = -np.sqrt(2/self.n_components) * np.sin(sin_input)

        B_blocks = []
        for i in range(0,X_constraint.shape[1]):
            B_i = base * W[i,:]
            B_blocks.append(B_i)
        B = np.vstack(B_blocks)

        Q = 2 * (A_centered.T @ A_centered + self.alpha * D)
        Q = 0.5 * (Q + Q.T)
        # solving min_c (1/2) c.T [2(A.T A + alpha D)] c + (-2 A.T y).T c
        # https://pypi.org/project/qpsolvers/
        self.coef_ = solve_qp(
                Q,
                -2 * A_centered.T @ y_centered,
                -B,
                np.zeros(B.shape[0]),
                solver="osqp"
        )

        if self.coef_ is None:
            raise RuntimeError("QP solver failed or monotonicity constraints are infeasible.")

        # q_hat = A*c + b => b = q_hat - A*c
        self.intercept_ = y_mean - A_mean @ self.coef_

        return self # feature_map is now fitted, model is now trained
    

    def predict(self, X):
        """
        After feature map is fitted, predicts value for specific point.
        """

        if self.coef_ is None:
            raise RuntimeError("ConstrainedRFFModel must be fit before prediction.")
        
        A = self.feature_map.transform(X) # apply approximate feature map to input
        return A @ self.coef_ + self.intercept_ # predict using linear model, q_hat = A*c + b
    

    def predict_dq_dX(self, X):
        """
        in RBFSampler, mapping function is phi(x):
            phi(x) = sqrt(2/n_components) cos(W^T x + offset)  where W is random_weights_ and offset is random_offset_
        A is made by applying this mapping function to every training/input point.

        regression model predicts q_hat = A*c + b where c = coef_ and b = intercept_ 
        this means that q_hat_i = phi(x_i)^T c + b  => q_hat(x) = phi(x)^T c + b
        => grad_x(q_hat) = c^T * grad_x(phi(x))
        => grad_x(q_hat) = c^T * -sqrt(2/n_components) sin(W^T x + offset) * W^T
        """

        if self.coef_ is None:
            raise RuntimeError("ConstrainedRFFModel must be fit before prediction.")
        
        W = self.feature_map.random_weights_ # shape (n_features, n_components)
        offset = self.feature_map.random_offset_ # shape n_components, )
        c = self.coef_

        sin_input = X @ W + offset
        scalar = -np.sqrt(2/self.n_components) * c
        dq_dX = (scalar * np.sin(sin_input)) @ W.T
        
        return dq_dX