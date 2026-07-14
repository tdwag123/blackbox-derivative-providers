import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression

"""
RFF without regularization.
"""

class LinRegRFFDerivativeProviderST:
    def __init__(self, s_data, T_data, q_data, **kwargs):
        X = np.column_stack([s_data, T_data])
        self.model = LinRegRFFModel(**kwargs)
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
    

class LinRegRFFModel():
    """
    Random Fourier Features model for n-D constitutive laws using RBF kernel approximation and ridge regression
    (X) -> random Fourier -> phi(X) -> ridge regression -> predict q -> analytically solve for predicted dq_dX
    """

    # n_components should be meaningfully smaller (<<) than n_data
    def __init__(self, n_components=400, gamma=0.1, random_state=None, alpha=1e-6):
        """
        n_components: number of Random Fourier features

        feature_map: turns X into phi(X)    https://scikit-learn.org/stable/modules/kernel_approximation.html#rbf-kernel-approx
        reg_model: learns q from phi(X)     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        """

        self.n_components = n_components
        self.feature_map = RBFSampler(
            n_components=n_components, 
            gamma=gamma, 
            random_state=random_state
        )
        self.reg_model = LinearRegression(fit_intercept=True, copy_X=True, tol=0.0001)
        self.alpha = alpha
        # self.coef_ = None
        # self.intercept_ = None


    def fit(self, X, y):
        """
        Fit linear ridge regression model on mapped feature matrix against target vector y
        with regularization parameter alpha.

        X = data matrix; array of shape (n_samples, n_features) holding training samples
            Note: In the 1D case q = q(s), n_features = 1
                  In the 2D case q = q(s,T), n_features = 2
        y = target vector; 1D array of shape (n_samples, ) holding target values
        """

        # Fit ridge regression model onto mapped feature matrix (the transformed version of X)
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(f"y has shape {y.shape}, which looks like {y.shape[1]} targets."
            "RFFModel.predict_dq_dX only supports single-output regression. "
            "Fit a separate RFFModel per output column instead."
        )

        new_X = self.feature_map.fit_transform(X, y)    # Fit feature_map to data, then transform it. 
                                                        # Fits "transformer" to X, returns transformed version of X
        self.reg_model.fit(new_X, y)

        return self # feature_map is now fitted, model is now trained
    

    def predict(self, X):
        """
        Based on reg_model formed by fit, predicts value for specific point.
        """
        
        new_X = self.feature_map.transform(X) # apply approximate feature map to input
        return self.reg_model.predict(new_X) # predict using linear model
    

    # idea: fit isotonic constraints for monotonicity????????
    def predict_dq_dX(self, X):
        """
        in RFF + ridge, can analytically compute predicted flux derivatives because model = sum of cosines + sines!
        https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf

        phi(x) = sqrt(2/n_components) cos(Wx + b)  where W is random_weights_ and b is random_offset_
        grad_x phi(x) = -sqrt(2/n_components) sin(W^T x + b) * W^T

        ridge model predicts q = phi(x)^T * beta + b_0  where beta = coef_ and b_0 = intercept_
                Note: coef_ is ndarray of shape (n_features,) or (n_targets, n_features)
        so grad_x q = beta^T * grad_x phi(x) = beta^T * -sqrt(2/n_components) sin(W^T x + b) * W^T
        """
    
        W = self.feature_map.random_weights_ # shape (n_features, n_components)
        b = self.feature_map.random_offset_ # shape n_components, )

        beta = self.reg_model.coef_

        sin_input = X @ W + b
        scalar = (-1) * np.sqrt(2/self.n_components)
        scalar_sine = scalar * np.sin(sin_input)
        weighted = scalar_sine * beta
        dq_dX = weighted @ W.T
        
        return dq_dX



# ---------------------------------------------------------------------------------------------------------------------
def example_simple_1d():
    """
    Creates sample s_data -> reshape into (N,1) -> compute q_data -> fit() -> predict()
    s_data: vector of input values
    q_data: vector of output values
    """

    # very simple, let's say q = cos(s)
    def q_true(s):
        return np.cos(s)
    
    def dq_ds_true(s):
        return np.sin(s) * (-1)

    rng = np.random.default_rng(0)
    n_data = 5000
    s_data = rng.uniform(-2.0, 2.0, n_data)
    noise_scale = 0.035 * (1.0 + 0.25 * np.abs(s_data))
    q_data = q_true(s_data) + noise_scale * rng.standard_normal(n_data)

    X = s_data.reshape(-1, 1)

    model = LinRegRFFModel()
    model.fit(X, q_data)
    q_pred = model.predict(X)

    dq_dX_pred = model.predict_dq_dX(X).ravel()

    # root mean squared error normalized by local noise scale
    train_rmse_noisy = np.sqrt(np.mean(((q_pred - q_data)) ** 2))
    train_rmse_true = np.sqrt(np.mean(((q_pred - q_true(s_data))) ** 2))
    train_rmse_true_dq = np.sqrt(np.mean(((dq_dX_pred - dq_ds_true(s_data))) ** 2))

    print("1D Raw RMSE vs noisy data:", train_rmse_noisy)
    print("1D Raw RMSE vs true function:", train_rmse_true)
    print("1D Raw RMSE vs true derivative:", train_rmse_true_dq)


    s_test = np.linspace(-2.0, 2.0, 5000)
    X_test = s_test.reshape(-1, 1)
    noise_scale_test = 0.035 * (1.0 + 0.25 * np.abs(s_test))

    q_test_true = q_true(s_test)
    q_test_pred = model.predict(X_test)
    dq_test_true = dq_ds_true(s_test)
    dq_test_pred = model.predict_dq_dX(X_test).ravel()

    test_rmse = np.sqrt(np.mean(((q_test_pred - q_test_true)) ** 2))
    print("1D Raw Test RMSE:", test_rmse)
    test_dq_rmse = np.sqrt(np.mean(((dq_test_pred - dq_test_true)) ** 2))
    print("1D Raw Test RMSE for dq:", test_dq_rmse)



"""
Note: FOR REAL 2D TABULAR DATA, SHOULD NORMALIZE INPUT COLUMNS TO SIMILAR RANGES BEFORE PROCEEDING
No normalization of inputs occurs in this code
"""
def example_2d():
    def q_true(s, T):
        base = -((1.0 + 0.20 * T**2) + 0.04 * s**2) * s
        oscillation = 0.35 * np.sin(2.5 * s) * np.exp(-0.5 * (T - 1.4) ** 2)
        transition = -0.45 * np.tanh(3.0 * (s - 0.45)) * np.exp(
            -2.0 * (T - 2.1) ** 2
        )
        return base + oscillation + transition

    def dq_ds_true(s, T):
        base = -(1.0 + 0.20 * T**2) - 0.12 * s**2
        oscillation = 0.875 * np.cos(2.5 * s) * np.exp(-0.5 * (T - 1.4) ** 2)
        u = 3.0 * (s - 0.45)
        sech2 = 1.0 / np.cosh(u) ** 2
        transition = -1.35 * sech2 * np.exp(-2.0 * (T - 2.1) ** 2)
        return base + oscillation + transition

    def dq_dT_true(s, T):
        base = -0.4 * T * s
        oscillation = (
            -0.35
            * (T - 1.4)
            * np.sin(2.5 * s)
            * np.exp(-0.5 * (T - 1.4) ** 2)
        )
        transition = (
            1.8
            * (T - 2.1)
            * np.tanh(3.0 * (s - 0.45))
            * np.exp(-2.0 * (T - 2.1) ** 2)
        )
        return base + oscillation + transition

    rng = np.random.default_rng(0)
    n_data = 5000
    s_data = rng.uniform(-2.0, 2.0, n_data)
    T_data = rng.uniform(0.0, 3.0, n_data)
    noise_scale = 0.035 * (1.0 + 0.25 * np.abs(s_data))
    q_data = q_true(s_data, T_data) + noise_scale * rng.standard_normal(n_data)

    X = np.column_stack([s_data, T_data])

    model = LinRegRFFModel()
    model.fit(X, q_data)
    q_pred = model.predict(X)
    dq_dX_pred = model.predict_dq_dX(X)

    # dq_dX_pred[:, 0] = predicted dq/ds
    # dq_dX_pred[:, 1] = predicted dq/dT

    dq_ds_exact = dq_ds_true(s_data, T_data)
    dq_dT_exact = dq_dT_true(s_data, T_data)

    rmse_q = np.sqrt(np.mean(((q_pred - q_true(s_data, T_data)) ** 2)))
    rmse_dq_ds = np.sqrt(np.mean(((dq_dX_pred[:, 0] - dq_ds_exact)) ** 2))
    rmse_dq_dT = np.sqrt(np.mean(((dq_dX_pred[:, 1] - dq_dT_exact)) ** 2))

    print("2D Raw RMSE for q:", rmse_q)
    print("2D Raw RMSE for dq_ds:", rmse_dq_ds)
    print("2D Raw RMSE for dq_dT:", rmse_dq_dT)


    

if __name__ == "__main__":
    example_simple_1d()
    example_2d()