import numpy as np
# import pyrfm ---- incompatible with most current scikit-learn :(
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

"""
pyrfm: A library for random feature maps in Python, https://neonnnnn.github.io/pyrfm/.
"""

class RFFModel():
    """
    Random Fourier Features model for n-D constitutive laws using RBF kernel approximation and ridge regression
    (X) -> random Fourier -> phi(X) -> ridge regression -> predict q -> analytically solve for predicted dq_dX

    RFF is approximation of RBF kernel for large datasets to stop computer from running out of memory
    """

    def __init__(self, n_components=400, gamma=0.1, random_state=None, alpha=1e-6):
        """
        n_components: number of Random Fourier features

        feature_map: turns X into phi(X)    
                                            https://scikit-learn.org/stable/modules/kernel_approximation.html#rbf-kernel-approx
        reg_model: learns q from phi(X)
                                            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        """

        self.n_components = n_components
        self.feature_map = RBFSampler(
            n_components=n_components, 
            gamma=gamma, 
            random_state=random_state
        )
        self.reg_model = Ridge(alpha=alpha, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver="auto")


    def fit(self, X, y):
        """
        Fit linear ridge regression model on mapped feature matrix against target vector y
        with regularization parameter alpha.

        X = data matrix; array of shape (n_samples, n_features) holding training samples
            Note: In the 1D case q = q(s), n_features = 1
                  In the 2D case q = q(s,T), n_features = 2
        y = target vector; 1D array of shape (n_samples, ) holding target values
        """

        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(f"y has shape {y.shape}, which looks like {y.shape[1]} targets."
            "RFFModel.predict_dq_dX only supports single-output regression. "
            "Fit a separate RFFModel per output column instead."
        )
    
        new_X = self.feature_map.fit_transform(X, y) # Fit feature_map to data, then transform it. Fits "transformer" to X, returns transformed version of X
        self.reg_model.fit(new_X, y) # Fits ridge regression model onto mapped feature matrix (the transformed version of X)

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

        RBFSampler approximates an RBF kernel feature map using random Fourier features. The mapping relies on a 
        Monte Carlo approximation to the kernel values. The fit function performs the Monte Carlo sampling, whereas
        the transform method performs the mapping of the data.

        so according to RBFSampler documentation, it uses mapping function z for single input vector x:
            phi(x) = sqrt(2/n_components) cos(Wx + b)  where W is random_weights_ and b is random_offset_

        ridge model predicts q = phi(x)^T * beta + b_0  where beta = coef_ and b_0 = intercept_
                Note: coef_ is ndarray of shape (n_features,) or (n_targets, n_features)
        so grad_x q = beta^T * grad_x phi(x)

        and since phi(x) = sqrt(2/n_components) cos(W^T x + b)
        grad_x phi(x) = -sqrt(2/n_components) sin(W^T x + b) * W^T

        so grad_x q = beta^T * -sqrt(2/n_components) sin(W^T x + b) * W^T
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
    n_data = 2000
    s_data = rng.uniform(-2.0, 2.0, n_data)
    noise_scale = 0.035 * (1.0 + 0.25 * np.abs(s_data))
    q_data = q_true(s_data) + noise_scale * rng.standard_normal(n_data)

    X = s_data.reshape(-1, 1)

    model = RFFModel()
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


    s_test = np.linspace(-2.0, 2.0, 400)
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
    n_data = 2000
    s_data = rng.uniform(-2.0, 2.0, n_data)
    T_data = rng.uniform(0.0, 3.0, n_data)
    noise_scale = 0.035 * (1.0 + 0.25 * np.abs(s_data))
    q_data = q_true(s_data, T_data) + noise_scale * rng.standard_normal(n_data)

    X = np.column_stack([s_data, T_data])

    model = RFFModel()
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


"""
"RBFSampler constructs approximate mapping for radial basis function kernel. 
can be used to explicitly model a kernel map, prior to applying a linear algorithm"

"SGDClassifier estimator implements regularized linear models with stochastic gradient descent
(SGD) learning: gradient of loss is estimated each sample at a time and the model is updated 
along the way with a decreasing strength schedule (aka learning rate)"

from sklearn.linear_model import SGDClassifier  # used for classification (discrete values) so we will not be using this


RF models are used in machine learning to approximate kernel methods.

Note: hyperparameters are set before training data; model parameters are learned from data

Basic structure:
    input(x)
        -> random nonlinear feature map z(x) in R^m where m is number of random features
        -> learned linear model
        -> prediction f(x) = z(x)^T a

prediction = w_1 * random_feature_1(x) + w_2 * random_feature_2(x) + ... + w_m * random_feature_m(x)

inputs:
    m = number of random features
    lambda = regularization strength

Other references:
https://proceedings.mlr.press/v70/avron17a.html

https://pages.cs.wisc.edu/~yudongchen/cs839_sp22/5_random_features.pdf
https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html

idea of RFF: construct explicit feature map which is of dimension much lower than number of
observations, but with resulting inner product which approximates desired kernel function k(x,y)
"""