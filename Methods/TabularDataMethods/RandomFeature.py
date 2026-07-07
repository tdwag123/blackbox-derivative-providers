import numpy as np
import pyrfm
from sklearn.linear_model import Ridge

"""
pyrfm: A library for random feature maps in Python, https://neonnnnn.github.io/pyrfm/.
"""

class RFFModel():
    """
    Random Fourier Features model for 1D constitutive law q = q(s) using RBF kernel and ridge regression
    (s) -> random Fourier -> phi(s) -> ridge regression -> q

    RFF is approximation of RBF kernel for large datasets to stop computer from running out of memory
    """

    def __init__(self, n_components=100, kernel='rbf', gamma='auto', use_offset=False, random_state=None, alpha=1.0):
        """
        n_components: number of Random Fourier features

        feature_map: turns X into phi(X)    Note: RandomFourier is feature map object!
                                            https://neonnnnn.github.io/pyrfm/generated/pyrfm.random_feature.RandomFourier.html#pyrfm.random_feature.RandomFourier
        model: learns q from phi(X)
                                            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        """

        self.feature_map = pyrfm.random_feature.RandomFourier(
            n_components=n_components, 
            kernel=kernel, 
            gamma=gamma, 
            use_offset=use_offset,
            random_state=random_state
        )
        self.model = Ridge(alpha=alpha, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver="auto")


    def fit(self, X, y):
        """
        Fit linear ridge regression model on mapped feature matrix against target vector y
        with regularization parameter alpha.

        X = data matrix; array of shape (n_samples, n_features) holding training samples
            Note: In the 1D case q = q(s), n_features = 1
        y = target vector; 1D array of shape (n_samples, ) holding target values
        """

        new_X = self.feature_map.fit_transform(X, y) # Fit feature_map to data, then transform it. Fits "transformer" to X, returns transformed version of X
        self.model.fit(new_X, y) # Fits ridge regression model onto mapped feature matrix (the transformed version of X)

        return self # feature_map is now fitted, model is now trained
    

    def predict(self, X):
        """
        Based on model formed by fit, predicts value for specific point.
        """
        new_X = self.feature_map.transform(X) # apply approximate feature map to input
        return self.model.predict(new_X) # predict using linear model


# ---------------------------------------------------------------------------------------------------------------------
def example():
    return 0

if __name__ == "__main__":
    example()


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