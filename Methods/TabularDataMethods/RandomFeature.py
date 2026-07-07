import numpy as np
import pyrfm
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier  # used for classification (discrete values) so we will not be using this

"""
pyrfm: A library for random feature maps in Python, https://neonnnnn.github.io/pyrfm/.
"""

class RFFModel():
    """
    Random Fourier Features model for constitutive law q = q(s,T) using RBF kernel and ridge regression
    (s,T) -> random Fourier -> phi(s,T) -> ridge regression -> q
    """
    def __init__(self, n_components=100, kernel='rbf', gamma='auto', use_offset=False, random_state=None):
        # RandomFourier is feature map object!
        self.feature_map = pyrfm.random_feature.RandomFourier(
            n_components=n_components, 
            kernel=kernel, 
            gamma=gamma, 
            use_offset=use_offset, 
            random_state=random_state
        )

        self.model = Ridge(alpha=1, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver="auto")

    def fit():
        """
        x = array of shape (n_samples, n_features) holding training samples
        y = array of shape (n_samples, ) holding target values
        """
        return 0


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
https://pages.cs.wisc.edu/~yudongchen/cs839_sp22/5_random_features.pdf
https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html

idea of RFF: construct explicit feature map which is of dimension much lower than number of
observations, but with resulting inner product which approximates desired kernel function k(x,y)
"""