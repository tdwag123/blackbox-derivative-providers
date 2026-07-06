import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

"""
"The RBFSampler constructs an approximate mapping for the radial basis function kernel, 
also known as Random Kitchen Sinks [RR2007]. This transformation can be used to explicitly 
model a kernel map, prior to applying a linear algorithm, for example a linear SVM"

"The SGDClassifier estimator implements regularized linear models with stochastic gradient descent
(SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated 
along the way with a decreasing strength schedule (aka learning rate)."


Random feature models are used in machine learning to approximate kernel methods. 
They "use Monte Carlo approximations to kernel functions by randomly sampled feature maps"

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

pyrfm: A library for random feature maps in Python, https://neonnnnn.github.io/pyrfm/.

Other references:
https://pages.cs.wisc.edu/~yudongchen/cs839_sp22/5_random_features.pdf
https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html
"""

def random_fourier_features_1d(x, y):
    """
    x = array of shape (n_samples, n_features) holding training samples
    y = array of shape (n_samples, ) holding target values
    """

    
    return 0

