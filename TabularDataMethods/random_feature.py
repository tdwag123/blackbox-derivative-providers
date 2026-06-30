"""
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

pyrfm: A library for random feature maps in Python.
"""


def random_feature_1d():
    return 0

