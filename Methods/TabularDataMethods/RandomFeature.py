import numpy as np
import pyrfm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from GaussianProcessesWrap import KISSGPFluxST

"""
pyrfm: A library for random feature maps in Python, https://neonnnnn.github.io/pyrfm/.
"""

class RFFModel():
    def __init__(
            self, 
            n_components=100, 
            kernel='rbf', 
            gamma='auto', 
            use_offset=False, 
            random_state=None,
            n_data=900,
            s_data,
            T_data):
        # RandomFourier is feature map object!
        self.feature_map = pyrfm.random_feature.RandomFourier(n_components, kernel, gamma, use_offset, random_state)

        rng = np.random.default_rng(0)
        n_data = 900
        s_data = rng.uniform(-2.0, 2.0, n_data)
        T_data = rng.uniform(0.0, 3.0, n_data)

        self.model = KISSGPFluxST(
            s_data,
        T_data,
        q_data,
        grid_size=40,
        training_iter=70,
        learning_rate=0.08,
    )



def random_fourier_features_1d(x, M):
    """
    idea of RFF: construct explicit feature map which is of dimension much lower than number of
    observations, but with resulting inner product which approximates desired kernel function k(x,y)

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
"""