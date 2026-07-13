from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, config
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial

# MLP code adapted from https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
# monotonicity component in loss function from https://arxiv.org/pdf/1909.10662
# could do this MUCH more easily probably.. (use NN library maybe flax that works with JAX)
# installing jax: conda install 'jax' and 'jaxlib'

config.update("jax_enable_x64", True)

def _random_layer_params(in_features, out_features, key):
    weight_key, _ = random.split(key)
    scale = jnp.sqrt(1.0 / in_features)
    weights = scale * random.normal(weight_key, (out_features, in_features))
    biases = jnp.zeros((out_features,))
    return weights, biases

def _init_network_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes) - 1)
    return [
        _random_layer_params(in_features, out_features, layer_key)
        for in_features, out_features, layer_key in zip(
            layer_sizes[:-1], layer_sizes[1:], keys
        )
    ]

def _predict_one(params, x, activation_func):
    activations = x
    for weights, biases in params[:-1]:
        activations = activation_func(jnp.dot(weights, activations) + biases)

    final_weights, final_biases = params[-1]
    return jnp.dot(final_weights, activations) + final_biases

_predict_batch = vmap(_predict_one, in_axes=(None, 0, None))

def _l2_penalty(params):
    return sum(jnp.sum(weights ** 2) for weights, biases in params)

def _mse_loss(params, x, y, l2_weight, monotonicity_weight, activation_func):
    predictions = _predict_batch(params, x, activation_func)
    data_loss = jnp.mean((predictions - y) ** 2)
    l2_loss = _l2_penalty(params)

    def f_single(xi):
        return _predict_one(params, xi, activation_func)[0]
    grads = vmap(grad(f_single))(x)
    mono_grads = grads[:,0]
    mono_penalty = jnp.mean(jnp.maximum(0.0, mono_grads)) # may want to swithc out max for smth smooth

    return data_loss + l2_weight * l2_loss + monotonicity_weight * mono_penalty

@partial(jit, static_argnums=(6,))
def _update(params, x, y, learning_rate, l2_weight, monotonicity_weight, activation_func):
    gradients = grad(_mse_loss)(params, x, y, l2_weight, monotonicity_weight, activation_func)
    return [
        (
            weights - learning_rate * d_weights,
            biases - learning_rate * d_biases,
        )
        for (weights, biases), (d_weights, d_biases) in zip(params, gradients)
    ]

class MLP:
    '''MLP implementation with L2 regularization and soft monotonicity constraint'''

    def __init__(
        self,
        hidden_layer_sizes = (32, 16),
        learning_rate = 1e-3,
        num_epochs = 100,
        batch_size = 32,
        l2_weight = 0.0,
        activation_func = "tanh",
        monotonicity_weight = 0.0,
        seed = 0,
        shuffle = True,
        scale_inputs=False,
        verbose = False
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.l2_weight = l2_weight

        if activation_func == "tanh":
            self.activation_func = jnp.tanh
        elif activation_func == "softplus":
            self.activation_func = nn.softplus
        elif activation_func == "relu":
            self.activation_func = nn.relu
        elif activation_func == "none":
            self.activation_func = lambda x: x
        else:
            raise ValueError(f"Unknown activation_func: {activation_func}")

        self.monotonicity_weight = monotonicity_weight
        self.seed = seed
        self.shuffle = shuffle
        self.scale_inputs = scale_inputs
        self.verbose = verbose

        self.params_ = None
        self.input_dim_ = None
        self.output_dim_ = None
        self.history_ = None
        self.x_mean_ = None
        self.x_std_ = None
        
        self.gradient = None

    def fit(self, x, y):
        x_train = self._prepare_x(x)
        y_train = self._prepare_y(y)

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        if self.scale_inputs:
            self.x_mean_ = jnp.mean(x_train, axis=0)
            self.x_std_ = jnp.std(x_train, axis=0)
            self.x_std_ = jnp.where(self.x_std_ == 0, 1.0, self.x_std_)
            x_train = self._scale_x(x_train)
        else:
            self.x_mean_ = jnp.zeros(x_train.shape[1])
            self.x_std_ = jnp.ones(x_train.shape[1])

        self.input_dim_ = x_train.shape[1]
        self.output_dim_ = y_train.shape[1]
        layer_sizes = [self.input_dim_, *self.hidden_layer_sizes, self.output_dim_]
        self.params_ = _init_network_params(layer_sizes, random.PRNGKey(self.seed))
        self.history_ = []

        rng = np.random.default_rng(self.seed)
        sample_count = x_train.shape[0]

        for epoch in range(self.num_epochs):
            indices = np.arange(sample_count)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, sample_count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                self.params_ = _update(self.params_, x_train[batch_indices], y_train[batch_indices], self.learning_rate, self.l2_weight, self.monotonicity_weight, self.activation_func)

            train_loss = float(_mse_loss(self.params_, x_train, y_train, self.l2_weight, self.monotonicity_weight, self.activation_func))
            record = {"epoch": float(epoch), "train_loss": train_loss}

            self.history_.append(record)

            if self.verbose:
                message = (
                    f"Epoch {epoch} - "
                    f"loss: {train_loss:.6f}"
                )
                print(message)

        self.gradient = grad(self.f)

        return self

    def predict(self, x):
        self._require_fitted()
        x_data = self._scale_x(x)
        if x_data.shape[1] != self.input_dim_:
            raise ValueError(
                f"x has {x_data.shape[1]} features, expected {self.input_dim_}."
            )

        predictions = np.asarray(_predict_batch(self.params_, x_data, self.activation_func))
        if self.output_dim_ == 1:
            return predictions.ravel()
        return predictions

    def score(self, y_true, y_pred):
        '''R^2'''
        y_true = np.asarray(self._prepare_y(y_true))
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        residual_sum = np.sum((y_true - y_pred) ** 2)
        total_sum = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        if total_sum == 0:
            return 0.0
        return float(1.0 - residual_sum / total_sum)

    def f(self, x):
        '''Wrapper for JAX grad'''
        return _predict_one(self.params_, x, self.activation_func)[0]

    def _scale_x(self, x):
        if self.x_mean_ is None or self.x_std_ is None:
            raise ValueError("Input scaling parameters have not been fitted yet.")

        x = self._prepare_x(x)
        return (x - self.x_mean_) / self.x_std_

    def evaluate(self, s_q, T_q):
        '''Wrapper function for returning flux and derivatives at quadrature states'''
        if self.gradient is None:
            raise ValueError("oops")

        if self.input_dim_ != 2:
            raise ValueError("mismatch :(")

        s_q = np.atleast_1d(np.asarray(s_q, dtype=float))
        T_q = np.atleast_1d(np.asarray(T_q, dtype=float))

        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        state = np.column_stack((s_q, T_q))
        state_sc = self._scale_x(state)
        grad_sc = vmap(self.gradient)(state_sc)
        grad = grad_sc / self.x_std_

        q_g = self.predict(state)
        a_g = grad[:,0]
        b_g = grad[:,1]

        return q_g, a_g, b_g

    @staticmethod
    def _prepare_x(x): # (watch out if passing in 1 sample)
        array = jnp.asarray(x, dtype=jnp.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("x must be a 1D or 2D array.")
        return array

    @staticmethod
    def _prepare_y(y):
        array = jnp.asarray(y, dtype=jnp.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("y must be a 1D or 2D array.")
        return array

    def _require_fitted(self):
        if self.params_ is None:
            raise ValueError("This BasicMLP instance is not fitted yet.")

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_pred-y_true)**2))

def test1():
    datapath = "../../Data/NoisyDeterministicOracles/datasets/nonlinear_low_noise.csv"
    features = np.loadtxt(datapath, delimiter=',', skiprows=1, usecols=(0,1,2))   # s, T, x
    targets = np.loadtxt(datapath, delimiter=',', skiprows=1, usecols=(7,8,9,10)) # q_clean, q_noisy, dq/ds_true, dq/dT_true
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.33, random_state=0)
    X_train = features_train[:,0:2] # s, T (could also train with x by setting 0:3)
    y_train = targets_train[:,1] # using noisy q here
    X_test = features_test[:,0:2]
    y_test = targets_test[:,1]

    model = MLP(
        scale_inputs=False, 
        num_epochs = 200, 
        l2_weight=0.0,           # set nonzero for regularization
        monotonicity_weight=0.0, # set nonzero for (closer to) monotone output
        hidden_layer_sizes=(16, 8), 
        verbose=False)
    model.fit(X_train, y_train)
    q_pred, a_pred, b_pred = model.evaluate(X_test[:,0], X_test[:,1])
    
    print("\nR2 scores on test set:")
    print(f"q: {model.score(y_test, q_pred)}")
    print(f"dq/ds: {model.score(targets_test[:,2], a_pred)}")
    print(f"dq/dT: {model.score(targets_test[:,3], b_pred)}")

    print("\nRMSE on test set:") 
    print(f"q: {rmse(y_test, q_pred)}")
    print(f"dq/ds: {rmse(targets_test[:,2], a_pred)}")
    print(f"dq/dT: {rmse(targets_test[:,3], b_pred)}")

    print("\nBut is it monotone?")
    print(f"Number of positive values for dq/ds: {np.sum(a_pred > 0)}")

if __name__ == "__main__":
    test1()
