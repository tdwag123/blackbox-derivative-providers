import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
from sklearn.model_selection import train_test_split

# MLP code adapted from https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html
# could do this MUCH more easily probably.. (use NN library maybe flax that works with JAX)
# installing jax: conda install 'jax' and 'jaxlib'

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

def _predict_one(params, x):
    activations = x
    for weights, biases in params[:-1]:
        activations = jnp.tanh(jnp.dot(weights, activations) + biases)

    final_weights, final_biases = params[-1]
    return jnp.dot(final_weights, activations) + final_biases

_predict_batch = vmap(_predict_one, in_axes=(None, 0))

def _mse_loss(params, x, y):
    predictions = _predict_batch(params, x)
    return jnp.mean((predictions - y) ** 2)

@jit
def _update(params, x, y, learning_rate):
    gradients = grad(_mse_loss)(params, x, y)
    return [
        (
            weights - learning_rate * d_weights,
            biases - learning_rate * d_biases,
        )
        for (weights, biases), (d_weights, d_biases) in zip(params, gradients)
    ]

class BasicMLP:

    def __init__(
        self,
        hidden_layer_sizes = (32, 16),
        learning_rate = 1e-3,
        num_epochs = 100,
        batch_size = 32,
        seed = 0,
        shuffle = True,
        scale_inputs=False,
        verbose = False
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
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
                self.params_ = _update(self.params_, x_train[batch_indices], y_train[batch_indices], self.learning_rate)

            train_loss = float(_mse_loss(self.params_, x_train, y_train))
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

        predictions = np.asarray(_predict_batch(self.params_, x_data))
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
        return _predict_one(self.params_, x)[0]

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
        array = jnp.asarray(x, dtype=jnp.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("x must be a 1D or 2D array.")
        return array

    @staticmethod
    def _prepare_y(y):
        array = jnp.asarray(y, dtype=jnp.float32)
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

if __name__ == "__main__":
    datapath = "../../Data/NoisyDeterministicOracles/datasets/nonlinear_low_noise.csv"
    features = np.loadtxt(datapath, delimiter=',', skiprows=1, usecols=(0,1,2))   # s, T, x
    targets = np.loadtxt(datapath, delimiter=',', skiprows=1, usecols=(7,8,9,10)) # q_clean, q_noisy, dq/ds_true, dq/dT_true
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.33, random_state=0)
    X_train = features_train[:,0:2] # s, T (could also train with x)
    y_train = targets_train[:,1] # using noisy q here
    X_test = features_test[:,0:2]
    y_test = targets_test[:,1]

    model = BasicMLP(scale_inputs=True, num_epochs = 200)
    model.fit(X_train, y_train)
    q, a, b = model.evaluate(X_test[:,0], X_test[:,1])
    
    print("\nR2 scores on test set:")
    print(f"q: {model.score(y_test, q)}")
    print(f"dq/ds: {model.score(targets_test[:,2], a)}")
    print(f"dq/dT: {model.score(targets_test[:,3], b)}")

    print("\nRMSE on test set:") # FIXME: should report scaled RMSE
    print(f"q: {rmse(y_test, q)}")
    print(f"dq/ds: {rmse(targets_test[:,2], a)}")
    print(f"dq/dT: {rmse(targets_test[:,3], b)}")