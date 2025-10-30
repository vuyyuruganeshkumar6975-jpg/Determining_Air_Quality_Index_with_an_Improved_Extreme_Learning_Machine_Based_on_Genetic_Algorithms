import numpy as np
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RandomLayer:
    """
    Implements a random layer for the Extreme Learning Machine.
    This generates random weights and biases for the hidden layer.
    """
    def __init__(self, n_hidden=10, activation=np.tanh, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = np.random.RandomState(random_state)

    def fit_transform(self, X):
        n_features = X.shape[1]
        self.random_weights_ = self.random_state.uniform(-1, 1, (n_features, self.n_hidden))
        self.bias_ = self.random_state.uniform(-1, 1, self.n_hidden)
        return self.transform(X)

    def transform(self, X):
        linear_output = np.dot(X, self.random_weights_) + self.bias_
        return self.activation(linear_output)


class GeneticELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=10, activation=np.tanh, random_state=None, mutation_rate=0.1, n_generations=10):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.coefs_ = None
        self.hidden_activations_ = None

    def fit(self, X, y):
        """
        Fits the model to the training data.
        """
        self.X_ = X  # Store input features for reuse in mutation
        X, y = check_X_y(X, y, ensure_2d=True)

        # Ensure hidden layer is initialized correctly with the number of input features
        self.hidden_layer_ = RandomLayer(n_hidden=self.n_hidden, activation=self.activation, random_state=self.random_state)
        
        # Fit and transform the input features to generate hidden activations
        self.hidden_activations_ = self.hidden_layer_.fit_transform(X)

        # Initial coefficient calculation (pseudo-inverse of hidden layer activations)
        self.coefs_ = pinv(self.hidden_activations_).dot(y)

        # Genetic optimization
        for generation in range(self.n_generations):
            self._mutate()
            preds = self.hidden_activations_.dot(self.coefs_)
            fitness = np.sqrt(np.mean((y - preds) ** 2))  # RMSE as fitness
            print(f"Generation {generation + 1}/{self.n_generations}, Fitness (RMSE): {fitness:.4f}")

        self.fitted_ = True
        return self

    def _mutate(self):
        """
        Applies mutation to the hidden layer's weights and biases.
        """
        mutation_mask_weights = np.random.rand(*self.hidden_layer_.random_weights_.shape) < self.mutation_rate
        mutation_mask_biases = np.random.rand(*self.hidden_layer_.bias_.shape) < self.mutation_rate

        # Apply mutation to the weights and biases
        self.hidden_layer_.random_weights_[mutation_mask_weights] += np.random.normal(size=np.sum(mutation_mask_weights))
        self.hidden_layer_.bias_[mutation_mask_biases] += np.random.normal(size=np.sum(mutation_mask_biases))

        # Update hidden activations after mutation
        self.hidden_activations_ = self.hidden_layer_.transform(self.X_)

    def predict(self, X):
        """
        Predicts the target values for the given input data.
        """
        check_is_fitted(self, "fitted_")
        X = check_array(X, ensure_2d=True)
        hidden_activations = self.hidden_layer_.transform(X)
        return hidden_activations.dot(self.coefs_)
