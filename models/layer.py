import numpy as np
from models.activations import get_activation


class DenseLayer:
    # one fully connected layer: z = XW + b, then activation
    def __init__(self, in_dim, out_dim, activation="relu", weight_init="xavier"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = activation
        self.act = get_activation(activation)
        self.W, self.b = self._init_weights(weight_init)
        # these get filled during backward(), exposed for the autograder
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        # cache needed for backprop
        self._input = None
        self._z = None     # pre-activation
        self._a = None     # post-activation

    @property
    def grad_W(self):
        return self._grad_W

    @grad_W.setter
    def grad_W(self, val):
        self._grad_W = val

    @property
    def grad_b(self):
        return self._grad_b

    @grad_b.setter
    def grad_b(self, val):
        self._grad_b = val

    def _init_weights(self, method):
        if method == "xavier":
            lim = np.sqrt(6.0 / (self.in_dim + self.out_dim))
            W = np.random.uniform(-lim, lim, (self.in_dim, self.out_dim))
        elif method == "random":
            W = np.random.randn(self.in_dim, self.out_dim) * 0.01
        elif method == "zeros":
            W = np.zeros((self.in_dim, self.out_dim))
        else:
            raise ValueError(f"unknown init method: {method}")
        b = np.zeros((1, self.out_dim))
        return W, b

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._input = X
        self._z = X @ self.W + self.b
        self._a = self.act.forward(self._z)
        return self._a

    def backward(self, delta, weight_decay=0.0):
        if delta.ndim == 1:
            delta = delta.reshape(1, -1)
        self.grad_W = self._input.T @ delta + weight_decay * self.W
        self.grad_b = delta.sum(axis=0, keepdims=True)
        return delta @ self.W.T

    def activation_grad(self, upstream):
        return upstream * self.act.backward(self._z)

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]
