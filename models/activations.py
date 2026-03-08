import numpy as np


# base class, mostly just so we have a consistent interface
class Activation:
    def forward(self, z):
        raise NotImplementedError

    def backward(self, z):
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, z):
        # clip to avoid overflow in exp, found -500/500 works fine
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Tanh(Activation):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        return 1 - np.tanh(z)**2


class ReLU(Activation):
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        # gradient is 1 where z > 0, else 0
        return (z > 0).astype(float)


# used for the output layer - just pass logits through, loss handles softmax
class Linear(Activation):
    def forward(self, z):
        return z

    def backward(self, z):
        return np.ones_like(z)


class Softmax(Activation):
    def forward(self, z):
        # subtract max per row for numerical stability (standard trick)
        shifted = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, z):
        # full jacobian is messy; for cross-entropy we handle it together in the loss
        s = self.forward(z)
        return s * (1 - s)


def get_activation(name):
    name = name.lower()
    options = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "linear": Linear,
        "softmax": Softmax,
    }
    if name not in options:
        raise ValueError(f"don't know activation '{name}', pick from {list(options.keys())}")
    return options[name]()
