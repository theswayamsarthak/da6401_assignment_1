import numpy as np
from models.layer import DenseLayer
from models.losses import get_loss, _softmax


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="relu", weight_init="xavier", loss="cross_entropy"):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.loss_name = loss
        self.loss_fn = get_loss(loss)
        self.layers = self._build()

    def _build(self):
        layers = []
        dims = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(dims) - 1):
            act = self.activation_name if i < len(dims) - 2 else "linear"
            layers.append(DenseLayer(dims[i], dims[i+1], act, self.weight_init))
        return layers

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_proba(self, X):
        return _softmax(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def backward(self, X, y_onehot, weight_decay=0.0):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y_onehot.ndim == 1:
            y_onehot = y_onehot.reshape(1, -1)

        logits = self.forward(X)
        loss = self.loss_fn.forward(logits, y_onehot)
        delta = self.loss_fn.backward(logits, y_onehot)

        for layer in reversed(self.layers):
            if layer.activation_name != "linear":
                delta = layer.activation_grad(delta)
            delta = layer.backward(delta, weight_decay)

        return loss

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_W"] = layer.W
            params[f"layer_{i}_b"] = layer.b
        np.save(path, params)
        print(f"saved model to {path}")

    def load(self, path):
        params = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            layer.W = params[f"layer_{i}_W"]
            layer.b = params[f"layer_{i}_b"]
        print(f"loaded weights from {path}")

    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "weight_init": self.weight_init,
            "loss": self.loss_name,
        }

    def __repr__(self):
        arch = [self.input_size] + self.hidden_sizes + [self.output_size]
        return f"MLP(layers={arch}, act={self.activation_name}, init={self.weight_init}, loss={self.loss_name})"
