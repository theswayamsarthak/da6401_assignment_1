import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
from models.layer import DenseLayer
from models.losses import _softmax
import argparse
import numpy as np

class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=[128,128,128],
                 output_size=10, activation="relu",
                 weight_init="xavier", loss="cross_entropy"):
        if isinstance(input_size, argparse.Namespace):
            input_size = 784
        super().__init__(input_size, hidden_sizes, output_size,
                        activation, weight_init, loss)

    def forward(self, X):
        self._last_X = X
        return super().forward(X)

    def backward(self, y_true, y_pred):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        if y_true.ndim == 1:
            onehot = np.zeros_like(y_pred)
            onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
            y_true = onehot

        probs = _softmax(y_pred)
        delta = (probs - y_true) / y_pred.shape[0]

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            if layer.activation_name != "linear":
                delta = layer.activation_grad(delta)
            delta = layer.backward(delta, weight_decay=0.0)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def set_weights(self, weights_or_key, value=None):
        if value is not None:
            return
        elif isinstance(weights_or_key, dict):
            d = weights_or_key
            keys = list(d.keys())
            if any(k.startswith("W") and k[1:].isdigit() for k in keys):
                n = sum(1 for k in keys if k.startswith("W") and k[1:].isdigit())
                new_layers = []
                expected_in = self.input_size
                for i in range(n):
                    W = np.array(d[f"W{i}"])
                    b = np.array(d[f"b{i}"])
                    act = self.activation_name if i < n-1 else "linear"
                    if W.ndim == 2 and W.shape[0] != expected_in:
                        W = W.T
                    in_dim, out_dim = W.shape
                    expected_in = out_dim
                    layer = DenseLayer(in_dim, out_dim, act, self.weight_init)
                    layer.W = W.copy()
                    layer.b = b.copy()
                    new_layers.append(layer)
                self.layers = new_layers
            elif "layer_0_W" in keys:
                for i, layer in enumerate(self.layers):
                    if f"layer_{i}_W" in d:
                        layer.W = np.array(d[f"layer_{i}_W"]).copy()
                        layer.b = np.array(d[f"layer_{i}_b"]).copy()
        elif isinstance(weights_or_key, (list, tuple)):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights_or_key[2*i]).reshape(layer.W.shape)
                layer.b = np.array(weights_or_key[2*i+1]).reshape(layer.b.shape)
        elif isinstance(weights_or_key, str):
            if os.path.exists(weights_or_key):
                params = np.load(weights_or_key, allow_pickle=True).item()
                for i, layer in enumerate(self.layers):
                    layer.W = params[f"layer_{i}_W"]
                    layer.b = params[f"layer_{i}_b"]

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

__all__ = ["NeuralNetwork"]
