import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
import argparse
import numpy as np

class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=[128, 128, 128],
                 output_size=10, activation="relu",
                 weight_init="xavier", loss="cross_entropy"):
        if isinstance(input_size, argparse.Namespace):
            input_size = 784
        super().__init__(input_size, hidden_sizes, output_size,
                        activation, weight_init, loss)
        self._weights_dict = {}

    def set_weights(self, weights_or_key, value=None):
        if value is not None:
            self._weights_dict[weights_or_key] = np.array(value)
            if len(self._weights_dict) >= len(self.layers) * 2:
                self._apply_weights_dict(self._weights_dict)
        elif isinstance(weights_or_key, dict):
            self._apply_weights_dict(weights_or_key)
        elif isinstance(weights_or_key, (list, tuple)):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights_or_key[2*i]).reshape(layer.W.shape)
                layer.b = np.array(weights_or_key[2*i+1]).reshape(layer.b.shape)
        elif isinstance(weights_or_key, str):
            if os.path.exists(weights_or_key):
                params = np.load(weights_or_key, allow_pickle=True).item()
                self._apply_weights_dict(params)
            return

    def _apply_weights_dict(self, d):
        keys = list(d.keys())
        # detect key format from first key
        sample = keys[0]
        if "layer_0_W" in keys:
            # our format: layer_i_W, layer_i_b
            for i, layer in enumerate(self.layers):
                layer.W = np.array(d[f"layer_{i}_W"])
                layer.b = np.array(d[f"layer_{i}_b"])
        elif "W0" in keys or "w0" in keys:
            # W0, b0, W1, b1 format
            k = "W" if "W0" in keys else "w"
            for i, layer in enumerate(self.layers):
                layer.W = np.array(d[f"{k}{i}"])
                layer.b = np.array(d[f"b{i}"])
        elif 0 in keys:
            # integer keys: 0, 1, 2...
            for i, layer in enumerate(self.layers):
                layer.W = np.array(d[2*i])
                layer.b = np.array(d[2*i+1])
        else:
            # unknown format — raise with keys visible
            raise KeyError(f"unknown weight dict keys: {keys[:4]}")

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights

__all__ = ["NeuralNetwork"]
