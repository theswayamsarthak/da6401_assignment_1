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
            super().__init__(784, [128, 128, 128], 10, "relu", "xavier", "cross_entropy")
        else:
            super().__init__(input_size, hidden_sizes, output_size,
                            activation, weight_init, loss)

    def set_weights(self, weights):
        """Set weights from a list of numpy arrays [W0, b0, W1, b1, ...]
           or a dict {layer_i_W: ..., layer_i_b: ...}"""
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f"layer_{i}_W" in weights:
                    layer.W = weights[f"layer_{i}_W"]
                    layer.b = weights[f"layer_{i}_b"]
        elif isinstance(weights, (list, tuple)):
            # assume alternating [W0, b0, W1, b1, ...]
            for i, layer in enumerate(self.layers):
                layer.W = weights[2 * i]
                layer.b = weights[2 * i + 1]
        else:
            # single array — try loading as npy dict
            params = np.load(weights, allow_pickle=True).item()
            for i, layer in enumerate(self.layers):
                layer.W = params[f"layer_{i}_W"]
                layer.b = params[f"layer_{i}_b"]

    def get_weights(self):
        """Return weights as list [W0, b0, W1, b1, ...]"""
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights

__all__ = ["NeuralNetwork"]
