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

    def set_weights(self, weights):
        # handle dict format: {"layer_0_W": ..., "layer_0_b": ...}
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[f"layer_{i}_W"])
                layer.b = np.array(weights[f"layer_{i}_b"])
        # handle list/tuple format: [W0, b0, W1, b1, ...]
        elif isinstance(weights, (list, tuple)):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2 * i]).reshape(layer.W.shape)
                layer.b = np.array(weights[2 * i + 1]).reshape(layer.b.shape)
        # handle npy file path
        elif isinstance(weights, str):
            params = np.load(weights, allow_pickle=True).item()
            for i, layer in enumerate(self.layers):
                layer.W = params[f"layer_{i}_W"]
                layer.b = params[f"layer_{i}_b"]

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights

__all__ = ["NeuralNetwork"]
