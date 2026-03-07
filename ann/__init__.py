import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
import argparse, json
import numpy as np

class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=[128, 128, 128],
                 output_size=10, activation="relu",
                 weight_init="xavier", loss="cross_entropy"):
        if isinstance(input_size, argparse.Namespace):
            input_size = 784
        # use exactly what is passed — no config overrides
        super().__init__(input_size, hidden_sizes, output_size,
                        activation, weight_init, loss)

    def set_weights(self, weights_or_key, value=None):
        if value is not None:
            return
        elif isinstance(weights_or_key, dict):
            d = weights_or_key
            keys = list(d.keys())
            if any(k.startswith("W") and k[1:].isdigit() for k in keys):
                for i, layer in enumerate(self.layers):
                    if f"W{i}" in d:
                        layer.W = np.array(d[f"W{i}"])
                    if f"b{i}" in d:
                        layer.b = np.array(d[f"b{i}"])
            elif "layer_0_W" in keys:
                for i, layer in enumerate(self.layers):
                    if f"layer_{i}_W" in d:
                        layer.W = np.array(d[f"layer_{i}_W"])
                    if f"layer_{i}_b" in d:
                        layer.b = np.array(d[f"layer_{i}_b"])
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
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights

__all__ = ["NeuralNetwork"]
