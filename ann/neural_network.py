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
        try:
            # unwrap 0-d numpy object array if needed
            if isinstance(weights_or_key, np.ndarray):
                if weights_or_key.ndim == 0:
                    weights_or_key = weights_or_key.item()
                else:
                    # plain numpy array list format
                    for i, layer in enumerate(self.layers):
                        layer.W = np.array(weights_or_key[2*i]).reshape(layer.W.shape)
                        layer.b = np.array(weights_or_key[2*i+1]).reshape(layer.b.shape)
                    return

            if value is not None:
                # set_weights("layer_0_W", array)
                self._weights_dict[weights_or_key] = np.array(value)
                if len(self._weights_dict) >= len(self.layers) * 2:
                    for i, layer in enumerate(self.layers):
                        layer.W = self._weights_dict[f"layer_{i}_W"]
                        layer.b = self._weights_dict[f"layer_{i}_b"]
            elif isinstance(weights_or_key, dict):
                for i, layer in enumerate(self.layers):
                    layer.W = np.array(weights_or_key[f"layer_{i}_W"])
                    layer.b = np.array(weights_or_key[f"layer_{i}_b"])
            elif isinstance(weights_or_key, (list, tuple)):
                for i, layer in enumerate(self.layers):
                    layer.W = np.array(weights_or_key[2*i]).reshape(layer.W.shape)
                    layer.b = np.array(weights_or_key[2*i+1]).reshape(layer.b.shape)
            elif isinstance(weights_or_key, str):
                if os.path.exists(weights_or_key):
                    params = np.load(weights_or_key, allow_pickle=True).item()
                else:
                    # it IS a key string but value is None — store empty and skip
                    return
                for i, layer in enumerate(self.layers):
                    layer.W = params[f"layer_{i}_W"]
                    layer.b = params[f"layer_{i}_b"]
        except Exception as e:
            raise RuntimeError(f"set_weights failed with input type {type(weights_or_key)}: {e}")

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights

__all__ = ["NeuralNetwork"]
