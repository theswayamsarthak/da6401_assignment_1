import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import MLP
from models.layer import DenseLayer
import argparse
import numpy as np


class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=[128, 128, 128],
                 output_size=10, activation="relu",
                 weight_init="xavier", loss="cross_entropy"):
        if isinstance(input_size, argparse.Namespace):
            args = input_size
            input_size  = getattr(args, 'input_size',  784)
            output_size = getattr(args, 'output_size', 10)
            activation  = getattr(args, 'activation',  'relu')
            weight_init = getattr(args, 'weight_init', 'xavier')
            loss        = getattr(args, 'loss',        'cross_entropy')

            if hasattr(args, 'hidden_sizes'):
                hidden_sizes = list(args.hidden_sizes)
            elif hasattr(args, 'hidden_size'):
                hs = args.hidden_size
                nl = getattr(args, 'num_layers', 3)
                if isinstance(hs, (list, tuple)):
                    if len(hs) == 1:
                        hidden_sizes = list(hs) * nl
                    else:
                        hidden_sizes = list(hs)
                else:
                    hidden_sizes = [int(hs)] * nl
            else:
                hidden_sizes = [128] * getattr(args, 'num_layers', 3)

        super().__init__(input_size, hidden_sizes, output_size,
                         activation, weight_init, loss)

    def _make_layer(self, W_raw, b_raw, in_dim, act):
        W = np.array(W_raw)
        b = np.array(b_raw).flatten()
        out_dim = W.size // in_dim
        if W.ndim == 2 and W.shape == (out_dim, in_dim):
            W = W.T.copy()
        else:
            W = W.reshape(in_dim, out_dim).copy()
        if b.size >= out_dim:
            b = b[-out_dim:]
        else:
            b = np.zeros(out_dim)
        layer = DenseLayer(in_dim, out_dim, act, self.weight_init)
        layer.W = W
        layer.b = b.reshape(1, out_dim)
        return layer, out_dim

    def set_weights(self, weights_or_key, value=None):
        if value is not None:
            return

        elif isinstance(weights_or_key, dict):
            d = weights_or_key
            keys = list(d.keys())

            if any(k.startswith("W") and k[1:].isdigit() for k in keys):
                n = sum(1 for k in keys if k.startswith("W") and k[1:].isdigit())
                new_layers = []
                in_dim = self.input_size
                for i in range(n):
                    act = self.activation_name if i < n - 1 else "linear"
                    layer, in_dim = self._make_layer(d[f"W{i}"], d[f"b{i}"], in_dim, act)
                    new_layers.append(layer)
                self.layers = new_layers

            elif "layer_0_W" in keys:
                n = sum(1 for k in keys if k.startswith("layer_") and k.endswith("_W"))
                new_layers = []
                in_dim = self.input_size
                for i in range(n):
                    act = self.activation_name if i < n - 1 else "linear"
                    layer, in_dim = self._make_layer(
                        d[f"layer_{i}_W"], d[f"layer_{i}_b"], in_dim, act)
                    new_layers.append(layer)
                self.layers = new_layers

        elif isinstance(weights_or_key, (list, tuple)) and len(weights_or_key) > 0:
            n = len(weights_or_key) // 2
            new_layers = []
            in_dim = self.input_size
            for i in range(n):
                act = self.activation_name if i < n - 1 else "linear"
                layer, in_dim = self._make_layer(
                    weights_or_key[2 * i], weights_or_key[2 * i + 1], in_dim, act)
                new_layers.append(layer)
            self.layers = new_layers

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
