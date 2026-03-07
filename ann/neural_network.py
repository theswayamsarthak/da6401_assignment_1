import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
import argparse
import numpy as np
import json

# load config to get correct architecture
_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_config.json")
try:
    with open(_cfg_path) as f:
        _cfg = json.load(f)
    _hidden  = _cfg["hidden_sizes"]
    _act     = _cfg.get("activation", "relu")
    _init    = _cfg.get("weight_init", "xavier")
    _loss    = _cfg.get("loss", "cross_entropy")
except Exception:
    _hidden, _act, _init, _loss = [128, 128, 128], "relu", "xavier", "cross_entropy"


class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=None,
                 output_size=10, activation=None,
                 weight_init=None, loss=None):
        if isinstance(input_size, argparse.Namespace):
            input_size = 784
        hs   = hidden_sizes if hidden_sizes is not None else _hidden
        act  = activation   if activation   is not None else _act
        wi   = weight_init  if weight_init  is not None else _init
        ls   = loss         if loss         is not None else _loss
        super().__init__(input_size, hs, output_size, act, wi, ls)

    def set_weights(self, weights):
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                layer.W = weights[f"layer_{i}_W"]
                layer.b = weights[f"layer_{i}_b"]
        elif isinstance(weights, (list, tuple)):
            for i, layer in enumerate(self.layers):
                W = np.array(weights[2 * i])
                b = np.array(weights[2 * i + 1])
                # reshape if needed
                layer.W = W.reshape(layer.W.shape)
                layer.b = b.reshape(layer.b.shape)
        else:
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
