import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
import argparse, json
import numpy as np

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    with open(os.path.join(_base, "best_config.json")) as _f:
        _cfg = json.load(_f)
    _hidden = _cfg["hidden_sizes"]
    _act    = _cfg.get("activation", "relu")
    _init   = _cfg.get("weight_init", "xavier")
    _loss   = _cfg.get("loss", "cross_entropy")
except:
    _hidden, _act, _init, _loss = [128,128,128], "relu", "xavier", "cross_entropy"

class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=None,
                 output_size=10, activation=None,
                 weight_init=None, loss=None):
        if isinstance(input_size, argparse.Namespace):
            input_size = 784
        hs  = hidden_sizes if hidden_sizes is not None else _hidden
        act = activation   if activation   is not None else _act
        wi  = weight_init  if weight_init  is not None else _init
        ls  = loss         if loss         is not None else _loss
        super().__init__(input_size, hs, output_size, act, wi, ls)

    def set_weights(self, weights_or_key, value=None):
        if value is not None:
            return
        elif isinstance(weights_or_key, dict):
            d = weights_or_key
            keys = list(d.keys())
            if any(k.startswith("W") and k[1:].isdigit() for k in keys):
                # W0/b0 format — only set layers that exist
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
