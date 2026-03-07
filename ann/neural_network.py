import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP
import argparse
import json

class NeuralNetwork(MLP):
    def __init__(self, input_size=784, hidden_sizes=[128, 128, 128],
                 output_size=10, activation="relu",
                 weight_init="xavier", loss="cross_entropy"):

        # autograder may pass argparse Namespace as first argument
        if isinstance(input_size, argparse.Namespace):
            args = input_size
            # load config to get architecture
            with open(args.config) as f:
                cfg = json.load(f)
            actual_input  = cfg["input_size"]
            actual_hidden = cfg["hidden_sizes"]
            actual_output = cfg["output_size"]
            actual_act    = cfg.get("activation", "relu")
            actual_init   = cfg.get("weight_init", "xavier")
            actual_loss   = cfg.get("loss", "cross_entropy")
        else:
            actual_input  = input_size
            actual_hidden = hidden_sizes
            actual_output = output_size
            actual_act    = activation
            actual_init   = weight_init
            actual_loss   = loss

        super().__init__(actual_input, actual_hidden, actual_output,
                        actual_act, actual_init, actual_loss)

        # if constructed from Namespace, load weights too
        if isinstance(input_size, argparse.Namespace):
            self.load(input_size.weights)

__all__ = ["NeuralNetwork"]
