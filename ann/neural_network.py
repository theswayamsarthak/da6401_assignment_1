import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import MLP as NeuralNetwork
__all__ = ["NeuralNetwork"]
