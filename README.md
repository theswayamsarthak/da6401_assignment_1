# DA6401 Assignment 1 — Neural Network from Scratch

Implementation of a fully connected neural network (MLP) in pure NumPy, trained on MNIST and Fashion-MNIST.

## Project Structure

```
.
├── ann/
│   └── neural_network.py      # NeuralNetwork class (autograder interface)
├── models/
│   ├── network.py             # MLP class
│   ├── layer.py               # DenseLayer with forward/backward
│   ├── activations.py         # Sigmoid, Tanh, ReLU, Linear, Softmax
│   └── losses.py              # CrossEntropy, MSE
├── optimizers/
│   └── optimizers.py          # SGD, Momentum, NAG, RMSProp, Adam, Nadam
├── utils/
│   ├── data_utils.py          # Dataset loading, one-hot, batching
│   ├── metrics.py             # Accuracy, precision, recall, F1
│   └── grad_check.py          # Numerical gradient checker
├── src/
│   └── train.py               # Training entry point
├── inference.py               # Load and evaluate a saved model
├── sweep_config.py            # W&B hyperparameter sweep config
├── best_model.npy             # Saved best model weights
└── best_config.json           # Config used for best model
```

## Installation

```bash
pip install numpy scikit-learn keras tensorflow wandb
```

## Wandb Project and Report link

Project - https://wandb.ai/theswayamsarthak-iitmaana/da6401-mlp
Report - https://wandb.ai/theswayamsarthak-iitmaana/da6401-mlp/reports/DA6401-Assignment-1--VmlldzoxNjEzNTMyNw
