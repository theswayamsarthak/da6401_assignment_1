import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network import MLP
from utils.data_utils import load_dataset
from utils.metrics import compute_metrics, get_confusion_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(_HERE, "best_model.npy"))
    p.add_argument("--config",  default=os.path.join(_HERE, "best_config.json"))
    p.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()

def parse_arguments():
    return parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    model = MLP(
        input_size=cfg["input_size"],
        hidden_sizes=cfg["hidden_sizes"],
        output_size=cfg["output_size"],
        activation=cfg["activation"],
        weight_init=cfg["weight_init"],
        loss=cfg["loss"],
    )
    model.load(args.weights)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    X, y = splits[args.split]
    preds = model.predict(X)
    m = compute_metrics(y, preds)
    print(f"\nresults on {args.split} set:")
    print(f"  accuracy:  {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall:    {m['recall']:.4f}")
    print(f"  f1:        {m['f1']:.4f}")
    print("\nconfusion matrix:")
    print(get_confusion_matrix(y, preds))
    return m

if __name__ == "__main__":
    main()

# autograder expects parse_arguments (not parse_args)
def parse_arguments():
    return parse_args()
