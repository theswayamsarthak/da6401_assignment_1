import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import MLP
from optimizers.optimizers import get_optimizer
from utils.data_utils import load_dataset, one_hot, get_batches
from utils.metrics import compute_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       default="mnist", choices=["mnist", "fashion"])
    p.add_argument("-e",   "--epochs",         type=int, default=10)
    p.add_argument("-b",   "--batch_size",     type=int, default=64)
    p.add_argument("-l",   "--loss",           default="cross_entropy",
                   choices=["cross_entropy", "mean_squared_error"])
    p.add_argument("-o",   "--optimizer",      default="adam",
                   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    p.add_argument("-nhl", "--num_layers",     type=int, default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int, nargs="+", default=[128])
    p.add_argument("-a",   "--activation",     default="relu",
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi",  "--weight_init",    default="xavier",
                   choices=["random", "xavier"])
    p.add_argument("--wandb_project",  default="da6401-mlp")
    p.add_argument("--wandb_entity",   default=None)
    p.add_argument("--no_wandb",       action="store_true", default=True)
    p.add_argument("--save_path",      default="best_model.npy")
    p.add_argument("--config_path",    default="best_config.json")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def train(args):
    np.random.seed(args.seed)

    if len(args.hidden_size) == 1:
        hidden_sizes = args.hidden_size * args.num_layers
    else:
        if len(args.hidden_size) != args.num_layers:
            raise ValueError("--hidden_size should have 1 value or exactly num_layers values")
        hidden_sizes = args.hidden_size

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)

    model = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
    )
    print(model)

    opt_kwargs = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
    if args.optimizer in ("momentum", "nag"):
        opt_kwargs["beta"] = 0.9
    elif args.optimizer == "rmsprop":
        opt_kwargs["beta"] = 0.9
    elif args.optimizer in ("adam", "nadam"):
        opt_kwargs["beta1"] = 0.9
        opt_kwargs["beta2"] = 0.999

    optimizer = get_optimizer(args.optimizer, **opt_kwargs)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       config=vars(args), reinit=True)
        except ImportError:
            print("wandb not found, skipping logging")
            use_wandb = False

    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for Xb, yb in get_batches(X_train, y_train, args.batch_size):
            yb_oh = one_hot(yb, 10)
            loss = model.backward(Xb, yb_oh, weight_decay=args.weight_decay)
            optimizer.step(model.layers)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        train_metrics = compute_metrics(y_train, model.predict(X_train))
        val_metrics   = compute_metrics(y_val,   model.predict(X_val))

        print(f"\nepoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")
        print_metrics(train_metrics, "train")
        print_metrics(val_metrics, "val")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            model.save(args.save_path)
            cfg = model.get_config()
            cfg.update({
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataset": args.dataset,
            })
            with open(args.config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"\nbest val f1={best_val_f1:.4f} at epoch {best_epoch}")
    model.load(args.save_path)
    test_metrics = compute_metrics(y_test, model.predict(X_test))
    print("\n--- test results (best model) ---")
    print_metrics(test_metrics, "test")

    if use_wandb:
        import wandb
        wandb.log({
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1"],
        })
        wandb.finish()

    return model, test_metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
