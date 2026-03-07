# W&B sweep config - run this once to register the sweep, then launch agents
# Usage:
#   python sweep_config.py          <- prints the sweep id
#   wandb agent <your-entity>/da6401-mlp/<sweep_id>

import wandb
import subprocess
import sys

sweep_cfg = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "epochs":        {"values": [10, 15, 20]},
        "batch_size":    {"values": [32, 64, 128]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "weight_decay":  {"values": [0.0, 0.0005, 0.001]},
        "num_layers":    {"values": [1, 2, 3, 4, 5]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":   {"values": ["random", "xavier"]},
        "loss":          {"values": ["cross_entropy"]},
    },
}


def run_sweep_agent():
    with wandb.init() as run:
        c = run.config
        cmd = [
            sys.executable, "train.py",
            "-d", "mnist",
            "-e", str(c.epochs),
            "-b", str(c.batch_size),
            "-l", c.loss,
            "-o", c.optimizer,
            "-lr", str(c.learning_rate),
            "-wd", str(c.weight_decay),
            "-nhl", str(c.num_layers),
            "-sz", str(c.hidden_size),
            "-a", c.activation,
            "-wi", c.weight_init,
            "--wandb_project", "da6401-mlp",
            "--save_path", f"sweep_{run.id}.npy",
            "--config_path", f"sweep_{run.id}.json",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="da6401-mlp")
    print(f"sweep id: {sweep_id}")
    print(f"start agents with: wandb agent {sweep_id}")
