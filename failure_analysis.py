"""
Creative failure visualization for best model.
Run in Colab:
    !python failure_analysis.py
"""

import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network import MLP
from utils.data_utils import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "best_model.npy"
CONFIG_PATH = "best_config.json"
DATASET     = "mnist"
WANDB_PROJECT = "da6401-mlp"
WANDB_ENTITY  = "theswayamsarthak-iitmaana"
CLASS_NAMES = [str(i) for i in range(10)]
# ──────────────────────────────────────────────────────────────────────────────

def load_best_model():
    import json
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    model = MLP(
        input_size=784,
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init="xavier",
        loss="cross_entropy",
    )
    model.load(MODEL_PATH)
    return model

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def get_predictions(model, X):
    logits = model.forward(X)
    probs  = softmax(logits)
    preds  = probs.argmax(axis=1)
    confs  = probs.max(axis=1)
    return preds, probs, confs

# ── 1. Confusion matrix (raw numpy) ──────────────────────────────────────────
def compute_confusion(y_true, y_pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ── 2. Wall of Shame: worst confusion pairs ───────────────────────────────────
def wall_of_shame(X_test, y_test, y_pred, probs, cm, top_k=6, imgs_per_pair=5):
    """Show actual test images for the top-k most confused pairs."""
    # get top-k off-diagonal confusion pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat = cm_copy.flatten()
    top_indices = flat.argsort()[::-1][:top_k]
    pairs = [(i // 10, i % 10) for i in top_indices]

    fig = plt.figure(figsize=(18, top_k * 2.5))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle("Wall of Shame — Most Confused Digit Pairs",
                 fontsize=18, color='white', fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(top_k, imgs_per_pair + 1,
                           hspace=0.5, wspace=0.15)

    for row, (true_cls, pred_cls) in enumerate(pairs):
        # label column
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.set_facecolor('#0f0f1a')
        ax_label.axis('off')
        count = cm[true_cls, pred_cls]
        ax_label.text(0.5, 0.5,
                      f"True: {true_cls}\n→ Pred: {pred_cls}\n({count}×)",
                      ha='center', va='center', fontsize=11,
                      color='#ff6b6b', fontweight='bold',
                      transform=ax_label.transAxes)

        # find misclassified examples for this pair
        mask = (y_test == true_cls) & (y_pred == pred_cls)
        idxs = np.where(mask)[0]

        for col in range(imgs_per_pair):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.set_facecolor('#1a1a2e')
            if col < len(idxs):
                idx = idxs[col]
                img = X_test[idx].reshape(28, 28)
                conf = probs[idx, pred_cls]
                ax.imshow(img, cmap='plasma', interpolation='nearest')
                ax.set_title(f"{conf:.0%} conf", fontsize=8,
                             color='#ffd700', pad=2)
            ax.axis('off')

    plt.tight_layout()
    return fig

# ── 3. Confidence histogram of correct vs wrong ───────────────────────────────
def confidence_histogram(y_test, y_pred, confs):
    correct = confs[y_test == y_pred]
    wrong   = confs[y_test != y_pred]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    bins = np.linspace(0, 1, 30)
    ax.hist(correct, bins=bins, alpha=0.75, color='#00d4aa',
            label=f'Correct ({len(correct)})', edgecolor='none')
    ax.hist(wrong,   bins=bins, alpha=0.85, color='#ff6b6b',
            label=f'Wrong ({len(wrong)})',   edgecolor='none')

    ax.set_xlabel("Model Confidence", color='white', fontsize=12)
    ax.set_ylabel("Count",            color='white', fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Misclassified",
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(fontsize=11, facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    return fig

# ── 4. Heatmap confusion matrix (styled) ─────────────────────────────────────
def styled_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # normalize by row for recall view
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

    # annotate cells
    for i in range(10):
        for j in range(10):
            color = 'white' if cm_norm[i, j] < 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, color='white')
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, color='white')
    ax.set_xlabel("Predicted", color='white', fontsize=12)
    ax.set_ylabel("True",      color='white', fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised recall)",
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data and model...")
    _, _, _, _, X_test, y_test = load_dataset(DATASET)
    model = load_best_model()

    y_pred, probs, confs = get_predictions(model, X_test)
    cm = compute_confusion(y_test, y_pred)

    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")
    print(f"Total errors:  {(y_pred != y_test).sum()} / {len(y_test)}")

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="failure-analysis",
        job_type="analysis",
    )

    # log all three figures
    fig1 = styled_confusion_matrix(cm)
    fig2 = wall_of_shame(X_test, y_test, y_pred, probs, cm)
    fig3 = confidence_histogram(y_test, y_pred, confs)

    wandb.log({
        "confusion_matrix_styled": wandb.Image(fig1),
        "wall_of_shame":           wandb.Image(fig2),
        "confidence_histogram":    wandb.Image(fig3),
    })

    # also log W&B native confusion matrix
    wandb.log({
        "conf_matrix_native": wandb.plot.confusion_matrix(
            y_true=y_test.tolist(),
            preds=y_pred.tolist(),
            class_names=CLASS_NAMES,
        )
    })

    print(f"\nAll plots logged. Run URL: {run.url}")
    run.finish()

if __name__ == "__main__":
    main()
