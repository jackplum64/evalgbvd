import csv
import math
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from pathlib import Path
import numpy as np
from typing import Dict, List
from scipy.stats import gaussian_kde

def load_metrics(csv_path: Path) -> Dict[str, List[float]]:
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        metrics: Dict[str, List[float]] = {
            name: [] for name in reader.fieldnames if name != 'idx'
        }
        for row in reader:
            for k in metrics:
                metrics[k].append(float(row[k]))
    return metrics

def plot_metrics_density(
    metrics: Dict[str, List[float]], save_path: Path
) -> None:
    names = list(metrics.keys())
    n_metrics = len(names)
    n_cols = 5
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 4),
        squeeze=False,
        sharey=False
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        vals = np.array(metrics[name], dtype=float)
        kde = gaussian_kde(vals)
        dens = kde(vals)
        ax.scatter(
            np.zeros_like(vals), vals,
            c=dens, s=12, cmap=colormap.jet, rasterized=True, zorder=10
        )
        ax.axvline(0.0, color='black', linewidth=1, zorder=5)
        ax.set_title(name, pad=6)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=0)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, format="pdf")
    plt.close(fig)

def plot_metrics_histograms(
    metrics: Dict[str, List[float]], save_path: Path
) -> None:
    names: List[str] = list(metrics.keys())
    n_metrics: int = len(names)
    n_cols: int = 5
    n_rows: int = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 4),
        squeeze=False,
        sharey=False
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        vals = np.array(metrics[name], dtype=float)
        mask = np.isfinite(vals)
        if not mask.all():
            print(f"[{name}] dropping {len(vals) - mask.sum()} non-finite samples")
            vals = vals[mask]
        n = vals.size
        bins = min(50, int(math.sqrt(n))) if n else 10
        try:
            ax.hist(
                vals, bins=bins,
                edgecolor='black', linewidth=0.5, zorder=2
            )
        except Exception as e:
            print(f"Skipping histogram for {name} at index {idx}: {e}")
            ax.text(
                0.5, 0.5, "error",
                ha="center", va="center",
                transform=ax.transAxes,
                color="red"
            )
        ax.set_title(name, pad=6)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=1)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, format="pdf")
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: Path
) -> None:
    """
    Plot a 2x2 confusion matrix in scikit-learn format,
    after swapping the TN (top-left) and TP (bottom-right) entries.

    Args:
        confusion_matrix: 2x2 array [[TN, FP],
                                     [FN, TP]]
        save_path:         Path to save the resulting figure
    """
    # copy and swap TN and TP
    cm = confusion_matrix.copy()
    cm[0, 0], cm[1, 1] = cm[1, 1], cm[0, 0]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_title("Pixelwise Confusion Matrix", pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # natural-order tick labels
    labels = ["0", "1"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)

    # annotate each cell
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        count = int(cm[i, j])
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(
            j, i, f"{count}",
            ha="center", va="center",
            color=color, fontsize=12
        )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=250, format="pdf")
    plt.close(fig)

def main():
    print('plotops.py was run but there is no code in main()')

if __name__ == '__main__':
    main()