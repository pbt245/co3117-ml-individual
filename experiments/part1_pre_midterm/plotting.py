"""Small plotting helpers shared by the Part I experiment scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def plot_curves(curves, title, ylabel, filename, logy=False):
    """curves: {label: list_of_values_per_epoch}."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in curves.items():
        ax.plot(range(1, len(values) + 1), values, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def plot_confusion(C, class_names, title, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, C[i, j], ha="center", va="center", fontsize=8,
                    color="white" if C[i, j] > C.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)
    