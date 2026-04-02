import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_slice(raw, labels, path):
    """
    Stub: creates a blank matplotlib figure and saves it to path.

    Args:
        raw:    np.ndarray — raw EM patch (ignored by stub)
        labels: np.ndarray — label patch (ignored by stub)
        path:   str — output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title("Raw EM (stub)")
    axes[1].set_title("Labels (stub)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
