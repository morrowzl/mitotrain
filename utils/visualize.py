import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_slice(raw, labels, path):
    """
    Save a side-by-side visualization of a raw EM slice and binary mito mask.

    Args:
        raw:    np.ndarray of shape (1, Z, Y, X), float32 in [0, 1]
        labels: np.ndarray of shape (1, Z, Y, X), uint8 binary mask
        path:   str — output file path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Take middle Z slice of channel 0
    z_mid = raw.shape[1] // 2
    raw_slice   = raw[0, z_mid]      # (Y, X), float32
    label_slice = labels[0, z_mid]   # (Y, X), uint8

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(raw_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Raw EM — z={z_mid}")
    axes[0].axis("off")

    axes[1].imshow(raw_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.5 * label_slice, interpolation="nearest")
    axes[1].set_title(f"Mito mask overlay — z={z_mid}")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
