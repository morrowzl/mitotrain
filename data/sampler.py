import numpy as np


def sample_patches(raw, labels, patch_size, n_patches):
    """
    Stub: returns a list of n_patches random (raw_patch, label_patch) tuples.
    Each patch has shape (1, patch_size, patch_size, patch_size).

    Args:
        raw:        np.ndarray of shape (1, D, H, W)
        labels:     np.ndarray of shape (1, D, H, W)
        patch_size: int — spatial extent of each cubic patch
        n_patches:  int — number of patches to return

    Returns:
        list of (raw_patch, label_patch) tuples, each of shape
        (1, patch_size, patch_size, patch_size)
    """
    patches = []
    for _ in range(n_patches):
        raw_patch = np.random.randint(
            0, 256, size=(1, patch_size, patch_size, patch_size), dtype=np.uint8
        )
        label_patch = np.zeros(
            (1, patch_size, patch_size, patch_size), dtype=np.uint8
        )
        patches.append((raw_patch, label_patch))
    return patches
