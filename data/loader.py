import numpy as np


def load_subvolume(dataset, layer, roi):
    """
    Stub: returns a random numpy array in place of a real subvolume.
    Shape: (1, 64, 64, 64) — one channel, 64^3 voxels.

    Args:
        dataset: dataset name (e.g. 'jrc_hela-2')
        layer:   layer name (e.g. 'raw' or 'labels/mito')
        roi:     (offset, shape) tuple — ignored by stub

    Returns:
        np.ndarray of shape (1, 64, 64, 64), dtype uint8
    """
    return np.random.randint(0, 256, size=(1, 64, 64, 64), dtype=np.uint8)
