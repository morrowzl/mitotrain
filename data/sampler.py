import warnings
import numpy as np

PATCH_SIZE  = 132
OUTPUT_SIZE = 40


def sample_patches(
    em_array,
    seg_array,
    patch_size,
    n_patches,
    min_fg_frac=0.05,
    max_attempts=50,
):
    """
    Sample n_patches paired (raw, label) crops from the full s2 volume.
    Foreground-biased: rejects patches below min_fg_frac until max_attempts.

    Args:
        em_array:     zarr array, shape (Z, Y, X), uint16 EM data
        seg_array:    zarr array, shape (Z, Y, X), instance-ID labels (not yet binarized)
        patch_size:   int — cubic patch edge length (e.g. 132)
        n_patches:    int — number of patches to return
        min_fg_frac:  float — minimum foreground fraction to accept a patch
        max_attempts: int — max resamples before giving up on a single patch

    Returns:
        list of (raw_patch, label_patch, fg_frac) triples
        raw_patch:   float32 numpy array, shape (1, P, P, P), percentile-normalized to [0, 1]
        label_patch: uint8 numpy array,   shape (1, P, P, P), binarized {0, 1}
        fg_frac:     float — foreground fraction of this patch
    """
    # Derive bounds from array.shape — do NOT use nm-based arithmetic.
    # Z/Y/X have different voxel sizes (21/16/16 nm at s2); a single divisor is wrong.
    vol_shape = em_array.shape  # (Z, Y, X)
    max_origins = [vol_shape[i] - patch_size for i in range(3)]

    if max_origins[1] < 0:
        raise ValueError(
            f"patch_size={patch_size} exceeds Y dimension {vol_shape[1]}"
        )

    patches = []
    for i in range(n_patches):
        accepted = False
        roi = None
        label = None
        fg_frac = 0.0

        for _ in range(max_attempts):
            z = np.random.randint(0, max_origins[0] + 1)
            y = np.random.randint(0, max_origins[1] + 1)
            x = np.random.randint(0, max_origins[2] + 1)
            roi = (
                slice(z, z + patch_size),
                slice(y, y + patch_size),
                slice(x, x + patch_size),
            )

            seg_crop = seg_array[roi]
            label    = (seg_crop > 0).astype(np.uint8)

            # Evaluate fg only in the center OUTPUT_SIZE³ region (matches BCE loss region)
            start  = (PATCH_SIZE - OUTPUT_SIZE) // 2   # = 46
            center = label[start:start+OUTPUT_SIZE,
                           start:start+OUTPUT_SIZE,
                           start:start+OUTPUT_SIZE]
            fg_frac = float(center.mean())

            if fg_frac >= min_fg_frac:
                accepted = True
                break

        if not accepted:
            warnings.warn(
                f"Patch {i}: max_attempts={max_attempts} reached without finding "
                f"fg_frac>={min_fg_frac}; using last sampled patch (fg_frac={fg_frac:.4f})"
            )

        raw_crop = em_array[roi].astype(np.float32)
        p_low, p_high = np.percentile(raw_crop, [1, 99])
        raw = np.clip((raw_crop - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)

        patches.append((raw[np.newaxis], label[np.newaxis], fg_frac))

    return patches
