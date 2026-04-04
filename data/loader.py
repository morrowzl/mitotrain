import numpy as np
import zarr
import dask.array as da

BUCKET    = "janelia-cosem-datasets"
DATASET   = "jrc_hela-2"
# .n5 suffix is required — omitting it causes PathNotFoundError (Sprint 1 bug)
N5_PATH   = f"s3://{BUCKET}/{DATASET}/{DATASET}.n5"
EM_LAYER  = "em/fibsem-uint16/s2"
SEG_LAYER = "labels/mito_seg/s2"   # was: labels/mito_pred/s2 (confirmed to label extracellular space, not mito)

# ROI origin confirmed by inspect_mito_seg.py; patch size set after U-Net shape test (Sprint 2)
PATCH_SIZE = 132
ROI = (
    slice(480, 480 + PATCH_SIZE),   # Z: confirmed mito-dense origin
    slice(80,  80  + PATCH_SIZE),   # Y: max is 400 at s2 — well within bounds
    slice(2382, 2382 + PATCH_SIZE), # X
)


def load_subvolume(dataset, layer, roi):
    """
    Load a subvolume from the janelia-cosem-datasets S3 bucket.

    Args:
        dataset: str, e.g. 'jrc_hela-2'
        layer:   str, e.g. 'em/fibsem-uint16/s2' or 'labels/mito_pred/s2'
        roi:     tuple of slices, e.g. (slice(0,96), slice(0,96), slice(0,96))

    Returns:
        numpy array of shape (1, Z, Y, X):
          - float32 in [0, 1] for uint16 EM layers (percentile-based normalization)
          - uint8 for label layers
    """
    store = zarr.N5FSStore(N5_PATH, anon=True)
    root = zarr.open(store, mode="r")
    zdata = root[layer]

    dask_arr = da.from_array(zdata, chunks=zdata.chunks)
    crop = dask_arr[roi].compute()  # only fetch the ROI, not the full volume

    # Sanity checks printed to console (Step 0 runtime confirmations)
    if zdata.dtype == np.uint16:
        print(f"      raw min/max: {crop.min()}, {crop.max()}")
        print(f"      raw 1st/99th pct: {np.percentile(crop, [1, 99])}")
        raw = crop.astype(np.float32)
        p_low, p_high = np.percentile(raw, [1, 99])
        raw = np.clip((raw - p_low) / (p_high - p_low), 0, 1)
        return raw[np.newaxis]  # (1, Z, Y, X)
    else:
        # Label layer: instance IDs (47, 110, 138, …) — binarize to foreground mask.
        # Binarize: any non-zero value = mitochondria present.
        label = (crop > 0).astype(np.uint8)
        print(f"      label unique values (after binarize): {np.unique(label)}")
        return label[np.newaxis]  # (1, Z, Y, X)


def open_arrays(dataset=DATASET):
    """
    Open EM and seg zarr arrays for lazy patch access. Does not load data.

    Returns:
        (em_array, seg_array): zarr arrays of shape (Z, Y, X)
    """
    store = zarr.N5FSStore(N5_PATH, anon=True)
    root = zarr.open(store, mode="r")
    return root[EM_LAYER], root[SEG_LAYER]
