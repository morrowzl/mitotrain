import numpy as np
import zarr
import dask.array as da

BUCKET    = "janelia-cosem-datasets"
DATASET   = "jrc_hela-2"
EM_LAYER  = "em/fibsem-uint16/s2"
SEG_LAYER = "labels/mito_pred/s2"

# ROI origin derived from neuroglancer coordinates (nm ÷ 16 for s2)
# Confirmed mitochondria-dense region from visual inspection
ROI = (
    slice(576,  672),   # Z: 96 voxels, chunk-aligned (6×96); mito-dense, confirmed at runtime
    slice(90,   186),   # Y: 96 voxels  (Y max is 400 at s2 — well within bounds)
    slice(2390, 2486),  # X: 96 voxels
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
    store = zarr.N5FSStore(f"s3://{BUCKET}/{dataset}/{dataset}.n5", anon=True)
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
        # Label layer: mito_pred/s2 has values 0–N (not strictly binary).
        # Binarize: any non-zero value = mitochondria present.
        label = (crop > 0).astype(np.uint8)
        print(f"      label unique values (after binarize): {np.unique(label)}")
        return label[np.newaxis]  # (1, Z, Y, X)
