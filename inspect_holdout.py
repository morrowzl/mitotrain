import zarr
import numpy as np

store = zarr.N5FSStore("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5", anon=True)
group = zarr.open(store, mode="r")
seg = group["labels/mito_seg/s2"]

# Sample several candidate regions and print fg counts
candidates = [
    (slice(700, 832), slice(80, 212), slice(1200, 1332)),
    (slice(600, 732), slice(80, 212), slice(1200, 1332)),
    (slice(400, 532), slice(80, 212), slice(600, 732)),
    (slice(200, 332), slice(80, 212), slice(1500, 1632)),
    (slice(100, 232), slice(80, 212), slice(2000, 2132)),
]
for roi in candidates:
    crop = (seg[roi] > 0)
    print(f"Z={roi[0].start}-{roi[0].stop}  X={roi[2].start}-{roi[2].stop}  fg={crop.mean():.4f}  ({crop.sum()} voxels)")