# mitotrain

A minimal pipeline for training a 3D U-Net to segment mitochondria in FIB-SEM electron microscopy data, using publicly available volumetric datasets from [OpenOrganelle](https://openorganelle.janelia.org/).

This project is motivated by an interest in volumetric biological image analysis and the data engineering patterns that support it. The goal is not to reproduce state-of-the-art segmentation results, but to work through the full pipeline — from raw data access to model training — using real scientific data and tooling from the field.

---

## Overview

The [COSEM/OpenOrganelle datasets](https://openorganelle.janelia.org/) provide whole-cell FIB-SEM volumes at 4–8 nm isotropic resolution, along with ML-generated segmentations for dozens of organelle classes. This project uses the mitochondria segmentation layer from the HeLa cell datasets (`jrc_hela-2`, `jrc_hela-3`) as ground truth labels to train a binary segmentation model on the raw EM signal.

The core task: given a 3D patch of raw FIB-SEM grayscale data, predict which voxels belong to mitochondria.

---

## Pipeline

1. **Data access** — Load raw EM and mitochondria segmentation arrays from the public S3 bucket using `zarr` and `s3fs`, without downloading the full volume
2. **Mask generation** — Convert instance segmentation labels to a binary mask (1 = mitochondria, 0 = everything else)
3. **Patch sampling** — Crop paired (raw, label) subvolumes for training, with balanced sampling to account for class imbalance
4. **Training** — Train a 3D U-Net using binary cross-entropy loss
5. **Evaluation** — Run inference on a held-out region and compare predicted mask to ground truth

---

## Data

Raw data and segmentations are accessed from the public Janelia COSEM S3 bucket:

```
s3://janelia-cosem-publications/heinrich-2021a/
```

No data is downloaded or stored in this repository. All access is lazy/remote via the Zarr chunked format.

**Datasets used:**
- `jrc_hela-2` — wild-type interphase HeLa cell, 8 nm voxels
- `jrc_hela-3` — wild-type interphase HeLa cell, 4 nm voxels

**Relevant layers:**
- `recon-1/em/` — raw FIB-SEM grayscale (model input)
- `recon-1/labels/mitochondria/` — instance segmentation (converted to binary mask for labels)

---

## Model

Uses the 3D U-Net implementation from [funkelab/funlib.learn.torch](https://github.com/funkelab/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/unet.py), developed at Janelia Research Campus.

---

## Dependencies

```
zarr
s3fs
torch
numpy
```

---

## References

- Heinrich et al. (2021). *Whole-cell organelle segmentation in volume electron microscopy.* Nature. https://doi.org/10.1038/s41586-021-03977-3
- OpenOrganelle: https://openorganelle.janelia.org
- funkelab/funlib.learn.torch: https://github.com/funkelab/funlib.learn.torch
