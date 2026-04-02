# mitotrain — Project Planning

## Approach

This project follows an incremental, agile-style development approach. The guiding constraint is that the codebase should always be in a runnable state — every sprint produces a working tool, even if functionality is minimal. Features are added as layers on top of a working skeleton rather than assembled from isolated components.

The single entry point — `train.py` — exists from Sprint 0 and is never broken between commits.

---

## Sprints

### Sprint 0 — Skeleton
**Goal:** A script that runs end to end without errors, touching every major component with pass-through or stub functionality.

- [ ] Initialize repo, add README and this planning document
- [ ] Create `train.py` as the single entry point
- [ ] Stub out major components: data loading, mask generation, patch sampling, model, training loop
- [ ] Confirm S3 access and print array shapes for a hardcoded small subvolume
- [ ] Commit: repo is immediately runnable by anyone who clones it

**Exit condition:** `python train.py` runs without errors.

---

### Sprint 1 — Real Data In, Real Labels Out
**Goal:** Load actual data from S3, convert to binary mask, and visualize.

- [ ] Load a small crop of raw EM from `jrc_hela-2` via `zarr` + `s3fs` (lazy/remote, no full download)
- [ ] Load the corresponding mitochondria segmentation layer
- [ ] Convert instance segmentation to binary mask (1 = mitochondria, 0 = everything else)
- [ ] Visualize a 2D slice of raw EM and binary mask side by side with matplotlib
- [ ] Save output figure to `outputs/`

**Exit condition:** Running `train.py` produces a side-by-side slice visualization of real data.

---

### Sprint 2 — Patch Sampler
**Goal:** Generate balanced (raw, label) patch pairs suitable for training.

- [ ] Implement a patch sampler that crops paired subvolumes from the loaded arrays
- [ ] Add balanced sampling: ensure a configurable minimum fraction of patches contain at least one mitochondria voxel
- [ ] Log sampling statistics: fraction of patches with mitochondria present
- [ ] Write a sanity check that visualizes a few sampled patch pairs

**Exit condition:** Sampler yields correctly shaped, paired patches with logged class balance stats.

---

### Sprint 3 — Model Wired Up
**Goal:** Confirm data flows through the U-Net end to end without shape errors.

- [ ] Add [funkelab U-Net](https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/unet.py) to the repo or install as dependency
- [ ] Instantiate model with appropriate input/output channels for binary segmentation
- [ ] Run a single forward pass on one batch
- [ ] Compute binary cross-entropy loss and print value
- [ ] Do not train yet — this sprint is purely about shape validation

**Exit condition:** Forward pass runs, loss value is printed, no shape errors.

---

### Sprint 4 — Training Loop
**Goal:** Train the model for a small number of epochs and save a checkpoint.

- [ ] Implement training loop: N epochs, one optimizer step per batch
- [ ] Log loss per epoch to console
- [ ] Save model checkpoint to `checkpoints/`
- [ ] Keep N small enough to complete in minutes on CPU (for portability)
- [ ] Add basic config (hardcoded constants at top of script, or simple config file) for patch size, N epochs, learning rate

**Exit condition:** `train.py` trains for N epochs, logs loss, and saves a checkpoint.

---

### Sprint 5 — Inference + Visualization
**Goal:** Run the trained model on a held-out region and visualize results.

- [ ] Load saved checkpoint
- [ ] Run inference on a held-out crop (not seen during training)
- [ ] Visualize predicted mask vs. ground truth side by side
- [ ] Save output figure to `outputs/`

**Exit condition:** Running `train.py` (or a separate `predict.py`) produces a results visualization suitable for the README.

---

## Backlog (Post-MVP)

These are enhancements worth pursuing if time allows, but not required for a complete, demonstrable project.

- Quantitative evaluation metrics (IoU, Dice coefficient)
- Config file instead of hardcoded values (e.g. YAML via `hydra` or `omegaconf`)
- Support for second dataset (`jrc_hela-3`)
- Weighted loss to further address class imbalance
- More training epochs / GPU support notes
- Data provenance logging (which S3 paths, which subvolume coordinates were used)
- Unit tests for mask generation and patch sampler

---

## Repo Structure (Target)

```
mitotrain/
├── train.py              # Single entry point — always runnable
├── predict.py            # Inference script (Sprint 5+)
├── data/
│   ├── loader.py         # S3 access, subvolume cropping
│   └── sampler.py        # Patch sampling with class balancing
├── model/
│   └── unet.py           # funkelab U-Net (vendored or imported)
├── utils/
│   └── visualize.py      # Slice visualization helpers
├── checkpoints/          # Saved model weights (gitignored)
├── outputs/              # Figures and results (gitignored)
├── PLANNING.md
└── README.md
```

---

## References

- Heinrich et al. (2021). *Whole-cell organelle segmentation in volume electron microscopy.* Nature. https://doi.org/10.1038/s41586-021-03977-3
- OpenOrganelle: https://openorganelle.janelia.org
- funkelab/funlib.learn.torch: https://github.com/funkelab/funlib.learn.torch
- COSEM S3 bucket: `s3://janelia-cosem-publications/heinrich-2021a/`
