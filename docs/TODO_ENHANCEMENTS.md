# To-Do and Enhancements Backlog

Living document. Items are moved here when they are not on the critical path to MVP.
**MVP is complete as of Sprint 5.** Priority should now be assessed for the next phase.

Items are grouped by theme. ✅ = completed during MVP sprints.

---

## Data & Labels

- [ ] **Investigate `mito_pred` vs `mito_seg` relationship across full volume** — zero overlap
  at Sprint 1 ROI suggests they label different things or have a coordinate mismatch.
  Not understood yet.

- [ ] **Confirm `mito_pred/s0` value encoding** — the s0 crop tested in `inspect_binarization.py`
  was all zeros (likely incorrect crop coordinates due to approximate nm→voxel scaling).
  A correctly-scaled s0 crop should be checked to determine if s0 is truly soft (0–255)
  or binary {0,1}.

- [ ] **GT crop coordinate alignment** — ground truth crops in `janelia-cosem-publications`
  are at s0 resolution. Aligning to s2 EM requires per-axis world-coordinate offset mapping.
  Label encoding is resolved (mito = IDs 3+4+5). Needed for evaluation sprint.

- [ ] **Pre-compute foreground location index** — the full-volume sampler required 70–200+
  attempts per epoch to fill 16 patches. Pre-computing a list of voxel coordinates where
  `mito_seg > 0` would eliminate rejection overhead and dramatically reduce sampling time,
  which was the dominant per-epoch cost (~80–110s per epoch).

- [ ] **Extend to second dataset (`jrc_hela-3`)** — for generalization testing.

---

## Model & Training

- ✅ **Weighted loss** — implemented as `pos_weight=4.0` in Sprint 4.2.

- [ ] **Extended training** — MVP used 30 epochs / 240 gradient steps on CPU. The model
  shows learned spatial structure but insufficient discrimination (prob range 0.27–0.32).
  Meaningful segmentation quality requires substantially more training. GPU is a prerequisite
  for extended runs — see GPU support item below.

- [ ] **Investigate growing backward pass time** — epoch 25 backward took 940s vs typical
  120–175s. Epoch 19 sampling took 2073s with MEM drop from 4.64→2.88 GB. Root cause
  unidentified. Must be resolved before any training run longer than 30 epochs is attempted.
  Possible causes: PyTorch autograd graph growth, OS memory paging, network interruption.

- [ ] **GPU support** — CPU training took 196 minutes for 30 epochs. Add `.to(device)` calls
  and a device selection flag. GPU is required for any training run that could produce
  meaningful segmentation quality.

- [ ] **Learning rate scheduling** — fixed `lr=1e-4` for MVP. Add cosine decay or step
  schedule if extended training shows plateau.

- [ ] **Reduce model size** — 93M parameters (`num_fmaps=12`, `fmap_inc_factor=5`) is large
  for a first training run. Reducing to `num_fmaps=6` cuts parameters ~4×, speeds up
  training, and may converge faster. Re-run shape test if changed — output size will differ.

- [ ] **Checkpoint resume** — currently saves `model.state_dict()` only. Full resume also
  needs `optimizer.state_dict()` and epoch number.

- [ ] **Remove sampler debug prints** — `fg in center crop:` lines still printing per
  attempt, producing thousands of log lines per run. Comment out in `data/sampler.py`.

---

## Evaluation & Metrics

- [ ] **Quantitative metrics** — IoU (Intersection over Union) and Dice coefficient.
  Add to `predict.py` once model quality is sufficient to produce meaningful numbers.

- ✅ **Held-out test region** — defined as `HOLDOUT_ROI = (slice(200,332), slice(80,212),
  slice(1500,1632))` after verifying 9.59% foreground. Note: original ROI at Z=800–932
  had zero foreground and was unusable — always verify before committing.

- [ ] **Comparison against `mito_pred` baseline** — compare trained model predictions
  against raw `mito_pred` predictions as a sanity baseline once model quality improves.

- [ ] **Threshold calibration** — current MVP model has prob range 0.27–0.32, making any
  threshold choice arbitrary. After extended training, calibrate threshold using a
  validation set rather than choosing manually.

---

## Pipeline & Engineering

- [ ] **Config file** — move hardcoded constants (`PATCH_SIZE`, `OUTPUT_SIZE`, `N_EPOCHS`,
  `LR`, `pos_weight`, layer paths, ROIs) to a single `config.py` or YAML file.

- [ ] **`fibsem_tools` integration** — Janelia's library wraps N5/Zarr access with spatial
  metadata via `xarray.DataArray`. Would simplify coordinate handling and is the
  Janelia-recommended access pattern.

- [ ] **zarr v3 migration** — `N5FSStore` is deprecated in zarr v3. Currently pinned to
  `zarr<3`. System Python 3.14 cannot use zarr v2 (`numcodecs` has no 3.14 wheel).
  Investigate `n5py` as the migration path before upgrading Python runtime.

- [ ] **Workflow manager** — the job description mentions workflow managers for production
  pipelines (e.g. Prefect, Luigi, Snakemake). Relevant for portfolio narrative.

- [ ] **Unit tests** — tests for mask generation, patch sampler bounds checking, and
  normalization correctness.

---

## Documentation & Portfolio

- [ ] **README visualization** — add `inference_preview_contrast_stretched.png` and
  `slice_preview.png` to the README with honest description of training stage.

- [ ] **Cover letter reference** — update cover letter to reference the completed project
  with GitHub link now that Sprint 5 is complete.

- ✅ **LESSONS_LEARNED.md** — updated through Sprint 5 (16 lessons).

- [ ] **Email Jan** — share the repo and describe what was built, what was found
  (activation bug, gradient explosion, background collapse), and what extended training
  would require. Frame as a natural follow-up to his suggestion.
