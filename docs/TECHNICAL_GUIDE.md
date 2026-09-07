# TERLS 3-D radar nowcasting: implementation guide

## Status and scientific scope

This repository implements an **adapted** TensorFlow 3-D ConvLSTM for TERLS
C-band radar reflectivity. The configured objective is ten observed volumes
followed by eight forecast volumes at a nominal 15-minute cadence: **T+15,
T+30, ..., T+120 minutes**.

The architecture is derived from Sun et al. (2022), but the local forecaster,
normalization, decoder, and kernel choices are not an exact reproduction.
Performance and baseline claims remain pending until a clean multi-day corpus
has been trained and evaluated on held-out days.

## Supported data path

The supported preprocessing route is:

```text
TERLS L2B polar NetCDF
  -> schema and scan-mode validation
  -> reflectivity QC and velocity dealiasing
  -> Py-ART 81 x 481 x 481 Cartesian cube
  -> linear-Z pooling to 16 x 120 x 120
  -> reflectivity tensor plus coverage sidecar and provenance manifest
```

Run it with:

```powershell
python 3d_data_scripts/run_l2b_to_ml_pipeline.py `
  --input "C:\path\to\full-volume-l2b" `
  --cube-output 3d_data/gridded `
  --tensor-output processed_data
```

Only full-volume scans meeting the configured sweep requirement should enter
the model corpus. Three-sweep surveillance scans are not interchangeable with
eleven-sweep volumes.

### Grid and tensor definitions

- Full grid: 81 vertical x 481 north-south x 481 east-west points.
- Full-grid spacing: 250 m vertically and 1 km horizontally.
- Model tensor: 16 x 120 x 120 `float32` reflectivity values in dBZ.
- Downsampling: average in linear reflectivity (`Z`), then convert back to dBZ.
- Reflectivity range supplied to the model: 0--80 dBZ, normalized by 80.
- Coverage: `_coverage/<scan>.npy`, with fractions from 0 to 1 for every pooled
  voxel. Missing coverage is not assumed to be clear air.

Every generated product must carry or reference source identity, processing
version, grid/QC settings, timestamps, and content hashes. Existing products
whose provenance does not match the requested configuration must be rejected
or explicitly regenerated with `--overwrite`.

## Data integrity rules

1. Preserve the decoded NetCDF mask before inspecting field values.
2. Never use physical `0 dBZ` as a missing-value sentinel.
3. Combine the DBZ and RHOHV masks during reflectivity QC.
4. Preserve the velocity field's own mask; do not replace it with the DBZ mask.
5. Use a nonphysical fill value when writing Cartesian NetCDF products.
6. Write cubes, tensors, masks, and manifests atomically.
7. Treat non-finite values, corrupt shapes, mismatched coordinates, and
   timestamp mismatches as validation failures.

## Sequence construction and data splitting

- Input length: 10 frames.
- Output length: 8 frames.
- Nominal cadence: 15 minutes.
- Continuity tolerance: configured in `nowcasting/training/train.py`.
- Echo filtering uses **input frames only**; future truth must not select the
  sample.
- Split complete calendar days (or independently identified storm events)
  before constructing temporal windows.
- No day or source frame may occur in more than one of train, validation, and
  test.

The current small June archive is useful for pipeline QA but cannot generate a
complete clean training sequence. A defensible experiment needs enough
continuous full-volume observations across many independent storm days.

## Model and training

The network accepts `(batch, 10, 16, 120, 120, 2)`—normalized reflectivity
plus observed-coverage fraction—and emits
`(batch, 8, 16, 120, 120, 1)`.

Training uses Adam with gradient clipping and a reflectivity-weighted
MAE-plus-MSE field:

| Observed reflectivity | Weight |
|---|---:|
| `< 15 dBZ` | 1 |
| `15--35 dBZ` | 10 |
| `35--45 dBZ` | 50 |
| `>= 45 dBZ` | 100 |

Coverage weights exclude unobserved voxels from the loss and verification.
Fresh training is the default. Resuming is allowed only when checkpoint
metadata matches the architecture, input/output lengths, tensor shape, and
schema version.

```powershell
python scripts/train.py

# Explicit resume, only from a compatible run:
$env:RETRAIN_RESUME = "1"
$env:RETRAIN_WEIGHTS = "C:\path\to\compatible.weights.h5"
python scripts/retrain_model.py
```

Set `NOWCAST_RANDOM_SEED` for deterministic Python, NumPy, and TensorFlow
initialization. `NOWCAST_OUTPUT_BIAS_INIT` controls the sigmoid-output prior;
recompute it from the clean training split instead of assuming the legacy
corpus value. Batch size and epochs remain environment-configurable for HPC.

## Evaluation

The held-out test set is evaluated after training. Machine-readable output
must include:

- weighted loss;
- CSI, POD, and FAR from globally accumulated contingency counts;
- the same scores at each lead time;
- prediction spread as a collapse diagnostic;
- a persistence baseline on exactly the same samples and coverage mask;
- split days, sequence counts, model/configuration metadata, and checkpoint
  identity.

Ratios must be computed after accumulating hits, misses, and false alarms over
the full evaluation set. Averaging independent batch ratios is invalid.
Comparisons with the published TERLS pySTEPS work must use the matched T+15 to
T+90 subset. FSS, HSS, pySTEPS/optical-flow, and a 2-D learning baseline are
still required for a publication-level comparison.

## Validation against MOSDAC L2C

L2B-derived cubes and L2C products differ in echo-selection policy. Report at
least:

- official-cell recall and generated-cell precision;
- intersection over union / CSI and thresholded POD/FAR;
- correlation, mean bias, MAE, and RMSE on co-valid voxels;
- results by altitude and reflectivity threshold;
- coordinate arrays, projection, grid origin, orientation, and timestamp
  agreement.

High recall alone is not spatial agreement. The existing four-scan comparison
is preliminary and must not be presented as validation of the legacy tensors
used in earlier experiments.

## Verification commands

```powershell
python -m pytest -q
python -m pip check
python -m compileall nowcasting scripts 3d_data_scripts
```

Before a paper or release, also regenerate the data manifest, verify held-out
days, archive the exact environment, run all baselines, and make the repository
and license status explicit.
