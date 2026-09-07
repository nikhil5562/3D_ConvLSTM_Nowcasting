# 3-D ConvLSTM Radar Nowcasting

This is a source-only academic project for forecasting the evolution of
three-dimensional convective radar echoes from TERLS/RCTLS Doppler weather
radar volumes.

The system accepts ten gridded radar volumes at a nominal 15-minute cadence
and predicts the next eight volumes, corresponding to lead times from T+15 to
T+120 minutes. Each model input contains normalized radar reflectivity and an
observed-coverage channel. The output contains predicted reflectivity.

## Processing workflow

```text
MOSDAC Level-2B polar NetCDF volumes
  -> input/schema validation
  -> reflectivity quality control and velocity dealiasing
  -> Py-ART Cartesian cubes (81 x 481 x 481)
  -> model tensors (16 x 120 x 120) plus coverage tensors
  -> 3-D ConvLSTM training and held-out evaluation
  -> eight forecast volumes and diagnostic images
```

## Folder structure

```text
3D_ConvLSTM_Nowcasting/
|-- nowcasting/
|   |-- data/                 Radar reading, QC, gridding and tensorization
|   |-- models/               3-D ConvLSTM architecture
|   `-- training/             Splitting, loss, metrics and training pipeline
|-- 3d_data_scripts/          Level-2B-to-model preprocessing commands
|-- scripts/                  Training, retraining, prediction and diagnosis
|-- tests/                    Focused regression tests for supported code
|-- docs/                     Technical implementation notes
|-- data/nc/                  Place source Level-2B NetCDF files here
|-- 3d_data/gridded/          Generated full-resolution Cartesian cubes
|-- processed_data/           Generated model tensors and coverage sidecars
|-- saved_models/             Checkpoints, split manifests and test metrics
|-- predictions/              Forecast comparison figures
`-- visualizations/           Optional visualization output
```

Radar datasets, trained weights, generated predictions, virtual environments,
and temporary files are intentionally not included in this shareable package.

## Environment

The checked environment uses Python 3.9.12 and TensorFlow 2.20.0. On Windows,
create an isolated environment from the project root:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow may already be supplied by an HPC environment. In that case, use
the site's recommended environment instead of installing a second TensorFlow
build.

## Prepare radar data

Place full-volume RCTLS Level-2B NetCDF files in `data/nc`, or pass another
input directory explicitly. A training sequence requires 18 continuous scans:
10 observations followed by 8 forecast targets. Three-sweep surveillance scans
must not be mixed with the required full-volume scans.

Run the complete supported preprocessing path:

```powershell
python 3d_data_scripts/run_l2b_to_ml_pipeline.py --input data/nc
```

The command performs a sequence-capacity preflight before expensive gridding.
For a one-scan preprocessing smoke test only, use:

```powershell
python 3d_data_scripts/run_l2b_to_ml_pipeline.py `
  --input data/nc `
  --limit 1 `
  --allow-insufficient-sequence
```

The gridding and tensorization stages can also be run separately:

```powershell
python 3d_data_scripts/create_l2b_gridded_cubes.py --input data/nc
python 3d_data_scripts/create_nowcasting_tensors.py
```

## Train and evaluate

Start a fresh training run:

```powershell
python scripts/train.py
```

Useful environment overrides include:

```powershell
$env:NOWCAST_BATCH_SIZE = "1"       # Useful on a memory-limited machine
$env:NOWCAST_EPOCHS = "20"
$env:NOWCAST_RANDOM_SEED = "42"
python scripts/train.py
```

Training saves versioned weights, a day-independent split manifest, the held-out
test sequences, and machine-readable test metrics in `saved_models`.

Resume only from a compatible, versioned checkpoint:

```powershell
$env:RETRAIN_RESUME = "1"
$env:RETRAIN_WEIGHTS = ".\saved_models\final_model.weights.h5"
python scripts/retrain_model.py
```

Generate forecast comparison figures from the saved held-out split:

```powershell
python scripts/predict.py
```

The prediction command deliberately refuses to run with random or incompatible
weights.

## Verification

Run the included source-code checks with:

```powershell
python -m pytest -q
python -m compileall nowcasting scripts 3d_data_scripts
python -m pip check
```

## Model and scientific scope

The network is an adapted encoder-forecaster 3-D ConvLSTM based on Sun et al.
(2022). The local architecture, masking policy, decoder and training protocol
differ from the reference implementation. It must therefore be described as an
adaptation rather than an exact reproduction.

The data pipeline preserves source masks, records coverage separately, pools
reflectivity in linear-Z space, writes products atomically, and records source
hashes and processing settings. Training divides complete UTC days before
forming temporal windows to reduce data leakage. Evaluation includes CSI, POD,
FAR, prediction spread and a persistence baseline on the held-out split.

Operational or publication-level skill should be claimed only after training
on a clean multi-day corpus and comparing the model against persistence,
optical-flow/pySTEPS and suitable two-dimensional learning baselines.

More detail is available in `docs/TECHNICAL_GUIDE.md`.

## Reference

N. Sun, Z. Zhou, Q. Li, and J. Jing, “Three-Dimensional Gridded Radar Echo
Extrapolation for Convective Storm Nowcasting Based on 3D-ConvLSTM Model,”
*Remote Sensing*, 14(17), 4256, 2022.

- Paper: https://www.mdpi.com/2072-4292/14/17/4256
- Reference repository: https://github.com/snl123/3D-storm-nowcasting
