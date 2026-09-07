import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import BoundaryNorm, ListedColormap

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.models.model_tf2 import build_model
from nowcasting.config import DBZ_MAX
from nowcasting.config import DBZ_MIN
from nowcasting.config import FORECAST_LEAD_MINUTES
from nowcasting.config import INPUT_CHANNELS
from nowcasting.config import INPUT_LENGTH
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.config import TARGET_SHAPE
from nowcasting.paths import MODEL_SAVE_DIR
from nowcasting.paths import PREDICTIONS_DIR
from nowcasting.paths import PROCESSED_DATA_DIR
from nowcasting.training.train import load_compatible_weights
from nowcasting.training.train import _prepare_coverage

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CONT_MIN_SEC = 800
CONT_MAX_SEC = 1000

WEIGHTS_PATH = str(MODEL_SAVE_DIR / "final_model.weights.h5")
DATA_DIR = str(PROCESSED_DATA_DIR)
OUTPUT_DIR = str(PREDICTIONS_DIR)
TEST_SPLIT_PATH = MODEL_SAVE_DIR / "test_sequences.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_timestamp(filename):
    basename = os.path.basename(filename)
    parts = basename.split("_")
    dt_str = parts[1] + "_" + parts[2]
    return datetime.strptime(dt_str, "%d%b%Y_%H%M%S")


def get_sorted_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    valid_files = []
    for file_path in files:
        try:
            timestamp = parse_timestamp(file_path)
            valid_files.append({"path": file_path, "time": timestamp})
        except Exception:
            continue
    valid_files.sort(key=lambda x: x["time"])
    return valid_files


def _ensure_3d_shape(data, target_shape):
    array = np.asarray(data)
    if array.ndim > 3:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    target_d, target_h, target_w = target_shape
    data_d, data_h, data_w = array.shape

    array = array[:target_d, :target_h, :target_w]
    pad_d = max(0, target_d - data_d)
    pad_h = max(0, target_h - data_h)
    pad_w = max(0, target_w - data_w)
    if pad_d or pad_h or pad_w:
        array = np.pad(array, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    return array


def load_and_normalize_sequence(file_paths, include_coverage=False):
    frames = []
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        data = _ensure_3d_shape(data, TARGET_SHAPE)
        data = np.nan_to_num(data, nan=0.0, posinf=DBZ_MAX, neginf=DBZ_MIN)
        data = np.clip(data, DBZ_MIN, DBZ_MAX).astype(np.float32)

        if idx == 0:
            print(f"  Input repaired shape: {data.shape}")
            print(f"  Input repaired range: [{data.min():.2f}, {data.max():.2f}] dBZ")

        data = data / DBZ_MAX
        data = np.expand_dims(data, axis=-1)
        if include_coverage:
            data = np.concatenate([data, _prepare_coverage(path)], axis=-1)
        frames.append(data)

    return np.stack(frames).astype(np.float32)


def select_contiguous_sequence(sorted_files, seq_len):
    valid = []
    for i in range(len(sorted_files) - seq_len + 1):
        is_contiguous = True
        for j in range(1, seq_len):
            diff = (sorted_files[i + j]["time"] - sorted_files[i + j - 1]["time"]).total_seconds()
            if not (CONT_MIN_SEC < diff < CONT_MAX_SEC):
                is_contiguous = False
                break
        if is_contiguous:
            valid.append(sorted_files[i : i + seq_len])

    if not valid:
        return None
    return valid[len(valid) // 2]


def load_held_out_sequence(sequence_index=0):
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Held-out split not found: {TEST_SPLIT_PATH}. Run training first."
        )
    with TEST_SPLIT_PATH.open("r", encoding="utf-8") as file_obj:
        sequences = json.load(file_obj)
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"Held-out split is empty or invalid: {TEST_SPLIT_PATH}")
    try:
        sequence_paths = sequences[sequence_index]
    except IndexError as exc:
        raise IndexError(
            f"Test sequence index {sequence_index} is out of range for "
            f"{len(sequences)} sequences."
        ) from exc
    expected_length = INPUT_LENGTH + OUTPUT_LENGTH
    if len(sequence_paths) != expected_length:
        raise ValueError(
            f"Held-out sequence has {len(sequence_paths)} frames; current experiment "
            f"requires {expected_length}. Regenerate the split with the current code."
        )
    missing = [path for path in sequence_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Held-out sequence contains missing file: {missing[0]}")
    return sequence_paths


def visualize_comparison_cartesian(y_true, y_pred, timestep, save_path):
    level_idx = 2  # ~2 km CAPPI when vertical resolution is 1 km.

    gt_slice = y_true[timestep, level_idx, :, :, 0] * DBZ_MAX
    pred_slice = y_pred[timestep, level_idx, :, :, 0] * DBZ_MAX

    print(f"\nTimestep {timestep + 1} statistics:")
    print(f"  Ground truth min/max/mean: {gt_slice.min():.2f}/{gt_slice.max():.2f}/{gt_slice.mean():.2f}")
    print(f"  Prediction   min/max/mean: {pred_slice.min():.2f}/{pred_slice.max():.2f}/{pred_slice.mean():.2f}")
    print(f"  GT pixels > 5 dBZ: {np.count_nonzero(gt_slice > 5)}")
    print(f"  Pred pixels > 5 dBZ: {np.count_nonzero(pred_slice > 5)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    extent_km = 240
    extent = [-extent_km, extent_km, -extent_km, extent_km]

    colors = [
        "#FFFFFF",
        "#00ECEC",
        "#01A0F6",
        "#0000F6",
        "#00FF00",
        "#00C800",
        "#009000",
        "#FFFF00",
        "#E7C000",
        "#FF9000",
        "#FF0000",
        "#D60000",
        "#C00000",
        "#FF00FF",
        "#9955C9",
    ]
    boundaries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    gt_masked = np.ma.masked_less(gt_slice, 5)
    pred_masked = np.ma.masked_less(pred_slice, 5)

    im = ax1.imshow(gt_masked, origin="lower", cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    lead_minutes = FORECAST_LEAD_MINUTES[timestep]
    ax1.set_title(f"Ground Truth (T+{lead_minutes} min) @ 2km CAPPI", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Distance from Radar (km)")
    ax1.set_ylabel("Distance from Radar (km)")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)

    ax2.imshow(pred_masked, origin="lower", cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    ax2.set_title(f"Prediction (T+{lead_minutes} min) @ 2km CAPPI", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Distance from Radar (km)")
    ax2.set_ylabel("Distance from Radar (km)")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)

    cbar = plt.colorbar(im, ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.02, extend="max")
    cbar.set_label("Reflectivity (dBZ)")

    plt.suptitle(f"Precipitation Nowcast Comparison - T+{lead_minutes} min", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    try:
        print("Building model...")
        input_shape = (INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS)
        model = build_model(input_shape, INPUT_LENGTH, OUTPUT_LENGTH)

        print(f"Loading trained weights from: {WEIGHTS_PATH}")
        load_compatible_weights(model, WEIGHTS_PATH)
        print("Compatible model weights loaded successfully.")

        sequence_index = int(os.environ.get("NOWCAST_TEST_SEQUENCE_INDEX", "0"))
        sequence_paths = load_held_out_sequence(sequence_index)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    input_files = sequence_paths[:INPUT_LENGTH]
    target_files = sequence_paths[INPUT_LENGTH:]

    print(f"Input files: {os.path.basename(input_files[0])} -> {os.path.basename(input_files[-1])}")
    print(f"Target files: {os.path.basename(target_files[0])} -> {os.path.basename(target_files[-1])}")

    print("\nLoading input sequence...")
    x_input = load_and_normalize_sequence(input_files, include_coverage=True)
    x_input = np.expand_dims(x_input, axis=0)

    print("Loading ground truth...")
    y_true = load_and_normalize_sequence(target_files)

    print("\nRunning model prediction...")
    y_pred = model.predict(x_input, verbose=1)[0]

    print(f"\nPrediction shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    if float(y_pred.max()) <= 1e-6:
        print("WARNING: Model output is near-zero everywhere. Retraining is required.")

    print("\nGenerating visualizations...")
    for timestep in range(OUTPUT_LENGTH):
        save_path = os.path.join(OUTPUT_DIR, f"comparison_t{timestep + 1}.png")
        visualize_comparison_cartesian(y_true, y_pred, timestep, save_path)

    print("\nPrediction complete.")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
