import os
import sys
import json
import glob
import hashlib
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Disable oneDNN optimizations to reduce platform-dependent numeric drift.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from nowcasting.models.model_tf2 import build_model
from nowcasting.config import DBZ_MAX
from nowcasting.config import DBZ_MIN
from nowcasting.config import FORECAST_HORIZON_MINUTES
from nowcasting.config import FORECAST_LEAD_MINUTES
from nowcasting.config import INPUT_CHANNELS
from nowcasting.config import INPUT_LENGTH
from nowcasting.config import NOMINAL_CADENCE_MINUTES
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.config import TARGET_SHAPE
from nowcasting.config import TARGET_CHANNELS
from nowcasting.paths import MODEL_SAVE_DIR as MODEL_SAVE_DIR_PATH
from nowcasting.paths import PROCESSED_DATA_DIR

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

# FIX 6: batch size 1 gave extremely noisy gradients. 4 is a good default on
# the HPC GPU; a CPU-only machine may OOM on the temporal unroll, so this is
# overridable:  set NOWCAST_BATCH_SIZE=1  before running locally.
BATCH_SIZE = int(os.environ.get("NOWCAST_BATCH_SIZE", "4"))
EPOCHS = int(os.environ.get("NOWCAST_EPOCHS", "20"))
LEARNING_RATE = 0.0001
GRAD_CLIP_NORM = 1.0
CONT_MIN_SEC = 800
CONT_MAX_SEC = 1000
RANDOM_SEED = int(os.environ.get("NOWCAST_RANDOM_SEED", "42"))
CHECKPOINT_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 2  # 8 leads plus reflectivity/coverage input channels

# FIX 3: only ~13.5% of cells contain any echo and >45 dBZ is 0.03%, so
# sequences that are almost entirely empty actively teach the model to output
# zeros. Require a minimum fraction of cells above ECHO_DBZ in every frame.
ECHO_DBZ = 15.0
MIN_ECHO_FRACTION = 0.02

DATA_DIR = str(PROCESSED_DATA_DIR)
MODEL_SAVE_DIR = str(MODEL_SAVE_DIR_PATH)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def parse_timestamp(filename):
    basename = os.path.basename(filename)
    parts = basename.split("_")
    dt_str = parts[1] + "_" + parts[2]
    return datetime.strptime(dt_str, "%d%b%Y_%H%M%S")


def get_sorted_files(data_dir):
    """Index the tensors by scan time, recording each frame's echo coverage.

    The echo fraction is computed once here (memory-mapped) so that
    get_sequences() can cheaply reject near-empty sequences (FIX 3).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    valid_files = []
    for file_path in files:
        if Path(file_path).stem.endswith("_coverage"):
            continue
        try:
            timestamp = parse_timestamp(file_path)
        except Exception:
            continue
        try:
            sample = np.load(file_path, mmap_mode="r")
            echo_fraction = float(np.mean(np.asarray(sample) >= ECHO_DBZ))
        except Exception:
            echo_fraction = 0.0
        valid_files.append(
            {"path": file_path, "time": timestamp, "echo": echo_fraction}
        )
    valid_files.sort(key=lambda x: x["time"])
    return valid_files


def _ensure_3d_shape(data, target_shape):
    array = np.asarray(data)
    if array.ndim > 3:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {array.shape}")

    target_d, target_h, target_w = target_shape
    data_d, data_h, data_w = array.shape

    array = array[:target_d, :target_h, :target_w]

    pad_d = max(0, target_d - data_d)
    pad_h = max(0, target_h - data_h)
    pad_w = max(0, target_w - data_w)
    if pad_d or pad_h or pad_w:
        array = np.pad(array, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")

    return array


def _prepare_frame(file_path):
    data = np.load(file_path)
    data = _ensure_3d_shape(data, TARGET_SHAPE)
    data = np.nan_to_num(data, nan=0.0, posinf=DBZ_MAX, neginf=DBZ_MIN)
    data = np.clip(data, DBZ_MIN, DBZ_MAX)
    data = data.astype(np.float32) / DBZ_MAX
    data = np.expand_dims(data, axis=-1)
    return data


def get_sequences(file_list, input_len, output_len):
    sequences = []
    seq_len = input_len + output_len

    if len(file_list) < seq_len:
        return sequences

    for i in range(len(file_list) - seq_len + 1):
        is_continuous = True
        for j in range(1, seq_len):
            diff = (file_list[i + j]["time"] - file_list[i + j - 1]["time"]).total_seconds()
            if not (CONT_MIN_SEC < diff < CONT_MAX_SEC):
                is_continuous = False
                break

        if not is_continuous:
            continue

        # Select on observations only. Looking at future target echo coverage
        # would leak ground truth and systematically remove initiation/decay.
        window = file_list[i : i + seq_len]
        if min(f.get("echo", 0.0) for f in window[:input_len]) < MIN_ECHO_FRACTION:
            continue

        sequences.append(
            {
                "start": i,
                "paths": [f["path"] for f in window],
            }
        )

    return sequences


def _allocate_split_units(units, train_ratio=0.8, val_ratio=0.1):
    """Chronologically allocate whole days, keeping at least one per split."""
    units = list(units)
    if len(units) < 3:
        raise ValueError(
            "At least three independent radar days are required for "
            "train/validation/test splitting."
        )
    n_train = min(max(1, int(len(units) * train_ratio)), len(units) - 2)
    n_val = min(max(1, int(len(units) * val_ratio)), len(units) - n_train - 1)
    return (
        units[:n_train],
        units[n_train : n_train + n_val],
        units[n_train + n_val :],
    )


def split_files_by_day(file_list, train_ratio=0.8, val_ratio=0.1):
    """Split complete calendar days before constructing temporal windows.

    This is intentionally chronological: validation and test data occur after
    training data. A window is built only from files belonging to one split,
    so no frame or day can leak across the boundaries.
    """
    by_day = defaultdict(list)
    for item in file_list:
        by_day[item["time"].date().isoformat()].append(item)

    train_days, val_days, test_days = _allocate_split_units(
        sorted(by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )

    def flatten(days):
        return [item for day in days for item in by_day[day]]

    return (
        flatten(train_days),
        flatten(val_days),
        flatten(test_days),
        {"train": train_days, "validation": val_days, "test": test_days},
    )


def create_day_independent_splits(
    file_list, input_len, output_len, train_ratio=0.8, val_ratio=0.1
):
    """Split days first, then construct sequences independently per day."""
    by_day = defaultdict(list)
    for item in file_list:
        by_day[item["time"].date().isoformat()].append(item)

    sequences_by_day = {}
    excluded_days = []
    for day in sorted(by_day):
        records = get_sequences(by_day[day], input_len, output_len)
        if records:
            sequences_by_day[day] = [record["paths"] for record in records]
        else:
            excluded_days.append(day)

    train_days, val_days, test_days = _allocate_split_units(
        sorted(sequences_by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )
    split_days = {
        "train": train_days,
        "validation": val_days,
        "test": test_days,
        "excluded_no_sequences": excluded_days,
    }
    flatten = lambda days: [seq for day in days for seq in sequences_by_day[day]]
    return flatten(train_days), flatten(val_days), flatten(test_days), split_days


def split_sequences_no_leakage(
    all_sequences, input_len, output_len, train_ratio=0.8, val_ratio=0.1
):
    """Compatibility wrapper that allocates already-built windows by day.

    New training code uses :func:`create_day_independent_splits`, which splits
    before windows are built. This wrapper remains for older callers and
    rejects cross-midnight windows rather than assigning them ambiguously.
    """
    by_day = defaultdict(list)
    for sequence in all_sequences:
        paths = sequence.get("paths", sequence)
        days = {parse_timestamp(path).date().isoformat() for path in paths}
        if len(days) == 1:
            by_day[next(iter(days))].append(list(paths))
    if len(by_day) < 3:
        return [], [], []
    train_days, val_days, test_days = _allocate_split_units(
        sorted(by_day), train_ratio=train_ratio, val_ratio=val_ratio
    )
    flatten = lambda days: [seq for day in days for seq in by_day[day]]
    return flatten(train_days), flatten(val_days), flatten(test_days)


def _coverage_path(frame_path):
    path = Path(frame_path)
    return path.parent / "_coverage" / path.name


def _prepare_coverage(frame_path):
    coverage_path = _coverage_path(frame_path)
    if not coverage_path.exists():
        return np.ones((*TARGET_SHAPE, 1), dtype=np.float32)
    coverage = np.load(coverage_path)
    coverage = _ensure_3d_shape(coverage, TARGET_SHAPE)
    coverage = np.nan_to_num(coverage, nan=0.0, posinf=0.0, neginf=0.0)
    coverage = np.clip(coverage, 0.0, 1.0).astype(np.float32)
    return np.expand_dims(coverage, axis=-1)


def sequence_generator(sequences, input_len, output_len, shuffle=False, seed=RANDOM_SEED):
    seq_len = input_len + output_len
    ordered_sequences = list(sequences)
    if shuffle:
        random.Random(seed).shuffle(ordered_sequences)
    for seq_paths in ordered_sequences:
        if len(seq_paths) != seq_len:
            raise ValueError(f"Expected {seq_len} paths, got {len(seq_paths)}")
        frames = [_prepare_frame(path) for path in seq_paths]
        stacked = np.stack(frames).astype(np.float32)
        input_coverage = np.stack(
            [_prepare_coverage(path) for path in seq_paths[:input_len]]
        ).astype(np.float32)
        x = np.concatenate([stacked[:input_len], input_coverage], axis=-1)
        y = stacked[input_len:]
        coverage = np.stack(
            [_prepare_coverage(path) for path in seq_paths[input_len:]]
        ).astype(np.float32)
        yield x, y, coverage


def create_dataset(sequences, training=False, batch_size=BATCH_SIZE):
    output_signature = (
        tf.TensorSpec(
            shape=(INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS), dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(OUTPUT_LENGTH, *TARGET_SHAPE, TARGET_CHANNELS), dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(OUTPUT_LENGTH, *TARGET_SHAPE, TARGET_CHANNELS), dtype=tf.float32
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(
            sequences,
            INPUT_LENGTH,
            OUTPUT_LENGTH,
            shuffle=training,
            seed=RANDOM_SEED,
        ),
        output_signature=output_signature,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def weighted_loss(y_true, y_pred):
    """Reflectivity-weighted MAE+MSE.

    FIX 3: with the old weights (1, 3, 8, 15) about 75% of the loss mass came
    from near-empty cells, so predicting all-zero was near-optimal and the
    model collapsed. Measured class frequencies are ~90.1% below 15 dBZ, 9.8%
    in 15-35, 0.08% in 35-45 and 0.03% above 45 dBZ, so the heavy classes need
    far larger weights to contribute a meaningful share of the gradient.
    """
    t1 = 15.0 / DBZ_MAX
    t2 = 35.0 / DBZ_MAX
    t3 = 45.0 / DBZ_MAX

    w1, w2, w3, w4 = 1.0, 10.0, 50.0, 100.0

    mask = tf.ones_like(y_true) * w1
    mask = tf.where((y_true >= t1) & (y_true < t2), tf.ones_like(y_true) * w2, mask)
    mask = tf.where((y_true >= t2) & (y_true < t3), tf.ones_like(y_true) * w3, mask)
    mask = tf.where(y_true >= t3, tf.ones_like(y_true) * w4, mask)

    # Return an unreduced field. Keras applies the target-coverage sample
    # weights before reduction, so unobserved radar cells contribute nothing.
    return (tf.abs(y_pred - y_true) + tf.square(y_pred - y_true)) * mask


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class CoverageWeightedLoss(tf.keras.losses.Loss):
    """Reflectivity loss normalized by the amount of observed radar coverage."""

    def __init__(
        self,
        name="coverage_weighted_loss",
        reduction="mean_with_sample_weight",
    ):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return weighted_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class ContingencyMetric(tf.keras.metrics.Metric):
    """Globally accumulated CSI, POD, or FAR at one threshold and lead."""

    def __init__(self, score, threshold_dbz, lead_index=None, name=None, **kwargs):
        if score not in {"csi", "pod", "far"}:
            raise ValueError(f"Unsupported contingency score: {score}")
        self.score = score
        self.threshold_dbz = float(threshold_dbz)
        self.lead_index = lead_index
        if name is None:
            suffix = "" if lead_index is None else f"_t{lead_index + 1:02d}"
            name = f"{score}{int(threshold_dbz)}{suffix}"
        super().__init__(name=name, **kwargs)
        self.hits = self.add_weight(name="hits", initializer="zeros", dtype=tf.float64)
        self.misses = self.add_weight(name="misses", initializer="zeros", dtype=tf.float64)
        self.false_alarms = self.add_weight(
            name="false_alarms", initializer="zeros", dtype=tf.float64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.lead_index is not None:
            y_true = tf.gather(y_true, self.lead_index, axis=1)
            y_pred = tf.gather(y_pred, self.lead_index, axis=1)
            if sample_weight is not None and sample_weight.shape.rank != 0:
                sample_weight = tf.gather(sample_weight, self.lead_index, axis=1)

        threshold = tf.cast(self.threshold_dbz / DBZ_MAX, y_true.dtype)
        observed = tf.cast(y_true >= threshold, tf.float64)
        forecast = tf.cast(y_pred >= threshold, tf.float64)
        if sample_weight is None:
            weight = tf.ones_like(observed, dtype=tf.float64)
        else:
            weight = tf.cast(sample_weight, tf.float64)
            weight = tf.broadcast_to(weight, tf.shape(observed))

        self.hits.assign_add(tf.reduce_sum(weight * observed * forecast))
        self.misses.assign_add(tf.reduce_sum(weight * observed * (1.0 - forecast)))
        self.false_alarms.assign_add(
            tf.reduce_sum(weight * (1.0 - observed) * forecast)
        )

    def result(self):
        if self.score == "csi":
            denominator = self.hits + self.misses + self.false_alarms
            numerator = self.hits
        elif self.score == "pod":
            denominator = self.hits + self.misses
            numerator = self.hits
        else:
            denominator = self.hits + self.false_alarms
            numerator = self.false_alarms
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        self.hits.assign(0.0)
        self.misses.assign(0.0)
        self.false_alarms.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "score": self.score,
                "threshold_dbz": self.threshold_dbz,
                "lead_index": self.lead_index,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="nowcasting")
class PredictionStd(tf.keras.metrics.Metric):
    """Coverage-weighted global prediction standard deviation."""

    def __init__(self, name="pred_std", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float64)
        self.total_sq = self.add_weight(
            name="total_sq", initializer="zeros", dtype=tf.float64
        )
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(y_pred, tf.float64)
        if sample_weight is None:
            weight = tf.ones_like(values, dtype=tf.float64)
        else:
            weight = tf.broadcast_to(tf.cast(sample_weight, tf.float64), tf.shape(values))
        self.total.assign_add(tf.reduce_sum(weight * values))
        self.total_sq.assign_add(tf.reduce_sum(weight * tf.square(values)))
        self.count.assign_add(tf.reduce_sum(weight))

    def result(self):
        mean = tf.math.divide_no_nan(self.total, self.count)
        variance = tf.math.divide_no_nan(self.total_sq, self.count) - tf.square(mean)
        return tf.sqrt(tf.maximum(variance, 0.0))

    def reset_state(self):
        self.total.assign(0.0)
        self.total_sq.assign(0.0)
        self.count.assign(0.0)


def build_metrics(include_per_lead=False):
    metrics = [PredictionStd()]
    for threshold_dbz in (20.0, 35.0):
        for score in ("csi", "pod", "far"):
            metrics.append(ContingencyMetric(score, threshold_dbz))
        if include_per_lead:
            for lead_index in range(OUTPUT_LENGTH):
                for score in ("csi", "pod", "far"):
                    metrics.append(
                        ContingencyMetric(score, threshold_dbz, lead_index=lead_index)
                    )
    return metrics


def _report_split_overlap(train_sequences, val_sequences, test_sequences):
    train_files = {path for seq in train_sequences for path in seq}
    val_files = {path for seq in val_sequences for path in seq}
    test_files = {path for seq in test_sequences for path in seq}

    overlap_train_val = len(train_files & val_files)
    overlap_train_test = len(train_files & test_files)
    overlap_val_test = len(val_files & test_files)

    print(
        "File overlap counts (must be zero): "
        f"train-val={overlap_train_val}, "
        f"train-test={overlap_train_test}, "
        f"val-test={overlap_val_test}"
    )

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Data leakage detected across train/val/test splits.")

    def sequence_days(sequences):
        return {parse_timestamp(path).date().isoformat() for seq in sequences for path in seq}

    train_days = sequence_days(train_sequences)
    val_days = sequence_days(val_sequences)
    test_days = sequence_days(test_sequences)
    if train_days & val_days or train_days & test_days or val_days & test_days:
        raise RuntimeError("Radar-day leakage detected across train/val/test splits.")


def set_global_determinism(seed=RANDOM_SEED):
    """Seed Python, NumPy, and TensorFlow before constructing the model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


class CheckpointCompatibilityError(RuntimeError):
    """Raised when checkpoint metadata does not match the current model."""


def _architecture_sha256(model):
    def sanitize(value):
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in sorted(value.items())
                if key not in {"name", "trainable", "dtype"}
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        return value

    descriptor = {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "layers": [
            {
                "class_name": layer.__class__.__name__,
                "config": sanitize(layer.get_config()),
                "weights": [list(weight.shape) for weight in layer.weights],
            }
            for layer in model.layers
        ],
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_metadata_path(weights_path):
    return Path(f"{weights_path}.metadata.json")


def training_backup_directory(model, run_tag=""):
    """Namespace crash recovery by schema and architecture compatibility."""
    architecture = _architecture_sha256(model)[:16]
    tag = run_tag or "main"
    namespace = f"v{CHECKPOINT_SCHEMA_VERSION}_m{MODEL_SCHEMA_VERSION}_{architecture}"
    return Path(MODEL_SAVE_DIR) / ".training_backup" / namespace / tag


def checkpoint_metadata(model):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "architecture_sha256": _architecture_sha256(model),
        "input_length": INPUT_LENGTH,
        "input_channels": INPUT_CHANNELS,
        "output_length": OUTPUT_LENGTH,
        "target_channels": TARGET_CHANNELS,
        "target_shape": list(TARGET_SHAPE),
        "nominal_cadence_minutes": NOMINAL_CADENCE_MINUTES,
        "forecast_lead_minutes": list(FORECAST_LEAD_MINUTES),
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "dbz_range": [DBZ_MIN, DBZ_MAX],
        "tensorflow_version": tf.__version__,
    }


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    os.replace(temporary_path, path)


def save_versioned_weights(model, weights_path):
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    metadata = checkpoint_metadata(model)
    metadata["weights_sha256"] = _file_sha256(weights_path)
    _write_json_atomic(checkpoint_metadata_path(weights_path), metadata)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compatible_weights(model, weights_path, allow_unversioned=False):
    """Load only weights explicitly proven compatible with this architecture."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    metadata_path = checkpoint_metadata_path(weights_path)
    if not metadata_path.exists():
        if not allow_unversioned:
            raise CheckpointCompatibilityError(
                f"Checkpoint metadata is missing: {metadata_path}. The weights predate "
                "architecture versioning; start a fresh run instead of loading them."
            )
    else:
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            saved = json.load(file_obj)
        expected = checkpoint_metadata(model)
        mismatches = {
            key: {"saved": saved.get(key), "expected": expected[key]}
            for key in (
                "schema_version",
                "model_schema_version",
                "architecture_sha256",
                "input_length",
                "input_channels",
                "output_length",
                "target_channels",
                "target_shape",
                "nominal_cadence_minutes",
            )
            if saved.get(key) != expected[key]
        }
        if mismatches:
            raise CheckpointCompatibilityError(
                "Checkpoint is incompatible with the current experiment: "
                + json.dumps(mismatches, sort_keys=True)
            )
        saved_digest = saved.get("weights_sha256")
        if saved_digest and saved_digest != _file_sha256(weights_path):
            raise CheckpointCompatibilityError(
                f"Checkpoint content hash does not match its metadata: {weights_path}"
            )
    try:
        model.load_weights(weights_path)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint tensors are incompatible with the current model: {exc}"
        ) from exc


def _new_evaluation_accumulator():
    return {
        "loss_numerator": 0.0,
        "weight_sum": 0.0,
        "pred_sum": 0.0,
        "pred_sq_sum": 0.0,
        "contingency": {
            threshold: np.zeros((OUTPUT_LENGTH, 3), dtype=np.float64)
            for threshold in (20.0, 35.0)
        },
    }


def _accumulate_evaluation(accumulator, y_true, y_pred, coverage):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape != coverage.shape:
        raise ValueError(
            f"Evaluation shapes differ: truth={y_true.shape}, prediction={y_pred.shape}, "
            f"coverage={coverage.shape}"
        )

    per_cell_loss = np.asarray(weighted_loss(y_true, y_pred), dtype=np.float64)
    accumulator["loss_numerator"] += float(np.sum(per_cell_loss * coverage))
    accumulator["weight_sum"] += float(np.sum(coverage))
    accumulator["pred_sum"] += float(np.sum(y_pred * coverage))
    accumulator["pred_sq_sum"] += float(np.sum(np.square(y_pred) * coverage))

    reduction_axes = tuple(axis for axis in range(y_true.ndim) if axis != 1)
    for threshold_dbz, counts in accumulator["contingency"].items():
        threshold = threshold_dbz / DBZ_MAX
        observed = y_true >= threshold
        forecast = y_pred >= threshold
        counts[:, 0] += np.sum(coverage * observed * forecast, axis=reduction_axes)
        counts[:, 1] += np.sum(coverage * observed * ~forecast, axis=reduction_axes)
        counts[:, 2] += np.sum(coverage * ~observed * forecast, axis=reduction_axes)


def _scores_from_counts(hits, misses, false_alarms):
    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    return {
        "csi": divide(hits, hits + misses + false_alarms),
        "pod": divide(hits, hits + misses),
        "far": divide(false_alarms, hits + false_alarms),
        "hits": float(hits),
        "misses": float(misses),
        "false_alarms": float(false_alarms),
    }


def _finalize_evaluation(accumulator):
    weight_sum = accumulator["weight_sum"]
    mean = accumulator["pred_sum"] / weight_sum if weight_sum else 0.0
    variance = accumulator["pred_sq_sum"] / weight_sum - mean * mean if weight_sum else 0.0
    result = {
        "coverage_weighted_loss": (
            accumulator["loss_numerator"] / weight_sum if weight_sum else 0.0
        ),
        "prediction_std": math.sqrt(max(variance, 0.0)),
        "evaluated_coverage_weight": weight_sum,
        "overall": {},
        "per_lead": [],
    }
    for lead_index, lead_minutes in enumerate(FORECAST_LEAD_MINUTES):
        lead_result = {"lead_index": lead_index + 1, "lead_minutes": lead_minutes}
        for threshold_dbz, counts in accumulator["contingency"].items():
            lead_result[f"{int(threshold_dbz)}_dbz"] = _scores_from_counts(*counts[lead_index])
        result["per_lead"].append(lead_result)

    for threshold_dbz, counts in accumulator["contingency"].items():
        result["overall"][f"{int(threshold_dbz)}_dbz"] = _scores_from_counts(
            *np.sum(counts, axis=0)
        )
    return result


def evaluate_test_set(model, test_dataset):
    """Evaluate the trained model and persistence on the same held-out data."""
    model_accumulator = _new_evaluation_accumulator()
    persistence_accumulator = _new_evaluation_accumulator()
    batches = 0
    for x_batch, y_batch, coverage_batch in test_dataset:
        y_prediction = model(x_batch, training=False)
        persistence = tf.repeat(x_batch[:, -1:, ..., :1], OUTPUT_LENGTH, axis=1)
        _accumulate_evaluation(model_accumulator, y_batch, y_prediction, coverage_batch)
        _accumulate_evaluation(
            persistence_accumulator, y_batch, persistence, coverage_batch
        )
        batches += 1
    if batches == 0:
        raise RuntimeError("Held-out test dataset produced no batches.")
    model_results = _finalize_evaluation(model_accumulator)
    persistence_results = _finalize_evaluation(persistence_accumulator)
    persistence_loss = persistence_results["coverage_weighted_loss"]
    return {
        "model": model_results,
        "persistence": persistence_results,
        "loss_skill_vs_persistence": (
            1.0 - model_results["coverage_weighted_loss"] / persistence_loss
            if persistence_loss > 0.0
            else None
        ),
        "batches": batches,
    }


def train(
    epochs=EPOCHS,
    run_tag="",
    max_steps_per_epoch=None,
    max_validation_steps=None,
    resume_weights_path=None,
    allow_unversioned_weights=False,
):
    set_global_determinism()
    input_shape = (INPUT_LENGTH, *TARGET_SHAPE, INPUT_CHANNELS)
    model = build_model(input_shape, INPUT_LENGTH, OUTPUT_LENGTH)
    if resume_weights_path:
        load_compatible_weights(
            model, resume_weights_path, allow_unversioned=allow_unversioned_weights
        )
        print(f"Loaded compatible warm-start weights from: {resume_weights_path}")
    model.compile(
        # FIX 6: clip gradients so a rare intense-echo batch cannot blow up
        # the weights now that the heavy classes carry large loss weights.
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=GRAD_CLIP_NORM
        ),
        # mean_with_sample_weight keeps loss magnitudes comparable when radar
        # coverage differs between batches instead of dividing by masked cells.
        loss=CoverageWeightedLoss(),
        metrics=[],
        weighted_metrics=build_metrics(include_per_lead=False),
    )

    model.summary()

    all_files = get_sorted_files(DATA_DIR)
    print(f"Total files found: {len(all_files)}")
    if not all_files:
        print("No files found. Exiting.")
        return False

    try:
        train_sequences, val_sequences, test_sequences, split_days = (
            create_day_independent_splits(all_files, INPUT_LENGTH, OUTPUT_LENGTH)
        )
    except ValueError as exc:
        print(f"Cannot create independent split: {exc}")
        return False
    print(
        "Day-independent split: "
        f"train={len(train_sequences)}, val={len(val_sequences)}, "
        f"test={len(test_sequences)}"
    )
    print(
        f"Split days: train={split_days['train']}, "
        f"validation={split_days['validation']}, test={split_days['test']}"
    )

    if not train_sequences:
        print("No valid training sequences found. Exiting.")
        return False
    if not val_sequences:
        print("No valid validation sequences found in val split. Exiting.")
        return False
    if not test_sequences:
        print("No valid test sequences found in test split. Exiting.")
        return False

    _report_split_overlap(train_sequences, val_sequences, test_sequences)

    train_dataset = create_dataset(train_sequences, training=True)
    val_dataset = create_dataset(val_sequences, training=False)
    test_dataset = create_dataset(test_sequences, training=False)

    steps_per_epoch = max(1, math.ceil(len(train_sequences) / BATCH_SIZE))
    validation_steps = max(1, math.ceil(len(val_sequences) / BATCH_SIZE))
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    if max_validation_steps is not None:
        validation_steps = min(validation_steps, max_validation_steps)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = f"{run_tag}_" if run_tag else ""
    log_filename = f"training_log_{tag_prefix}{run_timestamp}.csv"
    plot_filename = f"training_plot_{tag_prefix}{run_timestamp}.png"
    checkpoint_filename = f"best_model_{tag_prefix}{run_timestamp}.weights.h5"
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, checkpoint_filename)
    backup_dir = str(training_backup_directory(model, run_tag=run_tag))

    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(MODEL_SAVE_DIR, log_filename)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # Restores optimizer, epoch, and model state after an interrupted job.
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
    ]

    print(
        f"Training config: epochs={epochs}, "
        f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    if os.path.exists(checkpoint_path):
        # Evaluate and publish the actual minimum-validation-loss checkpoint,
        # even when training reached the final epoch without early stopping.
        model.load_weights(checkpoint_path)
        best_metadata = checkpoint_metadata(model)
        best_metadata["weights_sha256"] = _file_sha256(checkpoint_path)
        _write_json_atomic(
            checkpoint_metadata_path(checkpoint_path), best_metadata
        )
        print(f"Restored best validation checkpoint: {checkpoint_path}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss", marker="o")
        plt.plot(history.history["val_loss"], label="Val Loss", marker="s")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, plot_filename), dpi=150)
        plt.close()
        print(f"Training plot saved: {plot_filename}")
    except Exception as exc:
        print(f"Could not save training plot: {exc}")

    final_weights_path = os.path.join(MODEL_SAVE_DIR, "final_model.weights.h5")
    save_versioned_weights(model, final_weights_path)
    print(f"Model weights saved to {final_weights_path}")

    _write_json_atomic(
        os.path.join(MODEL_SAVE_DIR, "test_sequences.json"), test_sequences
    )
    print("Test sequences saved to test_sequences.json")

    split_manifest = {
        "schema_version": 1,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "strategy": "chronological_calendar_days_split_before_windows",
        "random_seed": RANDOM_SEED,
        "input_length": INPUT_LENGTH,
        "input_channels": INPUT_CHANNELS,
        "output_length": OUTPUT_LENGTH,
        "target_channels": TARGET_CHANNELS,
        "nominal_cadence_minutes": NOMINAL_CADENCE_MINUTES,
        "forecast_lead_minutes": list(FORECAST_LEAD_MINUTES),
        "days": split_days,
        "sequence_counts": {
            "train": len(train_sequences),
            "validation": len(val_sequences),
            "test": len(test_sequences),
        },
    }
    _write_json_atomic(
        os.path.join(MODEL_SAVE_DIR, "split_manifest.json"), split_manifest
    )

    print("Evaluating best restored weights on the untouched test split...")
    evaluation = evaluate_test_set(model, test_dataset)
    evaluation.update(
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "weights_path": final_weights_path,
            "weights_sha256": _file_sha256(final_weights_path),
            "checkpoint_metadata": checkpoint_metadata(model),
            "split_manifest": split_manifest,
        }
    )
    run_results_path = os.path.join(
        MODEL_SAVE_DIR, f"test_results_{tag_prefix}{run_timestamp}.json"
    )
    _write_json_atomic(run_results_path, evaluation)
    _write_json_atomic(os.path.join(MODEL_SAVE_DIR, "test_results.json"), evaluation)
    print(f"Held-out model and persistence results saved to {run_results_path}")
    return True


def main():
    return 0 if train(epochs=EPOCHS, run_tag="") else 1


if __name__ == "__main__":
    raise SystemExit(main())
