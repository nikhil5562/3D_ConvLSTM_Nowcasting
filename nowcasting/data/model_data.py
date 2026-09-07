"""Validated, coverage-aware model tensors shared by all model entrypoints."""

from pathlib import Path

import numpy as np

from nowcasting.config import DBZ_MAX, DBZ_MIN, TARGET_SHAPE
from nowcasting.data.provenance import ProvenanceMismatchError, validate_artifact_manifest
from nowcasting.data.timestamps import parse_timestamp


TENSOR_PIPELINE_VERSION = "cartesian-to-model-tensor-v2"
COVERAGE_DIRECTORY_NAME = "_coverage"


def tensor_processing_settings():
    """The tensor contract recorded by preprocessing and checked by consumers."""
    return {
        "input_crop": [64, 480, 480],
        "output_shape": list(TARGET_SHAPE),
        "pooling": "valid_only_linear_z_mean_4x4x4",
        "dbz_range": [DBZ_MIN, DBZ_MAX],
        "missing_output_dbz": DBZ_MIN,
        "coverage_companion": True,
    }


def coverage_path(frame_path):
    path = Path(frame_path)
    return path.parent / COVERAGE_DIRECTORY_NAME / path.name


def validate_tensor_artifact(frame_path):
    """Verify tensor/coverage hashes, processing settings, and source scan identity."""
    path = Path(frame_path)
    parse_timestamp(path)
    payload = validate_artifact_manifest(
        path,
        related_directory=coverage_path(path).parent,
        required_related_names=[path.name],
    )
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("pipeline_version") != TENSOR_PIPELINE_VERSION
        or provenance.get("stage") != "cartesian_grid_to_model_tensor"
        or provenance.get("settings") != tensor_processing_settings()
    ):
        raise ProvenanceMismatchError(
            f"Tensor processing settings are incompatible: {path}. Regenerate with --overwrite."
        )
    source = provenance.get("source")
    if (
        not isinstance(source, dict)
        or source.get("name") != f"{path.stem}_gridded.nc"
        or not isinstance(source.get("size_bytes"), int)
        or source["size_bytes"] < 0
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
    ):
        raise ProvenanceMismatchError(f"Tensor source identity does not match its scan: {path}")
    return payload


def _load_valid_array(path, minimum, maximum):
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Cannot load model tensor {path}: {exc}") from exc
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"Expected a single NumPy array: {path}")
    if array.shape != TARGET_SHAPE:
        raise ValueError(f"Expected tensor shape {TARGET_SHAPE}, got {array.shape}: {path}")
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"Tensor must contain finite real numeric values: {path}")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"Tensor values must be within [{minimum}, {maximum}]: {path}")
    return array.astype(np.float32, copy=False)


def load_frame(frame_path):
    """Return normalized reflectivity and coverage, both shaped (z, y, x, 1)."""
    validate_tensor_artifact(frame_path)
    reflectivity = _load_valid_array(frame_path, DBZ_MIN, DBZ_MAX)
    coverage = _load_valid_array(coverage_path(frame_path), 0.0, 1.0)
    return (reflectivity / DBZ_MAX)[..., None], coverage[..., None]


def load_and_normalize_sequence(file_paths, include_coverage=False):
    """Load a sequence with the same validation and normalization used in training."""
    frames = []
    for path in file_paths:
        reflectivity, coverage = load_frame(path)
        frames.append(
            np.concatenate([reflectivity, coverage], axis=-1)
            if include_coverage else reflectivity
        )
    if not frames:
        raise ValueError("Cannot load an empty model sequence.")
    return np.stack(frames).astype(np.float32)


def get_sorted_files(data_dir, echo_dbz=15.0):
    """Index and validate every model tensor before expensive model construction."""
    records = []
    for path in Path(data_dir).glob("*.npy"):
        timestamp = parse_timestamp(path)
        reflectivity, _ = load_frame(path)
        records.append({
            "path": str(path.resolve()),
            "time": timestamp,
            "echo": float(np.mean(reflectivity >= echo_dbz / DBZ_MAX)),
        })
    records.sort(key=lambda record: record["time"])
    if len({record["time"] for record in records}) != len(records):
        raise ValueError(f"Duplicate scan timestamps in model data: {data_dir}")
    return records
