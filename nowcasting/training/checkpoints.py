"""Versioned checkpoint identities and compatibility validation."""

import hashlib
import json
from pathlib import Path

import tensorflow as tf

from nowcasting.config import (
    DBZ_MAX, DBZ_MIN, FORECAST_HORIZON_MINUTES, FORECAST_LEAD_MINUTES,
    INPUT_CHANNELS, INPUT_LENGTH, NOMINAL_CADENCE_MINUTES, OUTPUT_LENGTH,
    TARGET_CHANNELS, TARGET_SHAPE,
)
from nowcasting.data.provenance import sha256_file as _file_sha256
from nowcasting.data.provenance import write_json_atomic as _write_json_atomic
from nowcasting.paths import MODEL_SAVE_DIR

CHECKPOINT_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 2


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


def save_versioned_weights(model, weights_path):
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    metadata = checkpoint_metadata(model)
    metadata["weights_sha256"] = _file_sha256(weights_path)
    _write_json_atomic(checkpoint_metadata_path(weights_path), metadata)


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
