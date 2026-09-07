"""Convert full 81x481x481 gridded cubes into compact ConvLSTM tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from nowcasting.data.provenance import (
    file_identity,
    save_npy_atomic,
    validate_reusable_artifact,
    write_artifact_manifest,
)

FULL_CUBE_SHAPE = (81, 481, 481)
MODEL_TENSOR_SHAPE = (16, 120, 120)
DBZ_MIN = 0.0
DBZ_MAX = 80.0
TENSOR_PIPELINE_VERSION = "cartesian-to-model-tensor-v2"
COVERAGE_DIRECTORY_NAME = "_coverage"


def _as_3d_cube(data: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    if np.ma.isMaskedArray(data):
        array = np.ma.asarray(data, dtype=np.float32).filled(np.nan)
    else:
        array = np.asarray(data)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D DBZ cube or single-time 4D cube, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def load_dbz_cube(cube_nc: str | Path) -> np.ndarray:
    """Load DBZ from a Py-ART grid NetCDF file as (z, y, x)."""
    with xr.open_dataset(cube_nc, decode_times=False) as dataset:
        if "DBZ" not in dataset:
            raise KeyError(f"No DBZ variable found in {cube_nc}")
        variable = dataset["DBZ"]
        data = variable.to_masked_array(copy=True)
        fill_value = variable.encoding.get(
            "_FillValue",
            variable.attrs.get("_FillValue", variable.attrs.get("missing_value")),
        )

    cube = _as_3d_cube(data)
    if fill_value is not None:
        cube = np.where(cube == fill_value, np.nan, cube)
    cube[cube < -32.0] = np.nan
    return cube


def downsample_dbz_to_model_tensor(
    dbz_cube: np.ndarray | np.ma.MaskedArray,
    dbz_min: float = DBZ_MIN,
    dbz_max: float = DBZ_MAX,
    return_coverage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Downsample in linear-Z space and optionally return valid-voxel fractions.

    Missing gates are excluded from the reflectivity average. A block with no
    observed gates receives ``dbz_min`` in the backward-compatible reflectivity
    tensor and 0 in the companion coverage tensor.
    """
    cube = _as_3d_cube(dbz_cube)
    if cube.shape[0] < 64 or cube.shape[1] < 480 or cube.shape[2] < 480:
        raise ValueError(
            f"Cube must be at least (64, 480, 480), got {cube.shape}"
        )

    cropped = cube[:64, :480, :480]
    valid = np.isfinite(cropped)
    clipped = np.clip(np.where(valid, cropped, dbz_min), dbz_min, dbz_max).astype(
        np.float32,
        copy=False,
    )

    linear_z = np.where(
        valid,
        np.power(10.0, clipped / 10.0, dtype=np.float32),
        0.0,
    ).astype(np.float32, copy=False)
    block_shape = (16, 4, 120, 4, 120, 4)
    valid_count = valid.reshape(block_shape).sum(axis=(1, 3, 5), dtype=np.int16)
    linear_sum = linear_z.reshape(block_shape).sum(axis=(1, 3, 5), dtype=np.float32)
    pooled_linear = np.divide(
        linear_sum,
        valid_count,
        out=np.ones(MODEL_TENSOR_SHAPE, dtype=np.float32),
        where=valid_count > 0,
    )
    pooled_dbz = 10.0 * np.log10(np.maximum(pooled_linear, 1e-6))
    pooled_dbz = np.clip(pooled_dbz, dbz_min, dbz_max).astype(np.float32)
    coverage = (valid_count.astype(np.float32) / 64.0).astype(np.float32)

    if pooled_dbz.shape != MODEL_TENSOR_SHAPE:
        raise RuntimeError(f"Tensor shape mismatch: {pooled_dbz.shape} != {MODEL_TENSOR_SHAPE}")
    if return_coverage:
        return pooled_dbz, coverage
    return pooled_dbz


def tensor_output_path(cube_nc: str | Path, output_dir: str | Path) -> Path:
    """Map `*_gridded.nc` cube names to model `.npy` tensor names."""
    cube_path = Path(cube_nc)
    stem = cube_path.stem
    if stem.endswith("_gridded"):
        stem = stem[: -len("_gridded")]
    return Path(output_dir) / f"{stem}.npy"


def coverage_output_path(cube_nc: str | Path, output_dir: str | Path) -> Path:
    """Return the companion coverage path outside root-level training discovery."""
    return Path(output_dir) / COVERAGE_DIRECTORY_NAME / tensor_output_path(cube_nc, output_dir).name


def _tensor_provenance(cube_nc: str | Path) -> dict:
    return {
        "pipeline_version": TENSOR_PIPELINE_VERSION,
        "stage": "cartesian_grid_to_model_tensor",
        "source": file_identity(cube_nc, include_path=False, include_mtime=False),
        "settings": {
            "input_crop": [64, 480, 480],
            "output_shape": list(MODEL_TENSOR_SHAPE),
            "pooling": "valid_only_linear_z_mean_4x4x4",
            "dbz_range": [DBZ_MIN, DBZ_MAX],
            "missing_output_dbz": DBZ_MIN,
            "coverage_companion": True,
        },
    }


def tensorize_cube(
    cube_nc: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Create one compact ConvLSTM tensor from a gridded cube file."""
    output_path = tensor_output_path(cube_nc, output_dir)
    coverage_path = coverage_output_path(cube_nc, output_dir)
    provenance = _tensor_provenance(cube_nc)
    if output_path.exists() and not overwrite:
        validate_reusable_artifact(
            output_path,
            provenance,
            related_directory=coverage_path.parent,
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube = load_dbz_cube(cube_nc)
    tensor, coverage = downsample_dbz_to_model_tensor(cube, return_coverage=True)
    save_npy_atomic(output_path, tensor)
    save_npy_atomic(coverage_path, coverage)
    write_artifact_manifest(output_path, provenance, related_paths=[coverage_path])
    return output_path


def iter_gridded_cubes(input_dir: str | Path) -> list[Path]:
    """Return timestamp-sortable gridded cube files from a directory."""
    return sorted(Path(input_dir).glob("*_gridded.nc"))


def tensorize_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Tensorize every gridded cube in a directory."""
    cubes = iter_gridded_cubes(input_dir)
    if limit is not None:
        cubes = cubes[: int(limit)]
    return [tensorize_cube(cube, output_dir=output_dir, overwrite=overwrite) for cube in cubes]
