"""Create full-resolution Cartesian cubes from L2B polar radar volumes."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pyart
import xarray as xr

from nowcasting.data.dealias_velocity import add_dealiased_velocity
from nowcasting.data.l2b_reader import (
    DEFAULT_MIN_FULL_SCAN_SIZE_MB,
    DEFAULT_MIN_SWEEPS,
    discover_l2b_candidates,
    is_full_volume_radar,
    read_l2b_radar,
    set_nonphysical_fill_value,
)
from nowcasting.data.provenance import (
    file_identity,
    validate_reusable_artifact,
    write_artifact_manifest,
)
from nowcasting.data.quality_control import apply_full_field_qc

FULL_CUBE_SHAPE = (81, 481, 481)
N_ALTITUDE, N_Y, N_X = FULL_CUBE_SHAPE
ALT_MAX_M = 20_000.0
HALF_HORIZ_M = 240_000.0
GRID_PIPELINE_VERSION = "l2b-cartesian-v2"


def _validate_written_grid(path: str | Path) -> None:
    """Reject a malformed temporary cube before it replaces the destination."""
    with xr.open_dataset(path, decode_times=False) as dataset:
        if "DBZ" not in dataset:
            raise ValueError(f"Written grid has no DBZ variable: {path}")
        variable = dataset["DBZ"]
        if tuple(variable.shape[-3:]) != tuple(FULL_CUBE_SHAPE):
            raise ValueError(
                f"Written DBZ shape {variable.shape} does not end with "
                f"{FULL_CUBE_SHAPE}: {path}"
            )
        fill_value = variable.encoding.get(
            "_FillValue",
            variable.attrs.get("_FillValue", variable.attrs.get("missing_value")),
        )
        if fill_value is None or float(fill_value) != -9999.0:
            raise ValueError(
                f"Written DBZ fill value is {fill_value!r}, expected -9999.0: {path}"
            )


def gridded_output_path(polar_nc: str | Path, output_dir: str | Path) -> Path:
    """Return the standard gridded cube path for an L2B input file."""
    return Path(output_dir) / f"{Path(polar_nc).stem}_gridded.nc"


def _fields_to_grid(radar: pyart.core.Radar) -> list[str]:
    fields = ["DBZ"]
    if "VEL_DEALIASED" in radar.fields:
        fields.append("VEL_DEALIASED")
    elif "VEL" in radar.fields:
        fields.append("VEL")
    return fields


def write_cartesian_grid(
    radar: pyart.core.Radar,
    output_nc: str | Path,
    roi_func: str = "dist_beam",
    constant_roi: float | None = None,
) -> Path:
    """Grid a corrected radar volume to the MOSDAC-compatible 81x481x481 cube."""
    roi_kwargs = {"roi_func": roi_func}
    if roi_func == "constant":
        roi_kwargs["constant_roi"] = float(constant_roi or 700.0)
    else:
        roi_kwargs.update({"h_factor": 1.0, "nb": 1.5, "bsp": 1.0, "min_radius": 500.0})

    for field_name in _fields_to_grid(radar):
        set_nonphysical_fill_value(radar.fields[field_name])

    grid_obj = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=FULL_CUBE_SHAPE,
        grid_limits=((0.0, ALT_MAX_M), (-HALF_HORIZ_M, HALF_HORIZ_M), (-HALF_HORIZ_M, HALF_HORIZ_M)),
        fields=_fields_to_grid(radar),
        weighting_function="Barnes2",
        **roi_kwargs,
    )

    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_nc.with_name(f".{output_nc.stem}.{uuid.uuid4().hex}.tmp.nc")
    try:
        pyart.io.write_grid(str(temporary), grid_obj)
        _validate_written_grid(temporary)
        os.replace(temporary, output_nc)
    finally:
        temporary.unlink(missing_ok=True)
    return output_nc


def _grid_provenance(
    polar_nc: Path,
    min_sweeps: int,
    roi_func: str = "dist_beam",
    constant_roi: float | None = None,
) -> dict:
    return {
        "pipeline_version": GRID_PIPELINE_VERSION,
        "stage": "l2b_to_cartesian_grid",
        "source": file_identity(polar_nc, include_path=False, include_mtime=False),
        "settings": {
            "grid_shape": list(FULL_CUBE_SHAPE),
            "grid_limits_m": [[0.0, ALT_MAX_M], [-HALF_HORIZ_M, HALF_HORIZ_M], [-HALF_HORIZ_M, HALF_HORIZ_M]],
            "weighting_function": "Barnes2",
            "roi_func": roi_func,
            "constant_roi": float(constant_roi or 700.0) if roi_func == "constant" else None,
            "min_sweeps": int(min_sweeps),
            "netcdf_fill_value": -9999.0,
        },
    }


def validate_gridded_cube_provenance(
    polar_nc: str | Path,
    gridded_nc: str | Path,
    min_sweeps: int = DEFAULT_MIN_SWEEPS,
) -> dict:
    """Verify that an existing cube exactly matches its source and settings."""
    provenance = _grid_provenance(Path(polar_nc), min_sweeps=min_sweeps)
    return validate_reusable_artifact(gridded_nc, provenance)


def create_gridded_cube(
    polar_nc: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    min_sweeps: int = DEFAULT_MIN_SWEEPS,
) -> Path | None:
    """Run the full-field L2B -> gridded cube pipeline for one scan."""
    polar_nc = Path(polar_nc)
    output_nc = gridded_output_path(polar_nc, output_dir)
    provenance = _grid_provenance(polar_nc, min_sweeps=min_sweeps)
    if output_nc.exists() and not overwrite:
        validate_gridded_cube_provenance(
            polar_nc,
            output_nc,
            min_sweeps=min_sweeps,
        )
        return output_nc

    radar = read_l2b_radar(polar_nc)
    if not is_full_volume_radar(radar, min_sweeps=min_sweeps):
        return None

    radar = apply_full_field_qc(radar)
    radar = add_dealiased_velocity(radar)
    output = write_cartesian_grid(radar, output_nc)
    write_artifact_manifest(output, provenance)
    return output


def create_gridded_cubes(
    input_dir: str | Path,
    output_dir: str | Path,
    min_size_mb: float = DEFAULT_MIN_FULL_SCAN_SIZE_MB,
    min_sweeps: int = DEFAULT_MIN_SWEEPS,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Batch-create gridded cubes for probable full-volume L2B scans."""
    candidates = discover_l2b_candidates(input_dir, min_size_mb=min_size_mb)
    if limit is not None:
        candidates = candidates[: int(limit)]

    outputs: list[Path] = []
    for candidate in candidates:
        output = create_gridded_cube(
            candidate,
            output_dir=output_dir,
            overwrite=overwrite,
            min_sweeps=min_sweeps,
        )
        if output is not None:
            outputs.append(output)
    return outputs
