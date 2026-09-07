"""Read and select MOSDAC RCTLS L2B standard radar volume files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyart

from nowcasting.config import CONT_MAX_SEC, CONT_MIN_SEC
from nowcasting.data.timestamps import parse_timestamp

NYQUIST_MS = 24.0
DEFAULT_MIN_FULL_SCAN_SIZE_MB = 100.0
DEFAULT_MIN_SWEEPS = 11
NETCDF_FILL_VALUE = np.float32(-9999.0)

_FILENAME_TIME_RE = re.compile(
    r"^RCTLS_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})(\d{2})_L2B_STD\.nc$",
    re.IGNORECASE,
)


def parse_timestamp_from_filename(filename: str | Path) -> datetime:
    """Parse the RCTLS timestamp embedded in an L2B filename."""
    name = Path(filename).name.upper()
    match = _FILENAME_TIME_RE.match(name)
    if not match:
        raise ValueError(f"Filename does not match RCTLS L2B standard pattern: {filename}")

    return parse_timestamp(filename)


def is_l2b_standard_file(path: str | Path) -> bool:
    """Return True only for raw RCTLS L2B standard NetCDF filenames."""
    return _FILENAME_TIME_RE.match(Path(path).name.upper()) is not None


def is_probable_full_volume_file(
    path: str | Path,
    min_size_mb: float = DEFAULT_MIN_FULL_SCAN_SIZE_MB,
) -> bool:
    """Fast size prefilter that rejects short interleaved scans."""
    path = Path(path)
    return path.is_file() and (path.stat().st_size / (1024 * 1024)) >= float(min_size_mb)


def discover_l2b_candidates(
    input_dir: str | Path,
    min_size_mb: float = DEFAULT_MIN_FULL_SCAN_SIZE_MB,
) -> list[Path]:
    """Find timestamp-sorted L2B files that pass the cheap full-volume prefilter."""
    root = Path(input_dir)
    candidates = [
        path
        for path in root.glob("RCTLS_*_L2B_STD.nc")
        if is_l2b_standard_file(path) and is_probable_full_volume_file(path, min_size_mb)
    ]
    return sorted(candidates, key=parse_timestamp_from_filename)


def is_full_volume_radar(radar: object, min_sweeps: int = DEFAULT_MIN_SWEEPS) -> bool:
    """Return True when a loaded radar object has enough sweeps for a full volume."""
    return int(getattr(radar, "nsweeps", 0)) >= int(min_sweeps)


def set_nonphysical_fill_value(field: dict) -> None:
    """Use a fill value that cannot be confused with valid 0 dBZ/0 m s-1 data."""
    field["_FillValue"] = NETCDF_FILL_VALUE
    field["missing_value"] = NETCDF_FILL_VALUE


def _fix_dbz_fill_value(radar: pyart.core.Radar) -> None:
    if "DBZ" not in radar.fields:
        return
    field = radar.fields["DBZ"]
    dbz = field["data"]
    raw = np.ma.getdata(dbz).astype(np.float32)
    original_mask = np.ma.getmaskarray(dbz)
    mask = original_mask | (raw < -32.0) | ~np.isfinite(raw)
    field["data"] = np.ma.masked_array(raw, mask=mask, fill_value=NETCDF_FILL_VALUE)
    set_nonphysical_fill_value(field)


def _inject_nyquist_velocity(radar: pyart.core.Radar) -> None:
    if radar.instrument_parameters is None:
        radar.instrument_parameters = {}
    if "nyquist_velocity" in radar.instrument_parameters:
        return

    radar.instrument_parameters["nyquist_velocity"] = {
        "data": np.full(radar.nrays, NYQUIST_MS, dtype=np.float32),
        "units": "meters_per_second",
        "long_name": "Nyquist velocity",
        "comments": "Inferred from TERLS VEL range; not present in source file.",
    }


def _fix_time_units(radar: pyart.core.Radar, path: Path) -> None:
    units = radar.time.get("units", "")
    if "yyyy" not in units.lower() and units.lower().startswith("seconds since"):
        return

    scan_dt = parse_timestamp_from_filename(path)
    radar.time["units"] = scan_dt.strftime("seconds since %Y-%m-%d %H:%M:%S")


def _fix_sweep_ray_indices(radar: pyart.core.Radar) -> None:
    if radar.nsweeps <= 1:
        return

    starts = np.asarray(radar.sweep_start_ray_index["data"], dtype=np.int64)
    ends = np.asarray(radar.sweep_end_ray_index["data"], dtype=np.int64)
    if starts.size != radar.nsweeps or ends.size != radar.nsweeps:
        raise ValueError(
            "Sweep index metadata length does not match nsweeps: "
            f"starts={starts.size}, ends={ends.size}, nsweeps={radar.nsweeps}"
        )
    if np.ptp(starts) != 0 or np.ptp(ends) != 0:
        expected_starts = np.concatenate(([0], ends[:-1] + 1))
        valid = (
            starts[0] == 0
            and ends[-1] == radar.nrays - 1
            and np.array_equal(starts, expected_starts)
            and np.all(starts <= ends)
        )
        if not valid:
            raise ValueError(
                "Sweep ray indices are non-contiguous or do not cover every ray; "
                "refusing to construct a potentially misregistered volume."
            )
        return

    if radar.nrays % radar.nsweeps != 0:
        raise ValueError(
            f"Cannot safely rebuild sweep indices: {radar.nrays} rays are not divisible "
            f"by {radar.nsweeps} sweeps."
        )
    rays_per_sweep = radar.nrays // radar.nsweeps
    rebuilt_starts = np.arange(radar.nsweeps, dtype=np.int32) * rays_per_sweep
    radar.sweep_start_ray_index["data"] = rebuilt_starts
    radar.sweep_end_ray_index["data"] = rebuilt_starts + rays_per_sweep - 1


def diagnose_sequence_capacity(
    paths: Iterable[str | Path],
    frames_per_sequence: int,
    min_gap_seconds: float = CONT_MIN_SEC,
    max_gap_seconds: float = CONT_MAX_SEC,
) -> dict[str, object]:
    """Summarize contiguous timestamp runs before expensive gridding/training."""
    if frames_per_sequence <= 0:
        raise ValueError("frames_per_sequence must be positive")
    if min_gap_seconds >= max_gap_seconds:
        raise ValueError("min_gap_seconds must be smaller than max_gap_seconds")

    timestamps = sorted(parse_timestamp_from_filename(path) for path in paths)
    if not timestamps:
        return {
            "frame_count": 0,
            "run_lengths": [],
            "longest_run": 0,
            "sequence_count": 0,
            "frames_per_sequence": int(frames_per_sequence),
        }

    run_lengths = []
    current_run = 1
    for previous, current in zip(timestamps, timestamps[1:]):
        gap = (current - previous).total_seconds()
        if float(min_gap_seconds) < gap < float(max_gap_seconds):
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    sequence_count = sum(max(0, length - frames_per_sequence + 1) for length in run_lengths)
    return {
        "frame_count": len(timestamps),
        "run_lengths": run_lengths,
        "longest_run": max(run_lengths),
        "sequence_count": sequence_count,
        "frames_per_sequence": int(frames_per_sequence),
    }


def read_l2b_radar(path: str | Path) -> pyart.core.Radar:
    """Read an L2B polar volume and apply MOSDAC schema fixes."""
    path = Path(path)
    radar = pyart.io.read_cfradial(str(path))
    _fix_dbz_fill_value(radar)
    _inject_nyquist_velocity(radar)
    _fix_time_units(radar, path)
    _fix_sweep_ray_indices(radar)
    return radar


def iter_full_volume_radars(
    paths: Iterable[str | Path],
    min_sweeps: int = DEFAULT_MIN_SWEEPS,
) -> Iterable[tuple[Path, pyart.core.Radar]]:
    """Yield loaded radar objects that pass the authoritative sweep-count check."""
    for path_like in paths:
        path = Path(path_like)
        radar = read_l2b_radar(path)
        if is_full_volume_radar(radar, min_sweeps=min_sweeps):
            yield path, radar
