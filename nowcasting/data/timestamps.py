"""Shared scan timestamps and cadence checks for preprocessing and model data."""

import re
from datetime import datetime
from pathlib import Path

from nowcasting.config import CONT_MAX_SEC, CONT_MIN_SEC


_SCAN_TIME_RE = re.compile(
    r"^RCTLS_(\d{2}[A-Z]{3}\d{4})_(\d{6})_L2B_STD(?:_gridded)?\.(?:nc|npy)$",
    re.IGNORECASE,
)


def parse_timestamp(filename):
    """Read the UTC scan timestamp from a supported RCTLS product name."""
    match = _SCAN_TIME_RE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Filename does not match an RCTLS L2B product: {filename}")
    return datetime.strptime("_".join(match.groups()), "%d%b%Y_%H%M%S")


def is_contiguous(timestamps):
    """Require every adjacent scan to match the configured cadence tolerance."""
    timestamps = list(timestamps)
    return all(
        CONT_MIN_SEC < (current - previous).total_seconds() < CONT_MAX_SEC
        for previous, current in zip(timestamps, timestamps[1:])
    )


def validate_sequence_paths(paths, expected_length):
    """Reject incorrectly sized, unordered, or mistimed forecast sequences."""
    if len(paths) != expected_length:
        raise ValueError(f"Expected {expected_length} paths, got {len(paths)}")
    if not is_contiguous(parse_timestamp(path) for path in paths):
        raise ValueError("Sequence timestamps do not match the configured scan cadence.")


def find_contiguous_window(records, length):
    """Return the first complete window in an already sorted scan index."""
    if length < 1:
        raise ValueError("Sequence length must be positive.")
    for index in range(len(records) - length + 1):
        window = records[index : index + length]
        if is_contiguous(record["time"] for record in window):
            return window
    return None
