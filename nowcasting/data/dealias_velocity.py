"""Velocity dealiasing helpers for TERLS DWR volumes."""

from __future__ import annotations

import warnings

import numpy as np
import pyart

from nowcasting.data.l2b_reader import NETCDF_FILL_VALUE, set_nonphysical_fill_value


def add_dealiased_velocity(radar: pyart.core.Radar) -> pyart.core.Radar:
    """Add VEL_DEALIASED when VEL exists; otherwise leave the radar unchanged."""
    if "VEL" not in radar.fields:
        return radar

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dealiased = pyart.correct.dealias_region_based(
            radar,
            vel_field="VEL",
            keep_original=False,
            centered=True,
        )

    if np.ma.isMaskedArray(dealiased["data"]):
        dealiased["data"].set_fill_value(NETCDF_FILL_VALUE)
    set_nonphysical_fill_value(dealiased)
    radar.add_field("VEL_DEALIASED", dealiased, replace_existing=True)
    return radar
