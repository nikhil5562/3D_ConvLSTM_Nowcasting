"""Full-field quality control for L2B polar radar volumes."""

from __future__ import annotations

import numpy as np
import pyart
from scipy.ndimage import median_filter

from nowcasting.data.l2b_reader import NETCDF_FILL_VALUE, set_nonphysical_fill_value

RHOHV_CLUTTER_THRESHOLD = 0.85
MEDIAN_FILTER_SIZE = (3, 3)
SPIKE_THRESHOLD_DBZ = 15.0
DBZ_PHYSICAL_MAX = 75.0


def apply_full_field_qc(
    radar: pyart.core.Radar,
    rhohv_threshold: float = RHOHV_CLUTTER_THRESHOLD,
    spike_threshold_dbz: float = SPIKE_THRESHOLD_DBZ,
    dbz_physical_max: float = DBZ_PHYSICAL_MAX,
) -> pyart.core.Radar:
    """Apply RHOHV gating plus per-sweep spike filtering in-place."""
    if "DBZ" not in radar.fields:
        raise KeyError("Radar volume does not contain required DBZ field.")

    dbz_field = radar.fields["DBZ"]
    dbz_data = dbz_field["data"]
    dbz = np.ma.getdata(dbz_data).astype(np.float32).copy()
    base_mask = np.ma.getmaskarray(dbz_data).copy()
    new_mask = base_mask | (dbz > float(dbz_physical_max)) | ~np.isfinite(dbz)

    if "RHOHV" in radar.fields:
        rhohv_data = radar.fields["RHOHV"]["data"]
        rhohv = np.ma.getdata(rhohv_data).astype(np.float32)
        if rhohv.shape != dbz.shape:
            raise ValueError(f"RHOHV shape {rhohv.shape} does not match DBZ shape {dbz.shape}")
        new_mask |= (
            np.ma.getmaskarray(rhohv_data)
            | ~np.isfinite(rhohv)
            | (rhohv < float(rhohv_threshold))
        )

    cleaned_dbz = dbz.copy()
    for sweep_slice in radar.iter_slice():
        sweep_dbz = dbz[sweep_slice].copy()
        sweep_mask = new_mask[sweep_slice]
        filled = np.where(sweep_mask, -999.0, sweep_dbz)
        smoothed = median_filter(filled, size=MEDIAN_FILTER_SIZE, mode="nearest")
        spike = (sweep_dbz - smoothed) > float(spike_threshold_dbz)
        new_mask[sweep_slice] |= spike
        cleaned_dbz[sweep_slice] = sweep_dbz

    dbz_field["data"] = np.ma.masked_array(
        cleaned_dbz,
        mask=new_mask,
        fill_value=NETCDF_FILL_VALUE,
    )
    set_nonphysical_fill_value(dbz_field)
    if "VEL" in radar.fields:
        vel_field = radar.fields["VEL"]
        vel_data = vel_field["data"]
        vel = np.ma.getdata(vel_data).astype(np.float32).copy()
        if vel.shape != dbz.shape:
            raise ValueError(f"VEL shape {vel.shape} does not match DBZ shape {dbz.shape}")
        vel_mask = np.ma.getmaskarray(vel_data) | ~np.isfinite(vel) | new_mask
        vel_field["data"] = np.ma.masked_array(
            vel,
            mask=vel_mask,
            fill_value=NETCDF_FILL_VALUE,
        )
        set_nonphysical_fill_value(vel_field)

    return radar
