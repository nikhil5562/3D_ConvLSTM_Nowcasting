import unittest

import numpy as np

from nowcasting.data.l2b_reader import NETCDF_FILL_VALUE
from nowcasting.data.quality_control import apply_full_field_qc


class RadarStub:
    def __init__(self, dbz, rhohv=None, vel=None):
        self.fields = {"DBZ": {"data": dbz}}
        if rhohv is not None:
            self.fields["RHOHV"] = {"data": rhohv}
        if vel is not None:
            self.fields["VEL"] = {"data": vel}

    def iter_slice(self):
        yield slice(0, self.fields["DBZ"]["data"].shape[0])


class QualityControlTests(unittest.TestCase):
    def test_unions_dbz_rhohv_and_velocity_masks(self):
        shape = (3, 3)
        dbz = np.ma.masked_array(np.full(shape, 20.0, dtype=np.float32), mask=False)
        dbz.mask[0, 0] = True

        rhohv_values = np.full(shape, 0.95, dtype=np.float32)
        rhohv_values[1, 0] = np.nan
        rhohv_values[1, 1] = 0.5
        rhohv = np.ma.masked_array(rhohv_values, mask=False)
        rhohv.mask[0, 1] = True

        vel = np.ma.masked_array(np.full(shape, 2.0, dtype=np.float32), mask=False)
        vel.mask[2, 2] = True

        radar = apply_full_field_qc(RadarStub(dbz, rhohv=rhohv, vel=vel))

        dbz_mask = np.ma.getmaskarray(radar.fields["DBZ"]["data"])
        vel_mask = np.ma.getmaskarray(radar.fields["VEL"]["data"])
        for index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.assertTrue(dbz_mask[index])
            self.assertTrue(vel_mask[index])
        self.assertTrue(vel_mask[2, 2])
        self.assertEqual(radar.fields["DBZ"]["_FillValue"], NETCDF_FILL_VALUE)
        self.assertEqual(radar.fields["VEL"]["_FillValue"], NETCDF_FILL_VALUE)

    def test_rejects_mismatched_qc_field_shape(self):
        dbz = np.ma.masked_array(np.zeros((3, 3), dtype=np.float32))
        rhohv = np.ma.masked_array(np.ones((2, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "RHOHV shape"):
            apply_full_field_qc(RadarStub(dbz, rhohv=rhohv))


if __name__ == "__main__":
    unittest.main()
