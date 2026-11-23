"""gnss_atmos.Ionex: a global ionosphere map's VTEC, interpolated in time, and its slant delay.

python3 -m unittest test_ionex
"""

import os
import tempfile
import unittest

import gnss_atmos


def _ionex(tecs, exponent=-1):
    """An IONEX body with one uniform map per value of `tecs` (TECU), an hour apart."""
    hdr = [
        "     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE",
        "  2026    11    23     0     0     0                        EPOCH OF FIRST MAP  ",
        "  6371.0                                                    BASE RADIUS         ",
        "   450.0 450.0   0.0                                        HGT1 / HGT2 / DHGT  ",
        "    87.5 -87.5  -2.5                                        LAT1 / LAT2 / DLAT  ",
        "  -180.0 180.0   5.0                                        LON1 / LON2 / DLON  ",
        "%6d                                                      EXPONENT            "
        % exponent,
        "                                                            END OF HEADER       ",
    ]
    body = []
    for k, tec in enumerate(tecs):
        body.append("%6d%54sSTART OF TEC MAP    " % (k + 1, ""))
        body.append(
            "  2026    11    23 %5d     0     0%24sEPOCH OF CURRENT MAP" % (k, "")
        )
        v = int(round(tec / 10.0**exponent))
        for i in range(71):
            body.append(
                "  %6.1f-180.0 180.0   5.0 450.0%28sLAT/LON1/LON2/DLON/H"
                % (87.5 - 2.5 * i, "")
            )
            row = [v] * 73
            for j in range(0, 73, 16):
                body.append("".join("%5d" % x for x in row[j : j + 16]))
        body.append("%6d%54sEND OF TEC MAP      " % (k + 1, ""))
    return "\n".join(hdr + body) + "\n"


class T(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".INX")
        with os.fdopen(fd, "w") as f:
            f.write(_ionex([10.0, 20.0]))
        self.g = gnss_atmos.Ionex(self.path)

    def tearDown(self):
        os.remove(self.path)

    def test_time_interpolation(self):
        t0 = self.g.t0
        self.assertAlmostEqual(self.g.vtec(t0, 40.0, -80.0), 10.0, places=6)
        self.assertAlmostEqual(self.g.vtec(t0 + 1800.0, 40.0, -80.0), 15.0, places=6)
        self.assertIsNone(self.g.vtec(t0 - 1.0, 40.0, -80.0))

    def test_zenith_and_slant_delay(self):
        t = self.g.t0 + 1800.0
        f = 1176.45e6
        zen = self.g.slant_delay_m(t, 40.0, -80.0, 90.0, 0.0, f)
        self.assertAlmostEqual(zen, 40.3e16 * 15.0 / f**2, places=6)
        self.assertGreater(
            self.g.slant_delay_m(t, 40.0, -80.0, 20.0, 135.0, f), 2.0 * zen
        )


if __name__ == "__main__":
    unittest.main()
