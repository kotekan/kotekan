#!/usr/bin/env python3
"""predict_all's range rate against the range it is the rate of.

Doppler seeds come from range_rate_mps and code seeds from range_m, so the rate must be the
time derivative of the range; the reference is a +-0.5 s central difference of range_m.

The position is rotated from the transmit-epoch frame into the reception frame (the Sagnac
rotation, omega*tau ~ 5 microradians), and the velocity must be rotated with it. Left in the
transmit frame it errs by up to |v|*omega*tau ~ 2 cm/s on the line of sight.

The difference of ranges also carries the light-time rate (tau changes by rr/c per second),
which the prediction leaves out: up to ~2 mm/s at the edge of the sky, a few 0.1 mm/s at most
where |rr| < 100 m/s. The comparison is made there, so a 1 mm/s tolerance tests the frame.

The record is one real GPS LNAV record, G05 from BRDC00WRD_R day 292 (toe 04:00 GPST).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge

HEADER = (
    "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
    "                                                            END OF HEADER\n"
)
G05 = (
    "G05 2026 10 19 04 00 00-2.374746836722e-04-2.273736754432e-13 0.000000000000e+00\n"
    "     2.600000000000e+01 1.721875000000e+01 4.077669851317e-09-2.789512033642e+00\n"
    "     9.331852197647e-07 5.250758957118e-03 4.529953002930e-06 5.153581094742e+03\n"
    "     1.008000000000e+05 4.656612873077e-08-6.686003317161e-02 1.490116119385e-08\n"
    "     9.824851700281e-01 3.043125000000e+02 1.515045159552e+00-7.939973588913e-09\n"
    "     3.785871982478e-11 1.000000000000e+00 2.441000000000e+03 0.000000000000e+00\n"
    "     2.000000000000e+00 0.000000000000e+00-1.024454832077e-08 2.600000000000e+01\n"
    "     9.361800000000e+04 4.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n"
)
# Receiver sites spread over the globe so the check sees many line-of-sight geometries.
SITES = [
    (lat, lon) for lat in (-60, -30, 0, 30, 49.32, 70) for lon in range(-180, 180, 30)
]
ALT = 500.0
MAX_RR = 100.0  # m/s: where the light-time rate is well under the tolerance
TOL = 1e-3  # m/s


def parse(text):
    """{(sys, prn): [record]} from RINEX nav record text, through parse_rinex_nav."""
    with tempfile.NamedTemporaryFile("w", suffix=".rnx", delete=False) as f:
        f.write(HEADER + text)
    try:
        return ge.parse_rinex_nav([f.name])
    finally:
        os.unlink(f.name)


def toe_utc(rec):
    """The record's toe as unix seconds (UTC)."""
    return rec["toe_gpst"] - ge.GPS_UTC_LEAP + ge.GPS_EPOCH.timestamp()


def worst_rate_error(predict, t0):
    """Largest |range_rate_mps - d(range_m)/dt| over SITES and t0 +- 2 h, counting only epochs
    with the satellite up and |rate| < MAX_RR. `predict(t_utc, lat, lon)` returns predict_all's
    per-satellite dict, or None. Returns (worst, samples)."""
    worst, n = 0.0, 0
    for lat, lon in SITES:
        for dt in range(-7200, 7201, 300):
            t = t0 + dt
            p = predict(t, lat, lon)
            if p is None or p["el"] < 0.0:
                continue
            fd = (
                predict(t + 0.5, lat, lon)["range_m"]
                - predict(t - 0.5, lat, lon)["range_m"]
            )
            if abs(fd) > MAX_RR:
                continue
            worst = max(worst, abs(p["range_rate_mps"] - fd))
            n += 1
    return worst, n


class TestPredictAllRangeRate(unittest.TestCase):
    def test_range_rate_is_the_derivative_of_range(self):
        eph = parse(G05)
        key = ("G", 5)

        def predict(t, lat, lon):
            return ge.predict_all(eph, lat, lon, ALT, t, mask_deg=-90.0).get(key)

        worst, n = worst_rate_error(predict, toe_utc(eph[key][0]))
        self.assertGreater(n, 50, "too few near-zero-rate samples to test")
        self.assertLess(worst, TOL, "range rate off d(range)/dt by %.4f m/s" % worst)


if __name__ == "__main__":
    unittest.main(verbosity=2)
