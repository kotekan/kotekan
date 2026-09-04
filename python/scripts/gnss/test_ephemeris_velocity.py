"""Task #52: the satellite velocity must be a CENTRAL difference, and here is what it costs.

gnss_ephemeris.sat_pos_clk used a FORWARD difference over h = 0.5 s and called it "adequate:
range-rate to ~mm/s". A forward difference does not estimate v(t); it estimates v(t + h/2). The
error is therefore a systematic TIME-TAG OFFSET of h/2, not noise -- it does not average down,
it does not shrink with SNR, and it is proportional to acceleration, so it is largest exactly
at zenith where the pass is best.

WHY IT MATTERED. A 0.25 s time offset in Doppler is dop_rate * 0.25. On sky the E5a fleet runs
dop_rate 0.036-0.369 Hz/s, so the induced Doppler error is 9-92 mHz -- against a cross-window
phase budget (#52) of ~15 mHz for 0.1 rad across one 1.049 s window. One first-order scheme was
up to 6x the entire budget.

These tests use a pure circular orbit, where the truth is analytic, so they measure the SCHEME
rather than the ephemeris.
"""
import math
import unittest

R_GPS = 26.56e6        # GPS orbit radius, m
T_GPS = 43082.0        # sidereal-ish period, s
CARRIER = 1176.45e6
C = 2.99792458e8


def pos(t, r=R_GPS, period=T_GPS):
    w = 2 * math.pi / period
    return (r * math.cos(w * t), r * math.sin(w * t), 0.0)


def vel_true(t, r=R_GPS, period=T_GPS):
    w = 2 * math.pi / period
    return (-r * w * math.sin(w * t), r * w * math.cos(w * t), 0.0)


def vel_forward(t, h=0.5):
    a, b = pos(t), pos(t + h)
    return tuple((b[k] - a[k]) / h for k in range(3))


def vel_central(t, h=0.5):
    a, b = pos(t - h), pos(t + h)
    return tuple((b[k] - a[k]) / (2 * h) for k in range(3))


def err(v, t):
    vt = vel_true(t)
    return math.sqrt(sum((v[k] - vt[k]) ** 2 for k in range(3)))


class TestSchemeNotEphemeris(unittest.TestCase):

    def test_forward_error_is_a_time_offset_of_h_over_2(self):
        """THE DIAGNOSIS, as an assertion: the forward difference's error is not random, it is
        v(t + h/2) - v(t). If this ever stops holding, the error has a different origin and the
        central-difference repair is not the right one."""
        h = 0.5
        t = 1234.0
        got = vel_forward(t, h)
        predicted = vel_true(t + h / 2.0)
        self.assertLess(err(got, t + h / 2.0), 0.05 * err(got, t),
                        "forward difference did not land on v(t+h/2): %r" % (got,))
        for k in range(3):
            self.assertAlmostEqual(got[k], predicted[k], delta=0.02)

    def test_central_beats_forward_by_orders(self):
        h = 0.5
        t = 1234.0
        ef = err(vel_forward(t, h), t)
        ec = err(vel_central(t, h), t)
        self.assertGreater(ef / max(ec, 1e-12), 1000.0,
                           "central %.3g m/s vs forward %.3g m/s -- expected >1000x" % (ec, ef))

    def test_the_forward_bias_equals_acceleration_times_h_over_2(self):
        """Closed form, so the size is predicted rather than measured: |a| h/2."""
        h = 0.5
        w = 2 * math.pi / T_GPS
        a_mag = R_GPS * w * w
        self.assertAlmostEqual(err(vel_forward(1234.0, h), 1234.0), a_mag * h / 2.0, delta=0.01)

    def test_it_exceeded_the_whole_cross_window_phase_budget(self):
        """The number that makes this worth fixing rather than noting. Line-of-sight
        acceleration implied by a real fleet Doppler rate, times h/2, in Hz."""
        budget_hz = 0.1 / (2 * math.pi * 1.0485760)      # 0.1 rad across one window
        self.assertAlmostEqual(budget_hz, 0.0152, places=4)
        for dop_rate in (0.036, 0.369):                   # measured E5a fleet range
            a_los = dop_rate * C / CARRIER                # m/s^2
            dop_err = (a_los * 0.25) * CARRIER / C        # Hz  == dop_rate * 0.25
            self.assertAlmostEqual(dop_err, dop_rate * 0.25, places=9)
            if dop_rate > 0.3:
                self.assertGreater(dop_err, 5 * budget_hz)

    def test_shrinking_h_is_not_the_fix(self):
        """⚠️ The tempting repair. The forward bias is FIRST order in h, so halving h only
        halves it and costs precision elsewhere; centring removes it outright at the same h."""
        t = 1234.0
        self.assertAlmostEqual(err(vel_forward(t, 0.25), t),
                               err(vel_forward(t, 0.5), t) / 2.0, delta=0.01)
        self.assertLess(err(vel_central(t, 0.5), t), err(vel_forward(t, 0.001), t))


class TestShippedFunction(unittest.TestCase):

    def test_sat_pos_clk_uses_a_central_difference(self):
        """Reads the shipped source rather than trusting the comment -- this defect WAS a
        comment that said "adequate: range-rate to ~mm/s" over a scheme that was not."""
        import os
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "gnss_ephemeris.py")) as fh:
            src = fh.read()
        self.assertIn("pm = _pos_only(e2, t_gpst - dt)", src)
        self.assertIn("(pp[k] - pm[k]) / (2.0 * dt)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
