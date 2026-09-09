"""gnss_pvt.hatch_smooth: carrier smoothing within an arc, never across one.

    python3 -m unittest test_pvt_hatch
"""
import math
import random
import unittest

import gnss_pvt

L_M = 299792458.0 / 10.23e6 * 10230      # one L5 code period, metres (~299.8 km)


def truth(t):
    """The model-removed range both observables share: the receiver clock, drifting slowly."""
    return 3.0 + 0.02 * t


def rows(n, code_sd, arc=1, t0=0.0, slope=0.0, const=-4.6e12, wrap=False):
    """n one-second rows: carrier residual = truth + a huge per-arc constant; code = truth +
    white noise + `slope`*t of code-carrier divergence."""
    random.seed(7)
    out = []
    for i in range(n):
        t = t0 + i
        carr = const + truth(t)               # -adr*lambda - range: the clock plus N*lambda
        code = truth(t) + slope * t + random.gauss(0.0, code_sd)
        if wrap:
            code += L_M * (i % 2)              # alternate rows one period apart
        out.append({"t": t, "code_resid_m": code, "carr_resid_m": carr, "adr_arc": arc})
    return out


class T(unittest.TestCase):
    def test_smoothing_beats_the_single_row(self):
        r = rows(90, code_sd=5.0)
        val, n, sd, method = gnss_pvt.hatch_smooth(r, 90.0)
        self.assertEqual(method, "hatch")
        self.assertEqual(n, 90)
        want = truth(r[-1]["t"])
        self.assertLess(abs(val - want), 2.0)                  # 5 m / sqrt(90) ~ 0.5 m
        self.assertGreater(abs(r[-1]["code_resid_m"] - want) + 1e-9, abs(val - want) * 0.2)
        self.assertAlmostEqual(sd, 5.0, delta=1.5)

    def test_never_across_an_arc_break(self):
        a = rows(60, code_sd=0.1, arc=1, const=-4.6e12)
        b = rows(30, code_sd=0.1, arc=2, t0=60.0, const=+1.2e11)   # new arc, new constant
        val, n, sd, method = gnss_pvt.hatch_smooth(a + b, 90.0)
        self.assertEqual(n, 30, "rows of the previous arc must not enter")
        self.assertLess(abs(val - truth(89.0)), 0.1)

    def test_window_is_honoured(self):
        _v, n, _sd, _m = gnss_pvt.hatch_smooth(rows(300, 1.0), 60.0)
        self.assertEqual(n, 61)

    def test_wrapped_code_is_unwrapped_to_the_newest(self):
        r = rows(40, code_sd=0.5, wrap=True)
        val, n, sd, _m = gnss_pvt.hatch_smooth(r, 90.0, wrap_m=L_M)
        self.assertEqual(n, 40)
        d = val - truth(r[-1]["t"])
        self.assertLess(abs(d - L_M * round(d / L_M)), 1.0)   # modulo one code period
        self.assertLess(sd, 1.0)

    def test_no_carrier_falls_back_to_a_boxcar(self):
        r = rows(30, code_sd=1.0)
        for x in r:
            x["carr_resid_m"] = None
        val, n, sd, method = gnss_pvt.hatch_smooth(r, 90.0)
        self.assertEqual(method, "boxcar")
        self.assertEqual(n, 30)

    def test_a_jumping_carrier_loses_to_the_boxcar(self):
        """The obs logs have carried an accumulated phase discontinuous by many kilometres
        inside one nominal arc. Hatch on that is catastrophic; the smoother must notice and
        use the code."""
        r = rows(60, code_sd=1.0)
        for i, x in enumerate(r):
            x["carr_resid_m"] += 1.0e10 * ((i * 7919) % 13)     # a phase reference that hops
        val, n, sd, method = gnss_pvt.hatch_smooth(r, 90.0)
        self.assertEqual(method, "boxcar")
        self.assertLess(abs(val - truth(r[-1]["t"])), 1.0)   # boxcar lag on a 2 cm/s clock
        self.assertLess(sd, 2.0)

    def test_single_row(self):
        val, n, sd, method = gnss_pvt.hatch_smooth(rows(1, 1.0), 90.0)
        self.assertEqual((n, method), (1, "single"))
        self.assertIsNone(gnss_pvt.hatch_smooth([], 90.0))


class TestWeights(unittest.TestCase):
    """A chain in its restart transient (sd 100 m) must not steer a fit that seven quiet chains
    (sd 3 m) are making; and a fit with no redundancy is not reported."""

    def _meas(self, noisy_bias):
        import math
        random.seed(11)
        out = []
        for g in range(8):
            for k in range(8):
                az, el = (g * 45 + k * 40) % 360, 20 + (k * 9) % 60
                res = -0.0 * az + 7.0 * g + random.gauss(0.0, 0.3)   # per-group clock only
                if g == 7:
                    res += noisy_bias + random.gauss(0.0, 30.0)
                    out.append({"group": "g%d" % g, "az": az, "el": el, "resid_m": res,
                                "sd_m": 100.0, "n": 45})
                else:
                    out.append({"group": "g%d" % g, "az": az, "el": el, "resid_m": res,
                                "sd_m": 3.0, "n": 45})
        return out

    def test_noisy_group_does_not_steer(self):
        r = gnss_pvt.solve(self._meas(noisy_bias=40.0), 49.32, -119.62, 545.0, min_el_deg=15.0)
        c = r["combined"]
        self.assertIsNotNone(c)
        self.assertLess(math.hypot(c["d_e"], c["d_n"]), 1.5, c)
        self.assertLess(abs(c["d_u"]), 3.0, c)
        # the quiet groups alone must agree with the combined answer
        g0 = r["groups"]["g0"]
        self.assertLess(math.hypot(g0["d_e"], g0["d_n"]), 2.0, g0)

    def test_no_redundancy_no_group_result(self):
        m = [{"group": "x", "az": a, "el": 45.0, "resid_m": 0.0, "sd_m": 1.0, "n": 10}
             for a in (0.0, 90.0, 180.0, 270.0)]
        r = gnss_pvt.solve(m, 49.32, -119.62, 545.0)
        self.assertNotIn("x", r["groups"])


if __name__ == "__main__":
    unittest.main()
