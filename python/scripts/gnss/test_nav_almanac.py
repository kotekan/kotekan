#!/usr/bin/env python3
"""nav_almanac: the almanac layouts, their placement in time, the pairing of the messages an
almanac spans, and the propagator.

The messages are live ones (11-24): an F/NAV page 5 and the page 6 after it from E30, an I/NAV
run of words 7 to 10 from E27, and a B-CNAV2 type 40 from C28. The values they are held to are
BRDC's for the satellites they describe (day 328), to the almanac fields' resolution.
"""

import math
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge
import nav_almanac as na


def bits(hexstr, n):
    v = int(hexstr, 16) >> (len(hexstr) * 4 - n)
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


P5 = bits("166625c00401d5eafee63413fc705b3ff43ffe3a00a028ae4bee874684c0", 238)
P6 = bits("1a656bf8fcf97fba7ffc7802c08c95bbcf378f7f214fdfc7fffa8ac97e4c", 238)
W7 = bits("1ee62df00800dfc2bf52de1dfcb178d5", 128)
W8 = bits("22ffe73ffe100020062600f731a09fe2", 128)
W9 = bits("26e62d68efffc3ffe1080240ddc36063", 128)
W10 = bits("2af1293f869c2fff3fffc000bc00120e", 128)
F40 = bits(
    "5e84d87c1e00c63dfb991063807810a514d649feccdb966994c2000000000000000577fa", 288
)
# BRDC day 328: (sqrtA, e, i0, OMEGA0, af0)
BRDC = {
    ("E", 28): (5440.60, 0.00045, 0.9503, 0.6391, None),
    ("E", 29): (5440.61, 0.00031, 0.9640, 0.6223, -2.672e-4),
    ("E", 30): (5440.63, 0.00054, 0.9586, 2.7273, -1.711e-3),
    ("E", 31): (5440.62, 0.00020, 0.9608, 2.7257, None),
    ("E", 32): (5440.60, 0.00019, 0.9503, 0.6390, -3.076e-5),
    ("E", 33): (5440.60, 0.00041, 0.9963, -1.4569, -4.757e-5),
    ("C", 28): (5282.63, 0.00023, 0.9455, None, 5.82e-4),
}
TOL = (0.1, 2e-5, 3e-4, 2e-3, 5e-6)


class TestLayouts(unittest.TestCase):
    def _check(self, alms):
        for sysc, prn, wna, toa, eph in alms:
            want = BRDC[(sysc, prn)]
            got = (eph["sqrtA"], eph["e"], eph["i0"], eph["OMEGA0"], eph["af0"])
            for f, w, g, tol in zip(
                ("sqrtA", "e", "i0", "OMEGA0", "af0"), want, got, TOL
            ):
                if w is not None:
                    self.assertLess(
                        abs(g - w),
                        tol,
                        "%s%02d %s %.6g, BRDC %.6g" % (sysc, prn, f, g, w),
                    )

    def test_fnav_pages_5_and_6(self):
        alms = [na.fnav_sv1(P5)] + na.fnav_sv2_sv3(P5, P6)
        self.assertEqual([a[1] for a in alms], [28, 29, 30])
        self.assertEqual({(a[2], a[3]) for a in alms}, {(2, 235800)})
        self._check(alms)

    def test_inav_words_7_to_10(self):
        alms = na.inav_svs(W7, W8, W9, W10)
        self.assertEqual([a[1] for a in alms], [31, 32, 33])
        self.assertEqual({(a[2], a[3]) for a in alms}, {(2, 237000)})
        self._check(alms)

    def test_bcnav2_type_40(self):
        a = na.bcnav2_type40(F40)
        self.assertEqual(a[:4], ("C", 28, 1089, 581632))
        self._check([a])


# GLONASS strings 6 and 7 (slot 16's almanac) and string 5 from R16, live 11-24, as
# glonass_nav_predictor keeps them (b[i] = bit i), packed with bit 0 least significant.
GLO_EVEN, GLO_ODD, GLO_S5 = (
    "0d61078b06b827720699da",
    "0ff34d0312ad3007857cac",
    "0b08c0000001080000311a",
)


def glo_bits(h):
    v = int(h, 16)
    return [(v >> i) & 1 for i in range(86)]


class TestGlonass(unittest.TestCase):
    """Held to BRDC on 11-24 the live almanacs of all 24 slots put the Doppler within 1.1 Hz and
    the range within 2.9 km, every channel as BRDC's."""

    def setUp(self):
        n_a = na._glo_u(glo_bits(GLO_S5), 80, 11)
        self.a = na.glo_almanac(glo_bits(GLO_EVEN), glo_bits(GLO_ODD), n_a)

    def test_fields(self):
        sysc, slot, n_a, t_lam, eph = self.a
        self.assertEqual((sysc, slot, n_a, eph["k"]), ("R", 16, 1059, -1))
        self.assertAlmostEqual(43200.0 + eph["dT"], 40544, delta=5)  # 11 h 15 m 44 s

    def test_node_passage(self):
        """The ascending node at t_lambda and one draconic period later, at GLONASS's radius."""
        eph = dict(self.a[4], toe=0.0)
        for t in (0.0, 43200.0 + eph["dT"]):
            x = na.sv_position_glo_alm(eph, t, wrap=False)
            self.assertLess(abs(x[2]), 20e3)
            self.assertAlmostEqual(math.dist(x, (0, 0, 0)), 25.51e6, delta=0.1e6)

    def test_day_of_the_span(self):
        # N^A 1059 of the span from 2024-01-01 is 2026-11-24; t_lambda is Moscow time (UTC + 3 h)
        t = na.glo_toa_gpst(1059, self.a[3], 2446 * 604800.0 + 2 * 86400.0)
        utc = na.ge_utc(t)
        self.assertEqual((utc.year, utc.month, utc.day), (2026, 11, 24))
        self.assertAlmostEqual(
            utc.hour * 3600 + utc.minute * 60 + utc.second + utc.microsecond * 1e-6,
            self.a[3] - 3 * 3600.0,
            delta=1e-3,
        )

    def test_collector_pairs_strings(self):
        f0 = 1795546800.0  # a frame start (2 s strings; frame 1 of its superframe)
        st = types.SimpleNamespace(
            utc_off=0.0,
            strings={
                f0 + 8.0: (5, glo_bits(GLO_S5), 0.0),
                f0 + 10.0: (6, glo_bits(GLO_EVEN), 0.0),
                f0 + 12.0: (7, glo_bits(GLO_ODD), 0.0),
            },
        )
        col = na.Collector("GLO_L2OF")
        col.scan(types.SimpleNamespace(_p={16: st}))
        self.assertEqual(col.got, [])  # read once: not yet
        # the same pair from another satellite at the same time is a second reading
        col.scan(types.SimpleNamespace(_p={9: st}))
        got = col.take(2446 * 604800.0 + 2 * 86400.0)
        self.assertEqual([(e["signal"], e["prn"]) for e in got], [("GLO_ALM", 16)])

    def test_collector_drops_a_pair_read_once(self):
        """A miscorrected even string differs from every other reading of its pair, so it never
        repeats; the true pair, read again in the next superframe, is taken."""
        f0 = 1795546800.0
        bad = glo_bits(GLO_EVEN)
        bad[76] ^= 1  # slot 16 read as 24
        col = na.Collector("GLO_L2OF")
        for f, even in ((f0, bad), (f0 + 150.0, glo_bits(GLO_EVEN))):
            st = types.SimpleNamespace(
                utc_off=0.0,
                strings={
                    f + 8.0: (5, glo_bits(GLO_S5), 0.0),
                    f + 10.0: (6, even, 0.0),
                    f + 12.0: (7, glo_bits(GLO_ODD), 0.0),
                },
            )
            col.scan(types.SimpleNamespace(_p={16: st}))
        self.assertEqual(col.got, [])
        st.strings = {
            k + 150.0: v for k, v in st.strings.items()
        }  # the superframe after
        col.scan(types.SimpleNamespace(_p={16: st}))
        self.assertEqual([a[1] for a in col.got], [16])


class TestPlacement(unittest.TestCase):
    WK = 2446 * 604800.0  # GPST week 2446, BDT week 1090, GST week 1422 (2 mod 4)

    def test_beidou_week_is_bdt(self):
        # BDT week 1089 starts 14 s into GPST week 2445
        got = na.toa_gpst("C", 1089, 581632, self.WK + 1000.0)
        self.assertEqual(got, self.WK - 604800.0 + 581632 + 14.0)

    def test_galileo_week_mod_4(self):
        self.assertEqual(
            na.toa_gpst("E", 2, 235800, self.WK + 300000.0), self.WK + 235800
        )
        # last week's (1 mod 4), read early in this week
        self.assertEqual(
            na.toa_gpst("E", 1, 590000, self.WK + 1000.0), self.WK - 604800.0 + 590000
        )
        # next week's (3 mod 4), uploaded ahead
        self.assertEqual(
            na.toa_gpst("E", 3, 1200, self.WK + 600000.0), self.WK + 604800.0 + 1200
        )


class TestPairing(unittest.TestCase):
    """A decoder holds each satellite's newest message of each type: the collector pairs a page
    6 with the page 5 50 s before it and a word with the one before it in the run, both of the
    same IODa, and nothing else."""

    def _feed(self, signal, msgs):
        col = na.Collector(signal)
        st = types.SimpleNamespace(pages={}, words={}, frames={})
        dec = types.SimpleNamespace(_p={30: st})
        for k, b, t in msgs:
            if signal == "GAL_E5AI_FNAV":
                st.pages[k] = (None, b, t, t)
            else:
                st.words[k] = (None, b, t)
            col.scan(dec)
        return sorted(e["prn"] for e in col.take(self.WK))

    WK = 2446 * 604800.0

    def test_fnav(self):
        self.assertEqual(
            self._feed("GAL_E5AI_FNAV", [(5, P5, 100.0), (6, P6, 150.0)]), [28, 29, 30]
        )
        # a page 6 a frame later than its page 5: satellite 1 only
        self.assertEqual(
            self._feed("GAL_E5AI_FNAV", [(5, P5, 100.0), (6, P6, 250.0)]), [28]
        )

    def test_inav(self):
        run = [(7, W7, 100.0), (8, W8, 102.0), (9, W9, 130.0), (10, W10, 132.0)]
        self.assertEqual(self._feed("GAL_E5BI_INAV", run), [31, 32, 33])
        # word 9 a run late: satellites 1 and 3 only
        late = [(7, W7, 100.0), (8, W8, 102.0), (9, W9, 190.0), (10, W10, 192.0)]
        self.assertEqual(self._feed("GAL_E5BI_INAV", late), [31, 33])


class TestPropagator(unittest.TestCase):
    def test_matches_sat_pos_clk_on_a_kepler_set(self):
        """With no harmonic terms, dn or idot, sat_pos_clk is the almanac algorithm: the two
        agree, weeks from t_oa with wrap=False."""
        eph = na.bcnav2_type40(F40)[4]
        eph = dict(eph, toe=581632.0, toc=581632.0, af2=0.0, dn=0.0)
        toe_gpst = na.toa_gpst("C", 1089, 581632, 0.0)
        rec = {
            "sys": "C",
            "sqrta": eph["sqrtA"],
            "dn": 0.0,
            "m0": eph["M0"],
            "ecc": eph["e"],
            "omega": eph["omega"],
            "cus": 0.0,
            "cuc": 0.0,
            "crs": 0.0,
            "crc": 0.0,
            "cis": 0.0,
            "cic": 0.0,
            "i0": eph["i0"],
            "idot": 0.0,
            "omega0": eph["OMEGA0"],
            "omegadot": eph["OMEGA_dot"],
            "toe_gpst": toe_gpst,
            "toe_sow": 581632.0,
            "toc_gpst": toe_gpst,
            "af0": 0.0,
            "af1": 0.0,
            "af2": 0.0,
        }
        for days in (0.0, 1.3, 20.0, -9.0):
            tk = days * 86400.0
            want = ge.sat_pos_clk(rec, toe_gpst + tk)[0]
            got = na.sv_position_alm(eph, 581632.0 + tk, wrap=abs(tk) <= 302400)
            self.assertLess(math.dist(want, got), 1e-3, "%+.1f days" % days)


if __name__ == "__main__":
    unittest.main(verbosity=2)
