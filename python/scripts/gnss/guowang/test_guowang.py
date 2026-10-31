#!/usr/bin/env python3
"""Checks for the GuoWang tools (#218).

- ts_unix: unix seconds -> skyfield time. ts.utc(1970, 1, 1, 0, 0, t) takes the leap-second offset
  of 1970 and lands 27 s early in 2026 (~8 deg along a DIGUI track); pinned against
  ts.from_datetime on a 10-26 burst time and across the 2016-12-31 leap second.
- The code families against the repo's code checks: B3I PRN 1-63 built from the G1 / G2-phase
  structure equal b1i_b3i_code_check.b3i() and its fingerprints; b1i_all holds the 63 B1I PRNs;
  L3OC PRN 1 data and pilot equal l3oc_code_check's.
- Berlekamp-Massey finds degree 26 (G1 x G2) on a B3I window, either polarity.
- A 0.2 s synthetic B3I signal with NH20 + data: the blind carrier, the fold (chip 0 at the code
  epoch, no chip errors) and the acquisition find PRN 23 at its code phase.

    /home/kvand/gnss/venv/bin/python test_guowang.py
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guowang_codes as gc  # noqa: E402
import guowang_passes as gp  # noqa: E402
import guowang_search as gs  # noqa: E402
import guowang_synth as gsy  # noqa: E402


class TimeTest(unittest.TestCase):
    def test_ts_unix(self):
        from skyfield.api import load

        ts = load.timescale()
        u = np.array(
            [1792973667.25, 1483228799.0, 1483228800.0]
        )  # 2026-10-26 00:14:27.25; 2016-12-31/2017-01-01
        got = gp.ts_unix(ts, u)
        for k, v in enumerate(u):
            want = ts.from_datetime(dt.datetime.fromtimestamp(v, dt.timezone.utc))
            self.assertLess(abs(got[k].tt - want.tt) * 86400.0, 1e-4)
        self.assertEqual(
            got[0].utc_strftime("%Y-%m-%d %H:%M:%S"), "2026-10-26 00:14:27"
        )
        # the trap this replaces
        bad = ts.utc(1970, 1, 1, 0, 0, u[0])
        self.assertAlmostEqual((got[0].tt - bad.tt) * 86400.0, 27.0, places=3)


class CodeTest(unittest.TestCase):
    def test_b3i(self):
        c = gc.b3i_prn()
        ref = np.array([gc._b13.b3i(p) for p in range(1, 64)], np.int8)
        self.assertTrue((c == ref).all())
        for prn, want in gc._b13._FP_B3I.items():
            self.assertEqual(gc._b13._fp(list(c[prn - 1])), want)

    def test_b1i_family_holds_prns(self):
        codes, labels = gc.b1i_all()
        self.assertEqual(codes.shape, (2048, 2046))
        self.assertEqual(sum(l.startswith("PRN") for l in labels), 63)

    def test_l3oc(self):
        codes, labels = gc.l3oc_all()
        dc1 = gc._l3._dc1()
        self.assertTrue((codes[0] == np.array(gc._l3._code(1, False, dc1))).all())
        self.assertTrue((codes[64] == np.array(gc._l3._code(1, True, dc1))).all())

    def test_berlekamp_massey(self):
        bits = (gc.b3i_prn([7])[0] < 0).astype(np.uint8)
        for b in (bits, bits ^ 1):
            r = gc.lfsr_report(b, windows=(100,), length=400)[0]
            self.assertEqual(r["L"], 26)
            self.assertEqual(sorted(r["divisors"]), ["B3I G1", "B3I G2"])

    def test_berlekamp_massey_unknown_lfsr(self):
        """A Gold-like code from two other degree-13 m-sequences: L = 26, no known divisor, and the
        connection polynomial is the product of the two generators'."""
        t1, t2 = (0, 1, 3, 4), (0, 1, 4, 6)
        m1, _ = gc.mseq_and_states(t1, 13, [1] * 13)
        m2, _ = gc.mseq_and_states(t2, 13, [1, 0] * 6 + [1])
        r = gc.lfsr_report(m1[:600] ^ m2[:600], windows=(0,), length=600)[0]
        self.assertEqual(r["L"], 26)
        self.assertEqual(r["divisors"], [])
        c1, c2 = gc.taps_to_connection(t1, 13), gc.taps_to_connection(t2, 13)
        prod = 0
        for i in range(c1.bit_length()):
            if c1 >> i & 1:
                prod ^= c2 << i
        self.assertEqual(r["poly"], prod)


class SearchTest(unittest.TestCase):
    def test_fold_epoch_anywhere(self):
        """The code epoch at different places in the fold, including the last sub-block (short
        folds, as period_scan runs them)."""
        fs, (f_rf, rc) = 25e6, gs.LOBES["b3"]
        code = gc.b3i_prn([7])[0]
        for chip0 in (100.5, 4000.25, 9000.75, 10200.5):
            x = gsy.synth(
                fs,
                0.06,
                code,
                rc,
                f_rf,
                3.0,
                dopp=-4e3,
                rate=-120.0,
                chip0=chip0,
                seed=3,
            )
            y, fsy = gs.extract_lobe(x, fs, f_rf, f_rf, rc, 2)
            f_c = -4e3 - 120.0 * 0.03
            per, n_chips, _ = gs.fold_period(1e-3, dict(detected=False), f_c, f_rf, rc)
            fo = gs.blind_fold(
                y, fsy, per, f_c, -120.0, 0.03, 0.0, f_rf, n_chips, t_use=0.05
            )
            r = np.dot(np.sign(fo["soft"]), code) / n_chips
            self.assertGreater(
                abs(r), 0.999, "chip0 %.2f: epoch-aligned agreement %.4f" % (chip0, r)
            )

    def test_synthetic_b3i(self):
        fs, (f_rf, rc) = 25e6, gs.LOBES["b3"]
        code = gc.b3i_prn([23])[0]
        x = gsy.synth(
            fs, 0.2, code, rc, f_rf, 3.0, dopp=12e3, rate=-80.0, chip0=1234.5, seed=2
        )
        y, fsy = gs.extract_lobe(x, fs, f_rf, f_rf, rc, 2)
        bc = gs.blind_carrier(y, fsy, 2, -30e3, 30e3, -150, 150, t_seg=0.05, t_use=0.2)
        car = bc["lines"][0]
        self.assertTrue(car["detected"])
        self.assertLess(abs(car["f_c"] - (12e3 - 80.0 * car["t_c"])), 2.0)
        self.assertLess(abs(car["rate"] + 80.0), 5.0)
        per, n_chips, _ = gs.fold_period(
            1e-3, dict(detected=False), car["f_c"], f_rf, rc
        )
        fo = gs.blind_fold(
            y,
            fsy,
            per,
            car["f_c"],
            car["rate"],
            car["t_c"],
            car["phase"],
            f_rf,
            n_chips,
            t_use=0.2,
        )
        err, n = gsy._truth_chips(code, fo["soft"])
        self.assertEqual(err, 0)
        # chip 0 at the code epoch (the NH20 / data sign changes): no rotation against the code
        r = np.dot(np.sign(fo["soft"]), code) / n
        self.assertAlmostEqual(abs(r), 1.0)
        fd0 = car["f_c"] - car["rate"] * car["t_c"]
        yw = gs.wipe(y, fsy, car["f_c"], car["rate"], car["t_c"])
        codes = gc.b3i_prn()
        res = gs.pcps(
            yw,
            fsy,
            codes,
            rc,
            f_rf,
            [0.0],
            [0.0],
            50,
            labels=["PRN%d" % p for p in range(1, 64)],
            drift=(fd0, car["rate"]),
        )
        best = res["results"][0]
        self.assertEqual(best["label"], "PRN23")
        self.assertTrue(best["detected"])
        self.assertTrue(res["identified"])
        self.assertLess(abs(best["chip0"] - 1234.5), 0.3)


if __name__ == "__main__":
    unittest.main()
