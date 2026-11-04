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
- The 1202 library merge (guowang_db.update): a close miss makes a no_transmission entry with no
  code, a far miss and a no_sky window count for nothing, a later find turns it transmitting.

    /home/kvand/gnss/venv/bin/python test_guowang.py
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guowang_codes as gc  # noqa: E402
import guowang_db as gdb  # noqa: E402
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

    def test_guowang_1202(self):
        """The codes read off DIGUI-26 (dump 2026-10-30) and DIGUI-79 (beam dump 2026-11-02),
        packed bits (1 = chip -1); every G2' phase gives a distinct code and the family labels
        the known satellites."""
        want = {
            1213: "7c69ad3164f0c5f7e073030bd30ac1c3cac03395bccd8632a0787a16cca62e8a1ea8f4016d6a429f1eb1b2a16aeb7ca576b03305ef3fc9d0869bb8fe4f46eded1ba6b148894cce558333b0c73e868aa0cc2b004a36f31cbc251a2223192b369ead19916c55daa4a10a50d681bf27437beb4b5e11c2ba40bd2119f8fb7679256d3c5775342f7ed91af2516b712c5c3d559363b1a86974a0000bbcef916af0a4e95cf40308bb9d08a3a2bf6de97a96b008fc47ddaa3c82716108ef22ac70804fa1b8f95936595303201660f2a72d5b27cf4bf339dac26d5568062290f39ba72cb5ce4a834269849ad33173940ea2fad9e181da0961b1968134acfd1e4d5db08788",
            2027: "30746637c4100cc9ef44a13aa55326e5ad8ffe9420209355d5f157aae61c4dc0045fff7dd4d9c21fa44f60535f71e20444586c0ce5e5d95a41a924291e0db8a3991ea46130b182a1588593394ba87bd804c72e1c92ff12c43ea290d664342c276a11499724565eebb77b191f50d9df45aab08841fc12abdbf239ec00eee56fa49fc0aa548fea987cbce9974825226f1d582bdd3cc1025e3be939f61177d2b2f93f23811772705308a35708a45c6c672d3f4ad2a7da9f27ff40cd7a706736854f7d54b6f28141621c6f34f39c1c8aa2ec8fc23b0b7d0abd79c35c2babe06c5e9d76c2321356598c100638b5c87803439630586220048ca56b4ce774dec9cba5e8",
        }
        for ph, hexbits in want.items():
            c = gc.guowang_1202(ph)
            self.assertEqual(len(c), 2046)
            self.assertEqual(np.packbits(c < 0).tobytes().hex(), hexbits)
        codes, labels = gc.guowang_1202_all()
        self.assertEqual(codes.shape, (2047, 2046))
        self.assertEqual(len({c.tobytes() for c in codes}), 2047)
        self.assertEqual(labels[2027], "DIGUI-79/g2ph2027")
        self.assertTrue((codes[2027] == gc.guowang_1202(2027)).all())
        m2 = gc.guowang_g2prime(2047 + 11)
        self.assertTrue(
            (m2[2047:] == m2[:11]).all()
        )  # period 2047: a maximal-length sequence

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


class DbTest(unittest.TestCase):
    @staticmethod
    def window(start, end, sats):
        return dict(
            complete=True,
            lobe_channels=64,
            utc_start=start,
            utc_end=end,
            satellites=sats,
        )

    def test_update(self):
        import json
        import tempfile

        db = {
            "about": "",
            "satellites": {
                "DIGUI-1": dict(
                    catnum=1,
                    g2_phase=10,
                    theta_us=-1.0,
                    measured_utc="2026-11-02T10:00:40Z",
                    sep_deg=1.0,
                    ref_input=116,
                )
            },
            "no_sky": [dict(start="2026-11-02T12:00:00Z", end="2026-11-02T13:00:00Z")],
        }
        logs = [
            self.window(
                "2026-11-02T11:00:00.000Z",
                "11:01:20.000",
                [
                    dict(sat="DIGUI-1", detected=True, sep_min_window=2.0),
                    dict(sat="DIGUI-2", detected=False, sep_min_window=1.5),
                    dict(sat="DIGUI-3", detected=False, sep_min_window=6.0),
                ],
            ),
            self.window(
                "2026-11-02T12:30:00.000Z",
                "12:31:20.000",
                [dict(sat="DIGUI-1", detected=False, sep_min_window=0.5)],
            ),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in logs))
            fh.flush()
            passes, skipped = gdb.read_passes(db, [fh.name], 4.0)
        self.assertEqual(skipped, {"incomplete": 0, "no_sky": 1, "far_miss": 1})
        gdb.update(db, passes, {}, {"DIGUI-2": 2})
        s = db["satellites"]
        self.assertNotIn("DIGUI-3", s)
        self.assertEqual(s["DIGUI-1"]["status"], "transmitting")
        self.assertEqual(s["DIGUI-1"]["passes_found"], 2)
        self.assertEqual(s["DIGUI-1"]["last_observed_utc"], "2026-11-02T11:00:40Z")
        self.assertEqual(s["DIGUI-2"]["status"], "no_transmission")
        self.assertIsNone(s["DIGUI-2"]["g2_phase"])
        self.assertIsNone(s["DIGUI-2"]["last_observed_utc"])
        self.assertEqual(s["DIGUI-2"]["catnum"], 2)
        self.assertEqual(sorted(gdb.coded(db)), ["DIGUI-1"])

        later = {
            "DIGUI-2": {"2026-11-02T14:00:40Z": ["2026-11-02T14:00:40Z", 3.0, True]}
        }
        prop = dict(
            sat="DIGUI-2", g2_phase=20, theta_us=2.0, sep_deg=3.0, ref_input=116
        )
        prop["measured_utc"] = "2026-11-02T14:00:38Z"
        changes = gdb.update(db, later, {"DIGUI-2": [prop]}, {})
        e = s["DIGUI-2"]
        self.assertEqual((e["status"], e["g2_phase"]), ("transmitting", 20))
        self.assertEqual((e["passes_found"], e["passes_silent"]), (1, 1))
        self.assertEqual(e["last_observed_utc"], "2026-11-02T14:00:40Z")
        self.assertTrue(any("no_transmission -> transmitting" in c for c in changes))
        self.assertEqual(json.loads(gdb.dumps(db)), db)

    def test_no_sky_set_after_the_fact(self):
        db = {
            "about": "",
            "satellites": {
                "DIGUI-1": dict(
                    catnum=1,
                    g2_phase=10,
                    measured_utc="2026-11-02T10:00:40Z",
                    sep_deg=1.0,
                    passes=[
                        ["2026-11-02T10:00:40Z", 1.0, True],
                        ["2026-11-02T18:00:40Z", 2.0, False],
                    ],
                    status="no_transmission",
                ),
            },
            "no_sky": [],
        }
        gdb.update(db, {}, {}, {})
        self.assertEqual(db["satellites"]["DIGUI-1"]["status"], "no_transmission")
        # the 18:00 miss fell in an outage noticed later: it no longer counts
        db["no_sky"] = [dict(start="2026-11-02T16:00:00Z", end=None)]
        gdb.update(db, {}, {}, {})
        e = db["satellites"]["DIGUI-1"]
        self.assertEqual(e["status"], "transmitting")
        self.assertEqual((e["passes_found"], e["passes_silent"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
