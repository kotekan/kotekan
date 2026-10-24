"""The seeding ephemeris reloads on the sky's 900-s cadence; the DCB table stays slow.

A broadcast record is valid for 4 h from its toe, and the hourly records publish late (BeiDou's
by more than an hour), so on a slow reload an in-view satellite's newest record can expire
before the next reload, taking a whole constellation's seeds and transit-sky rows with it. The DCB
walk-back is an HTTPS request per missing day, so it must not follow the ephemeris onto the
fast cadence.
"""
import ast
import os
import sys
import types
import unittest

from gnss_broker import deadreckon as dr

HERE = os.path.dirname(os.path.abspath(__file__))


class _Args(object):
    dcb_bias = 1
    dcb_max_age_days = 0.0
    dr_constellation = "C"
    chip_rate_hz = 10.23e6


class _EphMod(object):
    def __init__(self):
        self.fetches = 0

    def fetch_brdc(self, block=False):
        self.fetches += 1
        return "/nonexistent/brdc.rnx"

    def parse_rinex_nav(self, path):
        return {("C", 19): [{"toe_gpst": 0.0}]}

    def gpst_of_utc(self, t):
        return float(t)


def _ctx():
    c = types.SimpleNamespace()
    c.args = _Args()
    c.dr_eph_mod = _EphMod()
    c.dr_state = {"eph": None, "eph_t": 0.0}
    c.drp = types.SimpleNamespace(now_w=10000.0, t_code=0.001)
    c.lc_seg, c.code_len = 1, 10230
    c.utc0_sample0 = 1.0
    return c


class TestReloadCadence(unittest.TestCase):
    def setUp(self):
        self.calls = []
        fake = types.ModuleType("gnss_dcb")
        fake.fetch_dcb = lambda status=None: (self.calls.append("fetch"), None)[1]
        fake.parse_dcb = lambda p: None
        self._saved = sys.modules.get("gnss_dcb")
        sys.modules["gnss_dcb"] = fake
        self._log = dr._log
        dr._log = lambda *a, **k: None

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("gnss_dcb", None)
        else:
            sys.modules["gnss_dcb"] = self._saved
        dr._log = self._log

    def test_cadences(self):
        self.assertEqual(dr._DR_EPH_REFRESH_S, 900.0)
        self.assertGreaterEqual(dr._DR_DCB_REFRESH_S, 7200.0)

    def test_reload_without_dcb_never_touches_the_dcb_server(self):
        res = dr._dr_reload(_ctx(), first=False, with_dcb=False)
        self.assertEqual(self.calls, [])
        self.assertFalse(res["has_dcb"])
        self.assertIsNotNone(res["eph"])

    def test_first_load_fetches_the_dcb(self):
        res = dr._dr_reload(_ctx(), first=True)
        self.assertEqual(self.calls, ["fetch"])
        self.assertTrue(res["has_dcb"])

    def test_apply_keeps_the_dcb_across_an_ephemeris_only_reload(self):
        c = _ctx()
        dr._dr_apply_reload(c, dr._dr_reload(c, first=True))
        c.dr_state["dcb"] = {"marker": 1}
        t_dcb = c.dr_state["dcb_t"]
        c.drp.now_w += dr._DR_EPH_REFRESH_S + 1.0
        dr._dr_apply_reload(c, dr._dr_reload(c, first=False, with_dcb=False))
        self.assertEqual(c.dr_state["dcb"], {"marker": 1})
        self.assertEqual(c.dr_state["dcb_t"], t_dcb)
        self.assertEqual(c.dr_state["eph_t"], c.drp.now_w)

    def _dcb_lines(self, table, st):
        """One first load with gnss_dcb serving `table` and the status `st`: (res, the DCB:
        lines)."""

        def fetch(status=None):
            status.update(st)
            return "/nonexistent/dcb.bsx"

        sys.modules["gnss_dcb"].fetch_dcb = fetch
        sys.modules["gnss_dcb"].parse_dcb = lambda p: table
        logs = []
        dr._log = logs.append
        c = _ctx()
        c.args = _Args()
        c.args.dcb_max_age_days = 10.0
        res = dr._dr_reload(c, first=True)
        dr._dr_apply_reload(c, res)
        self.assertIs(c.dr_state["dcb"], res["dcb"])
        return res, [m for m in logs if m.startswith("DCB:")]

    def test_stale_dcb_stays_in_use_and_says_so(self):
        """A product past dcb_max_age_days is warn-only: its table is installed, so the warning
        must not announce the broadcast fall-back."""
        table = {("C", 19): 1.5e-9}
        res, lines = self._dcb_lines(table, {"age_days": 12.0})
        self.assertIs(res["dcb"], table)
        self.assertEqual(len(lines), 1)
        self.assertIn("12.0 days old (bar 10.0)", lines[0])
        self.assertNotIn("Falling back", lines[0])
        self.assertIn("stay in use", lines[0])

    def test_absent_dcb_falls_back(self):
        res, lines = self._dcb_lines(None, {"reason": "no-token"})
        self.assertIsNone(res["dcb"])
        self.assertEqual(len(lines), 1)
        self.assertIn("Falling back to the broadcast group delay", lines[0])

    def test_failed_reload_retries_in_ten_minutes(self):
        c = _ctx()
        dr._dr_apply_reload(
            c,
            {
                "eph": None,
                "dcb": None,
                "has_dcb": False,
                "error": "no network",
                "t0": 0.0,
            },
        )
        due_in = dr._DR_EPH_REFRESH_S - (c.drp.now_w - c.dr_state["eph_t"])
        self.assertAlmostEqual(due_in, 600.0)

    def test_the_gate_reads_the_constants(self):
        """The reload gate and the retry arithmetic use the named cadence, never a literal."""
        with open(os.path.join(HERE, "deadreckon.py")) as f:
            src = f.read()
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "stage_dead_reckon"
        )
        body = ast.unparse(fn)
        self.assertIn(
            'ctx.dr_state["eph_t"] > _DR_EPH_REFRESH_S'.replace('"', "'"), body
        )
        self.assertNotIn("> 7200", body)
        self.assertIn("_DR_DCB_REFRESH_S", body)


class TestT0mExact(unittest.TestCase):
    """The sample-0 GPST reduced mod the code period must be exact to the picosecond for every
    period: the double `% t_code` on a 1.48e9-s epoch was 82 ns short on the 100-ms codes and
    31 ns on the 20-ms ones, half a 10.23-Mcps chip between the two families (buglist #191)."""

    GPST0 = 1476826616.000004053  # the 2026-10-23 frame 0 (utc0 1792791398.000003815)

    def test_exact_for_every_code_period(self):
        from fractions import Fraction

        for lc_seg, code_len, rate in (
            (20, 10230, 10.23e6),  # gps_l5, bds_b3i
            (100, 10230, 10.23e6),  # gal_e5a, gal_e5b, bds_b2a
            (100, 5115, 5.115e6),  # gal_e6
            (10, 10230, 10.23e6),  # glo_l3oc
            (1, 10230, 10.23e6),  # bds_b2b
            (2, 10230, 10.23e6),  # sbas_l5
            (1, 10230, 0.5115e6),  # gps_l2c
        ):
            want = float(Fraction(self.GPST0) % Fraction(lc_seg * code_len, int(rate)))
            got = dr.t_mod_code(self.GPST0, lc_seg, code_len, rate)
            self.assertLess(abs(got - want), 1e-12, (lc_seg, code_len, rate))
        # The defect, pinned so the double reduction cannot come back: 0.1 as a double.
        naive = self.GPST0 % ((100 * 10230) / 10.23e6)
        exact = dr.t_mod_code(self.GPST0, 100, 10230, 10.23e6)
        self.assertGreater(abs(naive - exact), 50e-9)

    def test_reload_sets_t0m_exactly(self):
        c = _ctx()
        c.lc_seg, c.code_len = 100, 10230
        c.utc0_sample0 = self.GPST0  # _EphMod.gpst_of_utc is the identity
        dr._dr_apply_reload(
            c, {"eph": {}, "dcb": None, "has_dcb": False, "error": None, "t0": 0.0}
        )
        self.assertLess(
            abs(c.dr_state["t0m"] - dr.t_mod_code(self.GPST0, 100, 10230, 10.23e6)),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
