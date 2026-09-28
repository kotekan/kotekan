"""#151: the seeding ephemeris reloads on the sky's 900-s cadence; the DCB table stays slow.

The defect this pins: the dead-reckon copy of the broadcast ephemeris reloaded every 7200 s
while a record is valid for 4 h from its toe and the late-publishing hourly records (BeiDou's
by more than an hour) left in-view satellites with nothing newer than the reload's own
horizon -- so every 2 h a whole constellation's seeds and transit-sky rows expired together
until the next reload. The DCB walk-back is a live HTTPS request per missing day, so it must
not follow the ephemeris onto the fast cadence.
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
    dcb_require = 0
    dcb_max_age_days = 0.0
    dr_constellation = "C"


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

    def test_failed_reload_retries_in_ten_minutes(self):
        c = _ctx()
        dr._dr_apply_reload(c, {"eph": None, "dcb": None, "has_dcb": False,
                                "error": "no network", "fatal": None, "t0": 0.0})
        due_in = dr._DR_EPH_REFRESH_S - (c.drp.now_w - c.dr_state["eph_t"])
        self.assertAlmostEqual(due_in, 600.0)

    def test_the_gate_reads_the_constants(self):
        """The reload gate and the retry arithmetic use the named cadence, never a literal."""
        with open(os.path.join(HERE, "deadreckon.py")) as f:
            src = f.read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "stage_dead_reckon")
        body = ast.unparse(fn)
        self.assertIn('ctx.dr_state["eph_t"] > _DR_EPH_REFRESH_S'.replace('"', "'"), body)
        self.assertNotIn("> 7200", body)
        self.assertIn("_DR_DCB_REFRESH_S", body)


if __name__ == "__main__":
    unittest.main()
