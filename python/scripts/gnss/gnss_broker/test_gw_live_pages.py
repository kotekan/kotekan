"""The GuoWang data-sign wipe takes its message pages from the track-beam loop's live file.

    python3 -m gnss_broker.test_gw_live_pages      # from python/scripts/gnss

gw_live_pages re-reads nav_pages_live.json when it changes and keeps the last good read through
a malformed one; gw_pages_for takes the live pages of the satellite the covering table record
names while they are younger than the cap, then that record's own pages, never another
record's; dr_nav_pages attaches, replaces and drops a seed's pages from cycle to cycle while
the frame index stays; gw_nav_fields ships the frame index before any page exists.
"""
import json
import os
import tempfile
import types
import unittest

from gnss_broker import deadreckon as dr
from gnss_broker.seed import Seed

NOW = 1793930000.0  # 2026-11-06 01:13:20Z
G0 = 1000.0  # GPST of sample 0; ref_hop 100 at 1 hop/s puts the seed at g = 1100
LIVE = ["l1", "l2", "l3", "l4"]
LIVE2 = ["m1", "m2", "m3", "m4"]
TABLE = ["t1", "t2", "t3", "t4"]
MATE = ["x1", "x2", "x3", "x4"]


def _sow(t_unix):
    return (t_unix + dr.BDT_MINUS_UTC_S - dr.BDT_EPOCH_UNIX) % 604800


def _entry(pages, t_last):
    """A live entry whose four pages went out at t_last - 9, -6, -3 and 0 s."""
    return dict(
        nav_pages=pages,
        sow={str(p): _sow(t_last - 3 * (4 - p)) for p in (1, 2, 3, 4)},
        utc="2026-11-06T01:13:20Z",
    )


def _write(path, doc, mtime):
    with open(path, "w") as fh:
        json.dump(doc, fh)
    os.utime(path, (mtime, mtime))


def _rec(name, g_lo, g_hi, pages=None):
    r = dict(name=name, t0_gpst=g_lo, t1_gpst=g_hi)
    if pages:
        r["nav_pages"] = pages
    return r


class TestLivePages(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self._saved = (dr._log, dr._log_rl, dr._now)
        dr._log = self.logs.append
        dr._log_rl = lambda key, msg, every_s=10.0: self.logs.append(msg)
        dr._now = lambda: NOW
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "nav_pages_live.json")

    def tearDown(self):
        dr._log, dr._log_rl, dr._now = self._saved
        self.dir.cleanup()

    def _ctx(self, eph, seeds=None, max_age=None):
        c = types.SimpleNamespace()
        c.args = types.SimpleNamespace(
            dr_constellation="W",
            gw_table=os.path.join(self.dir.name, "gw_ephem.json"),
            gw_live_pages="",
            hops_per_sec=1.0,
            chip_rate_hz=2.046e6,
        )
        if max_age is not None:
            c.args.gw_live_pages_max_age_s = max_age
        c.dr_state = {"eph": eph}
        c.dr_eph_mod = types.SimpleNamespace(gpst_of_utc=float)
        c.utc0_sample0 = G0
        c.drp = types.SimpleNamespace(t_eph_age=100.0)
        c.seeds = seeds if seeds is not None else {}
        return c

    def test_page_transmit_time_from_its_second_of_week(self):
        self.assertAlmostEqual(dr.gw_page_tx_unix(_sow(NOW - 30.0), NOW), NOW - 30.0)
        # decoded across the week rollover: the sow is last week's
        self.assertAlmostEqual(
            dr.gw_page_tx_unix(_sow(NOW + 600.0), NOW), NOW + 600.0 - 604800
        )

    def test_store_reloads_on_change_and_keeps_the_last_good_read(self):
        c = self._ctx({})
        self.assertEqual(dr.gw_live_pages(c, NOW), {})  # no file yet
        _write(self.path, {"DIGUI-11": _entry(LIVE, NOW - 20.0)}, 1000.0)
        got = dr.gw_live_pages(c, NOW)
        self.assertEqual(got["DIGUI-11"][0], LIVE)
        self.assertAlmostEqual(got["DIGUI-11"][1], NOW - 20.0)
        self.assertIs(dr.gw_live_pages(c, NOW), got)  # unchanged file: no re-read
        with open(self.path, "w") as fh:
            fh.write("{not json")
        os.utime(self.path, (2000.0, 2000.0))
        self.assertIs(dr.gw_live_pages(c, NOW), got)
        self.assertIn("failed to load", self.logs[-1])
        _write(
            self.path,
            {"DIGUI-12": _entry(LIVE2, NOW - 5.0), "DIGUI-13": dict(nav_pages=["a"])},
            3000.0,
        )
        # three pages are not a template
        self.assertEqual(sorted(dr.gw_live_pages(c, NOW)), ["DIGUI-12"])

    def test_pages_follow_the_covering_record_name_never_its_code_mate(self):
        g = G0 + 100.0
        eph = {
            ("W", 11): [
                _rec("DIGUI-11", g - 100, g + 100, TABLE),
                _rec(
                    "DIGUI-99", g + 1000, g + 2000, MATE
                ),  # the code-mate's later pass
            ]
        }
        c = self._ctx(eph)
        _write(
            self.path,
            {"DIGUI-11": _entry(LIVE, NOW - 20.0), "DIGUI-99": _entry(MATE, NOW - 1.0)},
            1000.0,
        )
        pages, age = dr.gw_pages_for(c, 11, g, NOW)
        self.assertEqual(pages, LIVE)
        self.assertAlmostEqual(age, 20.0)
        # stale live pages: the record's own table pages, whatever the code-mate's age
        _write(
            self.path,
            {
                "DIGUI-11": _entry(LIVE, NOW - 400.0),
                "DIGUI-99": _entry(MATE, NOW - 1.0),
            },
            2000.0,
        )
        self.assertEqual(dr.gw_pages_for(c, 11, g, NOW), (TABLE, None))
        # and nothing from a record without them
        eph[("W", 11)][0].pop("nav_pages")
        self.assertIsNone(dr.gw_pages_for(c, 11, g, NOW))
        # outside every record: nothing, although the code-mate's record carries pages
        self.assertIsNone(dr.gw_pages_for(c, 11, g + 500.0, NOW))
        # the cap is an option
        _write(self.path, {"DIGUI-11": _entry(LIVE, NOW - 400.0)}, 3000.0)
        self.assertEqual(
            dr.gw_pages_for(self._ctx(eph, max_age=600.0), 11, g, NOW)[0], LIVE
        )

    def test_seed_pages_attach_refresh_and_drop_across_cycles(self):
        g = G0 + 100.0
        eph = {("W", 11): [_rec("DIGUI-11", g - 100, g + 100)]}
        seed = Seed.born(
            "dr_birth", epoch=100, ref_hop=100, doppler_hz=1.0, nav_frame_at_ref=7
        )
        other = Seed.born(
            "dr_birth", epoch=100, ref_hop=100, doppler_hz=1.0
        )  # no frame
        c = self._ctx(eph, seeds={11: seed, 12: other})
        dr.dr_nav_pages(c)
        self.assertNotIn("nav_pages", seed)
        _write(self.path, {"DIGUI-11": _entry(LIVE, NOW - 20.0)}, 1000.0)
        dr.dr_nav_pages(c)
        self.assertEqual(seed["nav_pages"], LIVE)
        self.assertEqual(seed["nav_frame_at_ref"], 7)
        self.assertEqual(seed.prov["nav_pages"], ("gw_nav", 100))
        self.assertIn("attached", self.logs[-1])
        _write(self.path, {"DIGUI-11": _entry(LIVE2, NOW - 8.0)}, 2000.0)
        dr.dr_nav_pages(c)
        self.assertEqual(seed["nav_pages"], LIVE2)
        self.assertIn("refreshed (live, 8 s old)", self.logs[-1])
        _write(self.path, {"DIGUI-11": _entry(LIVE2, NOW - 400.0)}, 3000.0)
        dr.dr_nav_pages(c)
        self.assertNotIn("nav_pages", seed)
        self.assertEqual(seed["nav_frame_at_ref"], 7)
        self.assertNotIn("nav_pages", other)
        self.assertIn("dropped", self.logs[-1])
        c.args.dr_constellation = "G"
        _write(self.path, {"DIGUI-11": _entry(LIVE, NOW - 20.0)}, 4000.0)
        dr.dr_nav_pages(c)
        self.assertNotIn("nav_pages", seed)  # not a GuoWang chain

    def test_nav_fields_ship_the_frame_index_before_any_page(self):
        g = G0 + 100.0
        eph = {("W", 11): [_rec("DIGUI-11", g - 100, g + 100)]}
        c = self._ctx(eph)
        v = dict(range_m=1.2e6, range_rate_mps=0.0, sat_clk_s=0.0)
        f = dr.gw_nav_fields(c, v, 11, 100, 0.0)
        self.assertEqual(sorted(f), ["nav_frame_at_ref", "nav_phase_at_ref"])
        _write(self.path, {"DIGUI-11": _entry(LIVE, NOW - 20.0)}, 1000.0)
        f2 = dr.gw_nav_fields(c, v, 11, 100, 0.0)
        self.assertEqual(f2["nav_pages"], LIVE)
        self.assertEqual(f2["nav_frame_at_ref"], f["nav_frame_at_ref"])
        c.args.dr_constellation = "G"
        self.assertIsNone(dr.gw_nav_fields(c, v, 11, 100, 0.0))


if __name__ == "__main__":
    unittest.main()
