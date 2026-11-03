"""GuoWang satellites in every chain's transit sky (--gw-transits).

    python3 -m gnss_broker.test_gw_transits      # from python/scripts/gnss

A synthetic table (gw_ephem_table.py's "transits"): one satellite held at the boresight. It joins
a BRDC chain's transit sky, so the dead-reckon clock freeze, the transit cone and the
assemblers' "_bore" post fire for it as they do for E14 on its 2026-10-31 pass (the in-repo
BRDC fixture). The store reloads when the file changes; a missing or unreadable file is no
GuoWang transits, and the sky is then the BRDC one exactly.
"""
import json
import math
import os
import tempfile
import types
import unittest
from datetime import datetime, timezone

import gnss_ephemeris as ge
from gnss_broker import almanac
from gnss_broker import deadreckon as dr
from gnss_broker import sky

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    *("..", "..", "..", "..", "scripts", "gnss", "fixtures"),
    "brdc_gal_e14_20261031.rnx",
)
LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
BORE = (180.0, 62.7)  # az, el
T_E14 = 1793407620.0  # 2026-10-31T00:47:00Z: E14 ~2 deg from boresight
T_GW = T_E14 + 7200.0  # E14 and E15 far from it; the GuoWang satellite at the boresight


def _transit(catnum, t0, t1, range_m=1.2e6):
    """A transit record held still at the boresight from unix t0 to t1."""
    la, lo = math.radians(LAT), math.radians(LON)
    az, el = (math.radians(x) for x in BORE)
    e = (-math.sin(lo), math.cos(lo), 0.0)
    n = (-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la))
    u = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    rx = ge._ecef_of_llh(LAT, LON, ALT)
    pos = [
        rx[i]
        + range_m
        * (
            math.cos(el) * math.sin(az) * e[i]
            + math.cos(el) * math.cos(az) * n[i]
            + math.sin(el) * u[i]
        )
        for i in range(3)
    ]
    g0 = ge.gpst_of_utc(t0)
    k = int((t1 - t0) / 30.0) + 1
    return dict(
        name="DIGUI-X",
        catnum=catnum,
        t0_gpst=g0,
        t1_gpst=g0 + (k - 1) * 30.0,
        dt=30.0,
        toe_gpst=g0,
        min_sep_deg=0.0,
        pos=[pos] * k,
        vel=[[0.0, 0.0, 0.0]] * k,
    )


def _write(path, transits, mtime, records=()):
    with open(path, "w") as fh:
        json.dump(dict(records=list(records), transits=transits), fh)
    os.utime(path, (mtime, mtime))


def _sky(eph, t, gw=None):
    return sky.transit_sky(
        ge, eph, LAT, LON, ALT, datetime.fromtimestamp(t, tz=timezone.utc), gw=gw
    )


class _Case(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self._log = (sky._log, sky._log_rl)
        sky._log = self.logs.append
        sky._log_rl = lambda key, msg, every_s=10.0: self.logs.append(msg)
        self._bore = (
            sky.BORESIGHT_AZ_DEG,
            sky.BORESIGHT_EL_DEG,
            sky.BORESIGHT_KNOWN,
            sky.BORESIGHT_SOURCE,
        )
        sky.set_boresight(BORE[0], BORE[1], "test")
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "gw_ephem.json")
        self.eph = ge.parse_rinex_nav([FIX])

    def tearDown(self):
        sky._log, sky._log_rl = self._log
        (
            sky.BORESIGHT_AZ_DEG,
            sky.BORESIGHT_EL_DEG,
            sky.BORESIGHT_KNOWN,
            sky.BORESIGHT_SOURCE,
        ) = self._bore
        self.dir.cleanup()


class TestStore(_Case):
    def test_a_missing_file_is_no_transits_until_it_lands(self):
        st = sky.gw_transit_store(self.path)
        self.assertEqual(sky.gw_transits(ge, st), {})
        self.assertIn("unreadable", self.logs[-1])
        _write(self.path, [_transit(99001, T_GW - 60.0, T_GW + 60.0)], 1000.0)
        self.assertEqual(list(sky.gw_transits(ge, st)), [("W", 99001)])

    def test_reload_on_change_and_a_bad_file_is_none(self):
        _write(self.path, [_transit(99001, T_GW - 60.0, T_GW + 60.0)], 1000.0)
        st = sky.gw_transit_store(self.path)
        eph = sky.gw_transits(ge, st)
        self.assertIs(sky.gw_transits(ge, st), eph)  # unchanged file: no reload
        _write(self.path, [_transit(99002, T_GW - 60.0, T_GW + 60.0)], 2000.0)
        self.assertEqual(list(sky.gw_transits(ge, st)), [("W", 99002)])
        with open(self.path, "w") as fh:
            fh.write("{not json")
        os.utime(self.path, (3000.0, 3000.0))
        self.assertEqual(sky.gw_transits(ge, st), {})
        self.assertIn("unreadable", self.logs[-1])
        os.remove(self.path)
        self.assertEqual(sky.gw_transits(ge, st), {})

    def test_transits_and_code_records_are_read_apart(self):
        rec = dict(_transit(99001, T_GW - 60.0, T_GW + 60.0), prn=7, clk_s=-1e-5)
        _write(self.path, [_transit(99001, T_GW - 60.0, T_GW + 60.0)], 1000.0, [rec])
        self.assertEqual(list(ge.load_gw_table(self.path)), [("W", 7)])
        self.assertEqual(
            list(ge.load_gw_table(self.path, transits=True)), [("W", 99001)]
        )


class TestTransitSky(_Case):
    def setUp(self):
        super().setUp()
        _write(self.path, [_transit(99001, T_GW - 120.0, T_GW + 120.0)], 1000.0)
        self.gw = sky.gw_transit_store(self.path)

    def test_a_guowang_satellite_joins_the_brdc_sky(self):
        pd = _sky(self.eph, T_GW, self.gw)
        self.assertIn(("E", 14), pd)
        self.assertEqual(pd[("W", 99001)]["health"], 0)
        near = sky.nearest_boresight(pd)
        self.assertEqual(near[1], ("W", 99001))
        self.assertLess(near[0], 0.01)
        # outside its segment it is gone, and the BRDC satellites are far
        pd = _sky(self.eph, T_GW + 600.0, self.gw)
        self.assertNotIn(("W", 99001), pd)
        self.assertGreater(sky.nearest_boresight(pd)[0], 10.0)

    def test_the_same_freezes_as_a_gnss_satellite(self):
        """E14 at 2 deg and the GuoWang satellite at the boresight each freeze a confirmed
        clock, fill the transit cone and set the "_bore" post; the BRDC sky alone at T_GW
        does none of it."""
        for t, gw, key in (
            (T_E14, None, ("E", 14)),
            (T_GW, self.gw, ("W", 99001)),
            (T_GW, None, None),
        ):
            pd = _sky(self.eph, t, gw)
            ctx = types.SimpleNamespace(
                args=types.SimpleNamespace(dr_clock_transit_freeze_deg=5.0),
                dr_state={"clk": 100.0, "clk_confirmed": True},
                drp=types.SimpleNamespace(transit_pd=pd, pd=None),
                transit_pd=pd,
            )
            for got in (dr.dr_clock_transit(ctx), dr._in_transit_cone(ctx)):
                self.assertEqual(got and got[1], key)
            bore = _posts([(1000.0, pd)])[0][1].get("_bore")
            if key is None:
                self.assertGreater(bore[0], 5.0)
            else:
                self.assertLess(bore[0], 2.5)

    def test_a_missing_file_leaves_the_brdc_sky_exactly(self):
        missing = sky.gw_transit_store(os.path.join(self.dir.name, "none.json"))
        for t in (T_E14, T_GW):
            want = _sky(self.eph, t)
            self.assertEqual(_sky(self.eph, t, missing), want)
            self.assertEqual(list(_sky(self.eph, t, missing)), list(want))


def _row(sys_, sep_deg):
    """A transit-sky row `sep_deg` north of the boresight, on its meridian."""
    return {(sys_, 7): {"az": BORE[0], "el": BORE[1] + sep_deg, "health": 0}}


class TestPostCadence(_Case):
    """The geometry post goes every 30 s, and every cycle (2 s) while a GuoWang satellite is
    within 10 deg of boresight, so the assemblers' freezes arm within a cycle of its entry."""

    CYCLES = range(1000, 1062, 2)

    def _times(self, sky_):
        """The cycles that post, with one transit sky throughout."""
        return [t for t, _ in _posts([(t, sky_) for t in self.CYCLES])]

    def test_gnss_satellites_keep_the_30_s_cadence(self):
        for sep in (0.5, 9.0, 40.0):
            self.assertEqual(self._times(_row("E", sep)), [1000, 1030, 1060])
        self.assertEqual(self._times({}), [1000, 1030, 1060])

    def test_a_guowang_satellite_near_boresight_posts_every_cycle(self):
        sky_ = {**_row("E", 30.0), **_row("W", 9.0)}
        self.assertEqual(self._times(sky_), list(self.CYCLES))
        self.assertEqual(self._times(_row("W", 11.0)), [1000, 1030, 1060])

    def test_a_guowang_pass_arms_the_freeze_within_a_cycle(self):
        """Across the boresight at 0.34 deg/s (1100 km): the first "_bore" inside the 6 deg
        freeze cone is posted within one cycle of the entry, where the 30 s cadence alone
        could post it up to 30 s late."""
        posts = _posts(
            [(t, _row("W", abs(15.0 - 0.34 * (t - 1000)))) for t in self.CYCLES]
        )
        entry = 1000 + 9.0 / 0.34
        armed = min(t for t, b in posts if b["_bore"][0] < 6.0)
        self.assertLessEqual(armed - entry, 2.0)
        self.assertEqual([t for t, b in posts if b["_bore"][0] > 10.0], [1000])


def _posts(cycles):
    """[(t0, body)]: what stage_almanac_predict posts to the assemblers over `cycles`, one
    (t0, transit sky) per cycle."""
    posts = []
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(
            almanac=True,
            doppler_sign=1,
            lat=LAT,
            lon=LON,
            alt=ALT,
            carrier_hz=1176.45e6,
            mask_deg=10.0,
            post_sat_geometry=1,
            bias_det_fresh_s=30.0,
            bias_min_snr=10.0,
            bias_min_sats=2,
        ),
        capable_t=None,
        capable=None,
        brdc_alm=object(),
        almanac_sats=[],
        alm_sys="G",
        alm_min_prn=1,
        alm_now=lambda: datetime.fromtimestamp(T_GW, tz=timezone.utc),
        dll_combiners=["http://cx19:12048/n2combine"],
        geom_post_t=[0.0],
        transit_pd=None,
        publisher=None,
        cb=types.SimpleNamespace(ema=None, value=0.0),
        best={},
        det_fresh={},
        t0=None,
        utc0_sample0=0.0,
    )
    row = (-1200.0, 0.4, 45.0, 2.1e7, 1e-5, 120.0, 0.01, -0.02, 1.7e9)
    saved = (almanac.brdc_predict, almanac._post, almanac._log_rl, almanac.timebase)
    almanac.brdc_predict = lambda *a: {5: row}
    almanac._post = lambda url, body, timeout=5.0: posts.append((ctx.t0, body))
    almanac._log_rl = lambda key, msg, every_s=10.0: None
    almanac.timebase = types.SimpleNamespace(observe=lambda c, log_rl: None)
    try:
        for t0, pd in cycles:
            ctx.t0, ctx.transit_pd = t0, pd
            almanac.stage_almanac_predict(ctx)
    finally:
        almanac.brdc_predict, almanac._post, almanac._log_rl, almanac.timebase = saved
    return posts


if __name__ == "__main__":
    unittest.main()
