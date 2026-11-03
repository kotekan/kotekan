"""A constellation W (GuoWang) chain takes its geometry from the TLE table, never from BRDC.

    python3 -m gnss_broker.test_gw_table      # from python/scripts/gnss

A synthetic table (load_gw_table's layout): PRNs 11-14 pass in the first 300 s, then nothing,
then a second satellite on PRN 11's code. The almanac store reloads when the file changes and
keeps the last table when a read fails, the empty sky between passes is not bridged as a
collapse, dead reckoning takes the almanac's table, and the model-owned seeds drop between
passes, when the dead-reckon stage does not run.
"""
import json
import math
import os
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

import gnss_ephemeris as ge
from gnss_broker import deadreckon as dr
from gnss_broker import sky

LAT, LON, ALT = 49.3199845, -119.62262691, 528.4
T0 = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)


def _rec(prn, name, t0, t1):
    """A satellite 1200 km above the site, held still from T0 + t0 to T0 + t1 (s)."""
    rx = ge._ecef_of_llh(LAT, LON, ALT)
    r = math.sqrt(sum(x * x for x in rx))
    pos = [x * (1.0 + 1.2e6 / r) for x in rx]
    g0 = ge.gpst_of_utc(T0) + t0
    n = int((t1 - t0) / 10.0) + 1
    return dict(
        prn=prn,
        name=name,
        t0_gpst=g0,
        t1_gpst=g0 + (n - 1) * 10.0,
        dt=10.0,
        toe_gpst=g0,
        clk_s=-1e-5,
        pos=[pos] * n,
        vel=[[0.0, 0.0, 0.0]] * n,
    )


TABLE = [_rec(p, "DIGUI-%d" % p, 0.0, 300.0) for p in (11, 12, 13, 14)] + [
    _rec(11, "DIGUI-99", 600.0, 900.0)
]


def _write(path, records, mtime):
    with open(path, "w") as fh:
        json.dump(dict(records=records), fh)
    os.utime(path, (mtime, mtime))


def _sky(store, t_s):
    return sky.brdc_predict(
        store, LAT, LON, ALT, "W", 1, T0 + timedelta(seconds=t_s), 1202.025e6
    )


class TestStore(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self._log = sky._log
        sky._log = self.logs.append
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "gw_ephem.json")

    def tearDown(self):
        sky._log = self._log
        self.dir.cleanup()

    def test_satellites_only_inside_their_segments(self):
        _write(self.path, TABLE, 1000.0)
        st = sky.gw_store(ge, self.path)
        self.assertEqual(sorted(st["eph"]), [("W", p) for p in (11, 12, 13, 14)])
        got = _sky(st, 100.0)
        self.assertEqual(sorted(got), [11, 12, 13, 14])
        self.assertGreater(got[11][2], 80.0)  # elevation
        self.assertEqual(got[11][4], -1e-5)  # the clock is clk_s
        # Between passes the sky is empty: four of four gone is not bridged as a collapse.
        self.assertEqual(_sky(st, 450.0), {})
        self.assertEqual(sorted(_sky(st, 700.0)), [11])

    def test_reload_on_change_and_keep_the_last_good_table(self):
        _write(self.path, TABLE, 1000.0)
        st = sky.gw_store(ge, self.path)
        eph = st["eph"]
        _sky(st, 100.0)
        self.assertIs(st["eph"], eph)  # unchanged file: no reload
        _write(self.path, [_rec(21, "DIGUI-21", 0.0, 300.0)], 2000.0)
        self.assertEqual(sorted(_sky(st, 100.0)), [21])
        eph = st["eph"]
        with open(self.path, "w") as fh:
            fh.write("{not json")
        os.utime(self.path, (3000.0, 3000.0))
        self.assertEqual(sorted(_sky(st, 100.0)), [21])
        self.assertIs(st["eph"], eph)
        self.assertIn("failed to load", self.logs[-1])
        os.remove(self.path)
        self.assertEqual(sorted(_sky(st, 100.0)), [21])

    def test_a_missing_table_predicts_nothing_until_it_lands(self):
        st = sky.gw_store(ge, self.path)
        self.assertEqual(st["eph"], {})
        self.assertEqual(_sky(st, 100.0), {})
        _write(self.path, TABLE, 1000.0)
        self.assertEqual(sorted(_sky(st, 100.0)), [11, 12, 13, 14])


def _ctx(store, pred):
    c = types.SimpleNamespace()
    c.args = types.SimpleNamespace(
        dr_constellation="W", almanac=False, dcb_bias=1, chip_rate_hz=2.046e6
    )
    c.brdc_alm = store
    c.dr_eph_mod = types.SimpleNamespace(gpst_of_utc=float)
    c.dr_state = {"eph": None, "eph_t": 0.0, "seeded": {11, 12}, "pin": {11: 0, 12: 0}}
    c.drp = types.SimpleNamespace(now_w=10000.0)
    c.lc_seg, c.code_len, c.utc0_sample0 = 1, 2046, 1.0
    c.pred = pred
    c.seeds = {11: {}, 12: {}, 20: {}}
    c.hold = types.SimpleNamespace(low_hits={11: 1, 20: 2})
    return c


class TestDeadReckon(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self._log = dr._log
        dr._log = self.logs.append

    def tearDown(self):
        dr._log = self._log

    def test_the_model_is_the_almanac_table(self):
        st = {"eph": {("W", 11): []}}
        c = _ctx(st, {})
        res = dr._dr_reload(
            c, first=True
        )  # no fetch_brdc on the stub: BRDC would raise
        self.assertIsNone(res["error"])
        self.assertIs(res["eph"], st["eph"])
        dr._dr_apply_reload(c, res)
        self.assertIs(c.dr_state["eph"], st["eph"])

    def test_model_owned_seeds_drop_between_passes(self):
        c = _ctx({"eph": {}}, {})
        dr.stage_dead_reckon(c)
        self.assertEqual(
            sorted(c.seeds), [20]
        )  # 20 is the search's, stage_coast_drop's
        self.assertEqual(c.dr_state["seeded"], set())
        self.assertEqual(c.dr_state["pin"], {})
        self.assertEqual(c.hold.low_hits, {20: 2})

    def test_seeds_stay_during_a_pass_and_on_brdc_chains(self):
        c = _ctx({"eph": {}}, {11: (0.0,) * 9})
        dr.stage_dead_reckon(c)
        self.assertEqual(sorted(c.seeds), [11, 12, 20])
        c = _ctx({"eph": {}}, {})
        c.args.dr_constellation = "C"
        dr.stage_dead_reckon(c)
        self.assertEqual(sorted(c.seeds), [11, 12, 20])
        self.assertEqual(c.dr_state["seeded"], {11, 12})


class RangeAccelerationTest(unittest.TestCase):
    def test_a_leo_range_is_carried_quadratically(self):
        """A W prediction carries range_acc_mps2, and range + rate dt + acc dt^2 / 2 lands within a
        few metres (0.03 chip) of the range 5 s on, where the linear carry is ~400 m (~3 chips) off
        (cp_predicted). The rest is the jerk."""
        rx = ge._ecef_of_llh(LAT, LON, ALT)
        r0 = math.sqrt(sum(x * x for x in rx))
        u = [x / r0 for x in rx]
        e = [-u[1], u[0], 0.0]
        n = math.sqrt(sum(x * x for x in e))
        e = [x / n for x in e]
        R = r0 + 1.2e6
        w = math.sqrt(3.986004418e14 / R ** 3)  # circular orbit through the zenith
        g0 = ge.gpst_of_utc(T0)
        pos, vel = [], []
        for k in range(61):
            a = w * (k * 10.0 - 300.0)
            pos.append(
                [R * (math.cos(a) * ui + math.sin(a) * ei) for ui, ei in zip(u, e)]
            )
            vel.append(
                [R * w * (-math.sin(a) * ui + math.cos(a) * ei) for ui, ei in zip(u, e)]
            )
        rec = dict(prn=5, t0_gpst=g0, t1_gpst=g0 + 600.0, dt=10.0, toe_gpst=g0 + 300.0)
        rec.update(clk_s=0.0, pos=pos, vel=vel, sys="W", health=0)
        eph = {("W", 5): [rec]}
        t = T0 + timedelta(seconds=250.0)
        v0 = ge.predict_all(eph, LAT, LON, ALT, t, mask_deg=-90.0)[("W", 5)]
        v5 = ge.predict_all(
            eph, LAT, LON, ALT, t + timedelta(seconds=5.0), mask_deg=-90.0
        )
        lin = v0["range_m"] + 5.0 * v0["range_rate_mps"]
        quad = lin + 0.5 * v0["range_acc_mps2"] * 25.0
        self.assertGreater(abs(v5[("W", 5)]["range_m"] - lin), 100.0)
        self.assertLess(abs(v5[("W", 5)]["range_m"] - quad), 5.0)


class DarkProbeTest(unittest.TestCase):
    def _stage(self, raw):
        from gnss_broker import almanac

        args = types.SimpleNamespace(
            almanac=True,
            doppler_sign=1,
            lat=LAT,
            lon=LON,
            alt=ALT,
            carrier_hz=1202.025e6,
            mask_deg=0.0,
            post_sat_geometry=0,
            bias_det_fresh_s=30.0,
            bias_min_snr=10.0,
            bias_min_sats=2,
            probe_dark=True,
            probe_dark_prns=[1, 2, 3],
        )
        ctx = types.SimpleNamespace(
            args=args,
            capable_t=None,
            capable=None,
            brdc_alm={},
            almanac_sats=[],
            alm_sys="W",
            alm_min_prn=1,
            alm_now=lambda: None,
            dll_combiners=[],
            geom_post_t=[0.0],
            transit_pd=None,
            publisher=None,
            cb=types.SimpleNamespace(ema=None, value=0.0),
            best={},
            det_fresh={},
            t0=1000.0,
        )
        saved = (almanac.brdc_predict, almanac._log_rl, almanac.timebase)
        almanac.brdc_predict = lambda *a: raw
        almanac._log_rl = lambda key, msg, every_s=10.0: None
        almanac.timebase = types.SimpleNamespace(observe=lambda c, log_rl: None)
        try:
            almanac.stage_almanac_predict(ctx)
        finally:
            almanac.brdc_predict, almanac._log_rl, almanac.timebase = saved
        return ctx

    def test_dark_prns_are_probes_with_an_empty_sky(self):
        """Between passes the table predicts nothing, and the dark PRNs are still probe rows."""
        ctx = self._stage({})
        self.assertEqual(ctx.pred, {})
        self.assertEqual(sorted(ctx.pred_dark), [1, 2, 3])
        self.assertEqual(ctx.pred_dark[1][2], -90.0)

    def test_a_predicted_prn_is_not_dark(self):
        row = (-1000.0, -120.0, 60.0, 1.3e6, 0.0, 180.0)
        ctx = self._stage({2: row})
        self.assertEqual(sorted(ctx.pred), [2])
        self.assertEqual(sorted(ctx.pred_dark), [1, 3])


if __name__ == "__main__":
    unittest.main()
