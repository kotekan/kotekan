#!/usr/bin/env python3
"""Tests for gnss_beam_cube.py build: what each veto and mask actually removes.

    /home/kvand/gnss/venv-ft/bin/python test_beam_cube_build.py

Offline: the BRDC fetch and orbit prediction are replaced by a fixed sky, and the archive
reader by synthetic rows, so each test controls exactly which satellite is where.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_beam_cube as bc  # noqa: E402

DAY = "20261001"
DAY_UNIX = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()


class StubGeom:
    """Every signal PRN at az 180, el 70, at the reference range; nothing vetoed."""

    bore = (bc.BORE_AZ, bc.BORE_EL)

    def vetoed(self, t):
        return False

    def at(self, sys_, prn, t):
        return (70.0, 180.0, bc.R_REF) if prn <= 5 else None


class TestPrnGainHealthMask(unittest.TestCase):
    """prn_gain_table must drop the chain-health mask's bins, as build_day_h5 does."""

    def setUp(self):
        # Two 5-min bins. Twenty noise PRNs (no geometry) set the pedestal at 1.0 in both; PRNs
        # 1-5 sit at 2.0, one pedestal above it. PRN 1 is in bin 0 for 100 s only, and all of
        # bin 1 at 101: if bin 1 is not masked its bright rows are most of PRN 1's samples.
        rows = []  # (t, prn, power)
        for k in range(60):
            tk = DAY_UNIX + 10.0 * k
            rows += [(tk, p, 1.0) for p in range(20, 40)]
            rows += [(tk, p, 2.0) for p in (2, 3, 4, 5)]
            if k < 10:
                rows.append((tk, 1, 2.0))
            elif k >= 30:
                rows.append((tk, 1, 101.0))
        t, prn, lev = (np.array(x) for x in zip(*rows))
        P = np.repeat(lev[:, None, None], 8, axis=2)
        h5 = ({"format": "gnss_cube_l0"}, t, prn.astype(int), P)
        self._orig = bc._h5_rows
        bc._h5_rows = lambda path: h5

    def tearDown(self):
        bc._h5_rows = self._orig

    def solve(self, health):
        args = Namespace(mask_deg=0.0, no_range_norm=False, _health=health)
        return bc.prn_gain_table(args, "gps_l5", ["x.h5"], DAY_UNIX, StubGeom())

    def test_masked_bin_does_not_reach_the_gain(self):
        g = self.solve({"gps_l5": (300.0, {int(DAY_UNIX) + 300})})
        self.assertLess(abs(g[1]), 1.0, "masked bin moved PRN 1's gain: %r" % g)

    def test_unmasked_bin_does(self):
        # The control: without the mask the bright bin does move the gain, so the masked case
        # above is a measurement and not a fit that returns ~0 whatever it is fed.
        g = self.solve({})
        self.assertGreater(g[1], 10.0, "%r" % g)


class TestPrnGainCells(unittest.TestCase):
    """prn_gain_table's theta x phi cells are measured about the boresight its Geometry carries."""

    class Geom(StubGeom):
        """PRN 1 at az 182, el 69.5 and PRNs 2-5 at az 182, el 66."""

        def __init__(self, bore):
            self.bore = bore

        def at(self, sys_, prn, t):
            if prn == 1:
                return (69.5, 182.0, bc.R_REF)
            return (66.0, 182.0, bc.R_REF) if prn <= 5 else None

    def setUp(self):
        # One 5-min bin: twenty noise PRNs set the pedestal at 1.0, PRN 1 sits 10 pedestals above
        # it and PRNs 2-5 one. About a boresight at el 70 both positions fall in one cell (theta
        # 0.9 and 4.1 deg, same quadrant), so PRN 1 solves 10 dB above the others. About el 81.41
        # they fall in the 10-15 and 15-20 deg rings, and PRN 1, alone in its cell, solves to 0.
        rows = []
        for k in range(30):
            tk = DAY_UNIX + 10.0 * k
            rows += [(tk, p, 1.0) for p in range(20, 40)]
            rows += [(tk, p, 2.0) for p in (2, 3, 4, 5)]
            rows.append((tk, 1, 11.0))
        t, prn, lev = (np.array(x) for x in zip(*rows))
        P = np.repeat(lev[:, None, None], 8, axis=2)
        h5 = ({"format": "gnss_cube_l0"}, t, prn.astype(int), P)
        self._orig = bc._h5_rows
        bc._h5_rows = lambda path: h5

    def tearDown(self):
        bc._h5_rows = self._orig

    def solve(self, bore):
        args = Namespace(mask_deg=0.0, no_range_norm=False, _health={})
        return bc.prn_gain_table(args, "gps_l5", ["x.h5"], DAY_UNIX, self.Geom(bore))

    def test_cells_follow_the_geometry_boresight(self):
        g = self.solve((180.0, 70.0))
        self.assertGreater(g[1], 5.0, "cells not about the el 70 boresight: %r" % g)

    def test_cells_about_p0(self):
        # The control: the same sky about the p0 boresight splits the two positions, so the
        # test above measures the cell geometry and not a fit that returns 10 dB regardless.
        g = self.solve((bc.BORE_AZ, bc.BORE_EL))
        self.assertLess(abs(g[1]), 1.0, "%r" % g)


class TestBuild(unittest.TestCase):
    """cmd_build on a two-row element archive (one probe, one sample) under a stub sky.

    G1 sits at az 180, el 60 except in the minutes listed in self.sky. The array epoch is a
    stub whose pointing has its boresight at self.bore_el.
    """

    STUBS = ("fetch_brdc", "parse_rinex_nav", "predict_all", "Geometry")

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.sky = {}
        self.bore_el = bc.BORE_EL
        self.made = []
        self._saved = {k: getattr(bc, k) for k in self.STUBS}
        self._for_day = bc.gnss_arraymap.for_day
        made = self.made

        class SpyGeometry(self._saved["Geometry"]):
            def __init__(self, *a):
                super().__init__(*a)
                made.append(self)

        bc.fetch_brdc = lambda when: None
        bc.parse_rinex_nav = lambda nav: {}
        bc.predict_all = self.predict_all
        bc.Geometry = SpyGeometry
        bc.gnss_arraymap.for_day = self.for_day
        t0 = DAY_UNIX + 3600.0
        with open(os.path.join(self.d, "elem_gps_l5_%s.jsonl" % DAY), "w") as fh:
            for row in (
                {"t": t0, "probe": True, "inst": "i0", "p2": [1.0] * 16},
                {"t": t0 + 1.0, "inst": "i0", "prn": 1, "p2": [3.0] * 16},
            ):
                fh.write(json.dumps(row) + "\n")

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(bc, k, v)
        bc.gnss_arraymap.for_day = self._for_day
        shutil.rmtree(self.d)

    def predict_all(self, eph, lat, lon, alt, t, mask_deg, max_age):
        g1 = {"el": 60.0, "az": 180.0, "range_m": bc.R_REF}
        return {("G", 1): self.sky.get(int(t // bc.GEOM_BIN_S), g1)}

    def for_day(self, daystr, be=None):
        pt = bc.gnss_arraymap.Pointing(
            "p_test",
            {"dec_deg": 0.0, "bore_az_deg": 180.0, "bore_el_deg": self.bore_el},
        )
        return SimpleNamespace(
            name="test", key=lambda: "test:00000000", verified=True, pointing=pt
        )

    def build(self, health_mask="auto"):
        args = Namespace(
            days=[DAY],
            chains=["gps_l5"],
            source="elem",
            archive=self.d,
            outdir=os.path.join(self.d, "out"),
            nside=8,
            veto_deg=5.0,
            mask_deg=0.0,
            health_mask=health_mask,
            no_range_norm=False,
            prn_norm=False,
            unbiased=False,
        )
        bc.cmd_build(args)
        path = os.path.join(args.outdir, "cube_%s_nside8.npz" % DAY)
        return path, json.loads(str(np.load(path)["meta"]))

    def test_veto_and_meta_follow_the_epoch_boresight(self):
        # Minute m: G1 0.5 deg from a boresight at el 70. Minute m + 1: G1 on the p0 boresight
        # (el 81.41), 11.4 deg from el 70. The sample row's G1 at el 60 is 10 deg off either way.
        self.bore_el = 70.0
        m = int(DAY_UNIX // bc.GEOM_BIN_S) + 600
        self.sky[m] = {"el": 70.5, "az": 180.0, "range_m": bc.R_REF}
        self.sky[m + 1] = {"el": 81.41, "az": 180.0, "range_m": bc.R_REF}
        _, meta = self.build()
        geom = self.made[0]
        self.assertTrue(geom.vetoed(m * bc.GEOM_BIN_S + 30.0), "not vetoed at the bore")
        self.assertFalse(
            geom.vetoed((m + 1) * bc.GEOM_BIN_S + 30.0), "vetoed at el 81.41"
        )
        self.assertEqual((meta["bore_az_deg"], meta["bore_el_deg"]), (180.0, 70.0))

    def test_export_carries_the_boresight(self):
        self.bore_el = 70.0
        path, _ = self.build()
        orig, lobe = bc.pix_centres, bc.lobe_level_db
        bc.pix_centres = lambda pix, nside: (
            np.full(len(pix), 180.0),
            np.full(len(pix), 60.0),
        )
        bores = []  # the boresight each chain's lobe annulus is centred on
        bc.lobe_level_db = lambda *a: bores.append(tuple(a[5])) or lobe(*a)
        try:
            web = os.path.join(self.d, "web")
            bc.cmd_export(
                Namespace(
                    masters=[path],
                    outdir=web,
                    nside=0,
                    subbands=0,
                    offset_ref="gps_l5",
                    offset_annulus=(2.0, 12.0),
                )
            )
        finally:
            bc.pix_centres, bc.lobe_level_db = orig, lobe
        with open(os.path.join(web, "cube_%s.json" % DAY)) as fh:
            man = json.load(fh)
        self.assertEqual((man["bore_az_deg"], man["bore_el_deg"]), (180.0, 70.0))
        self.assertEqual(bores, [(180.0, 70.0)])

    def test_lobe_annulus_is_about_the_given_boresight(self):
        # 20 pixels 5 deg south of a boresight at el 70: inside a 2..12 deg annulus about it,
        # 16.4 deg from the p0 boresight.
        az, el = np.full(20, 180.0), np.full(20, 65.0)
        n, s1 = np.full((1, 1, 20), 10, np.uint32), np.full((1, 1, 20), 100.0)
        self.assertIsNotNone(
            bc.lobe_level_db(n, s1, az, el, (2.0, 12.0), (180.0, 70.0))
        )
        self.assertIsNone(
            bc.lobe_level_db(n, s1, az, el, (2.0, 12.0), (bc.BORE_AZ, bc.BORE_EL))
        )

    def test_elem_build_records_no_health_mask(self):
        # build_day never applies the chain-health mask, so the master must not claim one.
        mask = os.path.join(self.d, "mask.json")
        with open(mask, "w") as fh:
            row = {"t": int(DAY_UNIX) + 3600, "ok": False}
            json.dump({"bin_s": 300, "chains": {"gps_l5": [row]}}, fh)
        _, meta = self.build(health_mask=mask)
        self.assertIsNone(meta["health_mask"])


if __name__ == "__main__":
    unittest.main()
