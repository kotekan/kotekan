#!/usr/bin/env python3
"""Tests for gnss_tec_chord.py: arc ids are unique across --pair sets.

    /home/kvand/gnss/venv-ft/bin/python test_tec_chord.py

gnss_tec_movie keys each arc on (sys, prn, arc) for its high-pass and its IONEX leveling, so
two pairs of one constellation must not hand out the same arc id for one satellite.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_tec_chord as gtc  # noqa: E402

DAY = "20261018"
T0 = 1792281600.0  # 2026-10-18 00:00 UTC
PRN = 11


def write_band(obs_dir, band, n=400):
    """One satellite, one fleet-ADR arc of n grid hops: carrier range only, no ionosphere."""
    lam = gtc.C / gtc.FREQ[band]
    with open(os.path.join(obs_dir, "%s_%s.jsonl" % (band, DAY)), "w") as fh:
        for k in range(n):
            rho = 2.2e7 + 300.0 * k
            row = {
                "t": T0 + k * gtc.GRID_SECONDS,
                "prn": PRN,
                "fadr_g_hop": (k + 1) * gtc.GRID_HOPS,
                "fadr_g_dop_cycles": -rho / lam,
                "fadr_arc": 1,
                "fadr_hop0": 1000,
                "cn0_kcoh_dbhz": 40.0,
                "az": 180.0,
                "el": 60.0,
            }
            fh.write(json.dumps(row) + "\n")


class TestArcIdsAcrossPairs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        for band in ("gal_e5a", "gal_e5b", "gal_e6"):
            write_band(self.d, band)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_two_galileo_pairs_get_distinct_arc_ids(self):
        out = os.path.join(self.d, "tec")
        argv = [
            "gnss_tec_chord.py",
            "--obs-dir",
            self.d,
            "--day",
            DAY,
            "--since-h",
            "1e6",
            "--pair",
            "gal_e5a:gal_e6",
            "--pair",
            "gal_e5b:gal_e6",
            "--no-plot",
            "--out",
            out,
        ]
        log = io.StringIO()
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(log):
                gtc.main()
        finally:
            sys.argv = old
        with open(out + "_arcs.json") as fh:
            summary = json.load(fh)
        self.assertEqual(len(summary), 2, log.getvalue())
        keys = [(s["sys"], s["prn"], s["arc"]) for s in summary]
        self.assertEqual(len(set(keys)), 2, keys)
        z = np.load(out + "_series.npz")
        self.assertEqual(
            len({(s, p, a) for s, p, a in zip(z["sys"], z["prn"], z["arc"])}), 2
        )
        # The per-pair line still counts that pair's arcs only.
        self.assertEqual(log.getvalue().count("  1 joint arc(s)"), 2, log.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
