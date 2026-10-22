#!/usr/bin/env python3
"""Tests for gnss_tec_movie.py: the site defaults and the IONEX 'not available' cells.

    /home/kvand/gnss/venv/bin/python test_tec_movie.py

The leveling test runs main() and needs matplotlib and scipy (the venv has them; venv-ft does
not, and skips it unless GNSS_REQUIRE_PLOT is set).
"""
import contextlib
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_beam_cube  # noqa: E402
import gnss_tec_movie as tm  # noqa: E402

HAVE_PLOT = all(importlib.util.find_spec(m) for m in ("matplotlib", "scipy"))
LAT, LON = 49.32075144444, -119.62081125
LATS = [55.0, 50.0, 45.0, 40.0]
LONS = [-130.0, -125.0, -120.0, -115.0, -110.0]
EPOCHS = [
    datetime(2026, 10, 18, 0, tzinfo=timezone.utc),
    datetime(2026, 10, 18, 2, tzinfo=timezone.utc),
]


def ionex_text(missing):
    """Two identical maps of 20.0 TECU (200 at EXPONENT -1), 9999 at each missing (lat, lon)."""
    out = [
        "%-60s%s"
        % ("     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"),
        "%-60s%s" % ("    -1", "EXPONENT"),
        "%-60s%s" % ("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        "%-60s%s" % ("", "END OF HEADER"),
    ]
    for n, ep in enumerate(EPOCHS, 1):
        out.append("%-60s%s" % ("%6d" % n, "START OF TEC MAP"))
        out.append(
            "%-60s%s"
            % (
                ep.strftime("  %Y    %m    %d    %H     0     0"),
                "EPOCH OF CURRENT MAP",
            )
        )
        for la in LATS:
            head = "  %6.1f%6.1f%6.1f%6.1f%6.1f" % (la, LONS[0], LONS[-1], 5.0, 450.0)
            out.append("%-60s%s" % (head, "LAT/LON1/LON2/DLON/H"))
            out.append(
                "".join("%5d" % (9999 if (la, lo) in missing else 200) for lo in LONS)
            )
        out.append("%-60s%s" % ("%6d" % n, "END OF TEC MAP"))
    out.append("%-60s%s" % ("", "END OF FILE"))
    return "\n".join(out) + "\n"


class TestSiteDefaults(unittest.TestCase):
    def test_lat_lon_default_to_the_station(self):
        a = tm.build_parser().parse_args(["series.npz"])
        self.assertEqual((a.lat, a.lon), (gnss_beam_cube.LAT, gnss_beam_cube.LON))
        self.assertAlmostEqual(a.lat, LAT, places=6)
        self.assertAlmostEqual(a.lon, LON, places=6)


class TestIonexMissing(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _ionex(self, missing):
        p = os.path.join(self.d, "gim.inx")
        with open(p, "w") as f:
            f.write(ionex_text(missing))
        return p

    def test_parse_marks_9999_not_available(self):
        eps, lats, lons, maps, hgt = tm.parse_ionex(self._ionex({(55.0, -120.0)}))
        self.assertEqual(maps.shape, (2, 4, 5))
        self.assertEqual(list(lats), LATS)
        self.assertEqual(list(lons), LONS)
        self.assertEqual(hgt, 450.0)
        bad = np.zeros(maps.shape, bool)
        bad[:, 0, 2] = True
        self.assertTrue(np.isnan(maps[bad]).all())
        np.testing.assert_allclose(maps[~bad], 20.0)

    # GNSS_REQUIRE_PLOT (the test runner sets it): no matplotlib or scipy fails rather than
    # skips, since a skip here only ever means the wrong interpreter.
    @unittest.skipUnless(
        HAVE_PLOT or os.environ.get("GNSS_REQUIRE_PLOT"), "needs matplotlib and scipy"
    )
    def test_leveling_skips_unavailable_cells(self):
        """One arc, most of it north where the GIM has a hole: the arc is leveled on its
        southern points, which read the GIM's 20 TECU exactly, and the northern ones drop."""
        gim = self._ionex({(55.0, -120.0)})
        t0 = EPOCHS[0].timestamp() + 1800.0
        n_n, n_s = 30, 10
        # North at el 45 pierces at lat ~53, beside the hole; south at el 70 at lat ~48.
        az = np.r_[np.zeros(n_n), np.full(n_s, 180.0)]
        el = np.r_[np.full(n_n, 45.0), np.full(n_s, 70.0)]
        _, _, M = tm.ipp(az, el, LAT, LON, 450.0)
        series = os.path.join(self.d, "s_series.npz")
        np.savez(
            series,
            t=t0 + 10.0 * np.arange(n_n + n_s),
            az=az,
            el=el,
            tecu=20.0 * M - 7.0,  # slant TEC minus an arbitrary arc offset
            sys=np.array(["G"] * (n_n + n_s)),
            prn=np.ones(n_n + n_s, int),
            arc=np.zeros(n_n + n_s, int),
        )
        argv = [
            "gnss_tec_movie.py",
            series,
            "--ionex",
            gim,
            "--lat",
            str(LAT),
            "--lon",
            str(LON),
            "--outdir",
            os.path.join(self.d, "movie"),
        ]
        out = io.StringIO()
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(out):
                tm.main()
        finally:
            sys.argv = old
        m = re.search(r"frames, (\d+) epochs, vmax ([0-9.]+) TECU", out.getvalue())
        self.assertIsNotNone(m, out.getvalue())
        self.assertEqual(int(m.group(1)), n_s, out.getvalue())
        self.assertAlmostEqual(float(m.group(2)), 1.15 * 20.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
