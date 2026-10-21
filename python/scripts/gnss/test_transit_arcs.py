#!/usr/bin/env python3
"""Tests for gnss_transit_arcs.py: the --bore-deg legend and the --broker-lines filter.

    /home/kvand/gnss/venv/bin/python test_transit_arcs.py

The figure test needs matplotlib (the venv has it; venv-ft does not, and skips it unless
GNSS_REQUIRE_PLOT is set).
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# gnss_transit_arcs puts the live tree first on sys.path for gnss_tec_chord and
# gnss_ephemeris: import this tree's copies first so it binds to them, then drop its entry so
# later imports in this process (one unittest run of several test files) come from this tree.
for _m in ("gnss_tec_chord", "gnss_ephemeris"):
    importlib.import_module(_m)
import gnss_transit_arcs as ta  # noqa: E402

sys.path.remove(ta.GNSS)

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None


# GNSS_REQUIRE_PLOT (the test runner sets it): no matplotlib fails rather than skips, since a
# skip here only ever means the wrong interpreter.
@unittest.skipUnless(
    HAVE_MPL or os.environ.get("GNSS_REQUIRE_PLOT"), "needs matplotlib"
)
class TestTransitLegend(unittest.TestCase):
    """The survival map's transit legend names the --bore-deg the windows were cut at."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _labels(self, bore_deg):
        import matplotlib.figure

        got = []
        orig = matplotlib.figure.Figure.legend

        def legend(fig, *a, **kw):
            got.extend(h.get_label() for h in kw.get("handles", []))
            return orig(fig, *a, **kw)

        t0 = 1790000000.0
        w = dict(t0=t0, t1=t0 + 120, t_ca=t0 + 60, sat="G01", min_sep=1.0, healthy=True)
        pr = dict(
            pair=("gps_l5", "gps_l2c"),
            sys="G",
            lockseg={},
            arcspan={},
            cls={0: {1: {"cls": "SURVIVED"}}},
            zoom={0: {}},
            arcs_ds={},
        )
        matplotlib.figure.Figure.legend = legend
        try:
            ta.figures(
                "before", [(0, w)], [pr, pr], os.path.join(self.d, "x"), 60.0, bore_deg
            )
        finally:
            matplotlib.figure.Figure.legend = orig
        return [s for s in got if s.startswith("transit")]

    def test_label_follows_bore_deg(self):
        self.assertEqual(self._labels(3.0), ["transit, < 3° of boresight"])
        self.assertEqual(self._labels(2.5), ["transit, < 2.5° of boresight"])


class TestBrokerFreezeLines(unittest.TestCase):
    """--broker-lines keeps the named chain's FROZEN/CLEAR lines, not every chain's."""

    LOG = [
        "[broker gps_l5 10:00:01.000] dead-reckon: receiver clock FROZEN for a boresight "
        "transit (G05 at 2.1 deg < dr_clock_transit_freeze_deg 5.0)\n",
        "[broker gal_e5a 10:00:02.000] dead-reckon: receiver clock FROZEN for a boresight "
        "transit (G05 at 2.1 deg < dr_clock_transit_freeze_deg 5.0)\n",
        "[broker gps_l5 10:00:03.000] DLL: PRN 3 disc +0.012\n",
        "[broker gal_e5a 10:04:02.000] dead-reckon: boresight transit CLEAR (nearest G05 at "
        "5.3 deg) after 240 s\n",
        "Oct 18 10:04:05 cf06 python[77]: [broker gps_l5 10:04:05.000] dead-reckon: boresight "
        "transit CLEAR (nearest G05 at 5.3 deg) after 244 s\n",
    ]

    def test_only_the_named_chain(self):
        got = ta.broker_freeze_lines(self.LOG, "gps_l5")
        self.assertEqual([t for t, _ in got], ["10:00:01.000", "10:04:05.000"])
        self.assertTrue(got[0][1].startswith("receiver clock FROZEN"))
        self.assertTrue(got[1][1].startswith("boresight transit CLEAR"))

    def test_other_chain(self):
        got = ta.broker_freeze_lines(self.LOG, "gal_e5a")
        self.assertEqual([t for t, _ in got], ["10:00:02.000", "10:04:02.000"])

    def test_untagged_single_chain_log_is_kept(self):
        log = [
            "[broker 09:00:00.500] dead-reckon: receiver clock FROZEN for a boresight transit\n"
        ]
        self.assertEqual(
            ta.broker_freeze_lines(log, "gps_l5"),
            [("09:00:00.500", "receiver clock FROZEN for a boresight transit\n")],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
