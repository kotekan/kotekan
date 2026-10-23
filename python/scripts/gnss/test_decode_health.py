#!/usr/bin/env python3
"""decode_health: the signal table covers what the broker observes, and a failed export loses
nothing.

  * Every signal id the broker passes to ctx.dh_obs (gnss_broker/navbits.py, including both
    I/NAV labels and the CNAV label gps_distributed_broker.py picks by carrier) has a
    SIGNAL_BAND entry, so none is exported with band "?". The viewer's SIG_NAME
    (app/panels/decode_health.js) names exactly those ids.
  * A flush whose write fails keeps the satellites it holds: the next good flush publishes
    them, rather than an empty list until each one is observed again.
  * read_all counts the broker files, fresh or not, and the viewer panel labels an empty
    table "no producer" only when there are none (the panel is rendered under node).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_health as dh

NAVBITS = os.path.join(HERE, "gnss_broker", "navbits.py")
BROKER = os.path.join(HERE, "gps_distributed_broker.py")
PANEL = os.path.join(HERE, "..", "js_viewer", "app", "panels", "decode_health.js")


def _observed_signals():
    nav = open(NAVBITS).read()
    sigs = set(re.findall(r'ctx\.dh_obs\(\s*"([A-Z0-9_]+)"', nav))
    m = re.search(r"_inav_sig = \((.*?)\)", nav, re.S)
    sigs |= set(re.findall(r'"([A-Z0-9_]+)"', m.group(1)))
    m = re.search(r"_nav\.cnav_sig = \((.*?)\)", open(BROKER).read(), re.S)
    sigs |= set(re.findall(r'"([A-Z0-9_]+)"', m.group(1)))
    return sigs


def _sig_name_keys():
    js = open(PANEL).read()
    m = re.search(r"const SIG_NAME = \{(.*?)\};", js, re.S)
    return set(re.findall(r"(\w+):", m.group(1)))


class TestSignalTable(unittest.TestCase):
    def test_every_observed_signal_has_a_band(self):
        sigs = _observed_signals()
        # The parse found the decoders: I/NAV on both bands, F/NAV, both CNAV carriers.
        for s in ("GAL_E1B_INAV", "GAL_E5BI_INAV", "GAL_E5AI_FNAV", "GPS_L2C_CNAV"):
            self.assertIn(s, sigs)
        self.assertEqual(sorted(sigs - set(dh.SIGNAL_BAND)), [])

    def test_viewer_names_every_exported_signal(self):
        self.assertEqual(_sig_name_keys(), set(dh.SIGNAL_BAND))


class TestFailedFlush(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_failed_write_keeps_the_satellites(self):
        path = os.path.join(self.d, "gps_l1.json")
        w = dh.DecodeHealthWriter(path, "gps_l1", sys="G")
        w.observe("GPS_L1_LNAV", 3, 1000.0, count=5, synced=True)
        w.observe("GPS_L1_LNAV", 7, 1000.0, count=2, synced=True)
        with mock.patch.object(dh.os, "replace", side_effect=OSError("disk full")):
            self.assertFalse(w.flush(1001.0, force=True))
        self.assertTrue(w.flush(1002.0, force=True))
        with open(path) as f:
            rec = json.load(f)
        self.assertEqual([s["prn"] for s in rec["sats"]], [3, 7])


class TestNoProducer(unittest.TestCase):
    """read_all counts the broker files present, fresh or not, and the panel names an empty
    table "no producer" only when there are none."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _write(self, chain, t):
        w = dh.DecodeHealthWriter(os.path.join(self.d, chain + ".json"), chain, sys="G")
        w.observe("GPS_L5_CNAV", 3, t, count=1, synced=True)
        self.assertTrue(w.flush(t, force=True))

    def test_file_count(self):
        missing = os.path.join(self.d, "absent")
        self.assertEqual(dh.read_all(missing, t_now=1000.0)["n_files"], 0)
        self.assertEqual(dh.read_all(self.d, t_now=1000.0)["n_files"], 0)
        open(os.path.join(self.d, "gps_l5.json.tmp.12"), "w").close()  # a writer's tmp
        self._write("gps_l5", 1000.0)
        out = dh.read_all(self.d, max_age_s=120.0, t_now=1500.0)
        self.assertEqual((out["n_files"], out["chains"], out["signals"]), (1, [], {}))
        out = dh.read_all(self.d, max_age_s=120.0, t_now=1010.0)
        self.assertEqual((out["n_files"], len(out["chains"])), (1, 1))

    # GNSS_REQUIRE_NODE (the test runner sets it): no node fails rather than skips.
    @unittest.skipUnless(
        shutil.which("node") or os.environ.get("GNSS_REQUIRE_NODE"), "needs node"
    )
    def test_panel_labels(self):
        d = self.d
        shutil.copy(PANEL, os.path.join(d, "panel.mjs"))
        with open(os.path.join(d, "run.mjs"), "w") as f:
            f.write(
                'import {DecodeHealthPanel} from "./panel.mjs";\n'
                "const p = Object.create(DecodeHealthPanel.prototype);\n"
                'p.root = {innerHTML: ""};\n'
                "p._render(JSON.parse(process.argv[2]));\n"
                "console.log(p.root.innerHTML);\n"
            )

        def render(data):
            r = subprocess.run(
                ["node", os.path.join(d, "run.mjs"), json.dumps(data)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout

        self.assertIn(
            "no producer", render({"signals": {}, "chains": [], "n_files": 0})
        )
        stale = render({"signals": {}, "chains": [], "n_files": 2})
        self.assertIn("2 broker files, none fresh", stale)
        self.assertNotIn("no producer", stale)
        fresh = {
            "signals": {},
            "chains": [{"chain": "gps_l5", "t_age_s": 3}],
            "n_files": 1,
        }
        self.assertIn("no satellite decoded yet", render(fresh))


if __name__ == "__main__":
    unittest.main()
