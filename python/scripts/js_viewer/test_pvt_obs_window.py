#!/usr/bin/env python3
"""_pvt_measurements opens only the obs logs that can hold a row inside its window.

The PVT globs match every day's obs log, and each matched file is tail-read (up to 4 MB) on
every compute, on the reactor thread. A log last written before the window cannot hold a row
inside it, so it must not be opened; today's log still is, and its rows still come back.

Needs the viewer's interpreter (twisted, autobahn):

    ~/gnss/venv/bin/python test_pvt_obs_window.py
"""
import builtins
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gnss"))
sys.path.insert(0, HERE)
import livebeam_server as lb


def _write(path, t):
    with open(path, "w") as f:
        for prn, res in [(1, 10.0), (2, -5.0), (3, 2.0)] * 2:
            row = {
                "t": t,
                "sys": "G",
                "prn": prn,
                "band": "gps_l5",
                "code_resid_m": res,
                "az": 30.0 * prn,
                "el": 40.0,
                "code_len": 10230,
                "chip_rate_hz": 10.23e6,
            }
            f.write(json.dumps(row) + "\n")
    os.utime(path, (t, t))


class TestObsWindow(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.now = time.time()
        self.today = os.path.join(self.d, "gps_l5_today.jsonl")
        self.old = os.path.join(self.d, "gps_l5_2daysago.jsonl")
        _write(self.today, self.now)
        _write(self.old, self.now - 2 * 86400.0)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_old_log_is_not_opened(self):
        opened = []
        real_open = builtins.open

        def spy(path, *a, **kw):
            opened.append(path)
            return real_open(path, *a, **kw)

        with mock.patch("builtins.open", side_effect=spy):
            m = lb._pvt_measurements(
                [os.path.join(self.d, "*.jsonl")], 30.0, self.now, smooth_s=90.0
            )
        self.assertNotIn(self.old, opened)
        self.assertIn(self.today, opened)
        self.assertEqual(sorted(x["prn"] for x in m), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
