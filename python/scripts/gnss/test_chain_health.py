#!/usr/bin/env python3
"""Tests for gnss_chain_health.py: one chain's files join into one series across midnight.

    /home/kvand/gnss/venv-ft/bin/python test_chain_health.py

In hour 00 chain_health_cron.sh passes today's files and then yesterday's. The mask must hold
both days' bins for the chain, and an episode that crosses 00:00 must be one episode.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_chain_health as gch  # noqa: E402

MIDNIGHT = 1792281600  # 2026-10-18 00:00 UTC


def write_obs(path, t0, t1, band="gps_l5"):
    """A record every 10 s over [t0, t1) at el 30, none fleet_present: every bin unhealthy."""
    with open(path, "w") as fh:
        for t in range(t0, t1, 10):
            row = dict(t=float(t), band=band, el=30.0, fleet_present=False, prn=1)
            fh.write(json.dumps(row) + "\n")


def run(argv):
    out = io.StringIO()
    old = sys.argv
    sys.argv = ["gnss_chain_health.py"] + argv
    try:
        with contextlib.redirect_stdout(out):
            gch.main()
        code = 0
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
    return code, out.getvalue()


class TestAcrossMidnight(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.today = os.path.join(self.d, "gps_l5_20261018.jsonl")
        self.yday = os.path.join(self.d, "gps_l5_20261017.jsonl")
        write_obs(self.today, MIDNIGHT, MIDNIGHT + 600)
        write_obs(self.yday, MIDNIGHT - 600, MIDNIGHT)
        self.mask = os.path.join(self.d, "m.json")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_cron_order_today_then_yesterday(self):
        code, out = run([self.today, self.yday, "--min-run", "900", "--out", self.mask])
        self.assertEqual(code, 1, out)
        ep = [ln for ln in out.splitlines() if ln.startswith("EPISODE")]
        self.assertEqual(len(ep), 1, out)
        self.assertIn("2026-10-17 23:50 -> 2026-10-18 00:10", ep[0])
        with open(self.mask) as fh:
            rows = json.load(fh)["chains"]["gps_l5"]
        self.assertEqual(
            [r["t"] for r in rows], [MIDNIGHT + k * 300 for k in (-2, -1, 0, 1)]
        )
        self.assertFalse(any(r["ok"] for r in rows))

    def test_each_side_alone_is_too_short(self):
        code, out = run([self.today, "--min-run", "900"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("EPISODE", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
