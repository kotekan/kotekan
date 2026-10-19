#!/usr/bin/env python3
"""Tests for gps_hoprate_validate.py: its exit status follows the generator it validates.

    /home/kvand/gnss/venv-ft/bin/python test_hoprate_validate.py

tests/boost/test_gnss_channelized_replica.cpp leaves nav-bit edges on code-period boundaries
to that script, so a broken hop-rate generator must make it return non-zero.
"""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps_hoprate_validate as hv  # noqa: E402


def run_main():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = hv.main()
    return rc, out.getvalue()


class TestHoprateValidate(unittest.TestCase):
    def setUp(self):
        self.orig = hv.hoprate_prefix

    def tearDown(self):
        hv.hoprate_prefix = self.orig

    def test_shipped_generator_passes(self):
        rc, out = run_main()
        self.assertEqual(rc, 0, out)
        self.assertNotIn("FAIL", out)

    def test_bit_per_hop_on_the_exact_path_fails(self):
        def per_hop(*a, **kw):
            kw["navbit_per_hop"] = True
            return self.orig(*a, **kw)

        hv.hoprate_prefix = per_hop
        rc, out = run_main()
        self.assertEqual(rc, 1, out)
        self.assertIn("per-chip", out.split("FAIL", 1)[-1])

    def test_bit_dropped_on_the_per_hop_path_fails(self):
        """A per-hop path that never applies the bit is off on every hop after the edge, not
        only on the straddling ones."""

        def no_bit(*a, **kw):
            if kw.get("navbit_per_hop"):
                kw["navbit"] = None
            return self.orig(*a, **kw)

        hv.hoprate_prefix = no_bit
        rc, out = run_main()
        self.assertEqual(rc, 1, out)
        self.assertIn("per-hop off on hops", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
