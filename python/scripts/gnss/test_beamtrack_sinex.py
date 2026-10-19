#!/usr/bin/env python3
"""gps_beamtrack._igs_prn_by_catnum must map a catalog number to the PRN assignment in force
today.

The IGS SINEX SATELLITE/PRN section gives each SVN -> PRN assignment a Valid_From and a
Valid_To. An assignment that has not started yet (pre-announced, starting next year) is not
current, even though its Valid_To is open, and it must not displace the assignment flying today.
An assignment that has ended is not current either.

The registry is read from $HOME/.cache/kotekan_gps; a fresh copy (mtime now) is never
re-downloaded, so the test runs offline against the file it writes.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps_beamtrack as bt


def _sinex(prn_lines):
    year = time.gmtime().tm_year
    body = [
        "%=SNX 2.02 IGS 26:001:00000 IGS 00:000:00000 00:000:00000 C 00000 0",
        "+SATELLITE/IDENTIFIER",
        "*SVN_ COSPAR ID SatCat Block__________ Comment__________________________________",
        " R801 2014-075A  40315 GLO-K1A         test",
        " R802 2011-064A  37829 GLO-M           test",
        "-SATELLITE/IDENTIFIER",
        "+SATELLITE/PRN",
        "*SVN_ Valid_From____ Valid_To______ PRN Comment_________________________________",
    ]
    body += [l % {"y0": year - 1, "y1": year + 1} for l in prn_lines]
    body += ["-SATELLITE/PRN", "%ENDSNX"]
    return "\n".join(body) + "\n"


class TestCurrentAssignment(unittest.TestCase):
    def setUp(self):
        self._home = os.environ.get("HOME")
        self.d = tempfile.mkdtemp()
        os.environ["HOME"] = self.d
        os.makedirs(os.path.join(self.d, ".cache", "kotekan_gps"))

    def tearDown(self):
        if self._home is not None:
            os.environ["HOME"] = self._home
        shutil.rmtree(self.d, ignore_errors=True)

    def _write(self, prn_lines):
        p = os.path.join(self.d, ".cache", "kotekan_gps", "igs_satellite_metadata.snx")
        with open(p, "w") as f:
            f.write(_sinex(prn_lines))

    def test_future_assignment_does_not_displace_the_current_one(self):
        # R801 flies R05 until next year, then is re-assigned to R09 (listed after it).
        self._write(
            [
                " R801 %(y0)d:001:00000 %(y1)d:001:00000 R05 ",
                " R801 %(y1)d:001:00000 0000:000:00000 R09 ",
            ]
        )
        self.assertEqual(bt._igs_prn_by_catnum().get(40315), ("R", 5))

    def test_ended_assignment_is_not_current(self):
        self._write(
            [
                " R802 2011:300:00000 %(y0)d:001:00000 R07 ",
                " R802 %(y0)d:001:00000 0000:000:00000 R12 ",
            ]
        )
        self.assertEqual(bt._igs_prn_by_catnum().get(37829), ("R", 12))

    def test_not_started_alone_maps_nothing(self):
        self._write([" R801 %(y1)d:001:00000 0000:000:00000 R09 "])
        self.assertNotIn(40315, bt._igs_prn_by_catnum())


if __name__ == "__main__":
    unittest.main()
