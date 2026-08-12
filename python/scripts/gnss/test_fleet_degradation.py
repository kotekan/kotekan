#!/usr/bin/env python3
"""fleet_coherent must degrade with the fleet, not with its worst member (#10).

THE INCIDENT, 2026-08-12. cx19's GPU-0 pipeline froze: all five of its chains stopped at
the same hop and kept serving that buffer for an hour, while the other seven instances
advanced together to within +-0.10 s. fleet_coherent intersected the hop sets of EVERY
contributor, so that one frozen instance emptied the common window for every satellite on
every cycle, and the estimator returned {} -- which is exactly what an empty sky returns.
No FLEET-COH line, no FLEET-INST line, no error: the cross-node coherent combine was dead
and nothing said so. It was found only by polling the instances by hand and noticing one
had not advanced in 20 s.

The tests below drive the SHIPPED fleet_coherent through a synthetic fleet whose staleness
is a parameter, because the real frozen instance was restarted before the fix existed --
and a fixture is better anyway: it can express the cases sky has not shown us yet (half the
fleet down, a slow drifter rather than a hard freeze, everyone stale together).

Run: python3 test_fleet_degradation.py
"""
import cmath
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.fleet import fleet_coherent   # noqa: E402

HOP0 = 49662562304
STEP = 2048          # one record: 2048 hops = 10.49 ms at CHORD
N_REC = 128          # what a combiner actually serves


def make_fleet(n_inst=8, stale=(), stale_by=N_REC * 4, prns=(23,), snr=30.0, seed=7):
    """A fleet of instances serving records for `prns`. Instances in `stale` are frozen
    `stale_by` records in the past -- the cx19 GPU-0 signature.

    Signal model: one common per-record sky phase (what the estimator locks onto) plus
    per-instance noise, so a correct combine sees coherent gain and the shuffled null does
    not. Returns {url: [record-dicts]} in /get_records wire shape.
    """
    rng = random.Random(seed)
    sky = {}
    out = {}
    for i in range(n_inst):
        url = "http://cx%02d:12049/gnss%d_n2combine" % (19 + (i // 2) * 8, i % 2)
        off = -stale_by * STEP if i in stale else 0
        recs = []
        for prn in prns:
            rows = []
            for k in range(N_REC):
                hop = HOP0 + (k * STEP) + off
                ph = sky.setdefault((prn, hop), rng.uniform(-3.14, 3.14))
                a = snr * cmath.exp(1j * ph)
                a += complex(rng.gauss(0, 1), rng.gauss(0, 1))
                rows.append([hop, a.real, a.imag, 1.0e9])
            recs.append({"prn": prn, "doppler_hz": -393.6,
                         "code_phase_chips": 165067.4, "records": rows})
        out[url] = recs
    return out


def run(fleet, min_instances=3, min_records=32):
    """Drive the shipped fleet_coherent against an in-memory fleet."""
    import gnss_broker.fleet as F
    orig = F._get
    F._get = lambda url: fleet.get(url.rsplit("/", 1)[0], [])
    try:
        return fleet_coherent(list(fleet), min_instances, min_records,
                              prns={23}, log=None, floor_margin=3.0, seed=1)
    finally:
        F._get = orig


class TestFleetDegradation(unittest.TestCase):

    def test_healthy_fleet_combines(self):
        r = run(make_fleet())
        self.assertIn(23, r, "a healthy 8-instance fleet produced nothing")
        self.assertEqual(len(r[23]["dropped"]), 0)
        self.assertEqual(r[23]["n_src"], 8)
        self.assertTrue(r[23]["present"])

    def test_one_frozen_instance_does_not_kill_the_combine(self):
        """THE INCIDENT. Under the old all-contributor intersection this returned {}."""
        r = run(make_fleet(stale={0}))
        self.assertIn(23, r, "one stale instance still empties the combine")
        self.assertEqual(r[23]["n_src"], 7, "the seven healthy instances must all be used")
        self.assertEqual(len(r[23]["dropped"]), 1)
        self.assertTrue(r[23]["present"])

    def test_three_frozen_of_eight(self):
        r = run(make_fleet(stale={0, 3, 6}))
        self.assertIn(23, r)
        self.assertEqual(r[23]["n_src"], 5)
        self.assertEqual(len(r[23]["dropped"]), 3)

    def test_down_to_min_instances(self):
        """Five of eight frozen: the remaining three still combine (min_instances=3)."""
        r = run(make_fleet(stale={0, 1, 2, 3, 4}))
        self.assertIn(23, r)
        self.assertEqual(r[23]["n_src"], 3)

    def test_below_min_instances_declines(self):
        """Six of eight frozen -> two live: too thin, and it must DECLINE rather than
        combine two instances into a confident-looking number."""
        r = run(make_fleet(stale={0, 1, 2, 3, 4, 5}))
        self.assertNotIn(23, r)

    def test_a_whole_fleet_stale_together_still_combines(self):
        """Staleness is only fatal when it is DISAGREEMENT. If every instance is equally
        behind (a lagging poll, not a freeze) the window is still common and usable."""
        r = run(make_fleet(stale=set(range(8)), stale_by=500))
        self.assertIn(23, r)
        self.assertEqual(r[23]["n_src"], 8)
        self.assertEqual(len(r[23]["dropped"]), 0)

    def test_partial_overlap_is_kept_not_dropped(self):
        """A drifter that still shares most of the window is a contributor, not a
        casualty: dropping it would throw away real fleet gain."""
        r = run(make_fleet(stale={0}, stale_by=8))     # 8 records behind of 128
        self.assertIn(23, r)
        self.assertEqual(r[23]["n_src"], 8, "a mostly-overlapping instance was discarded")

    def test_the_old_unanimous_rule_would_have_failed(self):
        """Pins WHY this changed: the intersection over all contributors is empty for the
        incident fleet, so any code that requires it returns nothing."""
        fleet = make_fleet(stale={0})
        sets = []
        for recs in fleet.values():
            sets.append({int(r[0]) for rec in recs for r in rec["records"]})
        self.assertEqual(len(set.intersection(*sets)), 0,
                         "fixture no longer reproduces the empty all-instance intersection")
        self.assertGreaterEqual(len(set.union(*sets)), N_REC + 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
