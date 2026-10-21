#!/usr/bin/env python3
"""fleet_coherent must degrade with the fleet, not with its worst member.

A frozen instance (a pipeline stopped at one hop, still serving that buffer) must not empty
the combine. Intersecting the hop sets of every contributor lets one such instance empty the
common window for every satellite on every cycle, and the estimator then returns {}, exactly
what an empty sky returns: no FLEET-COH line, no FLEET-INST line, no error.

The tests drive the shipped fleet_coherent through a synthetic fleet whose staleness is a
parameter, so they also cover cases sky rarely shows: half the fleet down, a slow drifter
rather than a hard freeze, everyone stale together.

The same fleet, with one instance carrying no sky signal, drives dark_instances: an instance at
the floor on every PRN its peers detect while its tap is fed is named, and a weak, starved or
lone one is not. The last class covers the DARK-INST line, get_rf's `dark` field and the
viewer's stream-health row (rendered under node).

Run: python3 test_fleet_degradation.py
"""
import cmath
import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gnss_broker.fleet as F  # noqa: E402
from gnss_broker.fleet import fleet_coherent  # noqa: E402

HOP0 = 49662562304
STEP = 2048  # one record: 2048 hops = 10.49 ms at CHORD
N_REC = 128  # what a combiner actually serves


def make_fleet(
    n_inst=8, stale=(), stale_by=N_REC * 4, prns=(23,), snr=30.0, seed=7, gain=None
):
    """A fleet of instances serving records for `prns`. Instances in `stale` are frozen
    `stale_by` records in the past, the signature of a frozen pipeline. `gain` scales an
    instance's sky signal ({index: factor}); 0 leaves it noise only, a dark instance.

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
                a = snr * (gain or {}).get(i, 1.0) * cmath.exp(1j * ph)
                a += complex(rng.gauss(0, 1), rng.gauss(0, 1))
                rows.append([hop, a.real, a.imag, 1.0e9])
            recs.append(
                {
                    "prn": prn,
                    "doppler_hz": -393.6,
                    "code_phase_chips": 165067.4,
                    "records": rows,
                }
            )
        out[url] = recs
    return out


def run(fleet, min_instances=3, min_records=32, prns={23}):
    """Drive the shipped fleet_coherent against an in-memory fleet."""
    import gnss_broker.fleet as F

    orig = F._get
    F._get = lambda url: fleet.get(url.rsplit("/", 1)[0], [])
    try:
        return fleet_coherent(
            list(fleet),
            min_instances,
            min_records,
            prns=prns,
            floor_margin=3.0,
            seed=1,
        )
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
        """One frozen instance must not empty the combine, as an all-contributor
        intersection would."""
        r = run(make_fleet(stale={0}))
        self.assertIn(23, r, "one stale instance still empties the combine")
        self.assertEqual(
            r[23]["n_src"], 7, "the seven healthy instances must all be used"
        )
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
        """Six of eight frozen -> two live: too thin, and it must decline rather than
        combine two instances into a confident-looking number."""
        r = run(make_fleet(stale={0, 1, 2, 3, 4, 5}))
        self.assertNotIn(23, r)

    def test_a_slightly_lagging_fleet_still_combines(self):
        """A fleet uniformly behind is a lagging poll, not a set satellite: the age test
        measures against the fleet's own newest record, so the window is common and current
        and it combines."""
        r = run(make_fleet(stale=set(range(8)), stale_by=100))  # ~1.0 s behind
        self.assertIn(23, r)
        self.assertEqual(r[23]["n_src"], 8)
        self.assertEqual(len(r[23]["dropped"]), 0)

    def test_a_SET_satellite_is_excluded_not_combined(self):
        """A satellite that has set must not be combined as current.

        Combiner buffers never expire a PRN, so once it stops being tracked every instance
        holds the same frozen window, they agree perfectly, and a test of agreement alone
        would publish a fossil as a current detection. Unlike a late poll, a set satellite
        trails the fleet's own newest record, which is the clock that catches it."""
        fleet = make_fleet(prns=(23, 31))  # 31 stays current
        for url, recs in fleet.items():  # 23 froze an hour ago
            for rec in recs:
                if rec["prn"] == 23:
                    for row in rec["records"]:
                        row[0] -= int(3600 * 195312.5)
        r = run(fleet, prns={23, 31})
        self.assertNotIn(23, r, "a satellite an hour stale was combined as current")
        self.assertIn(31, r, "the stale PRN took the healthy one down with it")

    def test_the_fleet_clock_survives_all_requested_prns_being_stale(self):
        """The clock is taken across every satellite the instances serve, not just the
        ones asked about; otherwise "now" is defined by the very PRNs under suspicion."""
        fleet = make_fleet(prns=(23, 31))
        for recs in fleet.values():  # 23 stale; only 23 is requested
            for rec in recs:
                if rec["prn"] == 23:
                    for row in rec["records"]:
                        row[0] -= int(3600 * 195312.5)
        self.assertNotIn(
            23,
            run(fleet, prns={23}),
            "a fossil was combined because nothing fresh was requested",
        )

    def test_partial_overlap_is_kept_not_dropped(self):
        """A drifter that still shares most of the window is a contributor, not a
        casualty: dropping it would throw away real fleet gain."""
        r = run(make_fleet(stale={0}, stale_by=8))  # 8 records behind of 128
        self.assertIn(23, r)
        self.assertEqual(
            r[23]["n_src"], 8, "a mostly-overlapping instance was discarded"
        )

    def test_a_frozen_instance_empties_the_all_instance_intersection(self):
        """The intersection over all contributors is empty for a fleet with one frozen
        instance, so any code that requires it returns nothing; this pins that the fixture
        reproduces that case."""
        fleet = make_fleet(stale={0})
        sets = []
        for recs in fleet.values():
            sets.append({int(r[0]) for rec in recs for r in rec["records"]})
        self.assertEqual(
            len(set.intersection(*sets)),
            0,
            "fixture no longer reproduces the empty all-instance intersection",
        )
        self.assertGreaterEqual(len(set.union(*sets)), N_REC + 8)


def tap_of(url):
    """The voltage-tap URL on the same node and GPU as a combiner URL."""
    return re.sub(r":\d+/(gnss\d)_.*$", r":12048/\1_srch_tap", url)


def taps_for(fleet, power=None):
    """get_rf instances (poll_rf_stats shape) for the fleet's GPUs: every tap on, two lobes of
    7 channels at unit power unless `power` ({combiner url: power}) says otherwise."""
    out = {}
    for url in fleet:
        p = (power or {}).get(url, 1.0)
        out[tap_of(url)] = {
            "state": "on",
            "lobes": [{"power": p, "n_chan": 7}, {"power": 1.1 * p, "n_chan": 7}],
        }
    return out


class TestDarkInstance(unittest.TestCase):
    """dark_instances: an instance at the floor on every PRN its peers detect while its tap is
    fed is named; a weak, starved or lone instance is not."""

    PRNS = (3, 5, 7)

    def fleet_and_fcoh(self, gain=None, snr=30.0, prns=PRNS):
        fleet = make_fleet(n_inst=11, prns=prns, snr=snr, gain=gain)
        return fleet, run(fleet, prns=set(prns))

    def test_the_dark_instance_is_named(self):
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0})
        dark_url = list(fleet)[3]
        self.assertEqual(
            len(fcoh[3]["per_inst"]), 11, "the dark instance left the combine"
        )
        got = F.dark_instances(fcoh, taps_for(fleet))
        self.assertEqual(list(got), [tap_of(dark_url)])
        d = got[tap_of(dark_url)]
        self.assertEqual((d["node"], d["gpu"], d["n_prn"]), ("cx27", 1, 3))
        self.assertLess(d["snr_max"], d["floor"])

    def test_a_healthy_fleet_has_none(self):
        fleet, fcoh = self.fleet_and_fcoh()
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet)), {})

    def test_a_weak_instance_under_a_raised_floor_is_not_dark(self):
        """Strong satellites raise the null floor above an instance that sees a few times less
        than its peers; the peer bar keeps it out."""
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.2})
        weak = list(fleet)[3]
        for v in fcoh.values():
            # Above the weak instance, while its peers still clear 3x it.
            peers = sorted(x for u, x in v["per_inst"].items() if u != weak)
            v["floor"] = 0.3 * peers[len(peers) // 2]
            self.assertLess(v["per_inst"][weak], v["floor"])
        rf = taps_for(fleet)
        self.assertEqual(F.dark_instances(fcoh, rf), {})
        # The peer bar is what keeps it out.
        self.assertEqual(
            list(F.dark_instances(fcoh, rf, peer_frac=1.0)), [tap_of(weak)]
        )

    def test_an_instance_above_the_floor_is_not_dark(self):
        """At a twentieth of its peers but above the null floor, the instance still despreads
        its satellites; the floor bar keeps it out."""
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.05})
        weak = list(fleet)[3]
        for v in fcoh.values():
            peers = sorted(x for u, x in v["per_inst"].items() if u != weak)
            self.assertLess(v["per_inst"][weak], 0.1 * peers[len(peers) // 2])
            self.assertGreater(v["per_inst"][weak], v["floor"])
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet)), {})

    def test_only_prns_the_peers_clear_the_margin_on_count(self):
        """The dark instance is named only on PRNs whose peer median clears floor_margin x the
        floor."""
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0})
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet), floor_margin=1e3), {})

    def test_a_dead_input_is_the_taps_to_report(self):
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0})
        dead = list(fleet)[3]
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet, {dead: 0.01})), {})
        rf = taps_for(fleet)
        rf[tap_of(dead)] = {"state": "off"}
        self.assertEqual(F.dark_instances(fcoh, rf), {})
        # The state decides, whatever lobes the entry still carries.
        rf = taps_for(fleet)
        rf[tap_of(dead)]["state"] = "unreachable"
        self.assertEqual(F.dark_instances(fcoh, rf), {})

    def test_a_sky_nobody_detects_names_nobody(self):
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0}, snr=0.3)
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet)), {})

    def test_one_prn_is_not_enough(self):
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0}, prns=(3,))
        self.assertEqual(F.dark_instances(fcoh, taps_for(fleet)), {})

    def test_telemetry_tags_find_their_taps(self):
        """With --telem-coherent the instances are <node>.<gpu> tags, not combiner URLs."""
        fleet, fcoh = self.fleet_and_fcoh(gain={3: 0.0})
        tag = {u: F.inst_key(u).replace("/", ".") for u in fleet}
        for v in fcoh.values():
            v["per_inst"] = {tag[u]: x for u, x in v["per_inst"].items()}
        got = F.dark_instances(fcoh, taps_for(fleet))
        self.assertEqual(list(got), [tap_of(list(fleet)[3])])


def synthetic_fcoh():
    """One chain's fleet_coherent rows: cx19/1 at the noise scale on two detected PRNs."""
    insts = ["cx19.0", "cx19.1", "cx27.0", "cx42.0", "cx42.1"]
    return {
        prn: {
            "per_inst": {u: (1.2 if u == "cx19.1" else lvl) for u in insts},
            "floor": 4.0,
        }
        for prn, lvl in ((9, 250.0), (13, 180.0))
    }


def synthetic_taps():
    """Every tap of synthetic_fcoh's instances on, at unit power."""
    out = {}
    for u in ("cx19.0", "cx19.1", "cx27.0", "cx42.0", "cx42.1"):
        node, gpu = u.split(".")
        out["http://%s:12048/gnss%s_srch_tap" % (node, gpu)] = {
            "state": "on",
            "lobes": [{"power": 1.0, "n_chan": 7}],
        }
    return out


class TestDarkInstLine(unittest.TestCase):
    """instr_dark_inst logs DARK-INST naming node, gpu and chain, publishes the verdict, and
    leaves the instance in the fleet's rows; get_rf serves it; the viewer shows it."""

    DARK_TAP = "http://cx19:12048/gnss1_srch_tap"

    def test_logged_and_published_not_dropped(self):
        import types
        from unittest import mock
        from gnss_broker import instruments

        fcoh = synthetic_fcoh()
        published = {}
        pub = types.SimpleNamespace(
            rf_instances=synthetic_taps,
            set_dark=lambda d, t: published.update(dark=d, t=t),
        )
        ctx = types.SimpleNamespace(
            publisher=pub,
            dllp=types.SimpleNamespace(fcoh=fcoh),
            args=types.SimpleNamespace(coh_floor_margin=3.0, signal="gal_e5b"),
            t0=1000.0,
        )
        logs = []
        with mock.patch.object(
            instruments, "_log_rl", side_effect=lambda k, m, every_s=0: logs.append(m)
        ):
            instruments.instr_dark_inst(ctx)
        self.assertEqual(list(published["dark"]), [self.DARK_TAP])
        self.assertEqual(published["t"], 1000.0)
        self.assertEqual(len(logs), 1)
        self.assertIn("DARK-INST: node cx19 gpu 1 chain gal_e5b", logs[0])
        self.assertIn("cx19.1", fcoh[9]["per_inst"], "the dark instance left the sums")

    def test_get_rf_serves_each_chains_verdict(self):
        import json
        import urllib.request
        from gnss_broker.publish import FleetPublisher

        pub = FleetPublisher(0, lambda m: None)
        try:
            e5b = pub.register("gal_e5b")
            l5 = pub.register("gps_l5")
            l5.set_rf(synthetic_taps(), 999.0)
            self.assertEqual(sorted(e5b.rf_instances()), sorted(synthetic_taps()))
            e5b.set_dark(F.dark_instances(synthetic_fcoh(), e5b.rf_instances()), 1000.0)
            l5.set_dark({}, 1000.0)
            port = pub._srv.server_address[1]
            with urllib.request.urlopen(
                "http://127.0.0.1:%d/gps_l5/get_rf" % port, timeout=5
            ) as r:
                got = json.loads(r.read())
        finally:
            pub._srv.shutdown()
            pub._srv.server_close()
        self.assertEqual(sorted(got["instances"]), sorted(synthetic_taps()))
        self.assertEqual(list(got["dark"]["gal_e5b"]["instances"]), [self.DARK_TAP])
        self.assertEqual(got["dark"]["gps_l5"]["instances"], {})

    def test_the_viewer_shows_it(self):
        import json
        import shutil
        import subprocess
        import tempfile

        # GNSS_REQUIRE_NODE (the test runner sets it): no node fails rather than skips.
        if not shutil.which("node") and not os.environ.get("GNSS_REQUIRE_NODE"):
            self.skipTest("no node")
        panel = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "js_viewer",
            "app",
            "panels",
            "airspy_stats.js",
        )
        rf = {
            "instances": synthetic_taps(),
            "dark": {
                "gal_e5b": {"t": 1.0, "instances": {self.DARK_TAP: {}}},
                "gps_l5": {"t": 1.0, "instances": {self.DARK_TAP: {}}},
                "bds_b2b": {"t": 1.0, "instances": {}},
            },
        }
        d = tempfile.mkdtemp()
        try:
            shutil.copy(panel, os.path.join(d, "panel.mjs"))
            with open(os.path.join(d, "run.mjs"), "w") as f:
                f.write(
                    'import {AirspyStatsPanel} from "./panel.mjs";\n'
                    "const p = Object.create(AirspyStatsPanel.prototype);\n"
                    "p._gnss = JSON.parse(process.argv[2]);\n"
                    "console.log(p._gnss_block());\n"
                )
            r = subprocess.run(
                ["node", os.path.join(d, "run.mjs"), json.dumps(rf)],
                capture_output=True,
                text=True,
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dark instances", r.stdout)
        self.assertIn("cx19/gnss1_srch_tap (gal_e5b, gps_l5)", r.stdout)
        self.assertIn("3 chain(s) checked", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
