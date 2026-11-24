"""eph_cache: writers keep their history across restarts, retention thins with age, and readers
merge every writer.

python3 -m unittest test_eph_cache
"""

import glob
import os
import tempfile
import unittest

import decoded_eph
import eph_cache as ec

T0 = 2446 * 604800.0 + 200000.0  # a GPST in week 2446


def entry(prn, toe_gpst, week_from="sky", signal="GPS_L5_CNAV", sysc="G", **eph):
    e = {"toe": toe_gpst % 604800.0}
    e.update(eph)
    return {
        "sys": sysc,
        "prn": prn,
        "signal": signal,
        "toe_field": "toe",
        "eph": e,
        "toe_gpst": toe_gpst,
        "week_from": week_from,
    }


class TestEphCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_history_survives_a_restart(self):
        w = ec.Writer("misc", self.dir)
        self.assertEqual(w.put([entry(1, T0), entry(2, T0)]), 2)
        self.assertEqual(w.put([entry(1, T0)]), 0)  # nothing new, nothing rewritten
        self.assertEqual(w.put([entry(1, T0 + 7200)]), 1)
        w2 = ec.Writer("misc", self.dir)
        self.assertEqual(len(w2.entries), 3)
        self.assertEqual(glob.glob(os.path.join(self.dir, "*.tmp")), [])

    def test_a_changed_issue_replaces_the_old(self):
        w = ec.Writer("misc", self.dir)
        w.put([entry(1, T0, af0=1e-4)])
        self.assertEqual(w.put([entry(1, T0, af0=2e-4)]), 1)
        (e,) = ec.load(self.dir)
        self.assertEqual(e["eph"]["af0"], 2e-4)

    def test_retention_thins_with_age(self):
        # every 10 min for 5 days: kept all for 6 h, hourly to 3 days, daily after
        ents = [entry(7, T0 - k * 600.0) for k in range(5 * 144)]
        kept = ec.thin(ents)
        ages = sorted(T0 - e["toe_gpst"] for e in kept)
        recent = [a for a in ages if a <= ec.RECENT_S]
        hourly = [a for a in ages if ec.RECENT_S < a <= ec.HOURLY_S]
        daily = [a for a in ages if a > ec.HOURLY_S]
        self.assertEqual(len(recent), 37)
        self.assertLessEqual(len(hourly), (ec.HOURLY_S - ec.RECENT_S) / 3600.0 + 1)
        self.assertLessEqual(len(daily), 3)
        self.assertGreater(len(daily), 0)
        # and nothing past KEEP_S
        self.assertEqual(ec.thin([entry(7, T0), entry(7, T0 - ec.KEEP_S - 1.0)]), [entry(7, T0)])

    def test_a_host_placed_issue_cannot_prune_the_rest(self):
        # one issue a wrong host clock put a year ahead: the sky's newest still rules
        ents = [entry(3, T0), entry(4, T0 + 365 * 86400.0, week_from="host")]
        self.assertEqual(len(ec.thin(ents)), 2)

    def test_readers_merge_every_writer(self):
        ec.Writer("misc", self.dir).put([entry(1, T0)])
        ec.Writer("fnav", self.dir).put(
            [entry(11, T0, signal="GAL_E5AI_FNAV", sysc="E")]
        )
        with open(os.path.join(self.dir, "broken.json"), "w") as fh:
            fh.write("{not json")
        self.assertEqual(len(ec.load(self.dir)), 2)
        self.assertEqual(len(decoded_eph.expand([self.dir])), 3)
        got = decoded_eph.load_files([self.dir])
        self.assertEqual(sorted(got), [("E", 11), ("G", 1)])

    def test_the_issue_nearest_now_is_predicted(self):
        # two issues of one satellite: the nearer one must be the one evaluated
        seen = []

        def sv_pos(eph, t_sow):
            seen.append(eph["toe"])
            return (2.0e7, 0.0, 0.0)

        t_utc = T0 - 18.0 + 315964800.0 + 600.0
        ents = [
            ("G", 1, "GPS_L5_CNAV", {"toe": (T0 - 7200) % 604800}, sv_pos, T0 - 7200),
            ("G", 1, "GPS_L5_CNAV", {"toe": T0 % 604800}, sv_pos, T0),
        ]
        decoded_eph.predict_from_decoders(ents[::-1], 0.0, 0.0, 0.0, t_utc)
        decoded_eph.predict_from_decoders(ents, 0.0, 0.0, 0.0, t_utc)
        self.assertEqual(set(seen), {T0 % 604800})


if __name__ == "__main__":
    unittest.main()
