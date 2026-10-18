"""_pvt_measurements must give each chain its own clock group, named by the obs row's band.

gnss_pvt.solve fits one receiver clock per group, and chains on different signals do not
share a clock offset: GPS L5 and L2C must land in separate groups, and no chain may be
relabelled from a substring of its name (an L5 row read as "G-L1").
"""
import json, os, sys, tempfile, time, unittest

sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/js_viewer")
sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")
import livebeam_server as lb

BANDS = [
    ("gps_l5", "G", 1176.45e6),
    ("gal_e5a", "E", 1176.45e6),
    ("bds_b2a", "C", 1176.45e6),
    ("gal_e5b", "E", 1207.14e6),
    ("bds_b2b", "C", 1207.14e6),
    ("bds_b3i", "C", 1268.52e6),
    ("gal_e6", "E", 1278.75e6),
    ("gps_l2c", "G", 1227.60e6),
]


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        now = time.time()
        for band, sysid, f in BANDS:
            with open(os.path.join(self.d, "%s_x.jsonl" % band), "w") as fh:
                for prn, res in ((1, 10.0), (2, -5.0), (3, 2.0)):
                    fh.write(
                        json.dumps(
                            {
                                "t": now,
                                "sys": sysid,
                                "prn": prn,
                                "band": band,
                                "carrier_hz": f,
                                "code_resid_m": res,
                                "az": 30.0 * prn,
                                "el": 40.0,
                                "code_len": 10230,
                                "chip_rate_hz": 10.23e6,
                            }
                        )
                        + "\n"
                    )

    def test_each_chain_is_its_own_group(self):
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time())
        groups = {x["group"] for x in m}
        for band, _s, _f in BANDS:
            self.assertIn(
                band,
                groups,
                "chain %s lost its own clock group; got %s" % (band, sorted(groups)),
            )
        self.assertNotIn(
            "G-L1",
            groups,
            "L5 data labelled L1 -- a group named from a substring of the chain name",
        )

    def test_l2c_does_not_share_a_clock_with_l5(self):
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time())
        g = {x["group"] for x in m}
        self.assertTrue(
            {"gps_l5", "gps_l2c"} <= g, "GPS L5 and L2C must be separate groups"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
