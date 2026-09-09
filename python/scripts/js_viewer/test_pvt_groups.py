"""_pvt_measurements must name a group by its CARRIER, not by a substring of the chain name.

The substring version classified every CHORD row as L1 (case-sensitive "L5" against "gps_l5"),
which mislabelled the panel and would have put L2C and L5 on one clock.
"""
import json, os, sys, tempfile, time, unittest
sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/js_viewer")
sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")
import livebeam_server as lb

BANDS = [("gps_l5", "G", 1176.45e6), ("gal_e5a", "E", 1176.45e6), ("bds_b2a", "C", 1176.45e6),
         ("gal_e5b", "E", 1207.14e6), ("bds_b2b", "C", 1207.14e6), ("bds_b3i", "C", 1268.52e6),
         ("gal_e6", "E", 1278.75e6), ("gps_l2c", "G", 1227.60e6)]

class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        now = time.time()
        for band, sysid, f in BANDS:
            with open(os.path.join(self.d, "%s_x.jsonl" % band), "w") as fh:
                for prn, res in ((1, 10.0), (2, -5.0), (3, 2.0)):
                    fh.write(json.dumps({"t": now, "sys": sysid, "prn": prn, "band": band,
                                         "carrier_hz": f, "code_resid_m": res,
                                         "az": 30.0 * prn, "el": 40.0,
                                         "code_len": 10230, "chip_rate_hz": 10.23e6}) + "\n")

    def test_each_chain_is_its_own_group(self):
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time())
        groups = {x["group"] for x in m}
        for band, _s, _f in BANDS:
            self.assertIn(band, groups, "chain %s lost its own clock group; got %s" % (band, sorted(groups)))
        self.assertNotIn("G-L1", groups, "L5 data labelled L1 -- the case-sensitive substring bug")

    def test_l2c_does_not_share_a_clock_with_l5(self):
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time())
        g = {x["group"] for x in m}
        self.assertTrue({"gps_l5", "gps_l2c"} <= g, "GPS L5 and L2C must be separate groups")

    def test_iono_free_is_off_by_default(self):
        """The co-hosted splits here are 51-102 MHz: the combination amplifies code noise
        8-17x to remove a few metres of ionosphere, so it is opt-in."""
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time())
        self.assertFalse(any(x["group"].endswith("-IF") for x in m))

    def test_iono_free_pairs_form_on_the_widest_split(self):
        m = lb._pvt_measurements([os.path.join(self.d, "*.jsonl")], 300.0, time.time(),
                                 iono_free=True)
        # Galileo has 1176.45 / 1207.14 / 1278.75 -> the widest split is E5a x E6
        self.assertTrue(any(x["group"] == "E-IF" for x in m), "no Galileo iono-free rows formed")
        self.assertTrue(any(x["group"] == "C-IF" for x in m), "no BeiDou iono-free rows formed")

if __name__ == "__main__":
    unittest.main(verbosity=2)
