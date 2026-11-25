"""gnss_skytime: a time mark gives the anchor error, whatever the host clock reads.

python3 -m unittest test_skytime
"""

import unittest

import gnss_ephemeris as ge
import gnss_skytime as st

ROW = {"range_m": 2.25e7, "sat_clk_s": 1.2e-4}
# a time count per signal: a CNAV TOW count, an F/NAV TOW, a B-CNAV2 SOW count, a GuoWang SOW,
# a GLONASS string's start (GPS time)
COUNT = {
    "GPS_L5_CNAV": 50000,
    "GPS_L2C_CNAV": 50000,
    "GAL_E5AI_FNAV": 300000,
    "BDS_B2A_BCNAV2": 100000,
    "GW_1202_NAV": 300003,
    "GLO_L2OF": 300018,
}


def mark(signal, err_s, count=None):
    """The mark a message would make with the labels err_s late: it started at tx_sow of
    satellite time, in GPS week 2400."""
    m = {"signal": signal, "prn": 1, "tow": COUNT[signal] if count is None else count}
    t_tx = 2400 * 604800.0 + st.tx_sow(m)
    t_rx = t_tx - ROW["sat_clk_s"] + ROW["range_m"] / ge.C_LIGHT
    m["t_host"] = t_rx + err_s + st.GPS0_UNIX - st.LEAP_S
    return m


class TestSkyTime(unittest.TestCase):
    def test_recovers_the_error(self):
        for sig in st.SIGNALS:
            for err in (0.0008, -0.0137, 2.5, -3600.0, 86400.0 * 3):
                got = st.anchor_error_s(mark(sig, err), ROW)
                self.assertAlmostEqual(got, err, delta=1e-6, msg="%s %s" % (sig, err))

    def test_week_rollover(self):
        # a message sent at the end of a week, labelled in the next; a B-CNAV2 frame at the end
        # of a BDT week starts 14 s into the next GPS week
        for sig, count in (
            ("GPS_L5_CNAV", 604800 // 6),
            ("BDS_B2A_BCNAV2", 201599),
            ("GW_1202_NAV", 604797),
        ):
            got = st.anchor_error_s(mark(sig, 7.0, count), ROW)
            self.assertAlmostEqual(got, 7.0, delta=1e-6, msg=sig)

    def test_a_week_number_sets_the_week(self):
        # host three years fast: alone the count lands in the host's week; with a broadcast
        # week number (GPS week 2400 in each system's own count) every mark reads the true error
        err = 3 * 365.25 * 86400.0
        for sig, wn in (
            ("GPS_L5_CNAV", 2400),
            ("GAL_E5AI_FNAV", 2400 - 1024),
            ("BDS_B2A_BCNAV2", 2400 - 1356),
            ("GW_1202_NAV", 2400 - 1356),
        ):
            m = mark(sig, err)
            w = dict(mark(sig, err, COUNT[sig] + 2), wn=wn)
            self.assertNotAlmostEqual(st.anchor_error_s(m, ROW), err, delta=1.0)
            g_ref = st.week_ref([m, w])
            self.assertAlmostEqual(
                st.anchor_error_s(m, ROW, g_ref), err, delta=1e-6, msg=sig
            )


if __name__ == "__main__":
    unittest.main()
