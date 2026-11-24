"""gnss_skytime: a CNAV time mark gives the anchor error, whatever the host clock reads.

python3 -m unittest test_skytime
"""

import unittest

import gnss_ephemeris as ge
import gnss_skytime as st

ROW = {"range_m": 2.25e7, "sat_clk_s": 1.2e-4}


def mark(signal, err_s, tow=50000):
    """The mark a message would make with the labels err_s late: it started (TOW * 6 - length)
    of satellite time, in GPS week 2400."""
    t_tx = 2400 * 604800.0 + tow * 6.0 - st.MSG_S[signal]
    t_rx = t_tx - ROW["sat_clk_s"] + ROW["range_m"] / ge.C_LIGHT
    t_host = t_rx + err_s + st.GPS0_UNIX - st.LEAP_S
    return {"signal": signal, "prn": 1, "t_host": t_host, "tow": tow}


class TestSkyTime(unittest.TestCase):
    def test_recovers_the_error(self):
        for sig in st.MSG_S:
            for err in (0.0008, -0.0137, 2.5, -3600.0, 86400.0 * 3):
                got = st.anchor_error_s(mark(sig, err), ROW)
                self.assertAlmostEqual(got, err, delta=1e-6, msg="%s %s" % (sig, err))

    def test_week_rollover(self):
        # a message sent at the end of a week, labelled in the next
        tow = 604800 // 6
        self.assertAlmostEqual(
            st.anchor_error_s(mark("GPS_L5_CNAV", 7.0, tow), ROW), 7.0, delta=1e-6
        )

    def test_a_type_10_sets_the_week(self):
        # host three years fast: alone the TOW lands in the host's week; with a type 10's week
        # number every mark reads the true error
        err = 3 * 365.25 * 86400.0
        m = mark("GPS_L5_CNAV", err)
        t10 = dict(mark("GPS_L5_CNAV", err, tow=50002), type=10, wn=2400)
        self.assertNotAlmostEqual(st.anchor_error_s(m, ROW), err, delta=1.0)
        g_ref = st.week_ref([m, t10])
        self.assertAlmostEqual(st.anchor_error_s(m, ROW, g_ref), err, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
