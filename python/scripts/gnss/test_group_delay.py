#!/usr/bin/env python3
"""The broadcast group-delay term (gnss_ephemeris.group_delay_s), with the controls that make
it a test rather than a restatement of the implementation.

Why each check exists:
  * Sign and scale against a hand-worked ICD number: the term is one multiply, so the only
    ways to get it wrong are the sign and the gamma, and both are silent.
  * The cross-type conversions. The BRDC carries both Galileo nav types and best_eph picks by
    toe, so a satellite's clock reference can change mid-run. E1's clock must be the same
    however it is reached; if it is not, a refresh steps that satellite.
  * The negative control. signal=None must return exactly 0.0: every non-broker caller (TEC,
    observables, the sky map) relies on that to keep the broadcast clock.
  * B2a must return zero from the broadcast path, as a requirement rather than an omission:
    TGD_B2ap lives in B-CNAV2 and RINEX 3 does not carry it. Borrowing TGD2 (a different
    band) would be a plausible-looking wrong answer, which is worse than none.
"""
import builtins
import gzip
import io
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_ephemeris import group_delay_s, GAMMA_L1L2, GAMMA_E1E5A, GAMMA_E1E5B

NS = 1e-9


def _g(tgd=0.0, tgd2=0.0, ds=0):
    return dict(sys="G", tgd=tgd, iodc=tgd2, l2_codes=ds)


def _e(bgd_a=0.0, bgd_b=0.0, fnav=True):
    return dict(sys="E", tgd=bgd_a, iodc=bgd_b, l2_codes=(0x100 if fnav else 0x200))


def _c(tgd1=0.0, tgd2=0.0):
    return dict(sys="C", tgd=tgd1, iodc=tgd2, l2_codes=0)


class TestGPS(unittest.TestCase):
    def test_l5_is_the_icd_term_without_isc(self):
        """IS-GPS-705: an L5 user's clock is dt_SV - TGD + ISC_L5I5. ISC_L5I5 is not in
        RINEX 3, so without a DCB pair the term is -TGD, unscaled: a TGD of -8.38 ns gives
        +8.38 ns. The 1/f^2 scaling (gamma_15 * TGD, +15.03 ns) is not the ICD's form, and a
        sign error moves the seeds the wrong way by twice the term."""
        self.assertAlmostEqual(
            group_delay_s(_g(tgd=-8.38 * NS), "gps_l5"), +8.38 * NS, places=15
        )

    def test_l2c_is_the_l2p_term(self):
        """TGD is defined on L1P/L2P, so the L2P term -gamma_12 * TGD is exact there, and L2C
        shares L2P's carrier. A TGD of -8.38 ns gives +13.80 ns; -TGD would give +8.38 ns and
        no term 0."""
        self.assertAlmostEqual(
            group_delay_s(_g(tgd=-8.38 * NS), "gps_l2c"),
            +GAMMA_L1L2 * 8.38 * NS,
            places=15,
        )
        self.assertAlmostEqual(GAMMA_L1L2, 1.64694, places=4)

    def test_l1_is_the_unscaled_term(self):
        self.assertAlmostEqual(
            group_delay_s(_g(tgd=-8.38 * NS), "gps_l1"), +8.38 * NS, places=15
        )

    def test_iodc_is_not_mistaken_for_a_second_delay(self):
        """orb[23] is IODC for GPS, not a group delay; the slot is shared with Galileo's
        BGD_E5b and BeiDou's TGD2. An IODC of 7, read as seconds, must not reach the clock."""
        self.assertEqual(group_delay_s(_g(tgd=0.0, tgd2=7.0), "gps_l5"), 0.0)


class TestGalileo(unittest.TestCase):
    def test_fnav_e5a_and_inav_e5b_are_the_native_cases(self):
        self.assertAlmostEqual(
            group_delay_s(_e(bgd_a=3.0 * NS, fnav=True), "gal_e5a"),
            -GAMMA_E1E5A * 3.0 * NS,
            places=15,
        )
        self.assertAlmostEqual(
            group_delay_s(_e(bgd_b=3.0 * NS, fnav=False), "gal_e5b"),
            -GAMMA_E1E5B * 3.0 * NS,
            places=15,
        )

    def test_e1_clock_is_reference_invariant(self):
        """The identity the cross-type conversions exist for: t_E1 = t_IF_a - BGD_a =
        t_IF_b - BGD_b, so an E1 user must land on the same clock from either record type.
        The two broadcast clocks differ by (BGD_a - BGD_b), so that offset is removed from
        the I/NAV side before comparing. This is what stops an F/NAV <-> I/NAV change at a
        refresh from stepping a satellite."""
        a, b = 3.03 * NS, 3.26 * NS
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e1")
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e1") - (a - b)
        self.assertAlmostEqual(from_fnav, from_inav, places=15)

    def test_e5a_agrees_across_record_types(self):
        """Same identity for the signal we actually track. Reaching E5a from an I/NAV
        record must equal reaching it from an F/NAV record, once the two records' own
        clock offset (BGD_a - BGD_b) is accounted for."""
        a, b = 3.03 * NS, 3.26 * NS
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e5a")
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e5a") - (a - b)
        self.assertAlmostEqual(from_fnav, from_inav, places=15)

    def test_e5b_agrees_across_record_types(self):
        a, b = 3.03 * NS, 3.26 * NS
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e5b") - (a - b)
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e5b")
        self.assertAlmostEqual(from_inav, from_fnav, places=15)

    def test_unflagged_record_defaults_to_fnav(self):
        e = dict(sys="E", tgd=3.0 * NS, iodc=0.0, l2_codes=0)
        self.assertAlmostEqual(
            group_delay_s(e, "gal_e5a"), -GAMMA_E1E5A * 3.0 * NS, places=15
        )


class TestBeiDou(unittest.TestCase):
    def test_b2a_is_zero_because_the_term_is_not_broadcast(self):
        """Not an omission: TGD_B2ap is a B-CNAV2 parameter and RINEX 3 carries only TGD1
        (B1I/B3I) and TGD2 (B2I/B3I). Borrowing TGD2 would apply a 1207 MHz delay to an
        1176 MHz signal; absent is better than plausibly wrong."""
        self.assertEqual(
            group_delay_s(_c(tgd1=-4.3 * NS, tgd2=-4.3 * NS), "bds_b2a"), 0.0
        )

    def test_b2b_uses_tgd2(self):
        self.assertAlmostEqual(
            group_delay_s(_c(tgd2=-4.3 * NS), "bds_b2b"), +4.3 * NS, places=15
        )

    def test_tgd1_is_b1i_only(self):
        """TGD1 is B1I's term. B1C needs TGD_B1Cp from B-CNAV1, which RINEX 3 does not
        carry, so it gets 0 like B2a rather than B1I's term."""
        self.assertEqual(group_delay_s(_c(tgd1=-10 * NS), "bds_b1c"), 0.0)
        self.assertAlmostEqual(
            group_delay_s(_c(tgd1=-10 * NS), "bds_b1i"), +10 * NS, places=15
        )


class TestNegativeControls(unittest.TestCase):
    def test_no_signal_is_exactly_zero(self):
        """Every non-broker caller passes no signal and must get the broadcast clock as is."""
        for e in (_g(tgd=-8.4 * NS), _e(3 * NS, 3 * NS), _c(-4.3 * NS, -4.3 * NS)):
            self.assertEqual(group_delay_s(e, None), 0.0)
            self.assertEqual(group_delay_s(e, ""), 0.0)

    def test_unknown_signal_is_zero_not_a_guess(self):
        self.assertEqual(group_delay_s(_e(3 * NS), "gal_e6"), 0.0)

    def test_zero_tgd_is_zero_correction(self):
        self.assertEqual(group_delay_s(_g(tgd=0.0), "gps_l5"), 0.0)

    def test_magnitude_stays_in_the_measured_band(self):
        """A guard against a units slip: broadcast group delays span about -45..+47 ns, so
        no correction may exceed ~100 ns (~1 chip at 10.23 Mcps). A ppm/seconds mix-up would
        blow straight past it."""
        for e, sig in (
            (_g(tgd=-45 * NS), "gps_l5"),
            (_g(tgd=-45 * NS), "gps_l2c"),
            (_e(-16 * NS, 4 * NS), "gal_e5a"),
            (_e(-16 * NS, 4 * NS, fnav=False), "gal_e5b"),
            (_c(tgd2=47 * NS), "bds_b2b"),
        ):
            self.assertLess(
                abs(group_delay_s(e, sig)),
                100 * NS,
                "%s correction implausibly large -- units?" % sig,
            )


class TestDcbSource(unittest.TestCase):
    """The DCB path (gnss_dcb) as a source for group_delay_s: it must override the broadcast
    term when present, fall back per satellite when not, and stay continuous across a Galileo
    record-type change, the same invariance the broadcast path is tested for above."""

    def setUp(self):
        import gnss_dcb

        self.m = gnss_dcb
        # b_C1C - b_C5Q = -4 ns, b_C1C - b_C7Q = -6 ns, b_C1P - b_C5P = +3, b_C1P - b_C6I = -2
        self.tab = {
            ("E", 7): {("C1C", "C5Q"): -4e-9, ("C1C", "C7Q"): -6e-9},
            ("C", 26): {("C1P", "C5P"): 3e-9, ("C1P", "C6I"): -2e-9},
        }

    def test_bds_b2a_is_reachable_where_broadcast_is_not(self):
        """B2a has no broadcast term, so the DCB path is its only source.
        b_C6I - b_C5P = DSB(C1P,C5P) - DSB(C1P,C6I)."""
        v = self.m.signal_bias_s(self.tab, "C", 26, "bds_b2a")
        self.assertAlmostEqual(v, 5e-9, places=15)
        self.assertEqual(
            group_delay_s(_c(tgd1=-4.3 * NS, tgd2=-4.3 * NS), "bds_b2a"),
            0.0,
            "broadcast path must still be zero for B2a",
        )

    def test_gps_l5_takes_an_exact_zero_dsb(self):
        """A DSB of exactly 0.0 is a measurement, not a missing pair: with no C5Q to fall
        back on, it must still give the L5 correction."""
        tab = {
            ("G", 1): {("C1C", "C5X"): 0.0, ("C1C", "C1W"): 1e-9, ("C1W", "C2W"): 2e-9}
        }
        v = self.m.signal_bias_s(tab, "G", 1, "gps_l5")
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v, -1e-9 + 2e-9 / (self.m.GAMMA_L1L2 - 1.0), places=18)

    def test_gps_l2c_reaches_c2l_through_c2w(self):
        """The datum is L1P/L2P IF and L2C is C2L:
        b_IF - b_C2L = gamma_12 * DSB(C1W,C2W) / (gamma_12 - 1) + DSB(C2W,C2L). Without the
        C2W-C2L pair the satellite falls back to the broadcast -gamma_12 * TGD."""
        g = self.m.GAMMA_L1L2
        tab = {("G", 1): {("C1W", "C2W"): 2e-9, ("C2W", "C2L"): 1e-9}}
        v = self.m.signal_bias_s(tab, "G", 1, "gps_l2c")
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v, g * 2e-9 / (g - 1.0) + 1e-9, places=18)
        e = dict(sys="G", prn=1, tgd=-8.38 * NS, iodc=0.0, l2_codes=0)
        self.assertAlmostEqual(group_delay_s(e, "gps_l2c", tab), v, places=18)
        part = {("G", 1): {("C1W", "C2W"): 2e-9}}
        self.assertIsNone(self.m.signal_bias_s(part, "G", 1, "gps_l2c"))
        self.assertAlmostEqual(
            group_delay_s(e, "gps_l2c", part), +g * 8.38 * NS, places=18
        )

    def test_gps_l2c_broadcast_term_tracks_the_dcb_path(self):
        """Sign and scale of both L2C paths against each other, on real numbers: DSB(C1W,C2W),
        DSB(C2W,C2L) from CAS0OPSRAP_20262880000 and TGD from BRDC00WRD_S day 292, for ten
        satellites. The DCB product is zero-mean and TGD is not, so the constellation mean is
        removed before comparing. -gamma_12 * TGD fits to 1.1 ns rms; -TGD leaves 5.6 ns, no
        term 13 ns and a flipped sign 26 ns."""
        rows = {  # prn: (DSB(C1W,C2W), DSB(C2W,C2L), TGD), ns
            1: (2.044, 0.508, -8.848),
            3: (-4.869, -0.756, 1.397),
            5: (3.647, -0.486, -10.245),
            6: (-6.211, -1.218, 3.725),
            8: (-6.956, -1.535, 4.657),
            10: (-5.397, 1.354, 2.328),
            12: (4.526, -0.604, -12.107),
            25: (-7.364, -1.106, 5.588),
            26: (-8.079, 0.589, 6.519),
            31: (4.762, -0.371, -13.504),
        }
        dcb = {
            ("G", p): {("C1W", "C2W"): a * NS, ("C2W", "C2L"): b * NS}
            for p, (a, b, _) in rows.items()
        }
        diff = []
        for p, (_, _, tgd) in rows.items():
            e = dict(sys="G", prn=p, tgd=tgd * NS, iodc=0.0, l2_codes=0)
            measured = group_delay_s(e, "gps_l2c", dcb)
            self.assertNotEqual(measured, group_delay_s(e, "gps_l2c"))
            diff.append(measured - group_delay_s(e, "gps_l2c"))
        mean = sum(diff) / len(diff)
        rms = (sum((d - mean) ** 2 for d in diff) / len(diff)) ** 0.5
        self.assertLess(rms, 1.5 * NS, "L2C DCB and broadcast paths disagree")

    def test_dcb_overrides_the_broadcast_term(self):
        e = dict(sys="C", prn=26, tgd=0.0, iodc=-4.3 * NS, l2_codes=0)
        self.assertAlmostEqual(group_delay_s(e, "bds_b2a", self.tab), 5e-9, places=15)

    def test_missing_satellite_falls_back_to_broadcast(self):
        """A partial product must degrade per-satellite, not all-or-nothing."""
        e = dict(sys="C", prn=99, tgd=0.0, iodc=-4.3 * NS, l2_codes=0)
        self.assertAlmostEqual(
            group_delay_s(e, "bds_b2b", self.tab), +4.3 * NS, places=15
        )

    def test_galileo_agrees_across_record_types(self):
        """Same invariance as the broadcast path: reaching E5a from an I/NAV record must
        equal reaching it from an F/NAV one. With full DSBs there is no approximation, so
        this should hold exactly rather than to a few ns."""
        fnav = self.m.signal_bias_s(self.tab, "E", 7, "gal_e5a", gal_inav=False)
        inav = self.m.signal_bias_s(self.tab, "E", 7, "gal_e5a", gal_inav=True)
        d15, d17 = -4e-9, -6e-9
        # the two datums differ by b_IF(1,5) - b_IF(1,7); removing it must reconcile them
        offset = GAMMA_E1E5A * d15 / (GAMMA_E1E5A - 1.0) - (
            d17 / (GAMMA_E1E5B - 1.0) + d15
        )
        self.assertAlmostEqual(fnav - inav, offset, places=18)

    def test_zero_mean_is_a_property_of_the_product(self):
        """Guards the zero-mean claim in --dcb-bias's help and gnss_dcb's docstring. If a
        product drops the zero-mean condition this fails, and the statement that the product
        cannot explain a common offset must be rewritten."""
        import os

        p = self.m.fetch_dcb()
        if not p or not os.path.exists(p):
            self.skipTest("no DCB product available (no token/network)")
        tab = self.m.parse_dcb(p)
        v = [
            self.m.signal_bias_s(tab, "C", k[1], "bds_b2a") for k in tab if k[0] == "C"
        ]
        v = [x for x in v if x is not None]
        self.assertGreater(len(v), 10)
        self.assertLess(
            abs(sum(v) / len(v)) * 1e9,
            0.5,
            "constellation mean is no longer ~0 -- the zero-mean condition "
            "changed and the 'per-sat only' caveat must be revisited",
        )


class TestDcbStaleness(unittest.TestCase):
    """A dead token is invisible because fetch_dcb serves the cache. These pin the three
    things that make it visible: the product's own epoch, the token's expiry, and a reason
    code that tells a missing credential from an unreachable network."""

    def setUp(self):
        import gnss_dcb

        self.m = gnss_dcb

    def test_product_age_comes_from_the_filename_not_the_mtime(self):
        from datetime import datetime, timezone

        when = datetime(2026, 9, 10, tzinfo=timezone.utc)  # doy 253
        a = self.m.product_age_days(
            "/x/CAS0OPSRAP_20262490000_01D_01D_DCB.BIA.gz", when
        )
        self.assertAlmostEqual(a, 4.0, places=6)
        b = self.m.product_age_days(
            "/x/CAS0MGXRAP_20262390000_01D_01D_DCB.BSX.gz", when
        )
        self.assertAlmostEqual(b, 14.0, places=6)
        self.assertIsNone(self.m.product_age_days("/x/not-a-product.gz", when))
        self.assertIsNone(self.m.product_age_days(None, when))

    def test_token_expiry_reads_a_jwt_and_declines_anything_else(self):
        import base64, json, time

        exp = time.time() + 30 * 86400
        pl = (
            base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
            .decode()
            .rstrip("=")
        )
        self.assertAlmostEqual(
            self.m.token_expiry_days("h.%s.sig" % pl), 30.0, places=2
        )
        self.assertIsNone(
            self.m.token_expiry_days("an-opaque-token")
        )  # not a JWT, not an error
        self.assertIsNone(self.m.token_expiry_days(""))

    def test_no_token_is_reported_as_such_not_as_unreachable(self):
        st = {}
        real = self.m._token
        self.m._token = lambda: None
        try:
            self.assertIsNone(self.m.fetch_dcb(status=st))
        finally:
            self.m._token = real
        self.assertEqual(st["reason"], "no-token")
        self.assertIsNone(st["path"])


class TestDcbFetch(unittest.TestCase):
    """What fetch_dcb puts in the cache, with the network faked: only the gzip product may be
    cached or served, and concurrent fetches of the same day must not share a temp file."""

    WHEN = datetime(2026, 9, 10, tzinfo=timezone.utc)

    def setUp(self):
        import gnss_dcb

        self.m = gnss_dcb
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        p = mock.patch.object(gnss_dcb, "_token", lambda: "x")
        p.start()
        self.addCleanup(p.stop)

    def _name(self, back):
        d = self.WHEN - timedelta(days=back)
        return "CAS0OPSRAP_%04d%03d0000_01D_01D_DCB.BIA.gz" % (
            d.year,
            d.timetuple().tm_yday,
        )

    def _fetch(self, urlopen, st):
        with mock.patch("urllib.request.urlopen", urlopen):
            return self.m.fetch_dcb(
                when=self.WHEN, cache_dir=self.dir, max_back_days=3, status=st
            )

    def test_a_non_gzip_200_is_not_cached(self):
        """A login or error page answered with 200 must not land under the product's name."""
        st = {}
        page = b"<html>" + b" " * 2000 + b"</html>"
        self.assertIsNone(self._fetch(lambda req, timeout: io.BytesIO(page), st))
        self.assertEqual(os.listdir(self.dir), [])
        self.assertEqual(st["reason"], "not-gzip")

    def test_a_cached_non_gzip_file_is_dropped_not_served(self):
        with open(os.path.join(self.dir, self._name(1)), "wb") as f:
            f.write(b"<html>" + b" " * 2000)
        good = os.path.join(self.dir, self._name(2))
        with open(good, "wb") as f:
            f.write(gzip.compress(os.urandom(2000)))

        def not_found(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

        st = {}
        self.assertEqual(self._fetch(not_found, st), good)
        self.assertFalse(os.path.exists(os.path.join(self.dir, self._name(1))))
        self.assertEqual(st["reason"], "ok")

    def test_concurrent_fetches_write_through_distinct_temp_files(self):
        """Two reload threads fetching the same day must not write one temp file: the second
        open truncates the file the first is about to publish."""
        payload = gzip.compress(os.urandom(2000))
        both_fetched = threading.Barrier(2, timeout=10)

        def urlopen(req, timeout):
            if not req.full_url.endswith(self._name(1)):
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            both_fetched.wait()  # both threads are past the cache check for day 1
            return io.BytesIO(payload)

        written = []
        real_open = builtins.open

        def recording_open(path, mode="r", *a, **kw):
            if "w" in mode and str(path).startswith(self.dir):
                written.append(str(path))
            return real_open(path, mode, *a, **kw)

        def fetch():
            self.m.fetch_dcb(when=self.WHEN, cache_dir=self.dir, max_back_days=3)

        # Patched once, here: a patch entered per thread would be undone by the first thread
        # to finish while the other still runs.
        with mock.patch("urllib.request.urlopen", urlopen), mock.patch(
            "builtins.open", recording_open
        ):
            th = [threading.Thread(target=fetch) for _ in range(2)]
            for t in th:
                t.start()
            for t in th:
                t.join()
        local = os.path.join(self.dir, self._name(1))
        self.assertEqual(len(written), 2)
        self.assertNotEqual(written[0], written[1], "the fetches shared a temp file")
        self.assertNotIn(local, written)
        with open(local, "rb") as f:
            self.assertEqual(f.read(), payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
