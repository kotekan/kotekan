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
from gnss_ephemeris import (  # noqa: E402
    C_LIGHT,
    GAMMA_E1E5A,
    GAMMA_E1E5B,
    GAMMA_L1L5,
    GLO_L3_BIAS_M,
    GLO_SLOT_SVN,
    glonass_l3_bias_s,
    glonass_svn,
    gpst_of_utc,
    group_delay_s,
)

NS = 1e-9


def _g(tgd=0.0, tgd2=0.0, ds=0):
    return dict(sys="G", tgd=tgd, iodc=tgd2, l2_codes=ds)


def _e(bgd_a=0.0, bgd_b=0.0, fnav=True):
    return dict(sys="E", tgd=bgd_a, iodc=bgd_b, l2_codes=(0x100 if fnav else 0x200))


def _c(tgd1=0.0, tgd2=0.0):
    return dict(sys="C", tgd=tgd1, iodc=tgd2, l2_codes=0)


def _r(slot, when="2026-10-19T00:00:00"):
    """A GLONASS record for `slot` with its epoch tb at `when` (UTC ISO)."""
    return dict(sys="R", prn=slot, toe_gpst=gpst_of_utc(datetime.fromisoformat(when).replace(tzinfo=timezone.utc)))


class TestGPS(unittest.TestCase):
    def test_l5_scales_by_gamma_and_flips_sign(self):
        """IS-GPS-705: an L5 user owes gamma_15 * TGD, subtracted from the clock. A typical
        TGD of -8.38 ns gives a positive correction of ~+15 ns; a sign error moves the seeds
        the wrong way by twice the term."""
        self.assertAlmostEqual(
            group_delay_s(_g(tgd=-8.38 * NS), "gps_l5"),
            +GAMMA_L1L5 * 8.38 * NS,
            places=15,
        )
        self.assertAlmostEqual(GAMMA_L1L5, 1.79329, places=4)

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
        self.assertEqual(group_delay_s(_g(tgd=-8.4 * NS), "gps_l2c"), 0.0)
        self.assertEqual(group_delay_s(_e(3 * NS), "gal_e6"), 0.0)

    def test_zero_tgd_is_zero_correction(self):
        self.assertEqual(group_delay_s(_g(tgd=0.0), "gps_l5"), 0.0)

    def test_magnitude_stays_in_the_measured_band(self):
        """A guard against a units slip: broadcast group delays span about -45..+47 ns, so
        no correction may exceed ~100 ns (~1 chip at 10.23 Mcps). A ppm/seconds mix-up would
        blow straight past it."""
        for e, sig in (
            (_g(tgd=-45 * NS), "gps_l5"),
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


class TestGLONASSL3(unittest.TestCase):
    """The L3OC per-satellite table (#210): a measured product, so the tests pin the sign, the
    routing by slot and epoch, and the controls, not the numbers."""

    def test_sign_late_code_negative_term(self):
        """A satellite whose L3OC code arrives LATE (P - model positive, the K1 three) owes a
        NEGATIVE clock term, an early one (the K2 pair) a positive term, of -value/c exactly:
        R19 is SVN 807 in 2026."""
        self.assertAlmostEqual(
            group_delay_s(_r(19), "glo_l3oc"), -GLO_L3_BIAS_M[807] / C_LIGHT, places=15
        )
        self.assertLess(group_delay_s(_r(19), "glo_l3oc"), -100 * NS)
        self.assertGreater(group_delay_s(_r(6), "glo_l3oc"), +400 * NS)

    def test_magnitudes_are_lobe_sized(self):
        """The point of the table: 1.3-4.7 chips at 10.23 Mcps, past the sparse comb's 95.9 m
        lobe for the K2 pair. A value that drifted under 30 m would no longer be the term
        that parks a tracker on a grating lobe, and one over 200 m is not what was measured."""
        for svn, m in GLO_L3_BIAS_M.items():
            self.assertTrue(30.0 < abs(m) < 200.0, (svn, m))
        self.assertTrue(all(m < -100.0 for svn, m in GLO_L3_BIAS_M.items() if svn in (803, 804)))
        self.assertTrue(all(m > 30.0 for svn, m in GLO_L3_BIAS_M.items() if svn in (807, 808, 813)))

    def test_slot_reassignment_follows_the_satellite(self):
        """SVN 803 moved from slot R26 to R08 on 2026-09-25 (IGS satellite metadata): before
        that R08 is at the datum and R26 carries the term; after it the reverse. Keying by
        slot alone would have put a K2 term on whatever flies in R08 next."""
        before, after = "2026-09-01T00:00:00", "2026-10-01T00:00:00"
        self.assertEqual(glonass_svn(8, gpst_of_utc(datetime.fromisoformat(before).replace(tzinfo=timezone.utc))), None)
        self.assertEqual(glonass_svn(8, gpst_of_utc(datetime.fromisoformat(after).replace(tzinfo=timezone.utc))), 803)
        self.assertEqual(group_delay_s(_r(8, before), "glo_l3oc"), 0.0)
        self.assertAlmostEqual(group_delay_s(_r(26, before), "glo_l3oc"), -GLO_L3_BIAS_M[803] / C_LIGHT, places=15)
        self.assertAlmostEqual(group_delay_s(_r(8, after), "glo_l3oc"), -GLO_L3_BIAS_M[803] / C_LIGHT, places=15)
        self.assertEqual(group_delay_s(_r(26, after), "glo_l3oc"), 0.0)

    def test_history_is_unambiguous(self):
        """No slot maps to two SVNs at once, and every SVN with a bias has a slot entry: an
        overlap or an orphan would be silent (the lookup returns the first match / 0.0)."""
        for i, (s1, v1, a1, b1) in enumerate(GLO_SLOT_SVN):
            for s2, v2, a2, b2 in GLO_SLOT_SVN[i + 1:]:
                if s1 != s2:
                    continue
                lo, hi = max(a1, a2), min(b1 or "9999", b2 or "9999")
                self.assertGreaterEqual(lo, hi, (s1, v1, v2))
        self.assertEqual(set(GLO_L3_BIAS_M), {v for _, v, _, _ in GLO_SLOT_SVN})

    def test_controls(self):
        """Off the table (slot R04, GLO-M+) is 0.0; GLONASS L2OC is 0.0 (its clock is the FDMA
        one it is referred to); signal=None is exactly 0.0; a loaded DCB table changes nothing
        for GLONASS (CAS publishes no C3 lines) and GPS L5 is untouched by the R branch."""
        self.assertEqual(group_delay_s(_r(4), "glo_l3oc"), 0.0)
        self.assertEqual(glonass_l3_bias_s(4, _r(4)["toe_gpst"]), 0.0)
        self.assertEqual(group_delay_s(_r(19), "glo_l2oc"), 0.0)
        self.assertEqual(group_delay_s(_r(19), None), 0.0)
        self.assertEqual(group_delay_s(_r(19), "glo_l3oc", dcb={("R", 19): {}}), group_delay_s(_r(19), "glo_l3oc"))
        self.assertAlmostEqual(group_delay_s(_g(tgd=-8.38 * NS), "gps_l5"), -GAMMA_L1L5 * -8.38 * NS, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
