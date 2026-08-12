#!/usr/bin/env python3
"""Pinning tests for track_vs_fit_chips (#42 / #45 step 1).

The retracted formula is reproduced here VERBATIM as `old_cp_err` so the discriminating
pair is permanent: on a pure bias-EMA step the old formula reads ~1700 chips per Hz and
the new one reads ~0; on a genuine lobe park both read +-3.27. If someone reintroduces a
currency translation into this comparator, test_bias_step_reads_zero is the tripwire.

Run: python3 test_track_vs_fit.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.fits import dr_cp0, dr_seed_phys, track_vs_fit_chips  # noqa: E402

HPS = 195312.5
CHIP = 10.23e6
CARR = 1176.45e6
SGN = 1.0
L = 10230.0
MOD = 20 * L                       # seeds live mod the long code
T0 = 195000.0                      # ~2.26 days of F-engine age -- the lever regime
H0 = int(round(T0 * HPS))


def phys_at(h, phys0, dop, h_ref):
    """True physical code phase at hop h for a trajectory pinned at (h_ref, phys0)."""
    dt = (h - h_ref) / HPS
    return (phys0 + dt * (CHIP + SGN * CHIP * dop / CARR)) % MOD


def make_held(phys0, dop_label, h_ref, rate=0.0, drate=0.0):
    """A held tuple whose cp0 is built self-consistently WITH its dop label."""
    return {"code_phase_chips": dr_cp0(phys0, h_ref / HPS, dop_label, CHIP, CARR,
                                       SGN, MOD),
            "doppler_hz": dop_label, "code_phase_rate": rate,
            "ref_hop": h_ref, "doppler_rate_hz_s": drate}


def old_cp_err(cand_seed, prev_seed, trim):
    """The RETRACTED sample-0 currency comparison, verbatim (broker, pre-2026-08-12)."""
    h_now = cand_seed["ref_hop"]
    cp_prev = (prev_seed["code_phase_chips"]
               + prev_seed["code_phase_rate"] * (h_now - prev_seed["ref_hop"]))
    dt_anchor = (h_now - prev_seed["ref_hop"]) / HPS
    cp_prev += (0.5 * SGN * float(prev_seed.get("doppler_rate_hz_s", 0.0) or 0.0)
                * CHIP / CARR * dt_anchor * dt_anchor)
    t_abs = h_now / HPS
    cp_prev += (t_abs * CHIP * SGN
                * (prev_seed["doppler_hz"] - cand_seed["doppler_hz"]) / CARR)
    return ((cand_seed["code_phase_chips"] - cp_prev - trim + L / 2.0) % L) - L / 2.0


class TestTrackVsFit(unittest.TestCase):

    def test_consistent_world_reads_zero(self):
        """Held on the true trajectory, detection measures the same sky -> ~0."""
        dop = 1234.5
        held = make_held(5000.0, dop, H0)
        h_det = H0 + int(7.5 * HPS)
        det_at_ref = phys_at(h_det, 5000.0, dop, H0)
        err = track_vs_fit_chips(held, det_at_ref, h_det, 0.0,
                                 HPS, CHIP, CARR, SGN, L)
        self.assertLess(abs(err), 1e-3)

    def test_bias_step_reads_zero_where_the_old_formula_read_1700_per_Hz(self):
        """THE #42 MECHANISM, as production actually produces it. A first version of
        this test built the candidate PAIR-CONSISTENTLY in the stepped currency and the
        old formula correctly read ~0 -- that was its design working. The 145 specimens
        show the opposite arriving at the comparator: the candidate's cp0 UNMOVED while
        its dop label stepped with the clock_bias EMA (cand ~= prev to chips, ddop ~Hz).
        Feed the old formula that -- a pair-inconsistent candidate -- and it manufactures
        t_abs*k*dBias of phantom. The new form never consumes the candidate tuple at
        all (it compares the search's at-epoch physical phase), so it is immune BY
        CONSTRUCTION, not by a better translation."""
        dop_true = 1234.5
        bias_step = 2.0
        held = make_held(5000.0, dop_true, H0)                    # frozen pre-step
        h_det = H0 + int(30.0 * HPS)
        cand_phys = phys_at(h_det, 5000.0, dop_true, H0)
        # production's defect: cp0 built in the OLD currency, label stepped post-hoc
        cand = make_held(cand_phys, dop_true, h_det)
        cand["doppler_hz"] = dop_true + bias_step
        old = old_cp_err(cand, held, 0.0)
        self.assertGreater(abs(old), 3000.0,
                           "the retracted formula should manufacture ~1700/Hz "
                           "on a pair-inconsistent candidate (got %+.1f)" % old)
        det_at_ref = cand_phys % MOD
        new = track_vs_fit_chips(held, det_at_ref, h_det, 0.0,
                                 HPS, CHIP, CARR, SGN, L)
        self.assertLess(abs(new), 1e-3,
                        "the at-epoch form must be blind to bias motion "
                        "(got %+.3f)" % new)

    def test_a_real_lobe_park_is_still_caught(self):
        """The referee's actual prey: the track parked on a correlation lobe 3.27
        chips from the true peak. Both formulas must see it; the new one must
        report the physical offset."""
        dop = -876.0
        held = make_held(5000.0 + 3.27, dop, H0)   # parked ON the lobe
        h_det = H0 + int(5.0 * HPS)
        det_at_ref = phys_at(h_det, 5000.0, dop, H0)   # sky truth
        err = track_vs_fit_chips(held, det_at_ref, h_det, 0.0,
                                 HPS, CHIP, CARR, SGN, L)
        self.assertAlmostEqual(err, -3.27, delta=0.01,
                               msg="fresh - held should read the park as -3.27")

    def test_noise_track_prn2_class_is_still_caught(self):
        """A non-capable PRN reading noise: detections wander thousands of chips.
        The residual must be large (the erratic guard's food), not smoothed away."""
        held = make_held(5000.0, 500.0, H0)
        h_det = H0 + int(3.0 * HPS)
        det_at_ref = (phys_at(h_det, 5000.0, 500.0, H0) + 2646.0) % MOD
        err = track_vs_fit_chips(held, det_at_ref, h_det, 0.0,
                                 HPS, CHIP, CARR, SGN, L)
        self.assertGreater(abs(err), 2000.0)

    def test_trim_is_differenced_out(self):
        """The DLL trim rides the commanded cp; the residual must subtract it."""
        dop = 1000.0
        held = make_held(5000.0, dop, H0)
        h_det = H0 + int(4.0 * HPS)
        det_at_ref = phys_at(h_det, 5000.0, dop, H0)
        err = track_vs_fit_chips(held, det_at_ref, h_det, 0.75,
                                 HPS, CHIP, CARR, SGN, L)
        self.assertAlmostEqual(err, -0.75, delta=1e-3)

    def test_missing_cp_at_ref_returns_none(self):
        """Pre-2026-08 payloads carry -1.0: the comparator must decline, not guess."""
        held = make_held(5000.0, 1000.0, H0)
        self.assertIsNone(track_vs_fit_chips(held, -1.0, H0, 0.0,
                                             HPS, CHIP, CARR, SGN, L))
        self.assertIsNone(track_vs_fit_chips(held, None, H0, 0.0,
                                             HPS, CHIP, CARR, SGN, L))

    def test_period_flicker_is_invisible_mod_L(self):
        """#41's whole-period assignment flips must NOT reach the escape referee."""
        dop = 1000.0
        held = make_held(5000.0, dop, H0)
        h_det = H0 + int(4.0 * HPS)
        base = phys_at(h_det, 5000.0, dop, H0)
        for k in (-2, 5, 9):
            det_at_ref = (base + k * L) % MOD
            err = track_vs_fit_chips(held, det_at_ref, h_det, 0.0,
                                     HPS, CHIP, CARR, SGN, L)
            self.assertLess(abs(err), 1e-3,
                            "a %+d-period flicker leaked into cp_err" % k)


class TestRetagSeedDoppler(unittest.TestCase):
    """retag_seed_doppler (#44 / #45 step 4): a dop re-tag must preserve the physical
    phase where the despread is RUNNING -- the present -- not where the seed happens to
    be anchored."""

    def _phys(self, cp0, dop, t):
        return (cp0 + t * CHIP * (1.0 + SGN * dop / CARR)) % MOD

    def test_retag_at_now_preserves_the_phase_now(self):
        cp0, dopA, dopB = 4321.0, 1000.0, 1002.0
        t_now = T0 + 600.0
        before = self._phys(cp0, dopA, t_now)
        from gnss_broker.fits import retag_seed_doppler
        cp0_new = retag_seed_doppler(cp0, dopA, dopB, t_now, CHIP, CARR, SGN, MOD)
        after = self._phys(cp0_new, dopB, t_now)
        # Bar sits just above the double-precision floor, not at zero: the phys
        # reconstruction runs through t*f_chip ~ 2e12 chips where one ulp is 2.4e-4,
        # and a few ulps accumulate across retag + rebuild. 2e-3 chips = ~0.2 mm of
        # code -- physics never sees it; a LOGIC error here is chips, not milli-chips.
        self.assertLess(abs(((after - before + MOD / 2) % MOD) - MOD / 2), 2e-3)

    def test_the_anchor_epoch_tripwire(self):
        """The #44 defect, pinned: translating at a 600-s-stale anchor steps the phase
        NOW by anchor_age * (f_chip/f_car) * ddop ~ 10.4 chips for a 2 Hz re-tag. If a
        future edit moves the epoch back to the anchor, this fails loudly."""
        cp0, dopA, ddop = 4321.0, 1000.0, 2.0
        t_anchor = T0
        t_now = T0 + 600.0
        before = self._phys(cp0, dopA, t_now)
        from gnss_broker.fits import retag_seed_doppler
        cp0_bad = retag_seed_doppler(cp0, dopA, dopA + ddop, t_anchor,
                                     CHIP, CARR, SGN, MOD)
        after_bad = self._phys(cp0_bad, dopA + ddop, t_now)
        err = abs(((after_bad - before + MOD / 2) % MOD) - MOD / 2)
        self.assertAlmostEqual(err, 600.0 * CHIP / CARR * ddop, delta=0.01,
                               msg="the anchor-epoch error law changed -- investigate")
        self.assertGreater(err, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
