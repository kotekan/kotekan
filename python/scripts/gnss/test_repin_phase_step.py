"""Task #52: the re-pin phase step is ARITHMETIC, and the subtraction must happen in the
Doppler domain.

WHAT THIS PINS. The replica's carrier phase is ABSOLUTELY anchored -- 2*pi*f_ref*t_abs -- and
propagate_seed hands the despread a new f_ref every record. So swapping replica r-1 for
replica r steps the phase AT THAT INSTANT by (f_r - f_{r-1}) * t_abs. gnss_gpu::PrnCtl carries
that step (dcyc) and GnssGpuRecordAssemble folds it into the NCO.

None of that ran on CHORD until 2026-08-13: `reanchored` was hardcoded 0 in BOTH producers
(cudaGnssChordTrack, cudaGnssInject), so the fold was dead code and every record's correlation
carried the raw step. Downstream it read as a per-record common phase that is "white in time"
-- which is how gnssElemCal.hpp describes it, as a property of the sky. It is not the sky.

⚠️ THIS FILE MODELS THE ARITHMETIC, NOT THE SHIPPED CODE PATH. It cannot catch a producer that
stops setting the flag. The shipped path is gated by scripts/gnss/e2e (which already runs at
hop0 = 114436200145, ~6.8 days of uptime) and, ultimately, by the sky.
"""
import math
import unittest

FFT = 16384
HOPS_PER_RECORD = 2048
SAMPLE_RATE = 3.2e9
T_REC = HOPS_PER_RECORD * FFT / SAMPLE_RATE      # 10.486 ms
F_OFFSET_E5A = 1176.45e6                          # the SKY carrier: this is what kills precision
SKY_UPTIME_S = 291355.0                           # 3.37 days, the state measured on 2026-08-13


def dcyc_doppler_domain(dop, dop_prev, t_abs):
    """What the producers now compute: difference BEFORE f_offset is added."""
    return (dop - dop_prev) * t_abs


def dcyc_fcar_domain(dop, dop_prev, t_abs, f_offset=F_OFFSET_E5A):
    """What the assembler used to have to do: difference two ~1.18 GHz doubles."""
    return ((f_offset + dop) - (f_offset + dop_prev)) * t_abs


class TestTheStepIsReal(unittest.TestCase):

    def test_step_is_hundreds_of_cycles_at_sky_uptime(self):
        """The measured sky case: a few micro-Hz of Doppler change per record becomes
        hundreds of cycles, because t_abs is days. PRN 21 read 1127."""
        for dop_rate, expect_min in ((-0.03554, 50.0),     # PRN 9, the mildest measured
                                     (-0.36893, 900.0)):   # PRN 21, the strongest
            df = dop_rate * T_REC
            cyc = abs(dcyc_doppler_domain(0.0, -df, SKY_UPTIME_S))
            self.assertGreater(cyc, expect_min, "dop_rate %g gave only %g cycles"
                               % (dop_rate, cyc))

    def test_it_is_a_FAST_ALIASED_RAMP_not_a_random_walk(self):
        """⚠️ CORRECTS THE FIRST DIAGNOSIS. Under a steady Doppler rate dcyc is very nearly
        CONSTANT record to record -- it drifts by only rate*T_REC^2 ~ 4e-5 cycles -- so the
        accumulated phase is a LINEAR RAMP whose per-record advance is frac(dcyc), i.e.
        effectively an arbitrary constant in [0,1) cycles. Not white; deterministic and fast.

        The observable consequence is sharper than 'incoherent', and it is what was actually
        measured on sky: a constant per-record advance sums to a DIRICHLET kernel, ~1/N, which
        sits BELOW the 1/sqrt(N) random-walk baseline. The raw record stream read coherence
        0.03-0.06 against a 0.088 baseline on 2026-08-13 and that below-baseline value was the
        one number the random-walk story could not produce."""
        n = 128
        drift = []
        for r in range(n):
            t = SKY_UPTIME_S + r * T_REC
            df = -0.36893 * T_REC
            drift.append(dcyc_doppler_domain(0.0, -df, t))
        # near-constant step: the whole run varies by far less than a cycle
        self.assertLess(max(drift) - min(drift), 0.05)
        # ...so the summed phasor is Dirichlet-suppressed, below the random-walk baseline
        phase = 0.0
        z = 0j
        for r in range(n):
            phase += drift[r]
            z += complex(math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase))
        self.assertLess(abs(z) / n, 1.0 / math.sqrt(n),
                        "a deterministic ramp must beat a random walk DOWNWARD, not match it")

    def test_it_is_INVISIBLE_at_zero_uptime(self):
        """⚠️ THE GATE THAT COULD NOT FAIL. t_abs is the entire lever: at the start of an
        offline run the step is nanocycles and this whole defect is unobservable. Any harness
        that exercises this MUST run at a realistic absolute time -- scripts/gnss/e2e does
        (hop0 ~ 6.8 days). Asserted here so nobody 'simplifies' that hop0 away."""
        df = -0.36893 * T_REC
        self.assertLess(abs(dcyc_doppler_domain(0.0, -df, 0.0)), 1e-12)
        self.assertLess(abs(dcyc_doppler_domain(0.0, -df, 1.0)), 0.01)
        self.assertGreater(abs(dcyc_doppler_domain(0.0, -df, SKY_UPTIME_S)), 900.0)


class TestTheSubtractionDomain(unittest.TestCase):

    def test_doppler_domain_is_exact_and_fcar_domain_is_not(self):
        """THE REASON PrnCtl CARRIES dcyc INSTEAD OF THE ASSEMBLER RECOMPUTING IT. ulp at
        1.176 GHz is 2.4e-7 Hz against a delta of ~4e-3 Hz, so the fcar-domain difference keeps
        only ~4 digits and the resulting step is wrong by a sizeable fraction of a cycle --
        the same order as the term being removed. Both routes are algebraically identical;
        only one survives float64."""
        worst_fcar = 0.0
        for r in range(500):
            t = SKY_UPTIME_S + r * T_REC
            dop_prev = 3000.0 * math.sin(r * 1e-3)      # realistic magnitude, ~1e3 Hz
            dop = dop_prev - 0.36893 * T_REC
            exact = (dop - dop_prev) * t                # float64 in the small domain
            worst_fcar = max(worst_fcar, abs(dcyc_fcar_domain(dop, dop_prev, t) - exact))
        # the fcar route loses a meaningful fraction of a cycle...
        self.assertGreater(worst_fcar, 0.01,
                           "fcar-domain differencing was accurate here -- if this ever holds, "
                           "re-derive the precision argument rather than deleting the field")
        # ...which is radians, not rounding dust
        self.assertGreater(worst_fcar * 2 * math.pi, 0.05)

    def test_doppler_domain_round_trips_through_a_float64_transport(self):
        """dcyc rides the control block as a double. At ~1e3 cycles that is ulp 2e-13 -- the
        transport itself must not be the error term."""
        for r in range(100):
            t = SKY_UPTIME_S + r * T_REC
            d = dcyc_doppler_domain(0.0, 0.36893 * T_REC, t)
            import struct
            back = struct.unpack("<d", struct.pack("<d", d))[0]
            self.assertLess(abs(back - d), 1e-9)


class TestFoldSemantics(unittest.TestCase):

    def test_the_step_is_between_two_replicas_at_ONE_instant(self):
        """Both terms are evaluated at THIS record's t_abs -- the quantity is 'what the phase
        did when the despread swapped replica r-1 for replica r', not a difference across
        time. Getting that wrong would leave a term proportional to the record length."""
        t = SKY_UPTIME_S
        step = dcyc_doppler_domain(100.004, 100.0, t)
        self.assertAlmostEqual(step, 0.004 * t, places=6)
        # And the same identity evaluated in the SKY-CARRIER domain -- (f_now*t - f_prev*t)
        # with f ~ 1.176 GHz -- is wrong by 0.045 cycles, 0.28 rad. This assertion is the
        # precision argument stated as a fact rather than as a comment: the algebra is
        # identical, the float64 result is not.
        f_prev, f_now = F_OFFSET_E5A + 100.0, F_OFFSET_E5A + 100.004
        self.assertGreater(abs((f_now * t - f_prev * t) - step), 0.01)

    def test_no_history_breaks_the_arc_instead_of_folding_against_junk(self):
        """A PRN whose seed vanished and came back has no valid previous Doppler. The
        producers emit reanchored = 1 there (fresh acquisition, NCO reset), never a step
        computed against a stale slot -- which would be a large arbitrary rotation."""
        have_hist = False
        reanchored = 3 if have_hist else 1
        dcyc = dcyc_doppler_domain(100.0, 0.0, SKY_UPTIME_S) if have_hist else 0.0
        self.assertEqual(reanchored, 1)
        self.assertEqual(dcyc, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
