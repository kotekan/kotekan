"""The fleet ADR (fleetadr): recovers the received carrier phase from commanded increments and
prompt phasors across instances, survives overlay flips, straddles and per-instance rotations,
breaks honestly on gaps, and never uses an increment that does not span its own step.

    python3 -m unittest gnss_broker.test_fleetadr
"""
import cmath
import math
import random
import unittest
from fractions import Fraction

from gnss_broker import fleetadr as fa

HPR = 2048
HPS = float(fa.HPS)
FC = 1176.45e6
HOP0 = 86_466_125_824


class Sky(object):
    """A satellite as the tracker sees it: a commanded phase that is slightly wrong in rate, a
    true received phase, twelve instances with private rotations, an overlay that flips."""

    def __init__(self, n_inst=12, resid_hz=0.3, noise_rad=0.0, seed=1):
        random.seed(seed)
        self.dop_cmd = -2182.9
        self.resid_hz = resid_hz           # true minus commanded carrier rate
        self.phi0 = [random.uniform(-math.pi, math.pi) for _ in range(n_inst)]
        self.amp = [random.uniform(0.5, 1.5) for _ in range(n_inst)]
        self.noise = noise_rad
        self.chip = 1.0
        self.fold_step = -0.97                 # rad per record, the assembler's in-frame fold
        self.bad_boundary = 0.0               # extra (wrong) fold at each frame's first record

    def t(self, hop):
        return hop / HPS

    def phi_res(self, hop):
        """Phi_rx - Phi_cmd: what the prompt's phase carries (cycles). Kept ANALYTIC: the two
        phases are ~5e14 cycles and their difference in doubles would be 0.06-cycle garbage."""
        t = self.t(hop)
        return self.resid_hz * t + 0.05 * math.sin(2 * math.pi * t / 30.0)

    def record(self, hop, prev_hop, insts=None, straddle=0.0, flip=False):
        """per_inst dict for `hop`: the record's Doppler (slot 1) and S_i from the phasors."""
        if flip:
            self.chip = -self.chip
        # arg(A) = 2pi (Phi_cmd - Phi_rx)
        ang = -2 * math.pi * self.phi_res(hop)
        # the prompt is rotated by exp(-i phi) and phi ACCUMULATES: a wrong step at each frame's
        # first record mis-rotates that record and every record after it (a random walk)
        nfr = (hop - HOP0) // (4 * HPR)
        ang -= self.bad_boundary * nfr
        out = {}
        for i in (insts if insts is not None else range(len(self.phi0))):
            a = self.amp[i] * cmath.exp(1j * (ang + random.gauss(0.0, self.noise) - self.phi0[i]))
            if straddle > 0.0:
                H = straddle * a * self.chip
                T = (1.0 - straddle) * a * (-self.chip)      # the chip flips inside the record
            else:
                H, T = a * self.chip, 0.0
            S = H * H + T * T                 # as exported: the instance's constant stays in
            # phi0: the assembler's NCO accumulator. A perfect fold steps it by the same
            # amount every record; `bad_boundary` mimics the live defect (a wrong step at r0).
            r = ((hop - HOP0) // HPR) % 4
            phi0 = self.fold_step * ((hop - HOP0) // HPR) + self.bad_boundary * ((hop - HOP0) // (4 * HPR))
            out[i] = (self.dop_cmd, 0.0, S, phi0, r)
        return out


def dop_only(sky, hop, hop0):
    """The received phase between two hops with the nominal f_c*dt removed: the commanded
    Doppler's advance plus what the residual did."""
    return sky.dop_cmd * (sky.t(hop) - sky.t(hop0)) + sky.phi_res(hop) - sky.phi_res(hop0)


def run(sky, hops, st=None, kw=None):
    st = st or fa.SatAdr()
    kw = kw or {}
    prev = None
    for hop in hops:
        rec = sky.record(hop, prev, **kw.get(hop, {}))
        fa.fold_record(st, hop, rec, HPR)
        prev = hop
    return st


class TestFold(unittest.TestCase):
    def test_recovers_the_received_phase(self):
        sky = Sky()
        hops = [HOP0 + k * HPR for k in range(400)]
        st = run(sky, hops)
        self.assertEqual(st.arc, 1)
        self.assertEqual(st.n, 400)
        want = dop_only(sky, hops[-1], hops[0])
        self.assertLess(abs(st.adr - want), 1e-3, (st.adr, want))

    def test_overlay_flips_and_straddles_do_not_matter(self):
        sky = Sky()
        hops = [HOP0 + k * HPR for k in range(300)]
        kw = {}
        for k, h in enumerate(hops):
            if k % 7 == 3:
                kw[h] = {"flip": True, "straddle": 0.5}     # the |2f-1| null of a linear sum
            elif k % 5 == 1:
                kw[h] = {"straddle": 0.3}
        st = run(sky, hops, kw=kw)
        want = dop_only(sky, hops[-1], hops[0])
        self.assertEqual(st.n, 300)
        self.assertLess(abs(st.adr - want), 1e-3)

    def test_fleet_sum_beats_one_instance(self):
        sky = Sky(noise_rad=0.3)
        hops = [HOP0 + k * HPR for k in range(300)]
        st_all = run(sky, hops)
        random.seed(1)
        sky1 = Sky(noise_rad=0.3)
        st_one = fa.SatAdr()
        prev = None
        for hop in hops:
            fa.fold_record(st_one, hop, sky1.record(hop, prev, insts=[0, 1]), HPR)
            prev = hop
        want = dop_only(sky, hops[-1], hops[0])
        self.assertLess(abs(st_all.adr - want), abs(st_one.adr - want))
        self.assertLess(abs(st_all.adr - want), 0.05)

    def test_varying_instance_weights_do_not_random_walk(self):
        """Prompt amplitudes swing record to record (where the secondary chip flips inside the
        record decides |S|), so a fleet increment formed from the weighted sum of cross-products
        does not telescope and its sum random-walks. Per-instance phases do telescope: with white
        per-record noise the fleet error must stay at the single-record level after thousands
        of records, not grow as sqrt(N)."""
        class Flicker(Sky):
            def record(self, hop, prev_hop, **kw):
                out = Sky.record(self, hop, prev_hop, **kw)
                return {i: (v[0], v[1], v[2] * random.uniform(0.05, 1.0) ** 2, v[3], v[4])
                        for i, v in out.items()}
        sky = Flicker(noise_rad=0.3)
        hops = [HOP0 + k * HPR for k in range(4000)]
        st = run(sky, hops)
        err = st.adr - dop_only(sky, hops[-1], hops[0])
        # 12 instances at 0.3 rad -> ~0.05 cycles each on the squared phasor; a random walk
        # over 4000 records would be ~0.9 cycles here
        self.assertLess(abs(err), 0.12, err)

    def test_gap_breaks_the_arc(self):
        sky = Sky()
        hops = [HOP0 + k * HPR for k in range(50)] + [HOP0 + k * HPR for k in range(60, 100)]
        st = run(sky, hops)
        self.assertEqual(st.arc, 2)
        self.assertEqual(st.breaks, 1)
        self.assertEqual(st.hop0, HOP0 + 60 * HPR)
        want = dop_only(sky, hops[-1], st.hop0)
        self.assertLess(abs(st.adr - want), 1e-3)

    def test_a_short_fleet_gap_is_bridged_by_vouching_increments(self):
        """Every instance skipped one record: their next slot-15 spans two records, which is
        exactly our step, so the arc continues and the phase is still right."""
        sky = Sky()
        hops = [HOP0 + k * HPR for k in range(100) if k != 40]
        st = run(sky, hops)
        self.assertEqual(st.arc, 1)
        want = dop_only(sky, hops[-1], hops[0])
        self.assertLess(abs(st.adr - want), 1e-3)

    def test_an_instance_that_skipped_cannot_vouch(self):
        """Instance 0 misses record 40; at record 41 its increment spans two records while
        the fleet's step is one. Its value must not enter -- median or not."""
        sky = Sky(n_inst=3)
        hops = [HOP0 + k * HPR for k in range(100)]
        st = fa.SatAdr()
        prev = {i: None for i in range(3)}
        for k, hop in enumerate(hops):
            insts = [1, 2] if k == 40 else [0, 1, 2]
            rec = {}
            for i in insts:
                r = sky.record(hop, prev[i], insts=[i])[i]
                rec[i] = r
                prev[i] = hop
            fa.fold_record(st, hop, rec, HPR)
        want = dop_only(sky, hops[-1], hops[0])
        self.assertEqual(st.arc, 1)
        self.assertLess(abs(st.adr - want), 1e-3)

    def test_no_voucher_ends_the_arc(self):
        """Instances 0-2 all missed record 40 but 3-5 were there: at 41 the fleet step is
        one record and only 3-5 can vouch. If those are ALSO absent at 41, nobody covers
        the step and the arc must end rather than guess."""
        sky = Sky(n_inst=6)
        hops = [HOP0 + k * HPR for k in range(60)]
        st = fa.SatAdr()
        prev = {i: None for i in range(6)}
        for k, hop in enumerate(hops):
            insts = [3, 4, 5] if k == 40 else ([0, 1, 2] if k == 41 else range(6))
            rec = {}
            for i in insts:
                rec[i] = sky.record(hop, prev[i], insts=[i])[i]
                prev[i] = hop
            fa.fold_record(st, hop, rec, HPR)
        self.assertEqual(st.arc, 2)
        self.assertEqual(st.hop0, hops[41])

    def test_instance_constants_may_differ_by_anything(self):
        """Each instance's exported prompt carries its own arbitrary constant; the increment
        must not depend on them (a phasor sum WOULD)."""
        random.seed(5)
        sky = Sky(n_inst=12)
        sky.phi0 = [random.uniform(-math.pi, math.pi) for _ in sky.phi0]
        hops = [HOP0 + k * HPR for k in range(200)]
        st = run(sky, hops)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_a_wrong_boundary_fold_is_repaired(self):
        """The live defect: the assembler's fold at each frame's first record is off by an
        arbitrary angle, mis-rotating that one record. The fold must notice (from REC_PHI0)
        and undo it, or the ADR walks by that angle every frame."""
        random.seed(9)
        for bad in (0.7, -2.1, 3.0):
            sky = Sky()
            sky.bad_boundary = bad
            hops = [HOP0 + k * HPR for k in range(400)]
            st = run(sky, hops)
            self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 2e-3, (bad, st.adr))
            self.assertGreater(st.n_bfix, 90)
            self.assertLess(abs(st.bfix + st.n_bfix * bad / (2 * math.pi)), 0.05)  # eps = -bad

    def test_too_few_instances_is_not_a_measurement(self):
        sky = Sky(n_inst=1)
        st = run(sky, [HOP0 + k * HPR for k in range(10)])
        self.assertIsNone(st.hop)
        self.assertEqual(st.n, 0)


class FakeClient(object):
    def __init__(self, frames):          # {win: {inst: TelemFrame}}
        self.f = frames

    def windows(self, chain, lag=1):
        return sorted(self.f)[:-lag] if lag else sorted(self.f)

    def frame_set(self, chain, win):
        return dict(self.f.get(win, {}))


class TestFrames(unittest.TestCase):
    """Through real TelemFrames (test_telem's builder) and the window ring."""

    def test_windows_fold_to_the_published_phase(self):
        import struct
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import test_telem
        from gnss_broker import telem
        sky = Sky(n_inst=2)
        n_rec = 4
        frames = {}
        for win in range(100, 112):
            for i, inst in enumerate(("cx19.0", "cx20.0")):
                rows = {}
                for r in range(n_rec):
                    hop = (win * n_rec + r) * HPR
                    # the frame carries A (not S): rebuild the phasor the builder expects
                    ang = -2 * math.pi * sky.phi_res(hop) - sky.phi0[i]
                    A = sky.amp[i] * cmath.exp(1j * ang)
                    rows[(r, 0, telem.REC_P_RE)] = 2.0 * A.real
                    rows[(r, 0, telem.REC_P_IM)] = 2.0 * A.imag
                    rows[(r, 0, telem.REC_PHI0)] = sky.fold_step * ((hop - HOP0) // HPR)
                    rows[(r, 0, telem.REC_DOPPLER)] = sky.dop_cmd
                    rows[(r, 0, telem.REC_TRIM_INC)] = 0.0
                raw = test_telem._make_frame(inst=inst, win=win, n_rec=n_rec, n_prn=1,
                                             hops_per_record=HPR, rows=rows)
                frames.setdefault(win, {})[inst] = telem.TelemFrame(
                    telem._HDR.unpack_from(raw, 0), raw, 0.0)
        # the builder's PRN in row 0 is 1
        fl = fa.FleetAdr(hpr=HPR)
        n = fl.fold_windows(FakeClient(frames), "gps_l5", {1}, now=1000.0)
        self.assertEqual(n, 11)
        pub = fl.publish(FC, 1000.0)
        self.assertIn(1, pub)
        st = fl.sats[1]
        self.assertEqual(st.arc, 1)
        self.assertEqual(st.n, 11 * n_rec)
        want = dop_only(sky, st.hop, st.hop0)
        # float32 phasors and Doppler in the frame: ~1e-6 cycles each, no accumulation
        self.assertLess(abs(st.adr - want), 1e-3, (st.adr, want))
        self.assertEqual(pub[1]["hop"], (111 * n_rec - 1) * HPR)
        self.assertLess(abs(pub[1]["dop_cycles"] - want), 1e-3)
        full = float(Fraction(want) + Fraction(st.hop - st.hop0) / fa.HPS * int(FC))
        self.assertLess(abs(pub[1]["cycles"] - full), 1e-2)


if __name__ == "__main__":
    unittest.main()
