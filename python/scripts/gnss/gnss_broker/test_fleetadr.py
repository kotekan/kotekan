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
        self.resid_hz = resid_hz  # true minus commanded carrier rate
        self.phi0 = [random.uniform(-math.pi, math.pi) for _ in range(n_inst)]
        self.amp = [random.uniform(0.5, 1.5) for _ in range(n_inst)]
        self.noise = noise_rad
        self.chip = 1.0
        self.fold_step = -0.97  # rad per record, the assembler's in-frame fold
        self.bad_boundary = 0.0  # extra (wrong) fold at each frame's first record

    def t(self, hop):
        return hop / HPS

    def phi_res(self, hop):
        """Phi_rx - Phi_cmd: what the prompt's phase carries (cycles). Kept analytic: the two
        phases are ~5e14 cycles, and their difference in doubles would carry 0.06-cycle
        rounding."""
        t = self.t(hop)
        return self.resid_hz * t + 0.05 * math.sin(2 * math.pi * t / 30.0)

    def record(self, hop, prev_hop, insts=None, straddle=0.0, flip=False):
        """per_inst dict for `hop`: the record's Doppler (slot 1) and S_i from the phasors."""
        if flip:
            self.chip = -self.chip
        # arg(A) = 2pi (Phi_cmd - Phi_rx)
        ang = -2 * math.pi * self.phi_res(hop)
        # bad_boundary models a wrong fold at each frame's first record; the rotation
        # accumulates, so it mis-rotates that record and every record after it
        nfr = (hop - HOP0) // (4 * HPR)
        ang -= self.bad_boundary * nfr
        out = {}
        for i in insts if insts is not None else range(len(self.phi0)):
            a = self.amp[i] * cmath.exp(
                1j * (ang + random.gauss(0.0, self.noise) - self.phi0[i])
            )
            if straddle > 0.0:
                H = straddle * a * self.chip
                T = (
                    (1.0 - straddle) * a * (-self.chip)
                )  # the chip flips inside the record
            else:
                H, T = a * self.chip, 0.0
            S = H * H + T * T  # as exported: the instance's constant stays in
            out[i] = (self.dop_cmd, 0.0, S)
        return out


def dop_only(sky, hop, hop0):
    """The received phase between two hops with the nominal f_c*dt removed: the commanded
    Doppler's advance plus what the residual did."""
    return (
        sky.dop_cmd * (sky.t(hop) - sky.t(hop0)) + sky.phi_res(hop) - sky.phi_res(hop0)
    )


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
                kw[h] = {
                    "flip": True,
                    "straddle": 0.5,
                }  # the |2f-1| null of a linear sum
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
                return {
                    i: (v[0], v[1], v[2] * random.uniform(0.05, 1.0) ** 2)
                    for i, v in out.items()
                }

        sky = Flicker(noise_rad=0.3)
        hops = [HOP0 + k * HPR for k in range(4000)]
        st = run(sky, hops)
        err = st.adr - dop_only(sky, hops[-1], hops[0])
        # 12 instances at 0.3 rad -> ~0.05 cycles each on the squared phasor; a random walk
        # over 4000 records would be ~0.9 cycles here
        self.assertLess(abs(err), 0.12, err)

    def test_clustered_half_cycle_slips_do_not_walk_the_median(self):
        """A weak satellite's instances slip by exactly +-0.5 cycle (the squared phasor's
        ambiguity), and they slip together: a common disturbance pushes several over the wrap in
        the same record, in the same direction. Re-seating them to a median taken with the
        slipped values still in the sorted list moves that median by their ranks, and the fleet
        phase random-walks. Repairing each slip by whole half-cycles first leaves the median
        where the unslipped majority puts it. (An even split, half the fleet slipping together,
        is undecidable by any median; it is rare and not tested.)"""
        errs = []
        for seed in (1, 2, 3, 4, 5):

            class Clustered(Sky):
                def record(self, hop, prev_hop, **kw):
                    out = Sky.record(self, hop, prev_hop, **kw)
                    if random.random() < 0.01:
                        for i in random.sample(
                            range(12), random.choice((2, 2, 3, 3, 4))
                        ):
                            v = out[i]
                            out[i] = (v[0], v[1], -v[2])  # S -> -S: a half-cycle slip
                    return out

            sky = Clustered(noise_rad=0.15, seed=seed)
            hops = [HOP0 + k * HPR for k in range(8000)]
            st = run(sky, hops)
            errs.append(st.adr - dop_only(sky, hops[-1], hops[0]))
        # judged on the rms over seeds: a walk is a distribution, not one draw. The bound sits
        # between the repair (~0.02) and a rank-moving re-seat (~0.09).
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        self.assertLess(
            rms, 0.04, "rms %.3f over seeds %s" % (rms, [round(e, 3) for e in errs])
        )

    def test_commanded_trim_does_not_enter_the_adr(self):
        """The export is relative to the model Doppler: the assembler rotates the commanded
        carrier trim back out. Slot 19 must therefore be recorded (trim_cycles) but never added
        to the accumulated phase."""

        class Trimmed(Sky):
            def record(self, hop, prev_hop, **kw):
                out = Sky.record(self, hop, prev_hop, **kw)
                return {i: (v[0], 0.004, v[2]) for i, v in out.items()}

        sky = Trimmed()
        hops = [HOP0 + k * HPR for k in range(400)]
        st = run(sky, hops)
        self.assertAlmostEqual(st.adr, dop_only(sky, hops[-1], hops[0]), places=3)
        self.assertAlmostEqual(st.trim, 0.004 * 399, places=6)

    def test_gap_breaks_the_arc(self):
        sky = Sky()
        hops = [HOP0 + k * HPR for k in range(50)] + [
            HOP0 + k * HPR for k in range(60, 100)
        ]
        st = run(sky, hops)
        self.assertEqual(st.arc, 2)
        self.assertEqual(st.last_break["hop"], HOP0 + 60 * HPR)
        self.assertEqual(st.hop0, HOP0 + 60 * HPR)
        want = dop_only(sky, hops[-1], st.hop0)
        self.assertLess(abs(st.adr - want), 1e-3)

    def test_a_short_fleet_gap_is_bridged_by_vouching_increments(self):
        """Every instance skipped one record, so each was present at the fleet's previous
        record and vouches for the two-record step: the arc continues and the phase is still
        right."""
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
        one record and only 3-5 can vouch. If those are also absent at 41, nobody covers
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
        must not depend on them (a phasor sum would)."""
        random.seed(5)
        sky = Sky(n_inst=12)
        sky.phi0 = [random.uniform(-math.pi, math.pi) for _ in sky.phi0]
        hops = [HOP0 + k * HPR for k in range(200)]
        st = run(sky, hops)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_too_few_instances_is_not_a_measurement(self):
        sky = Sky(n_inst=1)
        st = run(sky, [HOP0 + k * HPR for k in range(10)])
        self.assertIsNone(st.hop)
        self.assertEqual(st.n, 0)

    @staticmethod
    def _fade(rec, insts, power, phase_noise):
        """The instances' prompts scaled to `power` of their level; phase_noise True makes
        the phase a random angle (a prompt with no signal in it)."""
        out = {}
        for i, v in rec.items():
            if i in insts:
                s = v[2] * power
                if phase_noise:
                    s = abs(s) * cmath.exp(1j * random.uniform(-math.pi, math.pi))
                v = (v[0], v[1], s)
            out[i] = v
        return out

    def test_a_brownout_ends_the_arc(self):
        """A bright satellite crossing the beam takes every prompt 20 dB down for 300 records
        while the trackers hold their keys: the phases are noise and must not be folded. The
        arc ends there and the arc that starts when the signal returns is right."""
        sky = Sky(n_inst=12)
        hops = [HOP0 + k * HPR for k in range(700)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            rec = sky.record(hop, prev)
            if 200 <= k < 500:
                rec = self._fade(rec, range(12), 0.01, True)
            fa.fold_record(st, hop, rec, HPR)
            prev = hop
        self.assertEqual(st.arc, 2)
        self.assertEqual(st.hop0, hops[500])
        # the first SILENT_MIN_REC - 1 records of the brownout still fold
        self.assertEqual(st.last_break["dark_rec"], 300 - fa.SILENT_MIN_REC + 1)
        self.assertEqual(st.last_break["silent_rec"], 300 - fa.SILENT_MIN_REC + 1)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[500])), 1e-3)

    def test_a_one_record_dip_is_not_silence(self):
        """A seed re-pin costs every instance one record of correlation at once (20 dB down,
        phase still measured): the record is folded, nothing is silent, the arc goes on."""
        sky = Sky(n_inst=12)
        hops = [HOP0 + k * HPR for k in range(600)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            rec = sky.record(hop, prev)
            if k % 50 == 25:
                rec = self._fade(rec, range(12), 0.01, False)
            fa.fold_record(st, hop, rec, HPR)
            prev = hop
            self.assertEqual(st.n_silent, 0)
        self.assertEqual(st.arc, 1)
        self.assertIsNone(st.last_break)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_a_fade_inside_the_margin_keeps_the_arc(self):
        """A 5 dB fade still carries the signal: the arc continues and the phase is right."""
        sky = Sky(n_inst=12)
        hops = [HOP0 + k * HPR for k in range(700)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            rec = sky.record(hop, prev)
            if 200 <= k < 500:
                rec = self._fade(rec, range(12), 0.3, False)
            fa.fold_record(st, hop, rec, HPR)
            prev = hop
        self.assertEqual(st.arc, 1)
        self.assertEqual(st.n_silent, 0)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_a_coherent_dip_is_not_silence(self):
        """A strong satellite's prompts drop 14 dB for 300 records with the phase still measured
        (every instance steps together): nothing is silent and the arc goes on."""
        sky = Sky(n_inst=12)
        hops = [HOP0 + k * HPR for k in range(700)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            rec = sky.record(hop, prev)
            if 200 <= k < 500:
                rec = self._fade(rec, range(12), 0.2, False)
            fa.fold_record(st, hop, rec, HPR)
            prev = hop
            self.assertEqual(st.n_silent, 0)
        self.assertEqual(st.arc, 1)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_one_instance_losing_the_signal_keeps_the_arc(self):
        """One instance loses the signal for 200 records while eleven keep it: the record is
        not silent, and the median of the instances' phases keeps the arc right."""
        sky = Sky(n_inst=12)
        hops = [HOP0 + k * HPR for k in range(500)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            rec = sky.record(hop, prev)
            if 100 <= k < 300:
                rec = self._fade(rec, [0], 0.01, True)
            fa.fold_record(st, hop, rec, HPR)
            prev = hop
            self.assertEqual(st.n_silent, 0)
        self.assertEqual(st.arc, 1)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_the_longest_gap_bridged_is_max_gap_rec(self):
        """A fleet gap of max_gap_rec records is one step every instance vouches for. One
        record more and no Doppler covers it: the arc ends, and the break says why."""
        for gap, arcs in ((3, 1), (4, 2)):
            sky = Sky(n_inst=4)
            hops = [HOP0 + k * HPR for k in range(120)]
            hops += [hops[-1] + gap * HPR + k * HPR for k in range(30)]
            st = run(sky, hops)
            self.assertEqual(st.arc, arcs, gap)
            if arcs == 1:
                self.assertIsNone(st.last_break)
                self.assertEqual((st.hop0, st.n), (hops[0], 150))
                self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)
                continue
            self.assertEqual(
                st.last_break,
                {
                    "hop": hops[120],
                    "n_rec": 120,
                    "gap_rec": 4.0,
                    "dark_rec": 0,
                    "silent_rec": 0,
                    "noenergy_rec": 0,
                    "n_usable": 4,
                    "n_vouch": 0,
                    "inst_lag_rec": [4.0, 4.0, 4.0],
                },
            )
            self.assertEqual((st.hop0, st.n, st.n_inst), (hops[120], 30, 4))
            self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[120])), 1e-3)

    def test_an_instance_that_leaves_rejoins_at_the_fleet_phase(self):
        """An instance that stops arriving loses its own phase and the arc goes on with the
        rest. Back, it was not there at the previous record, so it cannot vouch for its first
        step: it takes the fleet's phase and counts again from the next record."""
        sky = Sky(n_inst=4)
        hops = [HOP0 + k * HPR for k in range(100)]
        st = fa.SatAdr()
        prev = None
        for k, hop in enumerate(hops):
            insts = [0, 1, 2] if 40 <= k < 60 else [0, 1, 2, 3]
            fa.fold_record(st, hop, sky.record(hop, prev, insts=insts), HPR)
            prev = hop
            self.assertEqual(sorted(st.inst_x), insts)
            self.assertEqual(st.n_inst, len(insts))
            if 40 <= k < 60:  # its last record stays remembered, at the hop it was seen
                self.assertEqual(st.inst_prev[3][0], hops[39])
            if k == 60:
                self.assertEqual(st.inst_x[3], st.res)
        self.assertEqual((st.arc, st.n), (1, 100))
        self.assertIsNone(st.last_break)
        self.assertLess(abs(st.adr - dop_only(sky, hops[-1], hops[0])), 1e-3)

    def test_an_excursion_left_after_the_repair_is_reseated(self):
        """Slips are undone by whole half-cycles against the first median and the median is
        taken again. An instance still more than 0.3 cycle from the new one, slipped or not,
        is on an excursion and takes the fleet's phase."""
        st = fa.SatAdr()
        rec = {i: (-2000.0, 0.0, 1 + 0j) for i in range(5)}
        fa.fold_record(st, HOP0, rec, HPR)
        fa.fold_record(st, HOP0 + HPR, rec, HPR)
        # the first median is 0.62: instances 0, 1, 3 and 4 are more than 0.3 away and step by
        # half cycles to 0.5, 0.81, 0.43 and 0.43; the median is then 0.5, and 0.81 is 0.31 off
        st.inst_x = dict(enumerate((0.0, 0.31, 0.62, 0.93, 0.93)))
        fa.fold_record(st, HOP0 + 2 * HPR, rec, HPR)  # the same prompts: no increment
        for i, want in enumerate((0.5, 0.5, 0.62, 0.43, 0.43)):
            self.assertAlmostEqual(st.inst_x[i], want, places=12)
        self.assertAlmostEqual(st.res, 0.5, places=12)
        self.assertEqual((st.arc, st.n), (1, 3))


class FakeClient(object):
    def __init__(self, frames):  # {win: {inst: TelemFrame}}
        self.f = frames

    def windows(self, chain, lag=1):
        return sorted(self.f)[:-lag] if lag else sorted(self.f)

    def frame_set(self, chain, win):
        return dict(self.f.get(win, {}))


class TestFrames(unittest.TestCase):
    """Through real TelemFrames (test_telem's builder) and the window ring."""

    def test_windows_fold_to_the_published_phase(self):
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
                    # a basis in force on the second sender only, from the last two windows
                    if i == 1 and win >= 110:
                        rows[(r, 0, telem.REC_PROJ_COST)] = 0.25
                        rows[(r, 0, telem.REC_PROJ_K)] = 2.0
                        rows[(r, 0, telem.REC_PROJ_SRC)] = 6.0
                    else:
                        rows[(r, 0, telem.REC_PROJ_COST)] = -1.0
                raw = test_telem._make_frame(
                    inst=inst,
                    win=win,
                    n_rec=n_rec,
                    n_prn=1,
                    hops_per_record=HPR,
                    rows=rows,
                )
                frames.setdefault(win, {})[inst] = telem.TelemFrame(
                    telem._HDR.unpack_from(raw, 0), raw, 0.0
                )
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
        # the projection slots ride through: windows 110 and 111 carry it, 111 is the ring's
        # unfinished newest window (lag 1), so this cycle folded window 110 only
        self.assertAlmostEqual(pub[1]["proj_b"], 0.25, places=6)
        self.assertEqual(pub[1]["proj_k"], 2)
        self.assertEqual(pub[1]["proj_src"], 6)

    @staticmethod
    def _frames(sky, wins, n_rec=4, row=None, insts=("cx19.0", "cx20.0"), present=None):
        """{win: {inst: TelemFrame}} for the senders carrying PRN 1 in row 0. row(win, r) may
        return extra {slot: value} for that record, present(win, inst) the frame's mask of
        filled record slots (None: all)."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import test_telem
        from gnss_broker import telem

        frames = {}
        for win in wins:
            for i, inst in enumerate(insts):
                rows = {}
                for r in range(n_rec):
                    hop = (win * n_rec + r) * HPR
                    ang = -2 * math.pi * sky.phi_res(hop) - sky.phi0[i]
                    A = sky.amp[i] * cmath.exp(1j * ang)
                    rows[(r, 0, telem.REC_P_RE)] = 2.0 * A.real
                    rows[(r, 0, telem.REC_P_IM)] = 2.0 * A.imag
                    rows[(r, 0, telem.REC_DOPPLER)] = sky.dop_cmd
                    rows[(r, 0, telem.REC_TRIM_INC)] = 0.0
                    rows[(r, 0, telem.REC_PROJ_COST)] = -1.0
                    for slot, v in (row(win, r) if row else {}).items():
                        rows[(r, 0, slot)] = v
                raw = test_telem._make_frame(
                    inst=inst,
                    win=win,
                    n_rec=n_rec,
                    n_prn=1,
                    hops_per_record=HPR,
                    present=present(win, inst) if present else None,
                    rows=rows,
                )
                frames.setdefault(win, {})[inst] = telem.TelemFrame(
                    telem._HDR.unpack_from(raw, 0), raw, 0.0
                )
        return frames

    def test_an_epoch_reset_restarts_the_fold(self):
        """An F-engine re-base restarts the window counter. The windows of the new epoch are all
        below the last one folded, and must still be folded: a new arc, not a dark chain."""
        sky = Sky(n_inst=2)
        lines = []
        fl = fa.FleetAdr(hpr=HPR, log=lines.append)
        old = FakeClient(self._frames(sky, range(100, 112)))
        self.assertEqual(fl.fold_windows(old, "gps_l5", {1}, now=1000.0), 11)
        self.assertEqual(fl.fold_windows(old, "gps_l5", {1}, now=1001.0), 0)
        new = FakeClient(self._frames(sky, range(0, 12)))
        self.assertEqual(fl.fold_windows(new, "gps_l5", {1}, now=1002.0), 11)
        st = fl.sats[1]
        self.assertEqual((st.arc, st.hop0, st.n), (1, 0, 11 * 4))
        self.assertLess(abs(st.adr - dop_only(sky, st.hop, st.hop0)), 1e-3)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("FADR EPOCH RESET gps_l5:"), lines)

    def test_a_break_counts_records_that_arrived_without_energy(self):
        """Records that arrived with no prompt energy (a zero row, or no row shipped) never reach
        the fold. The break counts them apart from the gap, so a stretch the trackers did not
        despread reads differently from telemetry that never came: window 130 has zero rows,
        131 is missing, 132 ships no row for the PRN."""
        from gnss_broker import telem

        def row(win, r):
            if win == 130:
                return {telem.REC_P_ENERGY: 0.0}
            if win == 132:
                return {telem.REC_PRN: 2.0}
            return {}

        sky = Sky(n_inst=2)
        frames = self._frames(sky, [w for w in range(100, 138) if w != 131], row=row)
        lines = []
        fl = fa.FleetAdr(hpr=HPR, log=lines.append)
        fl.fold_windows(FakeClient(frames), "gps_l5", {1}, now=1000.0)
        b = fl.sats[1].last_break
        self.assertEqual(b["hop"], 133 * 4 * HPR)
        self.assertEqual((b["gap_rec"], b["dark_rec"], b["noenergy_rec"]), (13, 0, 8))
        self.assertEqual(len(lines), 1)
        self.assertIn(" 1:gap13/dark0/silent0/noenergy8/", lines[0])

    def test_a_frame_decodes_to_the_slots_of_its_rows(self):
        """records_of_frame against the frame's own row(): only the wanted PRNs and the filled
        record slots, a row without prompt energy left out, S from the head and the tail."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import test_telem
        from gnss_broker import telem

        rows = {}
        for r in range(4):
            for p in range(3):
                rows[(r, p, telem.REC_DOPPLER)] = -1500.0 - 10 * p - r
                rows[(r, p, telem.REC_P_RE)] = 0.5 + p - 0.1 * r
                rows[(r, p, telem.REC_P_IM)] = -0.3 * (r + 1)
                rows[(r, p, telem.REC_TRIM_INC)] = 0.001 * (r + 1)
                rows[(r, p, telem.REC_PROJ_COST)] = -1.0 if p == 0 else 0.125 * r
                rows[(r, p, telem.REC_PROJ_K)] = float(p)
                rows[(r, p, telem.REC_PROJ_SRC)] = float(r)
            rows[(r, 1, telem.REC_PH_RE)] = 0.2  # PRN 2 is segmented: a head, so a tail
            rows[(r, 1, telem.REC_PH_IM)] = 0.1 * r
        rows[(3, 0, telem.REC_P_ENERGY)] = 0.0
        raw = test_telem._make_frame(n_rec=4, n_prn=3, present=0b1101, rows=rows)
        f = telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0)
        got = fa.records_of_frame(f, {1, 2, 9})  # PRN 3 is carried but not wanted
        self.assertEqual(
            {prn: sorted(v) for prn, v in got.items()}, {1: [0, 2], 2: [0, 2, 3]}
        )
        for prn, recs in got.items():
            for r, v in recs.items():
                a = f.row(r, prn)
                e = a[telem.REC_P_ENERGY]
                A = complex(a[telem.REC_P_RE] / e, a[telem.REC_P_IM] / e)
                H = complex(a[telem.REC_PH_RE] / e, a[telem.REC_PH_IM] / e)
                self.assertEqual(H != 0, prn == 2)
                H = H if prn == 2 else A
                want = (
                    a[telem.REC_DOPPLER],
                    a[telem.REC_TRIM_INC],
                    H * H + (A - H) * (A - H),
                    a[telem.REC_PROJ_COST],
                    prn - 1,
                    r,
                )
                self.assertEqual(v, want, (prn, r))
                self.assertEqual(
                    [type(x) for x in v], [float, float, complex, float, int, int]
                )
        self.assertEqual(fa.records_of_frame(f, {9}), {})

    def test_a_sender_with_a_hole_skips_one_step(self):
        """One of three senders misses a record. The other two fold it, and at the next record
        the one that skipped cannot vouch: the arc goes on, every record counted."""
        sky = Sky(n_inst=3)
        insts = ("cx19.0", "cx20.0", "cx21.0")
        frames = self._frames(
            sky,
            range(100, 131),
            insts=insts,
            present=lambda win, inst: 0b1011
            if (win, inst) == (110, "cx21.0")
            else None,
        )
        seen = {}  # hop -> instances behind the record
        fold = fa.fold_record

        def spy(st, hop, per_inst, *a, **k):
            ok = fold(st, hop, per_inst, *a, **k)
            seen[hop] = (st.n_inst, sorted(st.inst_x))
            return ok

        fl = fa.FleetAdr(hpr=HPR, log=self.fail)
        fa.fold_record = spy
        try:
            fl.fold_windows(FakeClient(frames), "gps_l5", {1}, now=1000.0)
        finally:
            fa.fold_record = fold
        hole = (110 * 4 + 2) * HPR
        self.assertEqual(seen[hole], (2, ["cx19.0", "cx20.0"]))
        self.assertEqual(seen[hole + HPR], (3, sorted(insts)))
        st = fl.sats[1]
        self.assertEqual((st.arc, st.n, st.n_inst), (1, 30 * 4, 3))
        self.assertIsNone(st.last_break)
        self.assertLess(abs(st.adr - dop_only(sky, st.hop, st.hop0)), 1e-3)

    def test_a_sender_dropping_out_leaves_too_few_and_the_arc_ends(self):
        """One of two senders stops for three windows. Its twelve records fold dark (one
        instance is not a measurement), and when it returns the step is thirteen records, more
        than max_gap_rec: a new arc, and the break line names the dark records."""
        sky = Sky(n_inst=2)
        frames = self._frames(sky, range(100, 140))
        for win in (130, 131, 132):
            del frames[win]["cx20.0"]
        lines = []
        fl = fa.FleetAdr(hpr=HPR, log=lines.append)
        fl.fold_windows(FakeClient(frames), "gps_l5", {1}, now=1000.0)
        st = fl.sats[1]
        self.assertEqual(
            st.last_break,
            {
                "hop": 133 * 4 * HPR,
                "n_rec": 30 * 4,
                "gap_rec": 13.0,
                "dark_rec": 12,
                "silent_rec": 0,
                "noenergy_rec": 0,
                "n_usable": 2,
                "n_vouch": 0,
                "inst_lag_rec": [13.0, 1.0],
                "logged": True,
            },
        )
        self.assertEqual((st.arc, st.hop0, st.n), (2, 133 * 4 * HPR, 6 * 4))
        self.assertLess(abs(st.adr - dop_only(sky, st.hop, st.hop0)), 1e-3)
        self.assertEqual(len(lines), 1)
        self.assertIn(
            " 1:gap13/dark12/silent0/noenergy0/usable2/vouch0/lag13,1", lines[0]
        )
        pub = fl.publish(FC, 1000.0)
        self.assertEqual((pub[1]["arc"], pub[1]["n_rec"], pub[1]["n_inst"]), (2, 24, 2))


if __name__ == "__main__":
    unittest.main()
