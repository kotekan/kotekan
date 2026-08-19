#!/usr/bin/env python3
"""Tests for the comb-derived fleet DLL (gnss_broker/combdll.py), task #63.

    python3 python/scripts/gnss/test_combdll.py

WHAT THESE ARE FOR. The live A/B (scripts/gnss/comb_dll_ab.py) shows the comb path agrees with
the polled one on sky, but agreement on today's sky cannot tell a coherent channel combine from
an incoherent one -- with every channel roughly in phase the two differ only in scale, and the
DLL's ratios divide the scale out. So the properties that MATTER are pinned here on constructed
data where they separate: a phase-opposed comb must cancel, instances must add in POWER, and a
missing record must be a hole rather than a zero.

The wire layout these frames assume is proved against the C++ header by test_telem.py's
test_wire_format_matches_cpp; if the layout moves, that test fails first and these stop being
meaningful rather than becoming quietly wrong.
"""
import cmath
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gnss_broker import combdll, telem  # noqa: E402


def make_frame(chain="gps_l5", inst="cx19.0", win=100, seq=0, n_rec=4, n_prn=2,
               chan_ids=(5972, 5988, 6004), taps=None, present=None,
               hops_per_record=2048, fft_len=16384):
    """One wire frame with an EXPLICIT comb.

    `taps` is {(record, prn_index, channel_index): (A_early, A_prompt, A_late, energy)} with
    complex A; anything unlisted defaults to (0.5, 1+0j, 0.25, 1.0). PRN numbers are 1+index.
    """
    if present is None:
        present = (1 << n_rec) - 1
    n_chan = len(chan_ids)
    ids = list(chan_ids) + [0] * (telem._MAX_CHAN - n_chan)
    wstart0 = win * n_rec * hops_per_record * fft_len
    hdr = telem._HDR.pack(telem._MAGIC, telem._VERSION, n_rec, n_prn, telem._ROW_FLOATS,
                          n_chan, 32, hops_per_record, fft_len, win, seq, wstart0, 0.0,
                          present, telem._MAX_CHAN, telem._ROW_TOTAL,
                          chain.encode(), inst.encode(), *ids)
    body = [0.0] * (n_rec * n_prn * telem._ROW_TOTAL)
    taps = taps or {}
    for r in range(n_rec):
        for p in range(n_prn):
            base = (r * n_prn + p) * telem._ROW_TOTAL
            body[base + telem.REC_PRN] = float(1 + p)
            body[base + telem.REC_P_ENERGY] = 1.0
            for ch in range(n_chan):
                E, P, L, en = taps.get((r, p, ch), (0.5 + 0j, 1 + 0j, 0.25 + 0j, 1.0))
                cb = base + telem._ROW_FLOATS + ch * telem._CHAN_FLOATS
                # The wire carries G = A*energy; comb_epl() divides it back out.
                body[cb + telem.CHAN_E_RE], body[cb + telem.CHAN_E_IM] = (E * en).real, (E * en).imag
                body[cb + telem.CHAN_E_ENERGY] = en
                body[cb + telem.CHAN_RE], body[cb + telem.CHAN_IM] = (P * en).real, (P * en).imag
                body[cb + telem.CHAN_ENERGY] = en
                body[cb + telem.CHAN_L_RE], body[cb + telem.CHAN_L_IM] = (L * en).real, (L * en).imag
                body[cb + telem.CHAN_L_ENERGY] = en
    return hdr + struct.pack("<%df" % len(body), *body)


def client_with(frames, depth=16):
    c = telem.TelemClient(depth=depth)
    for raw in frames:
        c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))
    return c


class TestInstancePowers(unittest.TestCase):
    """A. One instance: the powers are the combiner's expression, formed from the comb."""

    def test_matches_the_combiners_own_arithmetic(self):
        # 3 channels, A = 1 (prompt) / 0.5 (early) / 0.25 (late), unit energy each.
        # e2 = |sum G|^2/(sum E)^2 = (3*0.5)^2/9 = 0.25, p2 = 1.0, l2 = 0.0625.
        c = client_with([make_frame(win=w) for w in (10, 11)])
        t = combdll.instance_taps(c, "gps_l5", [10, 11])
        d = t[1]["cx19.0"]
        self.assertAlmostEqual(d["p"], 1.0, places=6)
        self.assertAlmostEqual(d["e"], 0.25, places=6)
        self.assertAlmostEqual(d["l"], 0.0625, places=6)
        self.assertEqual(d["n_rec"], 8)          # 2 windows x 4 records
        self.assertAlmostEqual(d["n_chan"], 3.0, places=6)

    def test_the_channel_combine_is_COHERENT(self):
        """Two channels in antiphase must CANCEL, not add.

        The one property an incoherent |A_c|^2 sum -- the obvious "simplification" -- gets
        wrong, and the reason it would never show up on sky: with the comb roughly in phase,
        coherent and incoherent differ only by a scale factor that the DLL's ratios divide
        straight back out. Here they differ by everything.
        """
        taps = {}
        for r in range(4):
            taps[(r, 0, 0)] = (0.5 + 0j, 1 + 0j, 0.25 + 0j, 1.0)
            taps[(r, 0, 1)] = (-0.5 + 0j, -1 + 0j, -0.25 + 0j, 1.0)
        c = client_with([make_frame(chan_ids=(5972, 5988), taps=taps, n_prn=1)])
        d = combdll.instance_taps(c, "gps_l5", [100])[1]["cx19.0"]
        self.assertAlmostEqual(d["p"], 0.0, places=9)
        self.assertAlmostEqual(d["e"], 0.0, places=9)
        # ...while the per-channel powers, which are single-channel by construction, are NOT
        # zero: |A|^2 = 1 on each. A cancelling full-band sum over live channels is the
        # signature the frequency axis exists to show.
        self.assertAlmostEqual(d["chan"][5972][1], 1.0, places=6)
        self.assertAlmostEqual(d["chan"][5988][1], 1.0, places=6)

    def test_energy_weighting_is_by_replica_energy_not_by_channel_count(self):
        # One loud channel (energy 9) and one quiet (energy 1), same amplitude: A stays 1.
        # A count-weighted mean would agree here; an UNWEIGHTED sum of A would not, so make
        # the amplitudes differ too: A = 1 at E=9 and A = 3 at E=1 -> (9+3)/10 = 1.2.
        taps = {}
        for r in range(4):
            taps[(r, 0, 0)] = (0j, 1 + 0j, 0j, 9.0)
            taps[(r, 0, 1)] = (0j, 3 + 0j, 0j, 1.0)
        c = client_with([make_frame(chan_ids=(5972, 5988), taps=taps, n_prn=1)])
        d = combdll.instance_taps(c, "gps_l5", [100])[1]["cx19.0"]
        self.assertAlmostEqual(d["p"], 1.2 ** 2, places=6)

    def test_a_missing_record_is_a_hole_not_a_zero(self):
        """present = 0b0101: two live records. The mean must be over the LIVE ones.

        Averaging the absent records in as zeros halves the power and reports a satellite as
        two flavours of fainter than it is -- the same pathology the deep fold's zero-padding
        had (a tiny |A| at an enormous fake significance).
        """
        c = client_with([make_frame(present=0b0101)])
        d = combdll.instance_taps(c, "gps_l5", [100])[1]["cx19.0"]
        self.assertEqual(d["n_rec"], 2)
        self.assertAlmostEqual(d["p"], 1.0, places=6)   # NOT 0.5


class TestFleetCombine(unittest.TestCase):
    """B. Across instances: powers add, and the discriminator is the summed-power ratio."""

    def _fleet(self, **kw):
        frames = [make_frame(inst="cx19.0", chan_ids=(5972, 5988, 6004), **kw),
                  make_frame(inst="cx42.0", chan_ids=(5976, 5992, 6008), **kw)]
        return combdll.fleet_dll_comb(client_with(frames), "gps_l5", n_win=4, lag=0)

    def test_powers_add_across_instances_and_the_ratio_does_not(self):
        out = self._fleet()
        v = out[1]
        self.assertEqual(v["n_src"], 2)
        self.assertAlmostEqual(v["p_pow"], 2.0, places=6)        # 1.0 + 1.0
        self.assertAlmostEqual(v["e_pow"], 0.5, places=6)
        self.assertAlmostEqual(v["l_pow"], 0.125, places=6)
        # disc and q are RATIOS: identical to one instance's. That is the documented property
        # of this combine -- summing shrinks the variance, it does not raise q.
        self.assertAlmostEqual(v["disc"], (0.25 - 0.0625) / 0.3125, places=6)
        self.assertAlmostEqual(v["q"], 2 * 1.0 / 0.3125, places=6)

    def test_one_instance_is_not_a_fleet(self):
        c = client_with([make_frame(inst="cx19.0")])
        self.assertEqual(combdll.fleet_dll_comb(c, "gps_l5", n_win=4, lag=0,
                                                min_instances=2), {})

    def test_channels_from_both_instances_appear_once_each(self):
        v = self._fleet()[1]
        self.assertEqual(sorted(v["chan"]), [5972, 5976, 5988, 5992, 6004, 6008])
        self.assertEqual(v["chan_dup"], [])
        self.assertEqual(len(combdll.chan_profile(v)), 6)

    def test_a_freq_id_claimed_twice_is_dropped_and_named(self):
        """Two instances must never hold one channel. If they do, say so and drop it.

        Adding the two together would report a power that is neither instance's measurement,
        and it would look perfectly ordinary in the profile.
        """
        frames = [make_frame(inst="cx19.0", chan_ids=(5972, 5988)),
                  make_frame(inst="cx42.0", chan_ids=(5988, 6004))]
        v = combdll.fleet_dll_comb(client_with(frames), "gps_l5", n_win=4, lag=0)[1]
        self.assertEqual(v["chan_dup"], [5988])
        self.assertEqual(sorted(v["chan"]), [5972, 6004])
        self.assertAlmostEqual(v["p_pow"], 2.0, places=6)   # the full-band sum is untouched

    def test_presence_policy_is_the_shared_one(self):
        """The keys apply_presence writes must be here -- it is the SAME function fleet_dll
        calls, so a comb row and a polled row cannot reach different verdicts from equal
        numbers."""
        v = self._fleet()[1]
        for k in ("q_floor", "q_med", "present", "present_gate", "p_floor", "p_med"):
            self.assertIn(k, v)

    def test_deep_statistics_are_supplied_never_invented(self):
        frames = [make_frame(inst="cx19.0", chan_ids=(5972,)),
                  make_frame(inst="cx42.0", chan_ids=(5988,))]
        c = client_with(frames)
        bare = combdll.fleet_dll_comb(c, "gps_l5", n_win=4, lag=0)[1]
        for k in combdll.COH_KEYS:
            self.assertIsNone(bare[k], k)
        polled = {1: {"coh_row": {"deep_snr": 40.0, "deep_floor": 2.0, "coherence_s": 1.0},
                      "coh_src": "http://cx19:12048/x", "coh_quad": (55.0, 7)}}
        fed = combdll.fleet_dll_comb(c, "gps_l5", n_win=4, lag=0, coh_from=polled,
                                     deep_gate_prns=True)[1]
        self.assertEqual(fed["present_gate"], "deep")
        self.assertTrue(fed["present"])
        # ALL THREE keys travel. coh_quad is the publisher's continuity fallback; losing it
        # is a 5-8 dB step in the served C/N0 that looks like a display quirk (docs 11.31).
        self.assertEqual(fed["coh_src"], "http://cx19:12048/x")
        self.assertEqual(fed["coh_quad"], (55.0, 7))


class TestPhaseSensitivity(unittest.TestCase):
    """C. What the frequency axis is FOR: a delay across the comb shows up here."""

    def test_a_ramp_across_channels_suppresses_the_full_band_sum(self):
        """Per-channel powers are blind to a delay; the coherent sum is not.

        This is the measurement the tracker's cross-channel sum made impossible to see: with
        the comb summed on the node, a full-band sum suppressed by a phase ramp is
        indistinguishable from a satellite that is simply faint.
        """
        ids = (5972, 5988, 6004, 6020)
        taps = {}
        for r in range(4):
            for ch in range(len(ids)):
                rot = cmath.exp(2j * cmath.pi * ch / len(ids))   # a full turn across the comb
                taps[(r, 0, ch)] = (0j, rot, 0j, 1.0)
        c = client_with([make_frame(chan_ids=ids, taps=taps, n_prn=1)])
        d = combdll.instance_taps(c, "gps_l5", [100])[1]["cx19.0"]
        self.assertAlmostEqual(d["p"], 0.0, places=9)
        for fid in ids:
            self.assertAlmostEqual(d["chan"][fid][1], 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
