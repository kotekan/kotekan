#!/usr/bin/env python3
"""LnavPredictor sync/decode bookkeeping.

  * Each subframe is decoded and scored once, although every try rescans the trailing
    SCAN_MAX_BITS window: decoded_sf counts distinct subframes, and the predict-mismatch
    score counts each subframe's bits once.
  * A subframe that starts on the run's first bit decodes with D29*=D30*=0, as frame_sync
    synced it, whatever the run's last bits are.
  * Slots at a bit edge: when the bit edge sits near half a slot on the UTC grid, an emit whose
    start-time estimate moved across that point still writes the history's slots.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navbit_predictor as N  # noqa: E402

TOW0 = 4000
PRN = 7


def _words(rng):
    return [rng.randint(0, 2, 24).astype(np.int8) for _ in range(10)]


def _subframe(sfid, tow, words):
    """One transmitted subframe (300 bits 0/1) carrying these 10 data words."""
    data = [np.array(w, dtype=np.int8).copy() for w in words]
    data[0][:8] = N.PREAMBLE
    data[0][8:] = 0
    data[1][17:19] = 0
    data[1] = N.set_how_tow(data[1], tow)
    data[1][19:22] = [int(x) for x in format(sfid, "03b")]
    return N.encode_subframe_trimmed(data)


def _stream(n_frames, sf2_frame3=None):
    """n_frames LNAV frames: subframes 1-3 constant (subframe 2 replaced from frame 3 on by
    sf2_frame3 when given), subframes 4/5 a fresh page each frame."""
    rng = np.random.RandomState(7)
    sf = {k: _words(rng) for k in (1, 2, 3)}
    out = []
    for fr in range(n_frames):
        for sfid in range(1, 6):
            tow = TOW0 + 5 * fr + sfid - 1
            if sfid == 2 and fr >= 2 and sf2_frame3 is not None:
                words = sf2_frame3
            elif sfid <= 3:
                words = sf[sfid]
            else:
                words = _words(rng)
            out.append(_subframe(sfid, tow, words))
    return np.concatenate(out), sf


def _feed(bits01):
    """Feed 0/1 bits as 50-bit emits with one carrier sign (rot = 1)."""
    pred = N.LnavPredictor()
    pm = 1 - 2 * bits01.astype(int)
    for e in range(0, len(pm), 50):
        obs = {
            "utc_ref": e * N.BIT_S,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 20,
            "bits": [[i, int(pm[e + i])] for i in range(min(50, len(pm) - e))],
        }
        pred.ingest(PRN, obs)
    return pred


class TestDecodeOnce(unittest.TestCase):
    def test_each_subframe_decoded_and_scored_once(self):
        sf = _stream(1)[1]
        sf2_new = [w.copy() for w in sf[2]]
        sf2_new[5] = 1 - sf2_new[5]  # a data-set change in frame 3
        frames, _ = _stream(3, sf2_frame3=sf2_new)
        # 50 bits of 0 ahead (the chain bits before subframe 1 are 00, as after any
        # subframe) and 300 after, so the last subframe is complete before the last try
        bits01 = np.concatenate([np.zeros(50, np.int8), frames, np.zeros(300, np.int8)])
        pred = _feed(bits01)
        h = pred.health(PRN)
        self.assertEqual(h["decoded_sf"], 15)
        # Scored: frames 2 and 3, subframes 1-3 in full (predicted from the frame before)
        # and subframes 4/5 on TLM+HOW only (their pages are new).
        st = pred._p[PRN]
        self.assertEqual(st.n_pred_checked, 2 * (3 * 300 + 2 * 60))
        # The only wrong bits: frame 3's subframe 2, predicted from frame 2's.
        tow = TOW0 + 5 * 2 + 1
        w = int((_subframe(2, tow, sf[2]) != _subframe(2, tow, sf2_new)).sum())
        self.assertGreater(w, 0)
        self.assertEqual(st.n_pred_wrong, min(w, 300 - w))


class TestRunStartsOnSubframe(unittest.TestCase):
    def test_first_subframe_decodes_when_run_ends_in_ones(self):
        frames, _ = _stream(1)
        bits01 = np.concatenate([frames[: 3 * N.SF_BITS], [1, 1]]).astype(np.int8)
        pred = N.LnavPredictor()
        st = pred._p.setdefault(PRN, N._PrnState())
        st.hist = {s: 1 - 2 * int(b) for s, b in enumerate(bits01)}
        pred._try_sync(PRN, st)
        self.assertIn(1, st.sf_data)
        self.assertEqual(st.n_decoded, 3)


class TestSlotAtBitEdge(unittest.TestCase):
    UTC0 = 1.7e9  # an absolute UTC origin, as the combiner stamps utc_ref

    def _feed_edge(self, pm, slip):
        """100-bit emits every 50 bits; bit k starts 0.05 bit past half a slot on the UTC
        grid, except in emit `slip`, which reads its start time 0.1 bit early."""
        pred = N.LnavPredictor()
        for j, a in enumerate(range(0, len(pm) - 100 + 1, 50)):
            frac = 0.45 if j == slip else 0.55
            obs = {
                "utc_ref": self.UTC0 + (a + frac) * N.BIT_S,
                "rec_dt": 0.001,
                "phase": 0,
                "br": 20,
                "bits": [[i, int(pm[a + i])] for i in range(100)],
            }
            pred.ingest(PRN, obs)
        return pred

    def test_slipped_emit_lands_on_the_history(self):
        frames, _ = _stream(2)
        pm = 1 - 2 * frames.astype(int)
        pred = self._feed_edge(pm, slip=20)
        origin = int(round(self.UTC0 / N.BIT_S)) + 1
        want = {origin + k: int(v) for k, v in enumerate(pm)}
        self.assertEqual(pred._p[PRN].hist, want)
        clean = self._feed_edge(pm, slip=None).health(PRN)["decoded_sf"]
        self.assertGreater(clean, 0)
        self.assertEqual(pred.health(PRN)["decoded_sf"], clean)


if __name__ == "__main__":
    unittest.main()
