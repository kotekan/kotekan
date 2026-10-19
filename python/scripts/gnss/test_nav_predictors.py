#!/usr/bin/env python3
"""Nav-message predictors (I/NAV, F/NAV, B-CNAV1/2/3, CNAV): decode scheduling and health.

  * The decode throttle is per PRN: the broker ingests every PRN back to back each poll, so
    two PRNs whose emits land within the same second must both decode, and a PRN is still
    held to one decode per _min_gap.
  * Health counts distinct pages/frames: every decode rescans the cached emits, and a page
    found again on a later pass is not counted again.
  * B-CNAV2 stores a frame only under the PRN the frame itself names.
  * CNAV: a warm PRN whose decode window holds no resolved emit still tries the window's
    global sign.
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bcnav1_predictor  # noqa: E402
import bcnav2_predictor  # noqa: E402
import bcnav3_predictor  # noqa: E402
import cnav_predictor  # noqa: E402
import fnav_predictor  # noqa: E402
import inav_predictor  # noqa: E402

GATED = (
    inav_predictor.InavPredictor,
    fnav_predictor.FnavPredictor,
    bcnav2_predictor.Bcnav2Predictor,
    bcnav1_predictor.Bcnav1Predictor,
    bcnav3_predictor.Bcnav3Predictor,
)


def _bits(v, n):
    return [(v >> (n - 1 - k)) & 1 for k in range(n)]


def _emit(utc_ref, n=8):
    """A gap-free nav_obs emit of n symbols (br=1, 1 ms records)."""
    return {
        "utc_ref": utc_ref,
        "rec_dt": 0.001,
        "phase": 0,
        "br": 1,
        "bits": [[i, 1 - 2 * (i % 2)] for i in range(n)],
    }


class TestPerPrnGate(unittest.TestCase):
    def test_two_prns_in_one_second_both_decode(self):
        for cls in GATED:
            with self.subTest(cls=cls.__name__):
                pred = cls()
                seen = []
                pred._decode = lambda prn, st: seen.append(prn)
                with mock.patch("time.time", return_value=1000.0):
                    pred.ingest(5, _emit(10.0))
                    pred.ingest(7, _emit(10.0))
                self.assertEqual(seen, [5, 7])

    def test_one_prn_still_throttled(self):
        for cls in GATED:
            with self.subTest(cls=cls.__name__):
                pred = cls()
                seen = []
                pred._decode = lambda prn, st: seen.append(prn)
                for wall, utc in ((1000.0, 10.0), (1000.5, 11.0), (1001.5, 12.0)):
                    with mock.patch("time.time", return_value=wall):
                        pred.ingest(5, _emit(utc))
                self.assertEqual(seen, [5, 5])


def _run_selftest(mod, cls):
    """Run a predictor module's self-test quietly; return (passed, the predictor it fed)."""
    made = []

    class Recording(cls):
        def __init__(self):
            super().__init__()
            made.append(self)

    with mock.patch.object(mod, cls.__name__, Recording):
        with contextlib.redirect_stdout(io.StringIO()):
            ok = mod._selftest()
    return ok, made[0]


class TestDistinctCounts(unittest.TestCase):
    # module, class, CRC-valid pages/frames in the module's self-test stream
    CASES = (
        (inav_predictor, inav_predictor.InavPredictor, 12),  # 3 x word types 1-4
        (fnav_predictor, fnav_predictor.FnavPredictor, 12),  # 3 x page types 1-4
        (bcnav2_predictor, bcnav2_predictor.Bcnav2Predictor, 6),  # 3 x types 10, 11
        (bcnav1_predictor, bcnav1_predictor.Bcnav1Predictor, 2),  # 2 frames
        (bcnav3_predictor, bcnav3_predictor.Bcnav3Predictor, 6),  # 3 x types 10, 30
    )

    def test_rescans_do_not_count_again(self):
        for mod, cls, n_frames in self.CASES:
            with self.subTest(cls=cls.__name__):
                ok, pred = _run_selftest(mod, cls)
                self.assertTrue(ok)
                (prn,) = pred._p
                h0 = pred.health(prn)
                self.assertEqual(h0["words"], n_frames)
                for _ in range(2):
                    pred._decode(prn, pred._p[prn])
                h1 = pred.health(prn)
                self.assertEqual((h1["pages"], h1["words"]), (h0["pages"], h0["words"]))

    def test_seen_slots_pruned_with_the_emit_cache(self):
        for mod, cls, _n in self.CASES:
            with self.subTest(cls=cls.__name__):
                pred = cls()
                pred._decode = lambda prn, st: None
                pred.ingest(5, _emit(0.0))
                st = pred._p[5]
                names = [k for k in vars(st) if k.startswith("seen")]
                self.assertTrue(names)
                # -1 lies before every cached emit once the first one is evicted; 10**9 lies
                # after all of them, so it must be kept
                for k in names:
                    getattr(st, k).update((-1, 10 ** 9))
                for j in range(1, mod.EMIT_MAX + 1):
                    pred.ingest(5, _emit(float(j)))
                for k in names:
                    self.assertNotIn(-1, getattr(st, k))
                    self.assertIn(10 ** 9, getattr(st, k))


def _bcnav2_stream(prn_field):
    """B-CNAV2 frames of types 10 and 11 whose PRN field is prn_field, as +-1 symbols."""
    B = bcnav2_predictor.B
    B._NB.init_gf()
    out = []
    for mt in (10, 11):
        f = _bits(prn_field, 6) + _bits(mt, 6) + [0] * (B.FRAME_BITS - 12)
        f[B.CRC_AT : B.CRC_AT + 24] = _bits(B.crc24q(f[: B.CRC_AT]), 24)
        info = B._NB.bin2gf(np.array(f, dtype=np.uint8))
        body = B._NB.encode_systematic(
            B.H_BCNV2_IDX, B.H_BCNV2_ELE, B.N_CHECK, B.N_VAR, info
        )
        syms = np.concatenate([np.array(B.PREAMBLE, dtype=np.int8), body])
        out.append(np.where(syms == 0, 1, -1))
    return np.concatenate(out)


class TestBcnav2FramePrn(unittest.TestCase):
    def test_frame_for_another_prn_is_not_stored(self):
        pm = _bcnav2_stream(30)
        obs = {
            "utc_ref": 0.0,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 5,
            "bits": [[i, int(s)] for i, s in enumerate(pm)],
        }
        for prn, want in ((30, [10, 11]), (31, [])):
            with self.subTest(prn=prn):
                pred = bcnav2_predictor.Bcnav2Predictor()
                pred._min_gap = 0.0
                pred.ingest(prn, obs)
                self.assertEqual(pred.health(prn)["have"], want)


def _cnav_symbols(prn):
    """A continuous CNAV message stream as transmitted +-1 symbols (module convention)."""
    C = cnav_predictor.C
    rng = np.random.RandomState(11)
    bits = []
    for k, mt in enumerate((10, 11, 30, 33, 10, 11)):
        data = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
        bits += C.build_cnav_message(prn, mt, 1000 + k, data=data)
    sym = cnav_predictor.conv_encode(bits, flush=True)
    return np.where(sym == 1, -1, 1).astype(np.int8)


class TestCnavGlobalSign(unittest.TestCase):
    def _decode(self, sign):
        """Ten abutting 201-symbol emits, all with carrier sign `sign`. The PRN is locked
        (phase, g2) and only the oldest emit is resolved, so the freshest WIN_EMITS are all
        free. Returns the number of messages one decode attempt recovers."""
        prn = 7
        pm = _cnav_symbols(prn)
        pred = cnav_predictor.CnavPredictor()
        pred._g2 = False
        st = pred._p.setdefault(prn, cnav_predictor._PrnState())
        st.phase = 0
        starts = [200 * k for k in range(10)]
        for s0 in starts:
            st.emits[s0] = (pm[s0 : s0 + 201] * sign).astype(np.int8)
        st.pol[starts[0]] = sign
        self.assertGreater(len(starts), cnav_predictor.WIN_EMITS)
        pred._try_decode(prn, st)
        return st.n_decoded

    def test_window_with_true_sign_decodes(self):
        self.assertGreater(self._decode(1), 0)

    def test_window_with_negated_sign_decodes(self):
        self.assertGreater(self._decode(-1), 0)


if __name__ == "__main__":
    unittest.main()
