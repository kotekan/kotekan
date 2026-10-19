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
  * Slots at a symbol edge: when the symbol edge sits near half a slot on the UTC grid, an emit
    whose start-time estimate moved across that point still lands on the slots of the history
    it overlaps (navobs_align.emit_offset); an emit that overlaps nothing keeps its rounded
    slot.
  * B-CNAV2 across an ephemeris update: once type 10's IODE changes, the type 11 cached before
    it is not paired with it; a type 11 that follows it is.
  * B-CNAV1 and CNAV-2 polarity: emits whose carrier sign flips as the phase drifts are
    chained by rot, so frames spanning several flips decode; a wrong link (a phase step the
    chain misreads) is found by the boundary-flip repair, on abutting and on overlapping
    emits.
  * Rolling emits: CNAV-2 stores each symbol once and bounds its cache in symbols, so a run of
    overlapping emits holds whole frames; CNAV merges an emit into the island it overlaps, so
    its decode window spans enough symbols, and its cache cap trims the oldest island it keeps
    rather than dropping it whole.
"""
import contextlib
import io
import math
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
import cnav2_predictor  # noqa: E402
import cnav_predictor  # noqa: E402
import fnav_predictor  # noqa: E402
import inav_predictor  # noqa: E402
import navobs_align  # noqa: E402
from galileo_inav import crc24q  # noqa: E402

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
                names = [k for k in vars(st) if k.startswith("seen") or k == "tried"]
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
        (phase, g2) and only the oldest emit is resolved, so every island of the trailing
        WIN_SYMS window is free. Returns the number of messages one decode attempt
        recovers."""
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
        self.assertLess(starts[1], starts[-1] + 201 - cnav_predictor.WIN_SYMS)
        pred._try_decode(prn, st)
        return st.n_decoded

    def test_window_with_true_sign_decodes(self):
        self.assertGreater(self._decode(1), 0)

    def test_window_with_negated_sign_decodes(self):
        self.assertGreater(self._decode(-1), 0)


class TestEmitOffset(unittest.TestCase):
    def setUp(self):
        self.h = np.random.RandomState(3).choice([-1, 1], 200).astype(np.int8)

    def test_shifted_negated_copy_is_found(self):
        # the emit holds history slots 51..150, negated, but rounded one slot low (50)
        new = -self.h[51:151]
        self.assertEqual(navobs_align.emit_offset(50, new, [(0, self.h)]), 1)
        self.assertEqual(navobs_align.emit_offset(52, new, [(0, self.h)]), -1)
        self.assertEqual(navobs_align.emit_offset(51, new, [(0, self.h)]), 0)

    def test_no_overlap_keeps_the_rounded_slot(self):
        # abutting: a one-slot shift would share a single (always agreeing) symbol
        new = self.h[:100].copy()
        new[0] = self.h[199]
        self.assertEqual(navobs_align.emit_offset(200, new, [(0, self.h)]), 0)

    def test_tie_keeps_the_rounded_slot(self):
        alt = np.array([1, -1] * 100, dtype=np.int8)
        self.assertEqual(navobs_align.emit_offset(50, alt[:100], [(0, alt)]), 0)

    def test_no_agreeing_offset_keeps_the_rounded_slot(self):
        other = np.random.RandomState(4).choice([-1, 1], 100).astype(np.int8)
        self.assertEqual(navobs_align.emit_offset(50, other, [(0, self.h)]), 0)


EDGE = 0.05  # symbol edges sit this far (in symbols) past half a slot on the UTC grid
UTC0 = 1.7e9  # an absolute UTC origin, as the combiner stamps utc_ref


def _edge_obs(pm, a, n, br, rec_dt, sign=1, slipped=False):
    """nav_obs of symbols pm[a:a+n] (absolute slot UTC0/sym_s + a + 1 each): symbol k starts at
    (UTC0/sym_s + k + 0.5 + EDGE) * sym_s. A slipped emit reads its start time 2*EDGE symbols
    early, across the half-slot point, so its symbols round one slot low."""
    sym_s = br * rec_dt
    frac = 0.5 - EDGE if slipped else 0.5 + EDGE
    return {
        "utc_ref": UTC0 + (a + frac) * sym_s,
        "rec_dt": rec_dt,
        "phase": 0,
        "br": br,
        "bits": [[i, int(sign * pm[a + i])] for i in range(n)],
    }


# predictor class, its decode hook, br, rec_dt (each signal's nav_obs geometry)
EDGE_CASES = (
    (fnav_predictor.FnavPredictor, "_decode", 20, 0.001),
    (bcnav2_predictor.Bcnav2Predictor, "_decode", 5, 0.001),
    (bcnav1_predictor.Bcnav1Predictor, "_decode", 1, 0.010),
    (cnav_predictor.CnavPredictor, "_try_decode", 1, 0.020),
)


class TestSlotAtSymbolEdge(unittest.TestCase):
    def _placed(self, cls, hook, br, rec_dt, n, step, slip):
        """Feed 100-symbol emits every `step` symbols with random carrier signs, emit `slip`
        slipped; return the stored emits as (offset from the true slot of pm[0], signs)."""
        rng = np.random.RandomState(9)
        pm = rng.choice([-1, 1], n).astype(np.int8)
        pred = cls()
        setattr(pred, hook, lambda prn, st: None)
        for j, a in enumerate(range(0, n - 100 + 1, step)):
            sign = int(rng.choice([-1, 1]))
            pred.ingest(5, _edge_obs(pm, a, 100, br, rec_dt, sign, slipped=j == slip))
        origin = int(round(UTC0 / (br * rec_dt))) + 1
        return pm, {s0 - origin: arr for s0, arr in pred._p[5].emits.items()}

    def test_slipped_emit_lands_on_the_history(self):
        for cls, hook, br, rec_dt in EDGE_CASES:
            with self.subTest(cls=cls.__name__):
                pm, emits = self._placed(cls, hook, br, rec_dt, 600, 50, slip=4)
                # CNAV merges overlapping emits into one island; the others keep each emit
                held = sorted(
                    {a + i for a, arr in emits.items() for i in range(len(arr))}
                )
                self.assertEqual(held, list(range(600)))
                for a, arr in emits.items():
                    seg = pm[a : a + len(arr)].astype(int)
                    self.assertEqual(abs(int(np.dot(arr, seg))), len(arr), a)

    def test_emit_without_overlap_keeps_its_rounded_slot(self):
        for cls, hook, br, rec_dt in EDGE_CASES:
            with self.subTest(cls=cls.__name__):
                _pm, emits = self._placed(cls, hook, br, rec_dt, 600, 100, slip=2)
                self.assertEqual(sorted(emits), [0, 100, 199, 300, 400, 500])


class TestBcnav2DecodeAtSymbolEdge(unittest.TestCase):
    def test_every_frame_decodes_across_a_slipped_emit(self):
        rng = np.random.RandomState(12)
        pm = np.concatenate([_bcnav2_frame(10 + k % 2, rng=rng) for k in range(6)])
        pred = bcnav2_predictor.Bcnav2Predictor()
        pred._decode = lambda prn, st: None
        for j, a in enumerate(range(0, len(pm) - 200 + 1, 100)):
            pred.ingest(30, _edge_obs(pm, a, 200, 5, 0.001, slipped=j == 13))
        del pred._decode
        pred._decode(30, pred._p[30])
        self.assertEqual(pred.health(30)["words"], 6)


def _bcnav2_frame(mt, iode=0, word=0, rng=None):
    """One B-CNAV2 frame (PRN 30) as +-1 symbols: type 10 carries `iode` in its IODE field,
    type 11 carries `word` in its Omega_0 field, the rest of the payload is zero or, given
    rng, random."""
    B = bcnav2_predictor.B
    B._NB.init_gf()
    f = _bits(30, 6) + _bits(mt, 6) + [0] * (B.FRAME_BITS - 12)
    if rng is not None:
        f[12 : B.CRC_AT] = [int(x) for x in rng.randint(0, 2, B.CRC_AT - 12)]
    if mt == 10:
        start, length = B._IODE_10
        f[start : start + length] = _bits(iode, length)
    else:
        _mt, start, length, _s, _sc = B.BCNV2_EPH_FIELDS["Omega_0"]
        f[start : start + length] = _bits(word, length)
    f[B.CRC_AT : B.CRC_AT + 24] = _bits(B.crc24q(f[: B.CRC_AT]), 24)
    info = B._NB.bin2gf(np.array(f, dtype=np.uint8))
    body = B._NB.encode_systematic(
        B.H_BCNV2_IDX, B.H_BCNV2_ELE, B.N_CHECK, B.N_VAR, info
    )
    return np.where(np.concatenate([np.array(B.PREAMBLE), body]) == 0, 1, -1)


class TestBcnav2IodeChange(unittest.TestCase):
    def setUp(self):
        self.pred = bcnav2_predictor.Bcnav2Predictor()
        self.pred._min_gap = 0.0
        self.e = 0

    def _feed(self, *frames):
        for pm in frames:
            obs = {
                "utc_ref": self.e * bcnav2_predictor.SYM_S,
                "rec_dt": 0.001,
                "phase": 0,
                "br": 5,
                "bits": [[i, int(s)] for i, s in enumerate(pm)],
            }
            self.pred.ingest(30, obs)
            self.e += len(pm)

    def _omega0(self):
        eph = self.pred.ephemeris(30)
        return None if eph is None else (eph["IODE"], eph["Omega_0"])

    def test_type_11_from_before_the_update_is_not_paired(self):
        old11, new11 = _bcnav2_frame(11, word=1 << 20), _bcnav2_frame(11, word=1 << 21)
        self._feed(_bcnav2_frame(10, iode=7), old11, _bcnav2_frame(10, iode=8))
        self.assertEqual(self.pred.health(30)["have"], [10])
        self.assertIsNone(self._omega0())
        self.pred._decode(30, self.pred._p[30])  # a rescan finds the old type 11 again
        self.assertIsNone(self._omega0())
        # the bounded emit cache drops its oldest emit, the old type 10, first: a rescan then
        # finds the old type 11 with no type 10 before it in the stream
        emits = self.pred._p[30].emits
        del emits[min(emits)]
        self.pred._decode(30, self.pred._p[30])
        self.assertIsNone(self._omega0())
        self._feed(new11)
        scale = bcnav2_predictor.B.BCNV2_EPH_FIELDS["Omega_0"][4]
        self.assertEqual(self._omega0(), (8, (1 << 21) * scale))
        self.pred._decode(30, self.pred._p[30])
        self.assertEqual(self._omega0(), (8, (1 << 21) * scale))

    def test_same_iode_keeps_the_type_11(self):
        self._feed(
            _bcnav2_frame(10, iode=7),
            _bcnav2_frame(11, word=1 << 20),
            _bcnav2_frame(10, iode=7),
        )
        self.assertEqual(self.pred.health(30)["have"], [10, 11])
        self.assertEqual(self._omega0()[0], 7)


def _carrier_obs(pm, a, n, rec_dt, phi):
    """nav_obs of symbols pm[a:a+n] (br=1) made as the combiner makes an emit at carrier phase
    phi: theta0 = phi reduced to [-pi/2, pi/2), rot = e^{-i*theta0}, and the signs taken
    against theta0, so the emit's sign is (-1)^k for phi = theta0 + k*pi."""
    k = math.floor(phi / math.pi + 0.5)
    theta0, sign = phi - math.pi * k, 1 - 2 * (k % 2)
    return {
        "utc_ref": a * rec_dt,
        "rec_dt": rec_dt,
        "phase": 0,
        "br": 1,
        "rot_re": math.cos(theta0),
        "rot_im": -math.sin(theta0),
        "bits": [[i, int(sign * pm[a + i])] for i in range(n)],
    }


def _feed_carrier(pred, prn, pm, n, step, rec_dt, drift, jump_at=None):
    """Feed n-symbol emits every `step` symbols while the carrier phase advances `drift` per
    emit, plus 2 rad before the emit starting at symbol jump_at: a step the chain reads as the
    wrong number of half-turns."""
    phi = 0.3
    for a in range(0, len(pm) - n + 1, step):
        if a == jump_at:
            phi += 2.0
        pred.ingest(prn, _carrier_obs(pm, a, n, rec_dt, phi))
        phi += drift


def _bcnav1_stream(prn, n, rng):
    """n B-CNAV1 frames for prn (SOH 0..n-1, each a random CRC-valid SF2) as +-1 symbols."""
    B = bcnav1_predictor.B
    B._NB.init_gf()
    out = []
    for soh in range(n):
        sf2 = [int(x) for x in rng.randint(0, 2, B.SF2_CRC_AT)]
        sf2 += _bits(crc24q(sf2), 24)
        info = B._NB.bin2gf(np.array(sf2, dtype=np.uint8))
        syms2 = B._NB.encode_systematic(B.H_SF2_IDX, B.H_SF2_ELE, 100, 200, info)
        il = B._interleave(syms2, np.zeros(528, dtype=np.uint8))
        bits = np.concatenate([B._sf1a(prn), B._sf1b_all()[soh], il])
        out.append(np.where(bits == 0, 1, -1))
    return np.concatenate(out)


class TestBcnav1Polarity(unittest.TestCase):
    PRN = 20

    def _decode(self, step, jump_at=None):
        """Three frames fed as 100-symbol emits every `step` symbols, the carrier drifting
        0.4 rad per emit (so every frame spans several sign flips). Returns the distinct
        frames decoded and the cached run's agreement with the truth."""
        bcnav1_predictor.B._INVERT[0] = None
        pm = _bcnav1_stream(self.PRN, 3, np.random.RandomState(7))
        pred = bcnav1_predictor.Bcnav1Predictor()
        pred._min_gap = 0.0
        _feed_carrier(pred, self.PRN, pm, 100, step, 0.010, 0.4, jump_at)
        st = pred._p[self.PRN]
        run0, signs = pred._runs(st)[-1][:2]
        agree = abs(int(np.dot(signs, pm[run0 : run0 + len(signs)])))
        return pred.health(self.PRN)["words"], agree == len(signs)

    def test_abutting_emits_decode_through_the_chain(self):
        self.assertEqual(self._decode(100), (3, True))

    def test_wrong_link_is_repaired(self):
        # the step lands inside the second frame (symbols 1800-3599)
        self.assertEqual(self._decode(100, jump_at=2300), (3, True))

    def test_wrong_link_is_repaired_on_overlapping_emits(self):
        self.assertEqual(self._decode(25, jump_at=2300), (3, True))


def _cnav2_stream(n, rng):
    """n CNAV-2 frames (TOI 100, 101, ...), the next frame's SF1 and 100 random symbols, as 0/1
    symbols."""
    C2 = cnav2_predictor.C2
    out = []
    for toi in range(100, 100 + n):
        parts = []
        for k, nk in ((C2.K_SF2, C2.N_SF2), (C2.K_SF3, C2.N_SF3)):
            m = [int(x) for x in rng.randint(0, 2, k - 24)]
            parts += (
                m + _bits(crc24q(m), 24) + [int(x) for x in rng.randint(0, 2, nk - k)]
            )
        il = np.array(parts, dtype=np.int8).reshape(C2.IL_COLS, C2.IL_ROWS).T.ravel()
        out.append(np.hstack([C2._sf1(toi), il]))
    out += [C2._sf1(100 + n), rng.randint(0, 2, 100)]
    return np.hstack(out).astype(np.int8)


class TestCnav2(unittest.TestCase):
    def _tois(self, step, drift, jump_at=None):
        """Three frames fed as 100-symbol emits every `step` symbols; returns the TOIs of the
        frames decoded."""
        pm = 1 - 2 * _cnav2_stream(3, np.random.RandomState(8))
        pred = cnav2_predictor.Cnav2Predictor()
        pred._min_gap = 0.0
        got = set()
        find = cnav2_predictor.C2.find_frames

        def recording(syms):
            for f in find(syms):
                got.add(f[1])
                yield f

        with mock.patch.object(cnav2_predictor.C2, "find_frames", recording):
            _feed_carrier(pred, 23, pm, 100, step, 0.010, drift, jump_at)
        self.pred = pred
        self.pm = pm
        return sorted(got)

    def _cache_agrees(self):
        """Whether the cached run matches the truth at one sign: the repair flipped the cache
        and the chain, so the emits after it splice on."""
        run0, syms = self.pred._runs(self.pred._p[23])[-1][:2]
        seg = self.pm[run0 : run0 + len(syms)].astype(int)
        return abs(int(np.dot(1 - 2 * syms.astype(int), seg))) == len(syms)

    def test_abutting_emits_decode_through_the_chain(self):
        self.assertEqual(self._tois(100, 0.4), [100, 101, 102])
        self.assertTrue(self._cache_agrees())

    def test_wrong_link_is_repaired(self):
        self.assertEqual(self._tois(100, 0.4, jump_at=2300), [100, 101, 102])
        self.assertTrue(self._cache_agrees())

    def test_wrong_link_is_repaired_on_overlapping_emits(self):
        self.assertEqual(self._tois(25, 0.1, jump_at=2300), [100, 101, 102])
        self.assertTrue(self._cache_agrees())

    def test_overlapping_emits_hold_whole_frames(self):
        # emits as a 0.25 s poll reads the rolling L1C-D combiner (1 s window), at one sign
        self.assertEqual(self._tois(25, 0.0), [100, 101, 102])
        emits = self.pred._p[23].emits
        held = [a + i for a, arr in emits.items() for i in range(len(arr))]
        self.assertEqual(len(held), len(set(held)))
        self.assertLessEqual(len(held), cnav2_predictor.SYM_MAX)

    def test_tried_pruned_with_the_cache(self):
        pred = cnav2_predictor.Cnav2Predictor()
        pred._decode = lambda prn, st: None
        pred.ingest(5, _emit(0.0, n=100))
        st = pred._p[5]
        st.tried.update((-1, 10 ** 9))
        for j in range(1, cnav2_predictor.SYM_MAX // 100 + 2):
            pred.ingest(5, _emit(j * 0.1, n=100))
        self.assertNotIn(0, st.emits)
        self.assertNotIn(-1, st.tried)
        self.assertIn(10 ** 9, st.tried)


class TestCnavRollingEmits(unittest.TestCase):
    """L2C emits as the rolling combiner makes them at integration_length 200: each holds the
    200-symbol deep window, a new one every 20 symbols, each at its own carrier sign."""

    def _feed(self, pred, pm):
        rng = np.random.RandomState(5)
        for a in range(0, len(pm) - 200 + 1, 20):
            sign = int(rng.choice([-1, 1]))
            pred.ingest(7, _edge_obs(pm, a, 200, 1, 0.020, sign=sign))

    def test_messages_decode(self):
        pm = _cnav_symbols(7)
        pred = cnav_predictor.CnavPredictor()
        pred._min_gap = 0.0
        self._feed(pred, pm)
        h = pred.health(7)
        self.assertGreaterEqual(h["messages"], 4)
        self.assertTrue(h["eph"])

    def test_one_island_bounded_in_symbols(self):
        pm = _cnav_symbols(7)
        pred = cnav_predictor.CnavPredictor()
        pred._try_decode = lambda prn, st: None
        with mock.patch.object(cnav_predictor, "SYM_MAX", 1000):
            self._feed(pred, pm)
        ((s0, arr),) = pred._p[7].emits.items()
        self.assertEqual(len(arr), 1000)
        # one past the newest emit's last symbol
        last = (len(pm) - 200) // 20 * 20 + 200
        origin = int(round(UTC0 / 0.020)) + 1
        seg = pm[last - 1000 : last].astype(int)
        self.assertEqual(s0 - origin, last - 1000)
        self.assertEqual(abs(int(np.dot(arr, seg))), 1000)

    def test_cap_trims_a_resolved_island(self):
        # A resolved island at the cap, then an emit that overlaps it but disagrees on 40% of
        # its symbols, so it stays an island of its own: the cap trims the resolved island's
        # head and keeps its sign rather than dropping it whole.
        pm = _cnav_symbols(7)
        pred = cnav_predictor.CnavPredictor()
        pred._try_decode = lambda prn, st: None
        with mock.patch.object(cnav_predictor, "SYM_MAX", 1000):
            for a in range(0, 1400, 20):
                pred.ingest(7, _edge_obs(pm, a, 200, 1, 0.020))
            st = pred._p[7]
            ((s0, arr),) = st.emits.items()
            st.pol[s0] = 1
            obs = _edge_obs(pm, 1400, 200, 1, 0.020)
            for k in np.random.RandomState(1).choice(200, 80, replace=False):
                obs["bits"][k][1] = -obs["bits"][k][1]
            pred.ingest(7, obs)
        self.assertEqual(
            sorted((s, len(a)) for s, a in st.emits.items()),
            [(s0 + 200, 800), (s0 + 820, 200)],
        )
        self.assertEqual(st.pol, {s0 + 200: 1})


if __name__ == "__main__":
    unittest.main()
