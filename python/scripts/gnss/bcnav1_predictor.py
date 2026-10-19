"""Streaming BeiDou B1C B-CNAV1 decoder for the broker.

Mirrors bcnav2_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them
into an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav1.py is the codec (SF1 framesync + 36x48 de-interleave + GF(64) NB-LDPC SF2 +
CRC-24Q + BDS-3 CNAV-style ephemeris).

Simpler to serve than B-CNAV2: the whole orbit is in SF2 (one subframe), so any CRC-valid SF2
yields a full ephemeris -- no cross-frame type matching. The frame is 1800 symbols (18 s), and
find_frames needs the PRN (the SF1 sync pattern is PRN-specific).

Polarity. Each combiner emit resolves its +-1 ambiguity on its own, and a frame spans ~18 one-
second emits, so the emits are chained as navbit_predictor chains LNAV: rot = e^{-i*theta0}
follows the carrier, and an odd number of pi-steps in theta0 from the last emit flips this
emit's sign relative to it. A run then splices in one sign, which find_frames resolves. One
wrong link flips everything after it; the frame that straddles it fails its CRC, and a
boundary-flip repair (_repair) finds the link.

nav_obs: {utc_ref, rec_dt, br, phase, bits}. For B1C-data the combiner emits one symbol per
10 ms B1C code period (navwipe_bit_records 1, rec_dt=0.010, br=1 -> br*rec_dt = 0.010 = SYM_S).
"""

import math
import time

import numpy as np

import beidou_bcnav1 as B
from navobs_align import emit_offset

SYM_S = B.SYM_S  # 0.010
FRAME = B.FRAME_SYMS  # 1800 symbols per frame (18 s)
EMIT_MAX = 96  # bound the stitched emit cache per PRN
SF2_TTL_S = 3600.0  # a cached SF2 older than this is stale (ephemeris ~ hourly)


def _sf2_crc_as_received(frame):
    """Whether one frame (+-1, SF1 at its true polarity) carries an SF2 whose systematic bits,
    the first 600 of the codeword, pass CRC as received."""
    syms2, _syms3 = B._deinterleave((frame < 0).astype(np.uint8))
    inverts = (B._INVERT[0],) if B._INVERT[0] is not None else (False, True)
    return any(
        B.check_crc(syms2[: B.SF2_BITS] ^ int(inv), B.SF2_CRC_AT, B.SF2_BITS)
        for inv in inverts
    )


class _PrnState:
    def __init__(self):
        self.emits = {}
        self.last_obs = None
        self.theta = None  # last emit's theta0 (rad, (-pi/2, pi/2])
        self.pol = 1  # chained polarity applied to incoming emits
        self.sf2 = None  # (sf2_bits[600], t_decoded)
        self.n_frames = 0
        self.n_crc = 0
        # frame start slots already counted: every decode rescans the cached emits, so a frame
        # is counted only the first time it is found
        self.seen = set()
        # frame start slots the repair has searched: a complete frame's stored symbols do not
        # change, so each is searched once
        self.tried = set()
        self.last_decode = 0.0


class Bcnav1Predictor:
    def __init__(self):
        self._p = {}
        self._min_gap = 1.0

    # ----------------------------------------------------------------- ingest
    def ingest(self, prn, obs):
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"])
            rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"])
            br = int(obs["br"])
            rot = complex(float(obs.get("rot_re", 1.0)), float(obs.get("rot_im", 0.0)))
            pairs = obs["bits"]
        except (KeyError, TypeError, ValueError):
            return
        if rec_dt <= 0 or br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        # theta0 = -arg(rot), reduced mod pi: an odd pi-step from the last emit flips the chain
        theta = -math.atan2(rot.imag, rot.real)
        if st.theta is not None and round((theta - st.theta) / math.pi) % 2:
            st.pol = -st.pol
        st.theta = theta
        sym_s = br * rec_dt
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            slot = int(round((utc_ref + (bi * br - phase) * rec_dt) / sym_s))
            m[slot] = sg * st.pol
        if len(m) < 2:
            return
        slot0, hi = min(m), max(m)
        arr = np.zeros(hi - slot0 + 1, dtype=np.int8)
        for s, v in m.items():
            arr[s - slot0] = v
        if (arr == 0).any():
            return
        # a symbol edge near half a slot can round this emit one slot off the history it
        # overlaps; line it up with that history
        slot0 += emit_offset(slot0, arr, st.emits.items())
        st.emits[slot0] = arr
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
            lo = min(st.emits)
            st.seen = {s for s in st.seen if s >= lo}
            st.tried = {s for s in st.tried if s >= lo}
        # Per-PRN decode throttle (see inav_predictor.ingest): a shared gate starves every PRN but
        # the first in the broker's row order.
        now = time.time()
        if self._min_gap and now - st.last_decode < self._min_gap:
            return
        st.last_decode = now
        self._decode(prn, st)

    # ----------------------------------------------------------------- decode
    def _runs(self, st):
        """Contiguous runs of the cached emits: (start slot, signs, cuts), cuts holding
        (index into signs, emit slot0) where each later emit's first new symbol lands."""
        if not st.emits:
            return []
        runs = []
        cur0 = None
        cur = None
        cuts = []
        for s0 in sorted(st.emits):
            a = st.emits[s0]
            if cur is None:
                cur0, cur = s0, a.copy()
                continue
            end = cur0 + len(cur)
            if s0 <= end:
                ov = end - s0
                if ov < len(a):
                    cuts.append((len(cur), s0))
                    cur = np.concatenate([cur, a[ov:]])
            else:
                runs.append((cur0, cur, cuts))
                cur0, cur, cuts = s0, a.copy(), []
        runs.append((cur0, cur, cuts))
        return runs

    def _decode(self, prn, st):
        for run0, signs, cuts in self._runs(st):
            if len(signs) < FRAME:
                continue
            signs = signs.astype(float)
            found = set()
            for i, _pol, _soh, sf2 in B.find_frames(prn, signs):
                found.add(i)
                self._store(st, run0 + i, sf2)
            self._repair(prn, st, run0, signs, cuts, found)

    def _store(self, st, at, sf2):
        st.sf2 = (sf2, time.time())
        if at not in st.seen:
            st.seen.add(at)
            st.n_frames += 1
            st.n_crc += 1

    def _repair(self, prn, st, run0, signs, cuts, found):
        """CRC-guided boundary-flip repair of the run's frames that find_frames did not decode.

        A complete frame whose SF1 PRN sequence is in the run but whose SF2 fails may straddle
        one wrong chain link: an emit whose sign came out flipped, and every later emit with
        it. Flip the frame from each emit boundary inside it and take the first flip under
        which SF2 passes its CRC as received, without LDPC iterations (an errored candidate
        costs ~0.3 s of iterations; at deep-integration SNR the right flip is error-free).
        That emit and every later one are negated in the cache, the chain with them, and the
        repaired frame is decoded. signs is updated in place for the run's later frames.
        """
        sf1a = 1.0 - 2.0 * B._sf1a(prn)  # SF1's PRN sequence as received (bit 0 -> +1)
        c = np.correlate(signs, sf1a, "valid")
        for i in np.flatnonzero(np.abs(c) == len(sf1a)):
            i = int(i)
            if i + FRAME > len(signs) or i in found or run0 + i in st.tried:
                continue
            st.tried.add(run0 + i)
            pol = 1.0 if c[i] > 0 else -1.0
            for b, s0 in cuts:
                if not i < b < i + FRAME:
                    continue
                w = pol * signs[i : i + FRAME]
                w[b - i :] *= -1
                if not _sf2_crc_as_received(w):
                    continue
                for s in st.emits:
                    if s >= s0:
                        st.emits[s] = -st.emits[s]
                st.pol = -st.pol
                signs[b:] *= -1
                for _j, _pol, _soh, sf2 in B.find_frames(prn, signs[i : i + FRAME]):
                    self._store(st, run0 + i, sf2)
                break

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from the cached SF2 (the full orbit) if decoded within SF2_TTL_S."""
        st = self._p.get(prn)
        if st is None or st.sf2 is None:
            return None
        if time.time() - st.sf2[1] > SF2_TTL_S:
            return None
        return B.parse_bcnav1_ephemeris(st.sf2[0])

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_frames,
            "words": st.n_crc,
            "have": ([2] if st.sf2 is not None else []),
            "eph": self.ephemeris(prn) is not None,
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    B._NB.init_gf()
    rng = np.random.RandomState(9)
    prn, soh = 20, 88

    sf2 = [0] * B.SF2_BITS
    truth = {}
    for name, (start, length, signed, scale) in B.BCNV1_SF2_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            sf2[start + k] = (code >> (length - 1 - k)) & 1
        truth[name] = B._field(sf2, start, length, signed, scale)
    crc = B.crc24q(sf2[0 : B.SF2_CRC_AT])
    sf2[B.SF2_CRC_AT : B.SF2_CRC_AT + 24] = [(crc >> (23 - k)) & 1 for k in range(24)]
    syms2 = B._NB.encode_systematic(
        B.H_SF2_IDX, B.H_SF2_ELE, 100, 200, B._NB.bin2gf(np.array(sf2, dtype=np.uint8))
    )
    syms3 = np.zeros(528, dtype=np.uint8)  # SF3 (not decoded): filler coded symbols
    il = B._interleave(syms2, syms3)
    sf1 = np.concatenate([B._sf1a(prn), B._sf1b_all()[soh]])
    frame_bits = np.concatenate([sf1, il]).astype(np.uint8)
    # two frames so a full 1800 lands within the stitched run
    stream = np.tile(frame_bits, 2)
    pm = np.where(stream == 0, 1.0, -1.0)

    B._INVERT[0] = None
    pred = Bcnav1Predictor()
    pred._min_gap = 0.0
    # 1 s emits (100 syms) as the combiner makes them: the carrier phase phi drifts 0.4 rad
    # per emit, each emit reports theta0 = phi reduced to [-pi/2, pi/2) as rot = e^{-i*theta0}
    # and takes its signs against theta0, so its sign flips whenever phi crosses an odd
    # multiple of pi/2. A frame spans ~18 emits and several flips, so it decodes only through
    # the chain. A 2 rad phase step before emit 23 (inside the second frame) is a link the
    # chain gets wrong; the boundary-flip repair must find it.
    ELEN = 100
    e = 0
    phi = 2 * math.pi * rng.rand()
    while e + ELEN <= len(pm):
        n = math.floor(phi / math.pi + 0.5)
        theta0, sgn = phi - math.pi * n, 1 - 2 * (n % 2)
        obs = {
            "utc_ref": e * SYM_S,
            "rec_dt": 0.010,
            "phase": 0,
            "br": 1,
            "rot_re": math.cos(theta0),
            "rot_im": -math.sin(theta0),
            "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(ELEN)],
        }
        pred.ingest(prn, obs)
        e += ELEN
        phi += 2.0 if e == 23 * ELEN else 0.4

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 1 and 2 in h["have"]):
        print("FAIL: SF2 not assembled")
        ok = False
    elif h["words"] != 2:
        print("FAIL: %d of 2 frames decoded" % h["words"])
        ok = False
    eph = pred.ephemeris(prn)
    if eph is None:
        print("FAIL: no ephemeris")
        ok = False
    else:
        bad = [k for k in truth if abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)]
        if bad:
            print("FAIL: ephemeris fields disagree:", bad)
            ok = False
        else:
            print(
                "ephemeris recovered, all fields match; SatType", round(eph["SatType"])
            )
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
