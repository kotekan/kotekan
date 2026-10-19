"""Streaming BeiDou B2a B-CNAV2 decoder for the broker.

Mirrors fnav_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav2.py is the codec (24-sym preamble framesync + GF(64) NB-LDPC + CRC-24Q + BDS-3
CNAV-style ephemeris); this is the per-PRN state machine around it.

Structurally like F/NAV -- a B-CNAV2 frame is a single self-contained 600-symbol unit (24
preamble + 576 LDPC) with its own CRC, no cross-frame pairing at the symbol level. The
ephemeris spans message types 10 (eph-I) and 11 (eph-II). F/NAV checks IODnav across its page
types, but type 11 has no IODE, so the two are paired by freshness (both cached within
FRAME_TTL_S; BDS eph ~hourly) and by order: when type 10's IODE changes, the cached type 11 is
dropped and only a type 11 that follows the new type 10 in the stream pairs with it.

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...]}. For B2a-data the combiner emits one
symbol per 5 ms (navwipe_bit_records 5 over 1 ms records -> br=5, rec_dt=0.001, br*rec_dt=0.005).
"""

import time

import numpy as np

import beidou_bcnav2 as B
from navobs_align import emit_offset

SYM_S = B.SYM_S  # 0.005
FRAME = B.FRAME_SYMS  # 600 symbols per frame
EMIT_MAX = 96  # bound the stitched emit cache per PRN (frame is 600 syms = 3 s)
FRAME_TTL_S = 3600.0  # a cached frame older than this is stale (ephemeris ~ hourly)
IODE_BITS = slice(B._IODE_10[0], B._IODE_10[0] + B._IODE_10[1])  # type 10's IODE bits


class _PrnState:
    def __init__(self):
        self.emits = {}  # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.last_obs = None
        self.frames = {}  # mestype -> (frame_bits[288], sow, t_decoded, start slot)
        self.n_frames = 0  # frames decoded (find_frames yields only CRC-OK frames)
        self.n_crc = 0  # frames with valid CRC
        # frame start slots already counted: every decode rescans the cached emits, so a frame
        # is counted only the first time it is found
        self.seen = set()
        # start slot of the type 10 whose IODE differs from the one before it: a type 11 found
        # earlier in the stream may belong to the old issue
        self.issue_at = None
        self.last_decode = 0.0


class Bcnav2Predictor:
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
            pairs = obs["bits"]
        except (KeyError, TypeError, ValueError):
            return
        if rec_dt <= 0 or br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        sym_s = br * rec_dt
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            slot = int(round((utc_ref + (bi * br - phase) * rec_dt) / sym_s))
            m[slot] = sg
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
        # Per-PRN decode throttle (see inav_predictor.ingest): a shared gate starves every PRN but
        # the first in the broker's row order.
        now = time.time()
        if self._min_gap and now - st.last_decode < self._min_gap:
            return
        st.last_decode = now
        self._decode(prn, st)

    # ----------------------------------------------------------------- decode
    def _runs(self, st):
        if not st.emits:
            return []
        runs = []
        cur0 = None
        cur = None
        for s0 in sorted(st.emits):
            a = st.emits[s0]
            if cur is None:
                cur0, cur = s0, a.copy()
                continue
            end = cur0 + len(cur)
            if s0 <= end:
                ov = end - s0
                if ov < len(a):
                    cur = np.concatenate([cur, a[ov:]])
            else:
                runs.append((cur0, cur))
                cur0, cur = s0, a.copy()
        runs.append((cur0, cur))
        return runs

    def _decode(self, prn, st):
        for run0, signs in self._runs(st):
            if len(signs) < FRAME:
                continue
            # find_frames does preamble sync (both polarities) + NB-LDPC + CRC internally
            for i, _pol, bits, fprn, mt, sow in B.find_frames(signs.astype(float)):
                # the frame names its own PRN: a CRC-valid frame from another satellite (a
                # mislabelled chain or a cross-correlation lock) must not be stored under this one
                if fprn != prn:
                    continue
                at = run0 + i
                if at not in st.seen:
                    st.seen.add(at)
                    st.n_frames += 1
                    st.n_crc += 1
                old = st.frames.get(mt)
                if old is not None and at < old[3]:
                    continue  # a rescan finds frames older than the cached one again
                if (
                    mt == 10
                    and old is not None
                    and bits[IODE_BITS] != old[0][IODE_BITS]
                ):
                    # an ephemeris update: the cached type 11 carries no IODE and may belong to
                    # the old issue, so drop it and wait for one that follows this frame
                    st.frames.pop(11, None)
                    st.issue_at = at
                if mt == 11 and st.issue_at is not None and at < st.issue_at:
                    continue
                st.frames[mt] = (bits, sow, time.time(), at)

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from message types 10 & 11 (the orbit) cached within FRAME_TTL_S."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {mt: v for mt, v in st.frames.items() if now - v[2] < FRAME_TTL_S}
        if not all(t in fresh for t in (10, 11)):
            return None
        eph = B.parse_bcnav2_ephemeris({mt: fresh[mt][0] for mt in fresh})
        return eph

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_frames,
            "words": st.n_crc,
            "have": sorted(st.frames),
            "eph": self.ephemeris(prn) is not None,
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(5)
    prn = 30

    # build valid LDPC codewords whose info bits are a chosen 288-bit frame (types 10, 11)
    # with a consistent ephemeris + valid CRC, then feed as symbol emits and recover it.
    B._NB.init_gf()
    # plant a known ephemeris across types 10/11
    frames = {10: [0] * B.FRAME_BITS, 11: [0] * B.FRAME_BITS}
    for mt in (10, 11):
        frames[mt][0:6] = [(prn >> (5 - k)) & 1 for k in range(6)]
        frames[mt][6:12] = [(mt >> (5 - k)) & 1 for k in range(6)]
    for name, (mt, start, length, signed, scale) in B.BCNV2_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            frames[mt][start + k] = (code >> (length - 1 - k)) & 1
    for mt in (10, 11):
        crc = B.crc24q(frames[mt][0 : B.CRC_AT])
        frames[mt][B.CRC_AT : B.CRC_AT + 24] = [
            (crc >> (23 - k)) & 1 for k in range(24)
        ]
    truth = B.parse_bcnav2_ephemeris(frames)

    # encode each 288-bit frame into a valid 576-bit LDPC codeword whose first 288 bits are the
    # frame (info-systematic): the decoder returns the first 288.
    def encode_frame(frame288):
        info = B._NB.bin2gf(np.array(frame288, dtype=np.uint8))  # 48 GF symbols
        return B._NB.encode_systematic(
            B.H_BCNV2_IDX, B.H_BCNV2_ELE, B.N_CHECK, B.N_VAR, info
        )

    sym = []
    for _ in range(3):
        for mt in (10, 11):
            body = encode_frame(frames[mt])
            frame_syms = np.concatenate([np.array(B.PREAMBLE, dtype=np.int8), body])
            sym.append(np.where(frame_syms == 0, 1.0, -1.0))
    pm = np.concatenate(sym)

    pred = Bcnav2Predictor()
    pred._min_gap = 0.0
    ELEN = FRAME
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {
            "utc_ref": e * SYM_S,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 5,
            "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(ELEN)],
        }
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 2 and set((10, 11)) <= set(h["have"])):
        print("FAIL: types 10 & 11 not both assembled")
        ok = False
    eph = pred.ephemeris(prn)
    if eph is None:
        print("FAIL: no ephemeris")
        ok = False
    else:
        bad = [
            k
            for k in truth
            if not k.startswith("_")
            and k not in ("A_ref", "IODE")
            and abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)
        ]
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
