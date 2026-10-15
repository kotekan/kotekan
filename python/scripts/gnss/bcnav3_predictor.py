"""Streaming BeiDou B2b B-CNAV3 decoder for the broker (the B2b primary-chain nav decoder).

Mirrors bcnav2_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav3.py is the codec (PRN-specific 22-sym framesync + GF(64) NB-LDPC(162,81) + CRC-24Q).

Unlike the aux-chain decoders, B2b is a primary acquired chain (not a pilot-derived data
sibling), so this runs on the B2b broker's own combiner via --nav-decoder bcnav3. A frame is a
single self-contained 1000-symbol unit (1 s at 1000 sps) with its own CRC -- no cross-frame
pairing as in I/NAV -- and the B2b symbol is 1 ms (br=1), so the export is contiguous.

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...]}. For B2b the combiner emits one symbol
per 1 ms (navwipe_bit_records 1 -> br=1, rec_dt=0.001, br*rec_dt=0.001).

The stitcher anchors slots to a per-PRN grid and keeps the contiguous segments of a gappy emit,
as inav_predictor does: neither matters for a clean br=1 stream; they keep a br>1 or gappy
export decodable.
"""

import time

import numpy as np

import beidou_bcnav3 as B

SYM_S = B.SYM_S  # 0.001
FRAME = B.FRAME_SYMS  # 1000 symbols per frame (1 s)
EMIT_MAX = 16  # bound the stitched emit cache per PRN; this also bounds the find_frames
# scan, which rescans every cached run on each decode
FRAME_TTL_S = 3600.0  # a cached frame older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}  # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.grid0 = None  # per-PRN symbol-grid origin (see inav_predictor.ingest)
        self.last_obs = None
        self.frames = {}  # mestype -> (frame_bits[486], sow, t_decoded)
        self.n_frames = 0  # frames decoded (preamble+LDPC found + CRC ok)
        self.last_sow = None
        self.last_decode = 0.0


class Bcnav3Predictor:
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
        # grid-anchored slotting (see inav_predictor): index relative to a per-PRN origin so
        # br>1 phase jitter can never fragment the run; a no-op for B2b's br=1.
        t0 = utc_ref - phase * rec_dt
        if st.grid0 is None:
            st.grid0 = t0
        base = int(round((t0 - st.grid0) / sym_s))
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            m[base + bi] = sg
        if len(m) < 2:
            return
        # store each maximal contiguous segment (salvage gappy emits)
        slots = sorted(m)
        seg = 0
        for i in range(len(slots)):
            if i + 1 == len(slots) or slots[i + 1] != slots[i] + 1:
                s0, s1 = slots[seg], slots[i]
                if s1 - s0 + 1 >= 2:
                    st.emits[s0] = np.array(
                        [m[s] for s in range(s0, s1 + 1)], dtype=np.int8
                    )
                seg = i + 1
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
        # Per-PRN decode throttle (not the shared self._last_decode bcnav2 uses): the broker ingests
        # every PRN each poll, so a single shared gate lets only the first PRN in dict order decode
        # -- if that one is weak, no frames ever land while strong sats go undecoded. Per-PRN gating
        # decodes each satellite once per _min_gap.
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
        for _run0, signs in self._runs(st):
            if len(signs) < FRAME:
                continue
            for _i, _pol, bits, mt, sow in B.find_frames(signs.astype(float), prn):
                st.n_frames += 1
                st.last_sow = sow
                st.frames[mt] = (bits, sow, time.time())

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from the cached message types (beidou_bcnav3.parse_bcnav3_ephemeris: type
        10 carries the whole orbit), or None until a type 10 within FRAME_TTL_S is cached."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {mt: v[0] for mt, v in st.frames.items() if now - v[2] < FRAME_TTL_S}
        return B.parse_bcnav3_ephemeris(fresh)

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_frames,
            "words": st.n_frames,
            "have": sorted(st.frames),
            "sow": st.last_sow,
            "eph": self.ephemeris(prn) is not None,
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    """Encode chosen 486-bit frames into valid LDPC codewords, wrap with the PRN preamble, feed as
    br=1 symbol emits, and recover the frames (message types + SOW) through the full predictor."""
    import beidou_bcnav3 as G

    rng = np.random.RandomState(7)
    prn = 33
    G._NB.init_gf()

    # info-systematic LDPC encode: solve H c = 0 with the first 81 GF symbols = the frame.
    def encode_frame(frame486):
        info = G._NB.bin2gf(np.array(frame486, dtype=np.uint8))  # 81 GF symbols
        return G._NB.encode_systematic(
            G.H_BCNV3_IDX, G.H_BCNV3_ELE, G.N_CHECK, G.N_VAR, info
        )

    truth = {}
    frames = {}
    for mt, sow in ((10, 123456), (30, 123457)):
        f = [int(x) for x in rng.randint(0, 2, G.CRC_AT)]
        f[0:6] = [(mt >> (5 - k)) & 1 for k in range(6)]
        f[6:26] = [(sow >> (19 - k)) & 1 for k in range(20)]
        crc = G.crc24q(f)
        f += [(crc >> (23 - k)) & 1 for k in range(24)]
        frames[mt] = f
        truth[mt] = sow

    sym = []
    for _ in range(3):
        for mt in (10, 30):
            body = encode_frame(frames[mt])  # 972 symbols
            pre = G.preamble_for(prn)  # 22
            resv = np.zeros(G.LDPC_OFFSET - G.N_PRE, dtype=np.int8)  # 6 reserved
            frame_syms = np.concatenate([pre, resv, body])  # 1000
            sym.append(np.where(frame_syms == 0, 1.0, -1.0))
    pm = np.concatenate(sym)

    pred = Bcnav3Predictor()
    pred._min_gap = 0.0
    e = 0
    while e + FRAME <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {
            "utc_ref": e * SYM_S,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 1,
            "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(FRAME)],
        }
        pred.ingest(prn, obs)
        e += FRAME

    h = pred.health(prn)
    print("health:", h)
    ok = bool(h and h["words"] >= 2 and set((10, 30)) <= set(h["have"]))
    msgs = {mt: v[1] for mt, v in pred._p[prn].frames.items()}
    if msgs.get(10) != truth[10] or msgs.get(30) != truth[30]:
        print("FAIL: SOW mismatch", msgs, truth)
        ok = False
    print("recovered message types + SOW:", msgs)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
