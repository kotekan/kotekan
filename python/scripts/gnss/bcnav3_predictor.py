"""Streaming BeiDou B2b B-CNAV3 decoder for the broker (the B2b primary-chain nav decoder).

Mirrors bcnav2_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav3.py is the codec (PRN-specific 22-sym framesync + GF(64) NB-LDPC(162,81) + CRC-24Q).

Unlike the S5 D-component siblings, B2b is a PRIMARY acquired chain (not a pilot-derived data
sibling), so this runs on the B2b broker's OWN combiner via --nav-decoder bcnav3. A frame is a
SINGLE self-contained 1000-symbol unit (1 s at 1000 sps) with its own CRC -- no cross-frame
pairing (the I/NAV even/odd trap does not apply here), and the B2b symbol is 1 ms (br=1) so the
export is contiguous, exactly the property E5b-I's 4 ms data channel lacked.

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...]}. For B2b the combiner emits one symbol
per 1 ms (navwipe_bit_records 1 -> br=1, rec_dt=0.001, br*rec_dt=0.001).

The stitcher carries the two weak-data fixes proven on E5b-I (grid-anchored slots + gappy-emit
salvage): no-ops for a clean br=1 stream, but robust if B2b ever exports with gaps.
"""

import time

import numpy as np

import beidou_bcnav3 as B

SYM_S = B.SYM_S               # 0.001
FRAME = B.FRAME_SYMS          # 1000 symbols per frame (1 s)
EMIT_MAX = 16                # bound the stitched emit cache per PRN (a 1000-sym frame is 1 s of
                            # emits; ~16 frames of history is ample AND bounds the find_frames scan)
FRAME_TTL_S = 3600.0        # a cached frame older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.grid0 = None        # per-PRN symbol-grid origin (E5b-I stitcher fix)
        self.last_obs = None
        self.pol = None
        self.frames = {}         # mestype -> (frame_bits[486], sow, t_decoded)
        self.n_frames = 0        # frames decoded (preamble+LDPC found + CRC ok)
        self.last_sow = None
        self.last_decode = 0.0


class Bcnav3Predictor:
    def __init__(self, log=None):
        self._p = {}
        self._log = log or (lambda m: None)
        self._min_gap = 1.0
        self._last_decode = 0.0

    # ----------------------------------------------------------------- ingest
    def ingest(self, prn, obs):
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"]); rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"]); br = int(obs["br"]); pairs = obs["bits"]
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
        # store each maximal CONTIGUOUS segment (salvage gappy emits)
        slots = sorted(m)
        seg = 0
        for i in range(len(slots)):
            if i + 1 == len(slots) or slots[i + 1] != slots[i] + 1:
                s0, s1 = slots[seg], slots[i]
                if s1 - s0 + 1 >= 2:
                    st.emits[s0] = np.array([m[s] for s in range(s0, s1 + 1)], dtype=np.int8)
                seg = i + 1
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
        # PER-PRN decode throttle (not the shared self._last_decode bcnav2 uses): the broker ingests
        # every PRN each poll, so a single shared gate lets only the FIRST PRN in dict order decode
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
        cur0 = None; cur = None
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
                runs.append((cur0, cur)); cur0, cur = s0, a.copy()
        runs.append((cur0, cur))
        return runs

    def _decode(self, prn, st):
        for _run0, signs in self._runs(st):
            if len(signs) < FRAME:
                continue
            for _i, pol, bits, mt, sow in B.find_frames(signs.astype(float), prn):
                st.n_frames += 1
                st.pol = pol
                st.last_sow = sow
                st.frames[mt] = (bits, sow, time.time())

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from the cached message types (Phase 2: BCNV3_EPH_FIELDS). None until the
        live message-type layout is mapped."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {mt: v[0] for mt, v in st.frames.items() if now - v[2] < FRAME_TTL_S}
        return B.parse_bcnav3_ephemeris(fresh)

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"mestype": mt, "sow": v[1]} for mt, v in sorted(st.frames.items())]

    def frame_hex(self, prn, mt):
        """Raw decoded frame as hex (for Phase 2 field mapping against live data)."""
        st = self._p.get(prn)
        if st is None or mt not in st.frames:
            return None
        b = st.frames[mt][0]
        return "".join("%x" % (B._uint(b[i:i + 4]) if i + 4 <= len(b)
                               else B._uint(b[i:] + [0] * (i + 4 - len(b)))) for i in range(0, len(b), 4))

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pol": st.pol, "pages": st.n_frames, "words": st.n_frames,
                "have": sorted(st.frames), "sow": st.last_sow,
                "eph": self.ephemeris(prn) is not None}


# ------------------------------------------------------------------- self-test
def _selftest():
    """Encode chosen 486-bit frames into valid LDPC codewords, wrap with the PRN preamble, feed as
    br=1 symbol emits, and recover the frames (message types + CRC) through the full predictor."""
    import beidou_bcnav3 as G
    rng = np.random.RandomState(7)
    prn = 33
    G._init_gf()

    # info-systematic LDPC encode: solve H c = 0 with the first 81 GF symbols = the frame.
    def encode_frame(frame486):
        info = G._bin2gf(np.array(frame486, dtype=np.uint8))            # 81 GF symbols
        H = [[0] * G.N_VAR for _ in range(G.N_CHECK)]
        for i in range(G.N_CHECK):
            for k in range(len(G.H_BCNV3_IDX[i])):
                H[i][G.H_BCNV3_IDX[i][k]] = G.H_BCNV3_ELE[i][k]
        rhs = [0] * G.N_CHECK
        for i in range(G.N_CHECK):
            for col in range(G.N_CHECK):
                if H[i][col]:
                    rhs[i] ^= G._gf_mul(H[i][col], int(info[col]))
        Hp = [[H[i][G.N_CHECK + j] for j in range(G.N_CHECK)] for i in range(G.N_CHECK)]
        p = [0] * G.N_CHECK
        r = 0; piv_col = [-1] * G.N_CHECK
        for col in range(G.N_CHECK):
            pr = next((rr for rr in range(r, G.N_CHECK) if Hp[rr][col] != 0), None)
            if pr is None:
                continue
            Hp[r], Hp[pr] = Hp[pr], Hp[r]; rhs[r], rhs[pr] = rhs[pr], rhs[r]
            inv = G.GF_VEC[(G.Q_GF - 1 - G.GF_POW[Hp[r][col]]) % (G.Q_GF - 1)]
            Hp[r] = [G._gf_mul(inv, x) for x in Hp[r]]; rhs[r] = G._gf_mul(inv, rhs[r])
            for rr in range(G.N_CHECK):
                if rr != r and Hp[rr][col]:
                    f = Hp[rr][col]
                    Hp[rr] = [Hp[rr][cc] ^ G._gf_mul(f, Hp[r][cc]) for cc in range(G.N_CHECK)]
                    rhs[rr] ^= G._gf_mul(f, rhs[r])
            piv_col[r] = col; r += 1
        for rr in range(r):
            p[piv_col[rr]] = rhs[rr]
        code = list(int(x) for x in info) + p
        return G._gf2bin(np.array(code, dtype=np.uint8))

    truth = {}
    frames = {}
    for mt, sow in ((10, 123456), (30, 123457)):
        f = [int(x) for x in rng.randint(0, 2, G.CRC_AT)]
        f[0:6] = [(mt >> (5 - k)) & 1 for k in range(6)]
        f[6:26] = [(sow >> (19 - k)) & 1 for k in range(20)]
        crc = G.crc24q(f)
        f += [(crc >> (23 - k)) & 1 for k in range(24)]
        frames[mt] = f; truth[mt] = sow

    sym = []
    for _ in range(3):
        for mt in (10, 30):
            body = encode_frame(frames[mt])                            # 972 symbols
            pre = G.preamble_for(prn)                                   # 22
            resv = np.zeros(G.LDPC_OFFSET - G.N_PRE, dtype=np.int8)     # 6 reserved
            frame_syms = np.concatenate([pre, resv, body])             # 1000
            sym.append(np.where(frame_syms == 0, 1.0, -1.0))
    pm = np.concatenate(sym)

    pred = Bcnav3Predictor(log=print)
    pred._min_gap = 0.0
    e = 0
    while e + FRAME <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {"utc_ref": e * SYM_S, "rec_dt": 0.001, "phase": 0, "br": 1,
               "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(FRAME)]}
        pred.ingest(prn, obs)
        e += FRAME

    h = pred.health(prn)
    print("health:", h)
    ok = bool(h and h["words"] >= 2 and set((10, 30)) <= set(h["have"]))
    msgs = {m["mestype"]: m["sow"] for m in pred.messages(prn)}
    if msgs.get(10) != truth[10] or msgs.get(30) != truth[30]:
        print("FAIL: SOW mismatch", msgs, truth); ok = False
    print("recovered message types + SOW:", msgs)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
