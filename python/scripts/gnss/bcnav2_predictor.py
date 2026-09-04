"""Streaming BeiDou B2a B-CNAV2 decoder for the broker (S5 D-component #3, first LDPC).

Mirrors fnav_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav2.py is the codec (24-sym preamble framesync + GF(64) NB-LDPC + CRC-24Q + BDS-3
CNAV-style ephemeris); this is the per-PRN state machine around it.

Structurally like F/NAV -- a B-CNAV2 frame is a SINGLE self-contained 600-symbol unit (24
preamble + 576 LDPC) with its own CRC, no cross-frame pairing at the symbol level. The
ephemeris spans message TYPES 10 (eph-I) and 11 (eph-II), matched by temporal freshness (BDS
eph ~hourly, type 11 has no IODE), exactly as F/NAV matches its page types.

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...]}. For B2a-data the combiner emits one
symbol per 5 ms (navwipe_bit_records 5 over 1 ms records -> br=5, rec_dt=0.001, br*rec_dt=0.005).
"""

import time

import numpy as np

import beidou_bcnav2 as B

SYM_S = B.SYM_S               # 0.005
FRAME = B.FRAME_SYMS          # 600 symbols per frame
EMIT_MAX = 96                 # bound the stitched emit cache per PRN (frame is 600 syms = 3 s)
FRAME_TTL_S = 3600.0         # a cached frame older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.last_obs = None
        self.pol = None
        self.frames = {}         # mestype -> (frame_bits[288], sow, t_decoded)
        self.n_frames = 0        # frames decoded (preamble+LDPC found)
        self.n_crc = 0           # frames with valid CRC
        self.last_decode = 0.0


class Bcnav2Predictor:
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
        st.emits[slot0] = arr
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
        now = time.time()
        if self._min_gap and now - self._last_decode < self._min_gap:
            return
        self._last_decode = now
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
            # find_frames does preamble sync (both polarities) + NB-LDPC + CRC internally
            for _i, pol, bits, fprn, mt, sow in B.find_frames(signs.astype(float)):
                st.n_frames += 1
                st.n_crc += 1
                st.pol = pol
                st.frames[mt] = (bits, sow, time.time())

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from cached message types 10 & 11 (the orbit), fresh."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {mt: v for mt, v in st.frames.items() if now - v[2] < FRAME_TTL_S}
        if not all(t in fresh for t in (10, 11)):
            return None
        eph = B.parse_bcnav2_ephemeris({mt: fresh[mt][0] for mt in fresh})
        return eph

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"mestype": mt, "sow": v[1]} for mt, v in sorted(st.frames.items())]

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pol": st.pol, "pages": st.n_frames, "words": st.n_crc,
                "have": sorted(st.frames), "eph": self.ephemeris(prn) is not None}


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(5)
    prn = 30

    # build valid LDPC codewords whose info bits are a chosen 288-bit frame (types 10, 11)
    # with a CONSISTENT ephemeris + valid CRC, then feed as symbol emits and recover it.
    B._init_gf()
    frames = {}
    truth_fields = {}
    for mt in (10, 11):
        f = [0] * B.FRAME_BITS
        f[6:12] = [(mt >> (5 - k)) & 1 for k in range(6)]
    # plant a known ephemeris across types 10/11
    tf = {}
    frames = {10: [0] * B.FRAME_BITS, 11: [0] * B.FRAME_BITS}
    for mt in (10, 11):
        frames[mt][6:12] = [(mt >> (5 - k)) & 1 for k in range(6)]
    for name, (mt, start, length, signed, scale) in B.BCNV2_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            frames[mt][start + k] = (code >> (length - 1 - k)) & 1
    for mt in (10, 11):
        crc = B.crc24q(frames[mt][0:B.CRC_AT])
        frames[mt][B.CRC_AT:B.CRC_AT + 24] = [(crc >> (23 - k)) & 1 for k in range(24)]
    truth = B.parse_bcnav2_ephemeris(frames)

    # encode each 288-bit frame into a valid 576-bit LDPC codeword: pick a null-space word,
    # then flip parity? No -- instead exploit that decode is systematic and we just need a
    # symbol stream whose LDPC-decode yields these info bits. Build a codeword whose FIRST 288
    # bits equal the frame by solving parity (info-systematic): the decoder returns first 288.
    def encode_frame(frame288):
        # solve H c = 0 with the first 48 GF symbols fixed to the frame's symbols
        info = B._bin2gf(np.array(frame288, dtype=np.uint8))       # 48 GF symbols
        # dense H, reduce parity columns (48..95) to solve for them given info (0..47)
        H = [[0] * B.N_VAR for _ in range(B.N_CHECK)]
        for i in range(B.N_CHECK):
            for k in range(len(B.H_BCNV2_IDX[i])):
                H[i][B.H_BCNV2_IDX[i][k]] = B.H_BCNV2_ELE[i][k]
        # move info contribution to RHS: for each check, rhs = sum ele*info over cols<48
        rhs = [0] * B.N_CHECK
        for i in range(B.N_CHECK):
            for col in range(48):
                if H[i][col]:
                    rhs[i] ^= B._gf_mul(H[i][col], int(info[col]))
        # solve the 48x48 system over parity cols 48..95 : Hp * p = rhs
        Hp = [[H[i][48 + j] for j in range(48)] for i in range(B.N_CHECK)]
        # gaussian elim over GF(64)
        p = [0] * 48
        rows = list(range(B.N_CHECK))
        piv_col = [-1] * B.N_CHECK
        r = 0
        for col in range(48):
            pr = next((rr for rr in range(r, B.N_CHECK) if Hp[rr][col] != 0), None)
            if pr is None:
                continue
            Hp[r], Hp[pr] = Hp[pr], Hp[r]; rhs[r], rhs[pr] = rhs[pr], rhs[r]
            inv = B.GF_VEC[(B.Q_GF - 1 - B.GF_POW[Hp[r][col]]) % (B.Q_GF - 1)]
            Hp[r] = [B._gf_mul(inv, x) for x in Hp[r]]; rhs[r] = B._gf_mul(inv, rhs[r])
            for rr in range(B.N_CHECK):
                if rr != r and Hp[rr][col]:
                    f = Hp[rr][col]
                    Hp[rr] = [Hp[rr][cc] ^ B._gf_mul(f, Hp[r][cc]) for cc in range(48)]
                    rhs[rr] ^= B._gf_mul(f, rhs[r])
            piv_col[r] = col; r += 1
        for rr in range(r):
            p[piv_col[rr]] = rhs[rr]
        code = list(int(x) for x in info) + p                       # 96 GF symbols
        return B._gf2bin(np.array(code, dtype=np.uint8))

    sym = []
    for _ in range(3):
        for mt in (10, 11):
            body = encode_frame(frames[mt])
            frame_syms = np.concatenate([np.array(B.PREAMBLE, dtype=np.int8), body])
            sym.append(np.where(frame_syms == 0, 1.0, -1.0))
    pm = np.concatenate(sym)

    pred = Bcnav2Predictor(log=print)
    pred._min_gap = 0.0
    ELEN = FRAME
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {"utc_ref": e * SYM_S, "rec_dt": 0.001, "phase": 0, "br": 5,
               "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 2 and set((10, 11)) <= set(h["have"])):
        print("FAIL: types 10 & 11 not both assembled"); ok = False
    eph = pred.ephemeris(prn)
    if eph is None:
        print("FAIL: no ephemeris"); ok = False
    else:
        bad = [k for k in truth if not k.startswith("_") and k not in ("A_ref", "IODE")
               and abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)]
        if bad:
            print("FAIL: ephemeris fields disagree:", bad); ok = False
        else:
            print("ephemeris recovered, all fields match; SatType", round(eph["SatType"]))
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
