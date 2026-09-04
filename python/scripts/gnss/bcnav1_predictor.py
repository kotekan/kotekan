"""Streaming BeiDou B1C B-CNAV1 decoder for the broker (S5 D-component #4, the LAST).

Mirrors bcnav2_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them
into an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
beidou_bcnav1.py is the codec (SF1 framesync + 36x48 de-interleave + GF(64) NB-LDPC SF2/SF3 +
CRC-24Q + BDS-3 CNAV-style ephemeris).

SIMPLER TO SERVE than B-CNAV2: the WHOLE orbit is in SF2 (one subframe), so any CRC-valid SF2
yields a full ephemeris -- no cross-frame type matching. The frame is 1800 symbols (18 s), and
find_frames needs the PRN (the SF1 sync pattern is PRN-specific).

nav_obs: {utc_ref, rec_dt, br, phase, bits}. For B1C-data the combiner emits one symbol per
10 ms B1C code period (navwipe_bit_records 1, rec_dt=0.010, br=1 -> br*rec_dt = 0.010 = SYM_S).
"""

import time

import numpy as np

import beidou_bcnav1 as B

SYM_S = B.SYM_S               # 0.010
FRAME = B.FRAME_SYMS          # 1800 symbols per frame (18 s)
EMIT_MAX = 96                 # bound the stitched emit cache per PRN
SF2_TTL_S = 3600.0           # a cached SF2 older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}
        self.last_obs = None
        self.pol = None
        self.soh = None
        self.sf2 = None          # (sf2_bits[600], t_decoded)
        self.n_frames = 0
        self.n_crc = 0
        self.last_decode = 0.0


class Bcnav1Predictor:
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
            for _i, pol, soh, sf2, _sf3 in B.find_frames(prn, signs.astype(float)):
                st.n_frames += 1
                st.n_crc += 1
                st.pol = pol
                st.soh = soh
                st.sf2 = (sf2, time.time())

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI ephemeris from the cached SF2 (the full orbit), fresh."""
        st = self._p.get(prn)
        if st is None or st.sf2 is None:
            return None
        if time.time() - st.sf2[1] > SF2_TTL_S:
            return None
        return B.parse_bcnav1_ephemeris(st.sf2[0])

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None or st.sf2 is None:
            return []
        return [{"subframe": 2, "soh": st.soh}]

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pol": st.pol, "pages": st.n_frames, "words": st.n_crc,
                "have": ([2] if st.sf2 is not None else []), "eph": self.ephemeris(prn) is not None}


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
    crc = B.crc24q(sf2[0:B.SF2_CRC_AT])
    sf2[B.SF2_CRC_AT:B.SF2_CRC_AT + 24] = [(crc >> (23 - k)) & 1 for k in range(24)]
    sf3 = [0] * B.SF3_BITS
    crc3 = B.crc24q(sf3[0:B.SF3_CRC_AT])
    sf3[B.SF3_CRC_AT:B.SF3_CRC_AT + 24] = [(crc3 >> (23 - k)) & 1 for k in range(24)]
    syms2 = B._NB.encode_systematic(B.H_SF2_IDX, B.H_SF2_ELE, 100, 200, B._NB.bin2gf(np.array(sf2, dtype=np.uint8)))
    syms3 = B._NB.encode_systematic(B.H_SF3_IDX, B.H_SF3_ELE, 44, 88, B._NB.bin2gf(np.array(sf3, dtype=np.uint8)))
    il = B._interleave(syms2, syms3)
    sf1 = np.concatenate([B._sf1a(prn), B._sf1b_all()[soh]])
    frame_bits = np.concatenate([sf1, il]).astype(np.uint8)
    # two frames so a full 1800 lands within the stitched run
    stream = np.tile(frame_bits, 2)
    pm = np.where(stream == 0, 1.0, -1.0)

    B._INVERT[0] = None
    pred = Bcnav1Predictor(log=print)
    pred._min_gap = 0.0
    # 1 s emits (100 syms) from a CONTIGUOUS tracking run: one global carrier-sign convention
    # across the whole run (continuous phase), which find_frames resolves via its pol=+-1. A
    # B-CNAV1 frame is 18 s / ~18 emits, so unlike the shorter signals it cannot re-sync
    # per-emit -- it needs a phase-continuous run (the realistic locked-tracking case).
    sgn = 1 - 2 * rng.randint(0, 2)
    ELEN = 100
    e = 0
    while e + ELEN <= len(pm):
        obs = {"utc_ref": e * SYM_S, "rec_dt": 0.010, "phase": 0, "br": 1,
               "bits": [[i, int(np.sign(pm[e + i]) * sgn)] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 1 and 2 in h["have"]):
        print("FAIL: SF2 not assembled"); ok = False
    eph = pred.ephemeris(prn)
    if eph is None:
        print("FAIL: no ephemeris"); ok = False
    else:
        bad = [k for k in truth if abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)]
        if bad:
            print("FAIL: ephemeris fields disagree:", bad); ok = False
        else:
            print("ephemeris recovered, all fields match; SatType", round(eph["SatType"]))
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
