"""Streaming GPS L1C-D CNAV-2 decoder for the broker (the L1C-D data sibling's nav decoder).

Mirrors bcnav3_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC cross-check.
gps_cnav2.py is the codec (SF1-TOI framesync + block deinterleave + systematic LDPC extract +
CRC-24Q + the CNAV Keplerian ephemeris).

Like B-CNAV1 the frame is 18 s (1800 symbols) -- long, so it needs a phase-CONTINUOUS run to
decode (find_frames needs 1852 contiguous symbols = frame + the next frame's SF1). The L1C-D
combiner emits one symbol per 10 ms (navwipe_bit_records 1 -> br=1, rec_dt=0.010); integration_
length 100 = a 1 s emit, longer than the broker's 0.2 s poll, so windows never drop (no B2b
count-vs-time trap). br=1 -> no slot jitter. The stitcher carries the E5b-I fixes anyway.
"""

import time

import numpy as np

import gps_cnav2 as C2

SYM_S = C2.SYM_S               # 0.010
FRAME = C2.FRAME_SYMS          # 1800
NEED = FRAME + C2.N_SF1        # 1852 (frame + next SF1) for a sync+decode
EMIT_MAX = 48                 # ~48 s of 1 s emits -> >2 frames of history, bounds the find scan
EPH_TTL_S = 3600.0           # a cached ephemeris older than this is stale (~hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(0/1) one per distinct nav_obs emit
        self.grid0 = None
        self.last_obs = None
        self.rev = None
        self.sf2 = None          # (bits[600], t_decoded)
        self.sf3 = None          # (bits[274], t_decoded)
        self.toi = None
        self.n_frames = 0
        self.last_decode = 0.0


class Cnav2Predictor:
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
        t0 = utc_ref - phase * rec_dt
        if st.grid0 is None:
            st.grid0 = t0
        base = int(round((t0 - st.grid0) / sym_s))
        # nav_obs signs are +-1; CNAV-2 symbols are 0/1 (SF1 patterns + LDPC), so map -1->1, +1->0.
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            m[base + bi] = 0 if sg > 0 else 1
        if len(m) < 2:
            return
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
        for _run0, syms in self._runs(st):
            if len(syms) < NEED:
                continue
            for _i, toi, sf2, sf3 in C2.find_frames(syms):
                st.n_frames += 1
                st.toi = toi
                st.sf2 = (sf2, time.time())
                st.sf3 = (sf3, time.time())

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        st = self._p.get(prn)
        if st is None or st.sf2 is None or time.time() - st.sf2[1] > EPH_TTL_S:
            return None
        return C2.parse_cnav2_ephemeris(st.sf2[0])

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pages": st.n_frames, "words": st.n_frames, "toi": st.toi,
                "have": ([2, 3] if st.sf2 else []), "eph": self.ephemeris(prn) is not None}


# ------------------------------------------------------------------- self-test
def _selftest():
    """Build CNAV-2 frames (systematic SF2/SF3 + CRC + arbitrary parity, SF1, interleave), feed as
    br=1 10 ms emits, recover the SF2/SF3 + TOI through the full predictor."""
    import gps_cnav2 as C
    from galileo_inav import crc24q
    rng = np.random.default_rng(4)
    prn = 23

    def crc_append(k):
        m = rng.integers(0, 2, k - 24).tolist()
        c = crc24q(m)
        return m + [(c >> (23 - j)) & 1 for j in range(24)]

    def build(toi):
        sf2 = crc_append(C.K_SF2); sf3 = crc_append(C.K_SF3)
        deint = np.array(sf2 + rng.integers(0, 2, C.N_SF2 - C.K_SF2).tolist()
                         + sf3 + rng.integers(0, 2, C.N_SF3 - C.K_SF3).tolist(), dtype=np.int8)
        il = deint.reshape(C.IL_COLS, C.IL_ROWS).T.ravel()
        return np.hstack([C._sf1(toi), il]).astype(np.int8), sf2, sf3

    # two consecutive frames so find_frames sees frame + next SF1
    f0, sf2_0, sf3_0 = build(100)
    f1, _, _ = build(101)
    stream = np.hstack([f0, f1]).astype(np.int8)          # 3600 symbols

    pred = Cnav2Predictor(log=print)
    pred._min_gap = 0.0
    e = 0
    ELEN = 100
    sgn = -1              # ONE polarity for the whole contiguous run (carrier stays locked);
                         # find_frames' both-polarity check resolves the global flip.
    while e + ELEN <= len(stream):
        obs = {"utc_ref": e * SYM_S, "rec_dt": 0.010, "phase": 0, "br": 1,
               "bits": [[i, int((1 - 2 * int(stream[e + i])) * sgn)] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = bool(h and h["words"] >= 1 and h["toi"] == 100 and st_sf2(pred, prn) == sf2_0)
    print("PASS" if ok else "FAIL")
    return ok


def st_sf2(pred, prn):
    st = pred._p.get(prn)
    return st.sf2[0] if st and st.sf2 else None


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
