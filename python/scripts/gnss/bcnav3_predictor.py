"""Streaming BeiDou B2b B-CNAV3 decoder (the B2b chain's nav decoder, run by nav_decode.py).

Ingests the B2b combiner's per-PRN nav_obs symbol emits, decodes the frames in each, and serves
the ephemeris for the BRDC cross-check. beidou_bcnav3.py is the codec (PRN-specific 22-sym
framesync + GF(64) NB-LDPC(162,81) + CRC-24Q).

A frame is a single self-contained 1000-symbol unit (1 s at 1000 sps) with its own CRC, and the
combiner's 1.5 s emit always holds one whole frame, so each emit is decoded on its own: emits are
never stitched. A stitched run mixes emits whose polarity and slot grid each emit sets for itself
(its carrier phase, its utc_ref), and a frame that straddles a flip or a one-symbol slip fails.
A frame found again in the next, overlapping emit is counted once, by its SOW.

nav_obs: {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...], soft=[10 x value, ...]}. For B2b the
combiner emits one symbol per 1 ms record (navwipe_bit_records 1 -> br=1). soft, when present,
carries each bit's soft value (its sign is the bit's), which weights the LDPC; without it the
decode is hard-decision.
"""

import time

import numpy as np

import beidou_bcnav3 as B

FRAME = B.FRAME_SYMS  # 1000 symbols per frame (1 s)
EMIT_MAX = 16  # emits held for the next decode while the per-PRN throttle defers it
FRAME_TTL_S = 3600.0  # a cached frame older than this is stale (ephemeris ~ hourly)
SOW_KEEP_S = 3600.0  # counted frames remembered, by SOW, for this long
MARKS_KEEP = 120  # frames' time marks kept per PRN (nav_decode --timemarks-file)


class _PrnState:
    def __init__(self):
        # (host-labelled UTC of its first symbol, symbol period s, a contiguous run)
        self.pending = []
        # sow -> (host-labelled UTC of the frame's first symbol, mestype)
        self.marks = {}
        self.last_obs = None
        self.frames = {}  # mestype -> (frame_bits[486], sow, t_decoded)
        self.n_frames = 0  # distinct frames decoded (preamble + LDPC + CRC ok)
        self.seen = set()  # SOWs of the frames counted
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
            phase = int(obs["phase"])
            br = int(obs["br"])
            rec_dt = float(obs.get("rec_dt", 0.001))
            pairs = obs["bits"]
            soft = obs.get("soft")
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        if br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        if soft is not None and len(soft) != len(pairs):
            soft = None
        m = {}
        for k, (bi, sg) in enumerate(pairs):
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            x = float(soft[k]) / 10.0 if soft is not None else float(sg)
            m[bi] = x if x * sg > 0 else 0.01 * sg  # the soft value, on the bit's side
        # each maximal contiguous run (a gappy emit keeps its runs)
        slots = sorted(m)
        seg = 0
        for i in range(len(slots)):
            if i + 1 == len(slots) or slots[i + 1] != slots[i] + 1:
                s0, s1 = slots[seg], slots[i]
                if s1 - s0 + 1 >= FRAME:
                    st.pending.append(
                        (
                            utc_ref + (s0 * br - phase) * rec_dt,
                            br * rec_dt,
                            np.array([m[s] for s in range(s0, s1 + 1)]),
                        )
                    )
                seg = i + 1
        del st.pending[:-EMIT_MAX]
        # Per-PRN decode throttle: one weak satellite must not starve the rest.
        now = time.time()
        if self._min_gap and now - st.last_decode < self._min_gap:
            return
        st.last_decode = now
        self._decode(prn, st)

    # ----------------------------------------------------------------- decode
    def _decode(self, prn, st):
        runs, st.pending = st.pending, []
        for t_first, sym_s, signs in runs:
            for i, _pol, bits, mt, sow in B.find_frames(signs, prn):
                st.last_sow = sow
                st.frames[mt] = (bits, sow, time.time())
                st.marks[sow] = (t_first + i * sym_s, mt)
                if sow not in st.seen:
                    st.seen.add(sow)
                    st.n_frames += 1
        if len(st.seen) > 2 * SOW_KEEP_S and st.last_sow is not None:
            st.seen = {x for x in st.seen if abs(x - st.last_sow) < SOW_KEEP_S}
        if len(st.marks) > MARKS_KEEP:
            for sow in sorted(st.marks)[:-MARKS_KEEP]:
                del st.marks[sow]

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

    def make_frame(mt, sow):
        f = [int(x) for x in rng.randint(0, 2, G.CRC_AT)]
        f[0:6] = [(mt >> (5 - k)) & 1 for k in range(6)]
        f[6:26] = [(sow >> (19 - k)) & 1 for k in range(20)]
        crc = G.crc24q(f)
        return f + [(crc >> (23 - k)) & 1 for k in range(24)]

    truth = {}  # mestype -> the SOW of its last frame, one second per frame as on air
    sym = []
    for rep in range(3):
        for k, mt in enumerate((10, 30)):
            truth[mt] = 123456 + 2 * rep + k
            body = encode_frame(make_frame(mt, truth[mt]))  # 972 symbols
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
            "utc_ref": e * B.SYM_S,
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
