"""Streaming Galileo E5a-I F/NAV decoder for the broker (S5 D-component #2).

Mirrors inav_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them
into an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC
cross-check. galileo_fnav.py is the codec (FEC / interleaver / CRC / page assembly /
ephemeris field tables); this is the per-PRN state machine around it.

SIMPLER THAN I/NAV. An F/NAV page is a SINGLE self-contained 500-symbol unit with its own
CRC-24 and 12-symbol sync -- there is NO even/odd split, so decoding is just find-sync ->
decode page -> CRC, with no cross-part pairing. Framesync is a sync correlation (both
polarities, since each emit's carrier sign is independent, exactly as for I/NAV). A page is
trustworthy the moment it lands with a valid CRC; the ephemeris assembles once page TYPES
2,3,4 (the orbit) are held with a consistent IODnav.

nav_obs interface (identical to CNAV/INAV): {utc_ref, rec_dt, br, phase, bits=[[idx,+-1],...]}.
For E5a-I the combiner emits one symbol per 20 ms F/NAV symbol (navwipe_bit_records 20 over
1 ms records -> br=20, rec_dt=0.001, so br*rec_dt = 0.020 = SYM_S).
"""

import time

import numpy as np

import galileo_fnav as G

SYM_S = G.SYM_S                 # 0.020
PAGE = G.PAGE_SYMS             # 500 symbols per page
EMIT_MAX = 64                  # bound the stitched emit cache per PRN
PAGE_TTL_S = 3600.0           # a cached page older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.last_obs = None     # (utc_ref, phase, br) dedup
        self.pol = None          # last resolved carrier polarity (health only)
        self.pages = {}          # page_type -> (iodnav, content_bits[238], t_decoded)
        self.n_pages = 0         # pages decoded (sync-valid)
        self.n_crc = 0           # pages assembled with valid CRC
        self.last_decode = 0.0


class FnavPredictor:
    def __init__(self, log=None):
        self._p = {}
        self._log = log or (lambda m: None)
        self._min_gap = 1.0      # s between decode attempts across all PRNs (0 = off, self-test)
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
        if (arr == 0).any():     # a gap breaks page contiguity: drop the emit
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
        """Merge the stitched emits into maximal contiguous symbol runs (slot0, signs)."""
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
            if s0 <= end:                     # overlap/abut: splice (shared slot signs agree)
                ov = end - s0
                if ov < len(a):
                    cur = np.concatenate([cur, a[ov:]])
            else:
                runs.append((cur0, cur)); cur0, cur = s0, a.copy()
        runs.append((cur0, cur))
        return runs

    def _decode(self, prn, st):
        for run0, signs in self._runs(st):
            soft = signs.astype(float)
            n = len(soft)
            i = 0
            while i + PAGE <= n:
                # ALWAYS try both polarities: each emit's carrier sign is independent, so a
                # single locked polarity would blind us to pages from opposite-sign emits.
                # The sync anchors each page to the TRUE polarity frame; decoded content is
                # absolute. Do NOT jump on a bare sync (a false 12-symbol sync ~1/4096 would
                # derail alignment) -- step by 1 and let the CRC be the true filter.
                for pol in (1.0, -1.0):
                    seg = pol * soft[i:i + PAGE]
                    hard = (seg[:G.N_SYNC] < 0).astype(np.int8)
                    if np.array_equal(hard, np.array(G.SYNC, dtype=np.int8)):
                        bits, ok = G.decode_page(seg, want_sync=True)
                        if ok and bits is not None:
                            st.n_pages += 1
                            crc_ok, pt = G.check_page(bits)
                            if crc_ok and pt is not None:
                                st.pol = int(pol)
                                iod = G._uint(bits[6:16])
                                st.pages[pt] = (iod, bits, time.time())
                                st.n_crc += 1
                            break
                i += 1

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI Keplerian set from cached page types 2,3,4 (orbit) with a CONSISTENT IODnav,
        fresh; folds in page 1 (clock) when held."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {pt: v for pt, v in st.pages.items() if now - v[2] < PAGE_TTL_S}
        if not all(t in fresh for t in (2, 3, 4)):
            return None
        eph = G.parse_fnav_ephemeris({pt: fresh[pt][1] for pt in fresh})
        if eph is None or not eph.get("_iod_consistent"):
            return None
        return eph

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"page_type": pt, "iodnav": v[0]} for pt, v in sorted(st.pages.items())]

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pol": st.pol, "pages": st.n_pages, "words": st.n_crc,
                "have": sorted(st.pages), "eph": self.ephemeris(prn) is not None}


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(7)
    prn = 11

    # a synthetic ephemeris: place known integer codes per field, build the pages
    pages = {}
    for pt in (1, 2, 3, 4):
        c = [0] * G.PAGE_BITS
        c[0:6] = [(pt >> (5 - i)) & 1 for i in range(6)]
        iod = 42
        c[6:16] = [(iod >> (9 - i)) & 1 for i in range(10)]
        pages[pt] = c
    for name, (pt, start, length, signed, scale) in G.FNAV_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            pages[pt][start + k] = (code >> (length - 1 - k)) & 1
    # re-CRC each page after planting fields (so decode's CRC gate passes)
    for pt in (1, 2, 3, 4):
        crc = G.crc24q(pages[pt][0:G.CRC_AT])
        pages[pt][G.CRC_AT:G.CRC_AT + 24] = [(crc >> (23 - i)) & 1 for i in range(24)]
    truth = G.parse_fnav_ephemeris(pages)

    # build the symbol stream: one page per type, repeated, INDEPENDENT random polarity/emit
    sym = []
    for _ in range(3):
        for pt in (1, 2, 3, 4):
            sym.append(np.where(G.encode_page(pages[pt]) == 0, 1.0, -1.0))
    pm = np.concatenate(sym).astype(np.int8)

    pred = FnavPredictor(log=print)
    pred._min_gap = 0.0
    ELEN = PAGE                                  # 500 = one page per emit (clean case)
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {"utc_ref": e * SYM_S, "rec_dt": 0.001, "phase": 0, "br": 20,
               "bits": [[i, int(pm[e + i]) * sgn] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 3 and set((2, 3, 4)) <= set(h["have"])):
        print("FAIL: orbit page types not all assembled"); ok = False
    eph = pred.ephemeris(prn)
    if eph is None:
        print("FAIL: no ephemeris"); ok = False
    else:
        bad = [k for k in truth if not k.startswith("_") and k != "IODnav"
               and abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)]
        if bad:
            print("FAIL: ephemeris fields disagree:", bad); ok = False
        else:
            print("ephemeris recovered, all fields match; IODnav", eph["IODnav"])
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
