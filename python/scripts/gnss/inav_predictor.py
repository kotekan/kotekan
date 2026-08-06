"""Streaming Galileo E1B I/NAV decoder for the broker (S5 D-component #1).

Mirrors cnav_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them
into an absolute-time symbol timeline, decode, and serve the ephemeris for the BRDC
cross-check. galileo_inav.py is the codec (FEC / interleaver / CRC / word assembly /
ephemeris field tables); this is the per-PRN state machine around it.

SIMPLER THAN CNAV IN ONE WAY, HARDER IN ANOTHER. Simpler: I/NAV carries a 10-symbol SYNC
pattern in every page part, so framesync is a correlation, not the CRC-guided phase
brute-force CNAV needs (CNAV has no in-stream sync). Harder: a nav word is split across an
EVEN page part and the ODD one 250 symbols later, and the CRC spans both -- so decoding is
find-sync -> decode page part -> pair even+odd -> assemble -> CRC, and a word is only
trustworthy once both halves land with a valid CRC.

Carrier polarity is a per-stream sign ambiguity (the tracker's phase lock is mod pi):
find_page_parts tries both, and once a page part decodes with valid CRC the resolved
polarity is remembered so later parts decode incrementally.

nav_obs interface (identical to CNAV): {utc_ref, rec_dt, br, phase, bits=[[idx, +-1],...]}.
For E1B the combiner emits one symbol per 4 ms record (br=1, rec_dt=0.004).
"""

import time

import numpy as np

import galileo_inav as G

SYM_S = G.SYM_S                 # 0.004
PAGE = G.PAGE_PART_SYMS         # 250 symbols per page part
EMIT_MAX = 64                   # bound the stitched emit cache per PRN
WORD_TTL_S = 3600.0            # a cached word older than this is stale (ephemeris ~ hourly)


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(+-1) one per distinct nav_obs emit
        self.grid0 = None        # per-PRN symbol-grid origin (abs time of the first emit's sym 0)
        self.last_obs = None     # (utc_ref, phase, br) dedup
        self.pol = None          # resolved carrier polarity (+1/-1), None until first CRC-OK
        self.words = {}          # word_type -> (iodnav, word_bits[128], t_decoded)
        self.n_pages = 0         # page parts decoded (sync-valid)
        self.n_words = 0         # words assembled with valid CRC
        self.last_decode = 0.0


class InavPredictor:
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
        # Symbol 0's absolute time, phase-corrected onto the symbol grid. Anchoring every emit to
        # a PER-PRN grid origin (grid0) and indexing slots RELATIVE to it keeps round() near an
        # integer, so consecutive emits abut exactly. Rounding each symbol's ABSOLUTE time instead
        # (the old code) jitters +-1 slot whenever phase/br is fractional (br>1, e.g. E5b-I br=4):
        # that 1-slot gap fragments the run and breaks I/NAV even/odd page pairing. br=1 (E1B) has
        # an integer phase shift and was immune; this is identical for it (grid0 absorbs its offset).
        t0 = utc_ref - phase * rec_dt
        if st.grid0 is None:
            st.grid0 = t0
        base = int(round((t0 - st.grid0) / sym_s))
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            slot = base + bi            # symbols are contiguous within an emit
            m[slot] = sg
        if len(m) < 2:
            return
        # Store every maximal CONTIGUOUS segment of the emit, not just gap-free emits. A weak data
        # channel (E5b-I) exports gappy per-symbol streams -- dropping the whole emit on any gap
        # (the old behaviour) discarded 100% of E5b-I emits. Its clean runs still carry decodable
        # page parts and stitch with neighbours; the _runs/_decode path already handles the rest.
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
        # decode gate (spread work across PRNs, as CNAV does)
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
            # collect sync-aligned page parts (both polarities unless already locked)
            parts = {}   # start_slot -> content_bits[114]
            i = 0
            n = len(soft)
            while i + PAGE <= n:
                # ALWAYS try both polarities: each emit's carrier sign is independent, so a
                # single locked polarity would blind us to page parts from opposite-sign
                # emits. The sync anchors each part to the TRUE polarity frame, so the
                # decoded content is absolute regardless of the raw emit sign.
                got = None
                for pol in (1, -1):
                    seg = pol * soft[i:i + PAGE]
                    hard = (seg[:G.N_SYNC] < 0).astype(np.int8)
                    if np.array_equal(hard, np.array(G.SYNC, dtype=np.int8)):
                        bits, ok = G.decode_page_part(seg, want_sync=True)
                        if ok and bits is not None:
                            got = (pol, bits); break
                if got is not None:
                    parts[run0 + i] = got
                    st.n_pages += 1
                # do NOT jump on a bare sync: a false sync (10 symbols matching 0101100000
                # by chance, ~1/1024 in FEC data) would jump PAST the real page parts and
                # derail alignment. The CRC at the WORD level (below) is the true filter, so
                # collect every sync-valid candidate and let CRC reject the false ones.
                i += 1
            # pair EVEN (flag 0) at slot s with ODD (flag 1) at slot s+PAGE; assemble+CRC
            for s in sorted(parts):
                pol_e, even = parts[s]
                nxt = parts.get(s + PAGE)
                if nxt is None:
                    continue
                pol_o, odd = nxt
                # even/odd raw polarities MAY differ (opposite-sign emits) -- each part's
                # content is already in the true frame via its sync, so only the even/odd
                # FLAGS gate the pairing, not the raw signs.
                if int(even[0]) != 0 or int(odd[0]) != 1:
                    continue
                res = G.assemble_word(even, odd)
                if res is None:
                    continue
                word, crc_ok, wt = res
                if not crc_ok:
                    continue
                st.pol = pol_e                       # last resolved sign (health only)
                iod = G._uint(word[6:16])
                st.words[wt] = (iod, word, time.time())
                st.n_words += 1

    # ----------------------------------------------------------------- serve
    def ephemeris(self, prn):
        """SI Keplerian set from cached Word Types 1-4 with a CONSISTENT IODnav, fresh."""
        st = self._p.get(prn)
        if st is None:
            return None
        now = time.time()
        fresh = {wt: v for wt, v in st.words.items() if now - v[2] < WORD_TTL_S}
        if not all(t in fresh for t in (1, 2, 3, 4)):
            return None
        iods = {fresh[t][0] for t in (1, 2, 3, 4)}
        if len(iods) != 1:                            # words from different ephemeris sets
            return None
        eph = G.parse_inav_ephemeris({t: fresh[t][1] for t in (1, 2, 3, 4)})
        if eph is None or not eph.get("_iod_consistent"):
            return None
        return eph

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"word_type": wt, "iodnav": v[0]} for wt, v in sorted(st.words.items())]

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {"pol": st.pol, "pages": st.n_pages, "words": st.n_words,
                "have": sorted(st.words), "eph": self.ephemeris(prn) is not None}


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(7)
    prn = 11

    # a synthetic ephemeris: place known integer codes per field, build the four words
    words = {}
    for wt in (1, 2, 3, 4):
        w = [0] * G.WORD_BITS
        w[0:6] = [(wt >> (5 - i)) & 1 for i in range(6)]
        w[6:16] = [0] * 10                      # IODnav = 42 -> set below (consistent)
        iod = 42
        w[6:16] = [(iod >> (9 - i)) & 1 for i in range(10)]
        words[wt] = w
    for name, (wt, start, length, signed, scale) in G.INAV_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            words[wt][start + k] = (code >> (length - 1 - k)) & 1
    truth = G.parse_inav_ephemeris(words)

    # build the symbol stream: even+odd page parts per word, repeated
    sym = []
    for _ in range(3):
        for wt in (1, 2, 3, 4):
            even, odd = G.build_page(words[wt])
            sym.append(np.where(G.encode_page_part(even) == 0, 1.0, -1.0))
            sym.append(np.where(G.encode_page_part(odd) == 0, 1.0, -1.0))
    pm = np.concatenate(sym).astype(np.int8)     # transmitted +-1 (0->+1, 1->-1)

    # Feed as page-aligned emits (one 250-symbol page part each) with an INDEPENDENT
    # random polarity per emit -- the combiner's deep-integration sign is arbitrary per
    # emit. I/NAV's sync lets each page part self-resolve its own polarity, so as long as
    # a page part lands within one single-polarity span it decodes; page-length emits are
    # the clean case. (A sub-page emit that splits a page part across a polarity flip
    # corrupts THAT part -- a graceful loss: the ephemeris still assembles from the parts
    # that land clean. Handling straddles needs per-emit polarity resolution -- a TODO to
    # settle against the real E1B combiner's emit cadence, not a correctness gap here.)
    pred = InavPredictor(log=print)
    pred._min_gap = 0.0
    ELEN = PAGE                                  # 250 = one page part per emit
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        obs = {"utc_ref": e * SYM_S, "rec_dt": SYM_S, "phase": 0, "br": 1,
               "bits": [[i, int(pm[e + i]) * sgn] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN

    h = pred.health(prn)
    print("health:", h)
    ok = True
    if not (h and h["words"] >= 4 and set((1, 2, 3, 4)) <= set(h["have"])):
        print("FAIL: not all four word types assembled"); ok = False
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
