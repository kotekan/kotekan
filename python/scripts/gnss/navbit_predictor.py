#!/usr/bin/env python3
"""LNAV nav-bit decode.

    combiner get_status "nav_obs"  ->  stitch  ->  frame_sync/decode  ->  ephemeris
                                                        (gps_nav_decode.py does the LNAV work)

Consumed by gnss_broker/navbits.py (the decode and the ephemeris cross-check against BRDC)
and by the decoded-eph fallback in gps_distributed_broker.py. It also scores how well each
newly decoded subframe could have been predicted from the earlier ones (health()["mismatch"],
logged as predict-mismatch).

Three design facts this leans on:
  * Polarity. Each combiner emit resolves its +-1 ambiguity independently (theta0 = the
    squaring estimate's half-angle, reduced to (-pi/2, pi/2]), so consecutive emits can be
    globally flipped relative to each other. The export carries rot = e^{-i*theta0}; the true
    carrier phase is continuous at lock, so the relative flip between emits is
    round(dtheta/pi) odd -- chained here, then verified by LNAV parity (frame_sync), with a
    boundary-flip repair search when parity disagrees. The absolute polarity stays arbitrary:
    the mismatch score compares each subframe at its better polarity.
  * Chain reset. LNAV trims words 2 (HOW) and 10 so their parity ends D29*=D30*=00 -- the
    parity chain resets at every subframe boundary. Prediction is therefore per-subframe
    independent, and an unknown subframe-4/5 page blanks only its own words 3-10.
  * Predictability. Subframes 1-3 repeat with only the HOW TOW advancing (+5/frame);
    subframes 4/5 cycle a 25-frame supercycle (page key = tow % 125). Once subframes 1-3 are
    held, coverage is over 60% of the frame (they plus TLM+HOW of subframes 4/5), and most
    of it after one 12.5 min supercycle. Bits that cannot be predicted are 0 = unknown and
    are not scored.

Self-test: python3 navbit_predictor.py   (synthetic multi-frame stream, chopped into
polarity-scrambled emits, must sync, predict the next frame exactly for subframes 1-3 and
extract the ephemeris).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_nav_decode import (
    PREAMBLE,
    decode_subframe,
    frame_sync,
    parity_encode,
    parse_lnav_ephemeris,
    set_how_tow,
    trim_word,
)  # noqa: E402


def encode_subframe_trimmed(data_words):
    """Encode 10x24 data words into the 300 transmitted bits the ICD way: the chain starts
    clean each subframe, and words 2 (HOW) and 10 are trimmed (their 2 non-information
    bits solved so parity ends D29=D30=00), which is what makes per-subframe prediction
    valid. It does not rebuild the HOW, so the stored alert/AS flags survive; the caller sets
    the TOW."""
    bits, d29s, d30s = [], 0, 0
    for w, src in enumerate(data_words):
        src = [int(x) for x in src]
        if w in (1, 9):
            src = trim_word(src, d29s, d30s)
        word = parity_encode(src, d29s, d30s)
        d29s, d30s = int(word[28]), int(word[29])
        bits.extend(int(x) for x in word)
    return np.array(bits, dtype=np.int8)


BIT_S = 0.020  # nominal LNAV bit (s, capture clock)
SYNC_MIN_BITS = 620  # ~2 subframes + slack before attempting frame_sync
SCAN_MAX_BITS = (
    3000  # 60 s: only the newest bits can hold an undecoded subframe, and frame_sync
)
# is a per-bit Python loop, too slow to rescan the whole history on every try
REPAIR_MIN_S = (
    30.0  # min seconds between polarity-repair searches per PRN. Long on purpose:
)
# a satellite that fails sync usually has bit errors, which no polarity assignment
# repairs, so the search mostly costs CPU on satellites it cannot rescue.
TRY_EVERY_BITS = (
    150  # attempt sync/decode at most once per ~3 s of new bits per PRN; subframes
)
# arrive every 6 s and the scan window holds 60 s, so nothing is missed.
REPAIR_WIN = 1000  # repair searches only the trailing bits: sync needs just
# SYNC_MIN_BITS consistent ones, and the freshest are likeliest to belong to the
# current arc.
HIST_MAX_SLOTS = (
    64000  # ~21 min of bit history; sync scans only the trailing SCAN_MAX_BITS
)
SF_BITS = 300


class FleetPages:
    """Cross-satellite sf4/5 page store, voted by unanimity.

    GPS subframe 4/5 pages are uploaded almanac data, so most words match across the
    constellation at matched TOW, but not all: satellites carry uploads of different
    vintages. No single donor can be trusted, so a word is taken only when at least
    MIN_DONORS satellites hold the page and all of them agree bit for bit.

    The parity chain is the hard constraint. Transmitted bits of word w+1 are
    data(w+1) XOR D30(w): an unknown word breaks the chain, so every later word in that
    subframe is untransmittable even if its data is known -- consumers must zero from the
    first unknown word onward (`prefix_mask`). The fraction of known words is therefore not
    the fraction of predictable bits. (Words 1-2, TLM+HOW, are always known: the chain
    re-seeds to D29=D30=0 at every subframe boundary by the ICD trim.)
    """

    MIN_DONORS = 2

    def __init__(self):
        self.votes = {}  # (sfid, page) -> {prn: [10x24 int8 words]}

    def offer(self, prn, sfid, page, data):
        """A freshly decoded page from a synced satellite (10x24 parity-clean words)."""
        self.votes.setdefault((sfid, page), {})[prn] = [
            np.array(d, dtype=np.int8).copy() for d in data
        ]

    def lookup(self, sfid, page):
        """(words[10], known[10]) by unanimity over donors, or None when fewer than
        MIN_DONORS donors hold the page or no data word is unanimous.
        Words 0-1 (TLM/HOW) are the caller's job -- TLM is constellation-common and HOW is
        computed per TOW -- so only data words 2-9 are voted here."""
        donors = self.votes.get((sfid, page))
        if not donors or len(donors) < self.MIN_DONORS:
            return None
        words = [np.zeros(24, dtype=np.int8) for _ in range(10)]
        known = [False] * 10
        for w in range(2, 10):
            vals = {}
            for d in donors.values():
                vals.setdefault(bytes(bytearray(int(b) & 1 for b in d[w])), []).append(
                    d[w]
                )
            if len(vals) == 1 and len(donors) >= self.MIN_DONORS:
                words[w] = np.array(next(iter(vals.values()))[0], dtype=np.int8)
                known[w] = True
        if not any(known[2:]):
            return None
        return words, known

    @staticmethod
    def prefix_mask(known):
        """known[10] -> the chain-valid prefix: True up to (excluding) the first unknown
        word; everything after an unknown word is untransmittable (see class docstring)."""
        out = [False] * 10
        for w in range(10):
            if not known[w]:
                break
            out[w] = True
        return out


class _PrnState:
    def __init__(self):
        self.hist = {}  # slot -> +-1 (chained polarity applied)
        self.theta = None  # last emit's theta0 (rad, (-pi/2, pi/2])
        self.pol = 1  # chained polarity applied to incoming emits
        self.bit_s = BIT_S  # measured bit duration (br * rec_dt)
        # sync state
        self.sync = None  # (slot0, tow0, sfid0): subframe start at slot0
        self.sf_data = {}  # sfid (1..3) -> 10x24 data words (latest)
        self.pages = {}  # (sfid, tow % 125) -> 10x24 data words (sf 4/5 supercycle)
        self.tlm = {}  # sfid -> word-1 data (TLM; near-constant, kept per sfid)
        self.how = {}  # sfid -> word-2 data template (alert/AS flags are
        # quasi-static; only the TOW field changes per subframe)
        self.n_decoded = 0
        self.repair_t = (
            0.0  # last polarity-repair attempt (rate limit, see REPAIR_MIN_S)
        )
        self.last_obs = (
            None  # (utc_ref, phase, br) of the last emit folded in -- the broker
        )
        # re-polls the same one several times over
        self.try_slot = (
            None  # newest slot at the last sync attempt (see TRY_EVERY_BITS)
        )
        self.n_pred_checked = (
            0  # prediction-health counters (bits compared / mismatched)
        )
        self.n_pred_wrong = 0


class LnavPredictor:
    """Per-PRN LNAV stitch/sync/decode. Thread-compat: call from one poll loop."""

    def __init__(self, log=None):
        self.fleet = FleetPages()
        self._p = {}
        self._log = log or (lambda m: None)

    # ---------------- ingest ----------------
    def ingest(self, prn, obs):
        """Fold one combiner nav_obs into the PRN's bit history."""
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
        # Same emit as last time? The broker polls faster than the combiner emits.
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        st.bit_s = br * rec_dt
        # Polarity chaining: theta0 = -arg(rot), reduced mod pi. An odd pi-step vs the last
        # emit means this emit's global sign is flipped relative to it.
        theta = -math.atan2(rot.imag, rot.real)
        if st.theta is not None:
            d = theta - st.theta
            k = round(d / math.pi)
            if k % 2:
                st.pol = -st.pol
        st.theta = theta
        for bi, sg in pairs:
            bi = int(bi)
            sg = int(sg)
            if sg not in (1, -1):
                continue
            utc_bit = utc_ref + (bi * br - phase) * rec_dt
            slot = int(round(utc_bit / st.bit_s))
            st.hist[slot] = sg * st.pol
        # Bound the history to the newest HIST_MAX_SLOTS slots: the cut point is a subtraction
        # from max(), with no sort.
        if len(st.hist) > HIST_MAX_SLOTS:
            cut = max(st.hist) - HIST_MAX_SLOTS
            st.hist = {k: v for k, v in st.hist.items() if k >= cut}
        newest = max(st.hist) if st.hist else None
        if newest is not None and (
            st.try_slot is None or newest - st.try_slot >= TRY_EVERY_BITS
        ):
            st.try_slot = newest
            self._try_sync(prn, st)

    # ---------------- sync + decode ----------------
    def _contig_run(self, st):
        """Trailing contiguous slot run (start_slot, bits +-1 list), capped at SCAN_MAX_BITS.

        Walks back from max() with dict probes: O(run), no sort. The cap bounds the work
        handed to frame_sync, a per-bit Python loop; only the newest bits can hold an
        undecoded subframe, and SYNC_MIN_BITS sits far inside the window.
        """
        if not st.hist:
            return None
        end = max(st.hist)
        start = end
        n = 1
        while n < SCAN_MAX_BITS and (start - 1) in st.hist:
            start -= 1
            n += 1
        return start, [st.hist[s] for s in range(start, end + 1)]

    def _try_sync(self, prn, st):
        run = self._contig_run(st)
        if run is None or len(run[1]) < SYNC_MIN_BITS:
            return
        slot0, bits_pm = run
        bits01 = ((1 - np.array(bits_pm, dtype=np.int8)) // 2).astype(
            np.int8
        )  # +-1 -> 0/1
        hits = frame_sync(bits01)
        if not hits:
            # Parity disagreed: possibly one wrong polarity-chain link inside the run. Repair
            # search: flip the suffix at each 50-bit step until parity syncs. Rate-limited to
            # one search per REPAIR_MIN_S per PRN: it runs hardest on the weak satellites it
            # least often helps, and the bits are still there on the next try.
            now = time.time()
            if now - st.repair_t < REPAIR_MIN_S:
                return
            st.repair_t = now
            n = len(bits01)
            base = max(0, n - REPAIR_WIN)  # search the trailing window only
            tail = bits01[base:]
            win = len(tail)
            for cut in range(50, win - SYNC_MIN_BITS + 1, 50):
                cand = tail.copy()
                cand[cut:] ^= 1
                hits = frame_sync(cand)
                if hits:
                    self._log(
                        "navbit PRN %d: polarity chain repaired at bit %d"
                        % (prn, base + cut)
                    )
                    for s in range(slot0 + base + cut, slot0 + n):
                        if s in st.hist:
                            st.hist[s] = -st.hist[s]
                    st.pol = -st.pol  # future emits chain off the repaired suffix
                    # continue with the repaired tail: slot0 moves with it so the decode
                    # below and st.sync stay in absolute slots.
                    bits01 = cand
                    slot0 = slot0 + base
                    break
            if not hits:
                return
        # Decode every complete subframe in the run; store content + the slot<->TOW anchor.
        for (idx, tow, sfid) in hits:
            if idx + SF_BITS > len(bits01):
                continue
            # Each try rescans the whole window, so decode and score a subframe only once:
            # st.sync holds the newest decoded subframe's start slot.
            if st.sync is not None and slot0 + idx <= st.sync[0]:
                continue
            # D29*/D30* before the run's first bit are unknown: take 0, as frame_sync does (the
            # ICD trim ends every subframe in 00); a negative index would read the run's end.
            d29 = int(bits01[idx - 2]) if idx >= 2 else 0
            d30 = int(bits01[idx - 1]) if idx >= 1 else 0
            data, ok, tow2, sfid2, _, _ = decode_subframe(bits01, idx, d29, d30)
            if not ok or tow2 != tow or sfid2 != sfid:
                continue
            # Prediction health: if we could have predicted this subframe, score it.
            pred = self._subframe_bits(st, tow, sfid)
            if pred is not None:
                got = 1 - 2 * bits01[idx : idx + SF_BITS].astype(int)  # 0/1 -> +-1
                known = pred != 0
                if known.any():
                    # Polarity-free score: the prediction lives in the canonical (ICD 0/1)
                    # frame while the chained history carries an arbitrary global sign, so a
                    # constant flip is not an error; score each subframe at its better
                    # polarity. A real content mismatch (data-set update, wrong page) still
                    # shows: it is wrong in both polarities.
                    w = int((pred[known] != got[known]).sum())
                    st.n_pred_checked += int(known.sum())
                    st.n_pred_wrong += min(w, int(known.sum()) - w)
            st.sync = (slot0 + idx, tow, sfid)
            st.tlm[sfid] = data[0]
            st.how[sfid] = data[1]
            if sfid <= 3:
                st.sf_data[sfid] = data
            else:
                st.pages[(sfid, tow % 125)] = data
                self.fleet.offer(prn, sfid, tow % 125, data)
            st.n_decoded += 1

    # ---------------- predict ----------------
    def _subframe_bits(self, st, tow, sfid):
        """Predicted +-1/0 bits (len 300) for the subframe with this HOW TOW/id, or None."""
        if sfid <= 3:
            data = st.sf_data.get(sfid)
            unknown_tail = False
        else:
            data = st.pages.get((sfid, tow % 125))
            unknown_tail = data is None
            fleet_known = None
            if data is None and sfid in st.tlm and sfid in st.how:
                # Page unseen by this satellite: TLM + HOW are still exact (the HOW template
                # carries the quasi-static alert/AS flags; TOW is set below), and the fleet
                # may know the data words -- unanimity across other satellites' decodes of
                # the same supercycle page (see FleetPages).
                data = [st.tlm[sfid], st.how[sfid]] + [np.zeros(24, dtype=np.int8)] * 8
                fl = self.fleet.lookup(sfid, tow % 125)
                if fl is not None:
                    fw, fk = fl
                    fleet_known = [True, True] + fk[2:]
                    for w in range(2, 10):
                        if fk[w]:
                            data[w] = fw[w]
        if data is None:
            return None
        data = [np.array(d, dtype=np.int8).copy() for d in data]
        data[1] = set_how_tow(data[1], tow)
        bits01 = encode_subframe_trimmed(
            data
        )  # ICD trim: chain resets at every subframe
        out = (1 - 2 * bits01.astype(int)).astype(np.int8)
        if unknown_tail:
            if fleet_known is None:
                out[60:] = 0  # words 3-10 unknown (only TLM+HOW predicted)
            else:
                # Fleet-filled: emit only the chain-valid prefix. Transmitted bits of word
                # w+1 depend on D30 of word w, so the first unknown word ends what can be
                # emitted -- a known word after a gap is real data but untransmittable.
                pm = FleetPages.prefix_mask(fleet_known)
                for w in range(10):
                    if not pm[w]:
                        out[30 * w :] = 0
                        break
        return out

    def ephemeris(self, prn):
        """Decoded LNAV clock+ephemeris (SI) from the latest subframes 1/2/3, or None until all
        three are decoded and their issue-of-data (IODE/IODC) agree -- guarding against a set
        stitched across a data-set upload. Fields per gps_nav_decode.LNAV_EPH_FIELDS (angles in
        radians; toe/toc in GPS seconds-of-week). Consumed by the decoded-eph BRDC fallback
        (gps_distributed_broker.py) and by gnss_broker/navbits.py (BRDC cross-check,
        decode-health export)."""
        st = self._p.get(prn)
        if st is None or not all(k in st.sf_data for k in (1, 2, 3)):
            return None
        eph = parse_lnav_ephemeris(st.sf_data[1], st.sf_data[2], st.sf_data[3])
        if eph is None or not eph["_iod_consistent"]:
            return None
        return eph

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        # run/hist diagnose the unsynced case: sync needs SYNC_MIN_BITS *contiguous* slots, so a
        # PRN can accumulate plenty of bits and still never sync if its stream is gappy (every
        # gap truncates the trailing run). run is capped at SCAN_MAX_BITS, so compare it with
        # min(hist, SCAN_MAX_BITS): well below it means gaps; near it but < SYNC_MIN_BITS means
        # not enough history yet.
        run = self._contig_run(st)
        return {
            "synced": st.sync is not None,
            "decoded_sf": st.n_decoded,
            "pages": len(st.pages),
            "run": len(run[1]) if run else 0,
            "hist": len(st.hist),
            "need": SYNC_MIN_BITS,
            "mismatch": (st.n_pred_wrong / st.n_pred_checked)
            if st.n_pred_checked
            else None,
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(7)
    # Build 4 frames (20 subframes) of a synthetic LNAV stream with proper structure:
    # sf 1-3 constant data; sf 4/5 pages varying by frame (so hold-prediction would fail
    # without the page store); TOW advancing; trimmed words 2/10 via encode_subframe_trimmed.
    sf_data = {
        k: [rng.randint(0, 2, 24).astype(np.int8) for _ in range(10)]
        for k in range(1, 4)
    }
    # Make the issue-of-data agree across subframes 1/2/3 so .ephemeris() accepts the set:
    # IODC 8 LSBs (sf1 word8 = index 7, bits 1-8) == IODE (sf2 word3 = index 2; sf3 word10 = 9).
    _iode = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    sf_data[1][7][0:8] = _iode
    sf_data[2][2][0:8] = _iode
    sf_data[3][9][0:8] = _iode
    pages = {}
    stream = []
    tow0 = 4000
    truth = []  # (tow, sfid, bits01)
    for fr in range(4):
        for k in range(5):
            sfid = k + 1
            tow = tow0 + fr * 5 + k
            if sfid <= 3:
                data = [d.copy() for d in sf_data[sfid]]
            else:
                pk = (sfid, tow % 125)
                if pk not in pages:
                    pages[pk] = [
                        rng.randint(0, 2, 24).astype(np.int8) for _ in range(10)
                    ]
                data = [d.copy() for d in pages[pk]]
            data[0][:8] = PREAMBLE
            data[0][8:] = 0
            data[1][17:19] = 0  # alert/AS flags quasi-static (ICD-realistic)
            data[1] = set_how_tow(data[1], tow)
            data[1][19:22] = [int(x) for x in format(sfid, "03b")]
            bits = encode_subframe_trimmed(data)
            truth.append((tow, sfid, bits))
            stream.extend(int(x) for x in bits)
    stream = np.array(stream, dtype=np.int8)
    pm = (1 - 2 * stream).astype(np.int8)  # 0/1 -> +-1 transmitted signs

    # Feed as 50-bit emits with a drifting carrier. The emit's global sign is not free: the
    # combiner decides signs against the mod-pi reduced axis e^{-i*theta_red}, so the flip is
    # exactly the branch the reduction removed -- sign = +1 when the true phasor projects onto
    # the +axis, -1 when onto the -axis. The reduced theta sequence then jumps by ~pi exactly
    # when the true phase crosses +-pi/2, which is what the chain detects.
    pred = LnavPredictor(log=print)
    theta_true = 0.3
    for e in range(0, len(pm) - 50, 50):
        theta_true += rng.uniform(
            -0.05, 0.25
        )  # drift; crosses pi/2 boundaries sometimes
        th_red = math.remainder(theta_true, math.pi)  # (-pi/2, pi/2]
        flip = round((theta_true - th_red) / math.pi) % 2
        rot = complex(math.cos(-th_red), math.sin(-th_red))
        sign = 1 - 2 * flip
        obs = {
            "utc_ref": e * BIT_S,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 20,
            "rot_re": rot.real,
            "rot_im": rot.imag,
            "bits": [[i, int(pm[e + i]) * sign] for i in range(50)],
        }
        pred.ingest(7, obs)
    st = pred._p[7]
    assert st.sync is not None, "no frame sync"
    # Predict each subframe of the frame after the ingested stream (what the mismatch score
    # compares against each new decode); compare vs a freshly encoded truth frame.
    ok = wrong = unknown = 0
    for k in range(5):
        sfid = k + 1
        tow = tow0 + 20 + k
        if sfid <= 3:
            data = [d.copy() for d in sf_data[sfid]]
        else:
            data = [
                d.copy()
                for d in pages.get(
                    (sfid, tow % 125), [np.zeros(24, dtype=np.int8)] * 10
                )
            ]
        data[0][:8] = PREAMBLE
        data[0][8:] = 0
        data[1][17:19] = 0  # alert/AS flags quasi-static (ICD-realistic)
        data[1] = set_how_tow(data[1], tow)
        data[1][19:22] = [int(x) for x in format(sfid, "03b")]
        bits = encode_subframe_trimmed(data)
        want = 1 - 2 * bits.astype(int)
        pred_sf = pred._subframe_bits(st, tow, sfid)
        seg = np.zeros(SF_BITS, dtype=int) if pred_sf is None else pred_sf.astype(int)
        # polarity-free comparison: the predictor's absolute polarity is arbitrary
        for pol in (1, -1):
            w = (seg != 0) & (seg * pol != want)
            if w.sum() <= (seg != 0).sum() // 2:
                break
        unknown += int((seg == 0).sum())
        wrong += int(w.sum())
        ok += int(((seg != 0) & ~w).sum())
    print(
        "self-test: predicted next frame: %d ok, %d wrong, %d unknown (of %d)"
        % (ok, wrong, unknown, 5 * SF_BITS)
    )
    # ephemeris extraction: .ephemeris() must accept the IOD-consistent set and reproduce a
    # direct parse of the injected subframes (the decode of sf 1-3 is stored in st.sf_data).
    eph = pred.ephemeris(7)
    assert (
        eph is not None and eph["_iod_consistent"]
    ), "ephemeris not extracted / IOD inconsistent"
    ref = parse_lnav_ephemeris(sf_data[1], sf_data[2], sf_data[3])
    assert all(
        abs(eph[k] - ref[k]) < 1e-12 * (abs(ref[k]) + 1)
        for k in ("sqrtA", "e", "M0", "toe", "OMEGA0", "i0", "af0")
    ), "ephemeris mismatch"
    print(
        "ephemeris: sqrtA=%.3f e=%.3e toe=%.0f  (IODE %d)"
        % (eph["sqrtA"], eph["e"], eph["toe"], int(eph["IODE"]))
    )
    print("health:", pred.health(7))
    assert wrong == 0, "prediction mismatches"
    assert ok >= 3 * SF_BITS, "subframes 1-3 must predict fully"
    print("PASS")


if __name__ == "__main__":
    _selftest()
