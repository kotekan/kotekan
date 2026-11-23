"""Slot alignment of nav_obs symbol emits against a predictor's stitched history.

The nav predictors place each emit's symbols on an absolute slot grid, slot = round(t / sym_s),
where t is the symbol's start time from the emit's utc_ref and phase. When a satellite's symbol
edge sits near half a symbol on that grid, a small change in the start-time estimate between
emits (a one-record phase change at br > 1, float jitter at br = 1) rounds the same symbol into
adjacent slots, and the stitched history slips by one. emit_offset measures the emit against
the history it overlaps and returns the slot shift that lines it up.
"""

import numpy as np

MIN_OVERLAP = 32  # symbols shared with the history before an offset is judged
# An offset that disagrees on more of its shared symbols than this is no match: a wrong offset
# of random symbols disagrees on about half, two emits with independent 15% symbol errors on
# about a quarter.
MAX_MISMATCH = 0.3


def emit_offset(slot0, arr, history):
    """Slot shift in {-1, 0, +1} that aligns a new emit with the stitched history.

    slot0, arr: the emit at its rounded slots, arr one entry per slot (+-1, 0 = no symbol).
    history: (start slot, array) pieces in the same encoding. Each piece has a carrier sign of
    its own (emit signs are independent), so a piece's agreement is counted at its better
    polarity.

    Returns the offset with the smallest mismatch fraction, the rounded slot winning ties; or
    0, the rounded slot, when fewer than MIN_OVERLAP symbols are shared there or when even
    the best offset disagrees on more than MAX_MISMATCH of its shared symbols.
    """
    a = np.asarray(arr, dtype=np.int64)
    best, best_bad = 0, None
    for d in (0, -1, 1):
        lo0 = slot0 + d
        n = bad = 0
        for s0, h in history:
            lo = max(lo0, s0)
            hi = min(lo0 + len(a), s0 + len(h))
            if hi <= lo:
                continue
            b = np.asarray(h[lo - s0 : hi - s0], dtype=np.int64)
            p = a[lo - lo0 : hi - lo0] * b
            k = int(np.count_nonzero(p))
            n += k
            bad += (k - abs(int(p.sum()))) // 2
        if n < MIN_OVERLAP:
            if d == 0:
                return 0
            continue
        if best_bad is None or bad / n < best_bad:
            best, best_bad = d, bad / n
    return best if best_bad <= MAX_MISMATCH else 0


class HistoryState:
    """One PRN's chained symbol history (SymbolHistory)."""

    def __init__(self):
        self.grid0 = None  # slot-grid origin (s)
        self.s0 = None  # first slot of the history
        self.hist = np.zeros(0)  # soft values from slot s0 (0 = no symbol)
        self.scan = None  # first slot the decoder has not scanned
        self.last_obs = None
        self.restarts = (
            0  # times the history restarted on an emit it could not chain to
        )
        self.off_grid = 0  # consecutive emits whose symbols fall off the history's grid


class SymbolHistory:
    """Chains a combiner's bit_export emits into one soft-symbol history per PRN. Each emit is
    slot-aligned to the history (emit_offset) and takes the polarity of the history it overlaps;
    one that shares fewer than min_overlap_pol symbols restarts the history, having nothing to
    chain its polarity to. The last hist_max symbols are kept. A subclass gives new_state()
    (a HistoryState) and _decode(prn, st), called after each new emit; soft values (nav_obs
    "soft", 10x) are used where present, else the signs."""

    hist_max = 1500
    min_overlap_pol = 16
    # An emit whose symbol boundaries fall off the history's grid (the combiner's bit sync
    # chose another phase for that window, pairing records across symbols) is dropped, unless
    # this many emits in a row agree on the new grid: then the grid itself moved and the history
    # restarts on it.
    grid_debounce = 5

    def __init__(self):
        self._p = {}

    def ingest(self, prn, obs):
        st = self._p.get(prn)
        if st is None:
            st = self._p[prn] = self.new_state()
        try:
            utc_ref = float(obs["utc_ref"])
            rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"])
            br = int(obs["br"])
            pairs = obs["bits"]
            soft = obs.get("soft")
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        if rec_dt <= 0 or br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        if soft is not None and len(soft) != len(pairs):
            soft = None
        sym_s = br * rec_dt
        t0 = utc_ref - phase * rec_dt
        if st.grid0 is None:
            st.grid0 = t0
        off = (t0 - st.grid0) / sym_s
        if abs(off - round(off)) > 0.25:
            st.off_grid += 1
            if st.off_grid < self.grid_debounce:
                return
            st.grid0, st.s0, st.hist = t0, None, np.zeros(0)
        st.off_grid = 0
        base = int(round((t0 - st.grid0) / sym_s))
        idx = np.array([int(p[0]) for p in pairs])
        sg = np.array([int(p[1]) for p in pairs], dtype=float)
        val = (
            sg
            if soft is None
            else np.where(np.array(soft) * sg > 0, np.array(soft) / 10.0, sg * 0.01)
        )
        lo, hi = int(idx.min()), int(idx.max())
        arr = np.zeros(hi - lo + 1)
        arr[idx - lo] = val
        slot0 = base + lo
        if st.s0 is not None and len(st.hist):
            slot0 += emit_offset(
                slot0, np.sign(arr).astype(int), [(st.s0, np.sign(st.hist).astype(int))]
            )
        self._merge(st, slot0, arr)
        self._decode(prn, st)

    def _merge(self, st, slot0, arr):
        if st.s0 is None or slot0 > st.s0 + len(st.hist) or slot0 + len(arr) < st.s0:
            # a new island: nothing to chain its polarity to, so the history restarts there
            st.s0, st.hist, st.scan = slot0, arr.copy(), slot0
            st.restarts += 1
            return
        lo = max(slot0, st.s0)
        hi = min(slot0 + len(arr), st.s0 + len(st.hist))
        a = arr[lo - slot0 : hi - slot0]
        h = st.hist[lo - st.s0 : hi - st.s0]
        both = (a != 0) & (h != 0)
        if (
            np.count_nonzero(both) >= self.min_overlap_pol
            and float(np.dot(np.sign(a[both]), np.sign(h[both]))) < 0
        ):
            arr = -arr
        end = max(slot0 + len(arr), st.s0 + len(st.hist))
        start = min(slot0, st.s0)
        new = np.zeros(end - start)
        new[st.s0 - start : st.s0 - start + len(st.hist)] = st.hist
        tail = arr[max(0, st.s0 + len(st.hist) - slot0) :]
        t0 = max(slot0, st.s0 + len(st.hist))
        new[t0 - start : t0 - start + len(tail)] = tail
        st.s0, st.hist = start, new
        if len(st.hist) > self.hist_max:
            cut = len(st.hist) - self.hist_max
            st.s0 += cut
            st.hist = st.hist[cut:]
            st.scan = max(st.scan, st.s0)
