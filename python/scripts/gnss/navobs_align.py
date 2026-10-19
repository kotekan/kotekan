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
