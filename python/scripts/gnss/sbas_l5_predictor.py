"""SBAS L5 (DFMC) message decode from the SBAS combiner's 500 sps symbols.

DFMC SBAS on L5 (ICAO Annex 10, ED-259) sends one 250-bit message per second: a 4-bit
preamble, a 6-bit message type, 216 data bits and a CRC-24Q over the first 226 bits, through the
rate-1/2 K=7 convolutional code that GPS CNAV uses (500 sps). The bi-binary (Manchester) halves
are wiped by the tracker code (SBAS_L5_I_MC), so each emit slot is one FEC symbol.

The combiner's emits are chained into a symbol history (navobs_align.SymbolHistory). Each new
stretch is Viterbi-decoded with an overlap of more than one message behind it, and messages are
found by their CRC and preamble at every bit offset, in both polarities. Which symbol of a pair
starts a data bit, and whether G2 is inverted, are searched until three messages land 500
symbols apart, then kept: the pair phase per PRN, the G2 convention for all (a link constant).

No ephemeris is decoded yet: the per-type counts show the stream (31 satellite mask,
32 clock-ephemeris corrections, 34-36 integrity, 37 OBAD/DFREI, 39-40 the GEO's own orbit,
42 time offsets, 47 almanac, 62 test, 63 null). The newest message of each type, and of MT32
for each satellite slot, is kept for the payload file (nav_fields reads MT31 and MT32).
"""

import collections
import time

import numpy as np

from galileo_inav import crc24q, viterbi_decode
from navobs_align import HistoryState, SymbolHistory

MSG_BITS = 250
CRC_AT = 226
# The six 4-bit preambles the messages cycle through: every word with two ones. A decode at the
# wrong pair phase turns a message into another that can still pass the CRC (CRC codewords are
# polynomial multiples); its preamble and spacing give it away.
# A zero-initialised CRC also passes a window shifted one bit whenever the bit it drops and
# the bit it takes in are both zero, so messages are counted only on the lattice of one start
# every 500 symbols, set where most are found.
PREAMBLES = {
    (0, 0, 1, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (1, 1, 0, 0),
}
# Viterbi output is trusted only this far into a stretch (no known start state) and this far
# from its end (no symbols after it): each message start is tried in one stretch only, where
# both margins clear it, so the stretches overlap by a message and both margins.
WARMUP_BITS = 16
TAIL_BITS = 48
OVERLAP_SYMS = 2 * (MSG_BITS + WARMUP_BITS + TAIL_BITS)
# An unlocked PRN decodes its whole history at most this often, and only while its symbols
# stand above noise (whose median |soft| is 0.67 sigma): the four-way decode of a dark GEO
# would hold up the others' emits.
RETRY_S = 2.0
# A right lock finds a message a second; at the wrong pair phase some messages still turn into
# CRC-valid ones with a valid preamble (about 3 in 8). A lock held this long with fewer new
# messages over the same span is dropped, as is one a tracker re-acquisition left behind (its
# emits chain through the gap, keeping a pair phase and lattice that no longer hold).
CHECK_S = 10.0
MIN_IN_CHECK = 6
MIN_MEDIAN_SOFT = 0.9


def _crc_bits(bits):
    v = crc24q(bits)
    return [(v >> (23 - j)) & 1 for j in range(24)]


# CRC-24Q starts from zero, so it is linear over GF(2): the CRC of every window is one product
# with this matrix, and a complemented message's differs from its own by the CRC of all ones.
_CRC_M = np.array(
    [_crc_bits([int(j == k) for j in range(CRC_AT)]) for k in range(CRC_AT)]
)
_CRC_INV = np.array(_crc_bits([1] * CRC_AT)) ^ 1


class _PrnState(HistoryState):
    def __init__(self):
        super().__init__()
        self.phase = None  # parity of the slots that start a bit, once locked
        self.lattice = None  # message start slots mod 500, once locked
        self.locked_at = None  # the history restart count the lock holds for
        self.lock_t = 0.0
        self.recent = collections.deque()  # times of the new messages
        self.next_try = 0.0
        self.seen = set()  # message start slots counted
        self.types = {}  # message type -> count
        self.newest = {}  # message type -> (host time, bits) of its newest message
        self.mt32 = (
            {}
        )  # MT32 satellite slot -> (host time, bits) of its newest correction
        self.n_msg = 0
        self.last_type = None


def brdc_xcheck(brdc, sysc, prn, eph, log=None):
    """No ephemeris is decoded yet, so there is nothing to cross-check."""
    return ""


class SbasL5Predictor(SymbolHistory):
    hist_max = 3000  # 6 s of symbols

    def __init__(self):
        super().__init__()
        self._g2 = None  # whether the second symbol of a pair is flipped before Viterbi

    def new_state(self):
        return _PrnState()

    def _decode(self, prn, st):
        now = time.time()
        while st.recent and now - st.recent[0] > CHECK_S:
            st.recent.popleft()
        if st.restarts != st.locked_at or (
            st.phase is not None
            and now - st.lock_t > CHECK_S
            and len(st.recent) < MIN_IN_CHECK
        ):
            # a restarted history is a new slot grid, and a silent lock a lost one: the pair
            # phase and lattice are unknown
            st.phase = st.lattice = None
        # Unlocked, the whole history, so that two messages can land in one decode.
        first = st.s0 if st.phase is None else max(st.s0, st.scan - OVERLAP_SYMS)
        seg = st.hist[first - st.s0 :]
        # unlocked: four messages, so that three whole ones clear the margins
        if len(seg) < (8 if st.phase is None else 2) * MSG_BITS + 2 * (
            WARMUP_BITS + TAIL_BITS
        ):
            return
        if st.phase is None:
            if time.time() < st.next_try:
                return
            st.next_try = time.time() + RETRY_S
            if np.median(np.abs(seg[-2 * MSG_BITS :])) < MIN_MEDIAN_SOFT:
                return
        best = None  # (messages, phase, g2, bits)
        for ph in (0, 1) if st.phase is None else ((st.phase - first) % 2,):
            for g2 in (False, True) if self._g2 is None else (self._g2,):
                x = np.array(seg[ph : ph + 2 * ((len(seg) - ph) // 2)], dtype=float)
                if g2:
                    x[1::2] = -x[1::2]
                bits = [int(b) for b in viterbi_decode(x, known_start=False)][
                    : len(x) // 2
                ]
                slots = self._scan(st, bits, first + ph, count=False)
                spaced = any(
                    x + 2 * MSG_BITS in slots and x + 4 * MSG_BITS in slots
                    for x in slots
                )
                if (st.phase is not None or spaced) and (
                    best is None or len(slots) > best[0]
                ):
                    best = (len(slots), ph, g2, bits)
        if best is not None:
            # the real pair phase finds every message; the wrong one only some transformed ones
            if st.phase is None:
                res = [
                    x % (2 * MSG_BITS)
                    for x in self._scan(st, best[3], first + best[1], False)
                ]
                st.lattice = max(set(res), key=res.count)
            if st.phase is None:
                st.lock_t = time.time()
            st.phase, self._g2 = (first + best[1]) % 2, best[2]
            st.locked_at = st.restarts
            self._scan(st, best[3], first + best[1], count=True)
        st.scan = st.s0 + len(st.hist)

    def _scan(self, st, bits, slot0, count):
        """The start slots of the messages in decoded bits whose first bit is at symbol slot0
        (CRC and preamble), counting them when @p count."""
        n = len(bits) - MSG_BITS - TAIL_BITS + 1 - WARMUP_BITS
        if n <= 0:
            return []
        w = np.lib.stride_tricks.sliding_window_view(
            np.array(bits, dtype=np.int64), MSG_BITS
        )
        w = w[WARMUP_BITS : WARMUP_BITS + n]
        d = ((w[:, :CRC_AT] @ _CRC_M) % 2) ^ w[:, CRC_AT:]
        found = []
        for k in np.nonzero(~d.any(axis=1) | (d == _CRC_INV).all(axis=1))[0]:
            m = w[k] ^ (0 if not d[k].any() else 1)
            if tuple(int(b) for b in m[:4]) not in PREAMBLES:
                continue
            slot = slot0 + 2 * (WARMUP_BITS + int(k))
            found.append(slot)
            if count and slot % (2 * MSG_BITS) == st.lattice and slot not in st.seen:
                st.seen.add(slot)
                st.recent.append(time.time())
                st.n_msg += 1
                mt = int("".join(str(int(b)) for b in m[4:10]), 2)
                st.types[mt] = st.types.get(mt, 0) + 1
                st.last_type = mt
                bits_m = [int(b) for b in m]
                st.newest[mt] = (time.time(), bits_m)
                if mt == 32:
                    st.mt32[int("".join(map(str, bits_m[11:19])), 2)] = st.newest[mt]
        if len(st.seen) > 4096:
            st.seen = {s for s in st.seen if s >= st.s0}
        return found

    def ephemeris(self, prn):
        return None

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_msg,
            "words": st.n_msg,
            "synced": st.phase is not None,
            "types": " ".join("%d:%d" % kv for kv in sorted(st.types.items())),
        }


def _selftest():
    """Random messages with valid CRCs, convolutionally encoded, with symbol noise, chopped
    into overlapping 500-symbol emits of random polarity starting on an odd symbol: every
    message must come back with its type."""
    from galileo_inav import conv_encode

    rng = np.random.RandomState(5)
    types = []
    bits = []
    for k in range(12):
        m = [int(x) for x in rng.randint(0, 2, CRC_AT)]
        m[:4] = sorted(PREAMBLES)[k % 6]
        mt = (31, 32, 35, 39, 40, 63)[k % 6]
        m[4:10] = [(mt >> (5 - j)) & 1 for j in range(6)]
        bits += m + _crc_bits(m)
        types.append(mt)
    sym = 1.0 - 2.0 * np.array(conv_encode(bits, flush=True), dtype=float)
    sym = np.concatenate([[1.0], sym]) + 0.5 * rng.randn(len(sym) + 1)
    pred = SbasL5Predictor()
    prn = 133
    for e in range(0, len(sym) - 500, 173):  # an odd step moves the segment's parity
        pol = 1 - 2 * rng.randint(0, 2)
        seg = sym[e : e + 500] * pol
        pred.ingest(
            prn,
            {
                "utc_ref": e * 0.002,
                "rec_dt": 0.001,
                "phase": 0,
                "br": 2,
                "bits": [[i, 1 if v >= 0 else -1] for i, v in enumerate(seg)],
                "soft": [int(round(10 * v)) for v in seg],
            },
        )
    st = pred._p[prn]
    want = {}
    for t in types[1:-1]:  # the first and last fall in the Viterbi warm-up
        want[t] = want.get(t, 0) + 1
    ok = all(st.types.get(t, 0) >= c for t, c in want.items()) and set(st.types) <= set(
        types
    )
    print(
        "messages %d (types %s), want at least %s: %s"
        % (st.n_msg, st.types, want, "PASS" if ok else "FAIL")
    )
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
