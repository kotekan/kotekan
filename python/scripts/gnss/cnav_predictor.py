#!/usr/bin/env python3
"""CNAV (L2C-CM / L5-I) decode.

The CNAV analog of navbit_predictor.LnavPredictor, but a different shape because CNAV is a
different problem:

    combiner get_status "nav_obs"  ->  per-emit polarity islands  ->  CRC-guided decode  ->  msgs
                              (gps_cnav.py frames the messages; galileo_inav.py holds the FEC/CRC)

Why it is not a copy of the LNAV predictor:

  * Not cheaply predictable. LNAV subframes 1-3 repeat every 30 s with only a TOW increment.
    CNAV interleaves message types on a control-segment schedule that is not contractually
    fixed, and the code is continuous rate-1/2 K=7, so there is no reliable future-symbol
    table. None is needed: CL already gives L2C a dataless pilot, so the decode is for the
    ephemeris (types 10+11 -> BRDC cross-check).

  * G2-inversion convention. On air the GPS CNAV symbols are not G2-inverted, while the
    module encoder inverts G2 (galileo_inav.G2_INVERT, the Galileo convention). With g2 set,
    the decoder flips the 2nd symbol of each phase-aligned pair (`x[1::2]`) before Viterbi;
    it searches g2 in {False, True} and locks it on the first CRC pass (a constant of the
    link).

  * Per-emit polarity is not chainable. Each nav_obs emit is a squaring-estimated window
    whose global sign is independent, and a single missed flip mid-run makes the
    convolutional decode garbage. The decoder is sized for abutting ~4 s (~200-symbol)
    emits: consecutive emits are then 4 s apart in carrier phase, so theta-chaining (what
    LNAV uses) misses flips, and they share only one boundary slot, too thin to vote
    reliably. So polarity is resolved by CRC: a search over the per-emit sign pattern in a
    short window (a message spans ~3 emits).

Consumed by gnss_broker/navbits.py, under --nav-decoder cnav or on the aux chain named by
--cnav-combiner (L5-I); navbits.py logs the decode health and the decoded ephemeris with its
BRDC cross-check.

Self-test: python3 cnav_predictor.py  (synthetic continuous CNAV stream chopped into
polarity-scrambled emits -> the CRC-guided decoder must recover the messages and the
ephemeris; g2=False wins there because the synthetic symbols use the module convention).
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps_cnav as C  # noqa: E402
from galileo_inav import (
    conv_encode,
    viterbi_decode,
)  # noqa: E402  (the shared CNAV FEC)

SYM_S = 0.020  # L2C-CM record = 20 ms = one CNAV symbol (50 sps)
MSG_SYMS = C.MSG_BITS * 2  # 600 symbols per 300-bit message (12 s)
WIN_EMITS = 7  # emits per warm decode window. A 300-bit message is 600 symbols, and the
# K=7 Viterbi needs ~35 bits to converge at each end, so a usable decode
# wants ~2 messages of contiguous symbols: ~6-7 emits of ~200 symbols
# (~4 s) each. A shorter window often fails to place the preamble. With
# abutting emits (one ~200-record window every 4 s), a decode that blocks
# the poll loop past the 4 s emit period misses an emit and fragments the
# run, which is why the decode is staggered and budget-capped below. The
# window takes the longest contiguous run. Polarity within a run is
# resolved by _try_decode (cold: rotated 2^k brute; warm: chained
# candidate + 1/2-emit-flip repair).
COLD_EMITS = 5  # window (emits) for a cold PRN's full 2^k brute (2^5=32 sign patterns).
VITERBI_BUDGET = (
    40  # hard cap on Viterbi calls per decode attempt. It covers a cold attempt
)
# (2^COLD_EMITS at one rotated phase/g2 = 32) and any warm attempt, and keeps
# an attempt well under the 4 s emit period, so a decode never stalls the poll
# loop long enough to miss an emit and fragment nav_obs.
COLD_TRIES = (
    6  # free quick cold-bootstrap attempts before exponential backoff kicks in.
)
# A decodable PRN locks within ~4 (the phase/g2 rotation cycles in 4); a
# PRN that never locks (too weak / no data) is the one to back off, because
# its full 2^COLD_EMITS brute every round-robin turn is a continuous load on
# the poll loop.
COLD_BACKOFF_BASE = (
    8.0  # s; retry a stuck cold PRN after BASE, then 2x each further failure...
)
COLD_BACKOFF_MAX = (
    300.0  # ...capped at 5 min. Locked PRNs are exempt (their incremental decode is
)
# a handful of Viterbis -- cheap -- so they stay responsive).
EMIT_MAX = 48  # keep ~48 emits (~3 min) per PRN
DECODED_MAX = 80  # keep the last ~80 decoded messages (~16 min)
TRY_EVERY_SYMS = (
    200  # attempt a decode once per ~4 s of new symbols per PRN (one new emit)
)


def _crc_msgs(bits):
    """CRC-24Q-valid CNAV messages in a decoded bit stream: [(bit_index, parsed)]."""
    out = []
    n = len(bits)
    i = 0
    while i + C.MSG_BITS <= n:
        if list(int(b) for b in bits[i : i + 8]) == C.PREAMBLE:
            w = [int(b) for b in bits[i : i + C.MSG_BITS]]
            m = C.parse_cnav_message(w)
            if m is not None:
                m["index"] = i
                m["bits"] = w
                out.append(m)
        i += 1
    return out


def _g2(x, on):
    """Apply the G2-inversion convention toggle to a phase-aligned symbol array: flip the 2nd
    symbol of every pair. `on` False decodes the module's native (G2-inverted) convention."""
    if not on:
        return x
    y = x.copy()
    y[1::2] *= -1
    return y


class _PrnState:
    def __init__(self):
        self.emits = {}  # slot0 -> np.array(+-1 signs), one per distinct nav_obs emit
        self.last_obs = None  # (utc_ref, phase, br) dedup
        self.try_slot = None
        self.phase = (
            None  # per-PRN locked pair_phase (pseudorange shifts the symbol grid)
        )
        self.pol = {}  # slot0 -> resolved emit sign (+-1) in one consistent frame
        self.decoded = {}  # start_slot -> data bits[300] (0/1), CRC-verified
        self.eph10 = None
        self.eph11 = None
        self.n_decoded = 0
        self.boot = 0  # cold-bootstrap (phase, g2) rotation counter
        self.cold_fail = (
            0  # consecutive cold decode attempts that did NOT lock (backoff)
        )
        self.cold_next_t = (
            0.0  # earliest wall time to retry a cold decode (exp backoff)
        )


class CnavPredictor:
    """Per-PRN CNAV stitch/CRC-decode. Single-threaded: call from one poll loop."""

    def __init__(self, log=None):
        self._p = {}
        self._log = log or (lambda m: None)
        # The G2-inversion convention is a link constant (same for every PRN of the signal); the
        # first PRN to pass CRC locks it here and the rest skip that half of the search.
        # pair_phase is not shared -- pseudorange shifts each PRN's symbol grid by up to a
        # symbol -- so it stays per-PRN.
        self._g2 = None
        self._min_gap = (
            1.0  # s between decode attempts across all PRNs (see ingest); 0 = off
        )
        self._last_decode = 0.0
        self._round = (
            set()
        )  # PRNs decoded this round -- round-robin fairness (see ingest)

    # ---------------- ingest ----------------
    def ingest(self, prn, obs):
        """Store one combiner nav_obs (hard +-1 symbols) as an absolute-slot polarity island."""
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"])
            rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"])
            br = int(obs["br"])
            pairs = obs["bits"]
        except (KeyError, TypeError, ValueError):
            return
        if rec_dt <= 0 or br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        sym_s = br * rec_dt
        # Build this emit's absolute slot -> sign map (no polarity applied: each emit is its own
        # sign island; the decoder resolves inter-emit polarity by CRC).
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            slot = int(round((utc_ref + (bi * br - phase) * rec_dt) / sym_s))
            m[slot] = sg
        if len(m) < 2:
            return
        slot0 = min(m)
        hi = max(m)
        arr = np.zeros(hi - slot0 + 1, dtype=np.int8)
        for s, v in m.items():
            arr[s - slot0] = v
        if (
            arr == 0
        ).any():  # a gap inside the emit breaks the convolutional stream: drop it
            return
        st.emits[slot0] = arr
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
        newest = hi
        # Stagger and round-robin across PRNs. The broker ingests every PRN back-to-back in one
        # poll loop; if every cold PRN ran a full decode in one cycle, the loop would stall for
        # seconds and miss the emits arriving meanwhile, fragmenting nav_obs so nothing decodes.
        # So gate on wall time (at most one decode per _min_gap, well under the 4 s emit period)
        # and rotate which PRN decodes: without the round set, the first PRN in iteration order
        # would win the gate every cycle and the rest would never decode. self._min_gap = 0
        # disables the gate (the single-PRN self-test). Cold backoff: a cold PRN (phase is None)
        # that keeps failing its full 2^k brute is a continuous load, so after COLD_TRIES
        # failures it retries on an exponential backoff (locked PRNs decode incrementally and
        # are never backed off).
        now = time.time()
        due = st.try_slot is None or newest - st.try_slot >= TRY_EVERY_SYMS
        cold = st.phase is None
        if (
            due
            and prn not in self._round
            and now - self._last_decode >= self._min_gap
            and not (cold and now < st.cold_next_t)
        ):
            st.try_slot = newest
            self._round.add(prn)
            # a round is complete once every currently-eligible PRN has had a turn (backed-off cold
            # PRNs are not eligible, so they don't stall the rotation among the active ones).
            eligible = sum(
                1
                for s2 in self._p.values()
                if s2.phase is not None or now >= s2.cold_next_t
            )
            if len(self._round) >= max(1, eligible):
                self._round.clear()
            self._try_decode(prn, st)
            self._last_decode = time.time()
            if cold:
                if st.phase is not None:  # locked this attempt -> full speed again
                    st.cold_fail = 0
                    st.cold_next_t = 0.0
                else:  # still cold -> exponential backoff
                    st.cold_fail += 1
                    over = st.cold_fail - COLD_TRIES
                    st.cold_next_t = now + (
                        0.0
                        if over <= 0
                        else min(COLD_BACKOFF_BASE * 2 ** (over - 1), COLD_BACKOFF_MAX)
                    )

    # ---------------- decode ----------------
    # Polarity is resolved by CRC, not chaining. Each nav_obs emit's global sign is independent and
    # abutting emits share only one slot, too thin to chain reliably: one bad link in a 7-emit
    # window ruins the decode. Instead, every attempt decodes the freshest emits of the longest
    # contiguous run while searching only the per-emit signs still unknown (st.pol holds the
    # rest); pair_phase (per-PRN, pseudorange-shifted) and g2 (a global link constant) lock on the
    # first CRC. In steady state only the newest 1-2 emits are free -> 2-4 Viterbi passes; a cold
    # PRN pays a 2^k over the window's k emits, which covers the global sign.
    def _try_decode(self, prn, st):
        starts = sorted(st.emits)
        if len(starts) < 3:
            return
        # Group emits into maximal contiguous runs (each next emit abuts/overlaps the prior, i.e.
        # starts no later than one slot past its end), then take the longest run, not simply the
        # freshest: the newest emit is often isolated by a gap, and insisting on it collapses the
        # window to one emit. Ties go to the fresher run.
        runs = [[starts[0]]]
        for s in starts[1:]:
            prev = runs[-1][-1]
            if prev < s <= prev + len(st.emits[prev]):
                runs[-1].append(s)
            else:
                runs.append([s])
        best = max(runs, key=lambda r: (len(r), r[-1]))
        # Cold PRN (no emit of the run resolved yet): a small window and a full 2^k brute over
        # every sign pattern, bounded by 2^COLD_EMITS. Warm PRN: the freshest WIN_EMITS,
        # resolved emits fixed from st.pol, only the free ones searched.
        cold = not any(s in st.pol for s in best)
        chain = best[-(COLD_EMITS if cold else WIN_EMITS) :]
        if len(chain) < 3:
            return
        hist, owner = {}, {}
        for s in chain:
            arr = st.emits[s]
            for i in range(len(arr)):
                hist[s + i] = int(arr[i])
                owner[s + i] = s
        lo, hi = min(hist), max(hist)
        if lo % 2:  # even-anchor: the pair grid is absolute (utc)
            lo += 1
        if hi - lo + 1 < MSG_SYMS:
            # Say so once per PRN. This window can never hold a message, and returning quietly
            # is indistinguishable from "decoding, just not yet": the chain would report NO
            # DECODE forever. It is reachable by configuration: the window is counted in emits
            # (WIN_EMITS / COLD_EMITS) while what matters is the symbol span, so a combiner
            # emitting shorter windows than L2C's 4 s starves it. The fix is on the producer
            # (raise that chain's integration_length so one emit spans more symbols); raising
            # the emit count is not available, because a cold bootstrap is 2^COLD_EMITS
            # Viterbis.
            if len(chain) >= (COLD_EMITS if cold else WIN_EMITS):
                self._warn_short = getattr(self, "_warn_short", set())
                if prn not in self._warn_short:
                    self._warn_short.add(prn)
                    self._log(
                        "cnav PRN %d: emit span too short to decode -- %d symbols over "
                        "%d emits, need %d. Raise this chain's integration_length "
                        "(emit span = integration_length x record period)."
                        % (prn, hi - lo + 1, len(chain), MSG_SYMS)
                    )
            return
        for x in range(lo, hi + 1):
            if x not in hist:  # gap inside the window -- bail (rare)
                return
        raw = np.array([hist[x] for x in range(lo, hi + 1)], dtype=float)
        own = np.array([chain.index(owner[x]) for x in range(lo, hi + 1)])
        base = np.array(
            [st.pol.get(s, 1) for s in chain]
        )  # known signs; frees overwritten below
        free_idx = [k for k, s in enumerate(chain) if s not in st.pol]
        phases = (st.phase,) if st.phase is not None else (0, 1)
        g2s = (self._g2,) if self._g2 is not None else (False, True)
        # A cold attempt does a full 2^k brute but over only one (phase, g2) per call, rotating
        # across calls, so each attempt is at most 2^COLD_EMITS Viterbis and stays under the 4 s
        # emit period (a longer stall misses emits and fragments the contiguity the brute needs).
        if cold:
            opts = [(ph, g2) for ph in phases for g2 in g2s]
            phases, g2s = (
                (opts[st.boot % len(opts)][0],),
                (opts[st.boot % len(opts)][1],),
            )
            st.boot += 1
        # Bounded candidate set. The decode runs in the broker's single poll loop; a long brute
        # force would make it miss the emits that arrive meanwhile, fragmenting the very
        # contiguity it needs. Cold: every sign pattern of the free emits (at most
        # 2^COLD_EMITS). Warm: the overlap-chained sign pattern (usually decodes as-is), plus
        # every 1- and 2-emit flip of the free emits (repairs one or two bad chain links). The
        # Viterbi calls are capped at VITERBI_BUDGET either way.
        chained = base.copy()
        acc = {}
        for k, s in enumerate(chain):
            arr = st.emits[s]
            m = {s + i: int(arr[i]) for i in range(len(arr))}
            if s in st.pol:
                sg = st.pol[s]
            elif acc:
                shared = set(m) & set(acc)
                ag = sum(1 for x in shared if m[x] == acc[x]) if shared else 0
                sg = 1 if not shared or ag >= len(shared) - ag else -1
            else:
                sg = 1
            chained[k] = sg
            for x, v in m.items():
                acc[x] = v * sg
        base_free = [int(chained[k]) for k in free_idx]
        nf = len(base_free)
        if cold:
            # Full brute over the small cold window: every sign pattern (covers the global sign).
            combos = list(itertools.product((1, -1), repeat=nf))
        else:
            # Warm: chained candidate + 1- and 2-emit-flip repairs of the (few) free emits.
            combos = [tuple(base_free)]
            for j in range(nf):
                c = list(base_free)
                c[j] = -c[j]
                combos.append(tuple(c))
            for j in range(nf):
                for jj in range(j + 1, nf):
                    c = list(base_free)
                    c[j] = -c[j]
                    c[jj] = -c[jj]
                    combos.append(tuple(c))
        budget = VITERBI_BUDGET  # hard cap on Viterbi calls per attempt -> never stalls the loop
        for combo in combos:
            if budget <= 0:
                return
            signs = base.copy()
            for j, k in enumerate(free_idx):
                signs[k] = combo[j]
            r = raw * signs[own]
            for ph in phases:
                for g2 in g2s:
                    if budget <= 0:
                        return
                    budget -= 1
                    msgs = _crc_msgs(viterbi_decode(_g2(r[ph:], g2), known_start=False))
                    if not msgs:
                        continue
                    st.phase = ph
                    if self._g2 is None:
                        self._g2 = g2
                        self._log(
                            "cnav PRN %d: locked g2_invert=%d (global)" % (prn, int(g2))
                        )
                    # store messages; record pol only for emits a decoded message actually spans
                    # (an unspanned free emit's sign was untested -> leave it for a later window).
                    spanned = set()
                    for m in msgs:
                        ss = lo + ph + 2 * m["index"]
                        st.decoded[ss] = np.array(m["bits"], dtype=np.int8)
                        if m["type"] == 10:
                            st.eph10 = st.decoded[ss]
                        elif m["type"] == 11:
                            st.eph11 = st.decoded[ss]
                        st.n_decoded += 1
                        for c in chain:
                            if (
                                c + len(st.emits[c]) - 1 >= ss
                                and c <= ss + MSG_SYMS - 1
                            ):
                                spanned.add(c)
                    for k, s in enumerate(chain):
                        if s in spanned:
                            st.pol[s] = int(signs[k])
                    if len(st.decoded) > DECODED_MAX:
                        for s in sorted(st.decoded)[:-DECODED_MAX]:
                            st.decoded.pop(s, None)
                    for s in list(st.pol):  # bound pol to live emits
                        if s not in st.emits:
                            st.pol.pop(s, None)
                    return

    # ---------------- serve ----------------
    def ephemeris(self, prn):
        """CNAV ephemeris (SI) from the latest type-10 + type-11, consistency-gated toe==toe2."""
        st = self._p.get(prn)
        if st is None or st.eph10 is None or st.eph11 is None:
            return None
        eph = C.parse_cnav_ephemeris(list(st.eph10), list(st.eph11))
        if abs(eph["toe"] - eph["toe2"]) > 1e-6:
            return None
        return eph

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "synced": bool(st.decoded),
            "decoded": st.n_decoded,
            "messages": len(st.decoded),
            "eph": st.eph10 is not None and st.eph11 is not None,
            "emits": len(st.emits),
            "g2": self._g2,
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(11)
    prn = 7
    types = [10, 11, 30, 33, 10, 11, 30, 34]
    tow0 = 1000
    p10 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    p11 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    toe = np.array([(100 >> (10 - i)) & 1 for i in range(11)], dtype=np.int8)
    p10[32:43] = toe  # toe (msg10 bits 71..81) == toe2 (msg11 bits 39..49)
    p11[0:11] = toe
    msg_bits = []
    for i, mt in enumerate(types):
        payload = (
            p10
            if mt == 10
            else p11
            if mt == 11
            else rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
        )
        msg_bits += C.build_cnav_message(prn, mt, tow0 + i, data=payload)
    sym = conv_encode(msg_bits, flush=True)
    pm = np.where(sym == 1, -1, 1).astype(
        np.int8
    )  # transmitted +-1 (module convention)

    # Feed as abutting ~201-symbol emits (the geometry the window constants assume) with an
    # independent random global sign each (the squaring estimate's arbitrary per-emit
    # polarity) -- the CRC search must resolve it.
    pred = CnavPredictor(log=print)
    pred._min_gap = (
        0.0  # the self-test feeds all emits in <1 s of wall time; no stagger
    )
    ELEN = 201
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        # abutting emits: share one boundary slot (start at e, cover e..e+ELEN-1)
        obs = {
            "utc_ref": e * SYM_S,
            "rec_dt": SYM_S,
            "phase": 0,
            "br": 1,
            "bits": [[i, int(pm[e + i]) * sgn] for i in range(ELEN)],
        }
        pred.ingest(prn, obs)
        e += ELEN - 1  # the next emit starts on this one's last slot
    h = pred.health(prn)
    print("health:", h)
    assert h["synced"], "no CNAV decode"
    assert h["messages"] >= len(types) - 3, "decoded too few messages"
    assert h["eph"], "ephemeris (10+11) not assembled"
    assert (
        h["g2"] is False
    ), "synthetic stream is module convention -> g2 must lock False"

    eph = pred.ephemeris(prn)
    assert eph is not None
    truth = C.parse_cnav_ephemeris(
        list(C.build_cnav_message(prn, 10, tow0, data=p10)),
        list(C.build_cnav_message(prn, 11, tow0 + 1, data=p11)),
    )
    for k in ("e", "M0", "toe", "OMEGA0"):
        assert abs(eph[k] - truth[k]) < 1e-9, "ephemeris field %s mismatch" % k
    print("PASS")


if __name__ == "__main__":
    _selftest()
