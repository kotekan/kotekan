#!/usr/bin/env python3
"""CNAV (L2C-CM / L5-I) decode-and-serve for the shared-knowledge broker (S3).

The CNAV analog of navbit_predictor.LnavPredictor, but a DIFFERENT shape because CNAV is a
different problem -- and because the LIVE symbol stream taught us two things the offline decode
gate could not (both diagnosed on-air 2026-07-28, see the header lessons below):

    combiner get_status "nav_obs"  ->  per-emit polarity islands  ->  CRC-guided decode  ->  msgs
                                                            (gps_cnav.py does the FEC/CRC work)

WHY IT IS NOT A COPY OF THE LNAV PREDICTOR:

  * NOT CHEAPLY PREDICTABLE. LNAV subframes 1-3 repeat every 30 s with only a TOW increment.
    CNAV interleaves message TYPES on a control-segment schedule that is NOT contractually
    fixed, and the code is continuous rate-1/2 K=7. So there is no reliable future-symbol
    table: predict() SERVES already-decoded spans and returns None at a forward horizon. That
    is correct -- CL (Mechanism A) already gives L2C a sub-floor DATALESS pilot, so CNAV's
    prizes are the live EPHEMERIS (types 10+11 -> BRDC cross-check) and the S4 L5-I cross-band
    feed, NOT the CM self-wipe.

  * ★ TWO LIVE-ONLY FAULTS the offline gate (synthetic, module-consistent symbols) could never
    show, both fixed here:
     1. G2-INVERSION CONVENTION. The on-air symbols are G2-*not*-inverted relative to
        gps_cnav's encoder (measured: a single strong emit round-trips through Viterbi at 0.86
        agreement raw but 0.99 with every-other-symbol flipped). The fix is to flip the 2nd
        symbol of each phase-aligned pair (`x[1::2]`) before Viterbi -- searched as g2 in {0,1}
        and locked once a CRC passes (it is a constant of the link).
     2. PER-EMIT POLARITY IS NOT CHAINABLE. Each nav_obs emit is a ~4 s squaring-estimated
        window whose global sign is independent; consecutive emits are 4 s apart in carrier
        phase, so theta-chaining (what LNAV uses) MISSES flips and a single missed flip mid-run
        makes the convolutional decode garbage. The emits only share ONE boundary slot, too thin
        to vote. So polarity is resolved the only way that works: CRC-GUIDED brute force over the
        per-emit sign pattern in a short window (a real message spans ~3 emits).

  * CONTINUOUS ENCODER. Re-encoding a decoded message to recover its symbols (serve()) carries
    the encoder state across message boundaries; the leading 6 pairs of each message depend on
    the previous message and are left unknown.

Consumed by gps_distributed_broker.py under --nav-decoder cnav (the L2C broker): it logs decode
health + the live ephemeris, and shadow-serves decoded-span signs; navbit_health scores before
any wipe.

Self-test: python3 cnav_predictor.py  (synthetic continuous CNAV stream chopped into
polarity-scrambled emits -> the CRC-guided decoder must recover the messages, ephemeris, and
re-encode the span; g2=0 wins there because the synthetic symbols ARE module convention).
"""
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps_cnav as C  # noqa: E402

SYM_S = 0.020            # L2C-CM record = 20 ms = one CNAV symbol (50 sps)
MSG_SYMS = C.MSG_BITS * 2   # 600 symbols per 300-bit message (12 s)
WIN_EMITS = 7            # emits per decode window. A 300-bit message is 600 symbols, and the K=7
                         # Viterbi needs ~35 bits to converge at each end, so a usable decode
                         # wants ~2 messages of contiguous symbols -- ~6-7 emits (~201 syms /
                         # ~4 s each). Fewer silently fails to place the preamble (found live: an
                         # 801-symbol / 4-emit window decodes NOTHING even under full brute force;
                         # ~1200 does). The combiner emits nav_obs CONTIGUOUSLY (block mode, every
                         # ~200 records / 4 s) -- but only while the box keeps up: a decode that
                         # blocks the poll loop past the 4 s emit period MISSES an emit and self-
                         # fragments the run (measured -- the whole reason the decode is staggered +
                         # budget-capped below). The window takes the LONGEST contiguous run.
                         # Polarity within a run is resolved by _try_decode (cold: rotated 2^k
                         # brute; warm: chained candidate + 1/2-emit-flip repair).
COLD_EMITS = 5          # window (emits) for a COLD PRN's full 2^k brute (2^5=32 sign patterns).
VITERBI_BUDGET = 40     # hard cap on Viterbi calls per decode attempt (~1.7 s at ~40 ms each) --
                        # covers a cold attempt (2^COLD_EMITS at ONE rotated phase/g2 = 32) and any
                        # warm attempt, while staying under the 4 s emit period so a decode never
                        # stalls the poll loop long enough to miss an emit and fragment nav_obs.
COLD_TRIES = 6          # free quick cold-bootstrap attempts before exponential backoff kicks in.
                        # A DECODABLE PRN locks within ~4 (the phase/g2 rotation cycles in 4); a
                        # PRN that never locks (too weak / no data) is the one to back off, because
                        # its full 2^COLD_EMITS brute every round-robin turn is a CONTINUOUS load
                        # that dropped L5 frames in the deployed broker (2026-07-28).
COLD_BACKOFF_BASE = 8.0 # s; retry a stuck cold PRN after BASE, then 2x each further failure...
COLD_BACKOFF_MAX = 300.0  # ...capped at 5 min. Locked PRNs are exempt (their incremental decode is
                        # a handful of Viterbis -- cheap -- so they stay responsive).
EMIT_MAX = 48           # keep ~48 emits (~3 min) per PRN
DECODED_MAX = 80        # keep the last ~80 decoded messages (~16 min) for serve/ephemeris
TRY_EVERY_SYMS = 200    # attempt a decode once per ~4 s of NEW symbols per PRN (one new emit)


def _crc_msgs(bits):
    """CRC-24Q-valid CNAV messages in a decoded bit stream: [(bit_index, parsed)]."""
    out = []
    n = len(bits)
    i = 0
    while i + C.MSG_BITS <= n:
        if list(int(b) for b in bits[i:i + 8]) == C.PREAMBLE:
            w = [int(b) for b in bits[i:i + C.MSG_BITS]]
            m = C.parse_cnav_message(w)
            if m is not None:
                m["index"] = i
                m["bits"] = w
                out.append(m)
        i += 1
    return out


def _g2(x, on):
    """Apply the G2-inversion convention toggle to a phase-aligned symbol array: flip the 2nd
    symbol of every pair. `on` False decodes gps_cnav's native (G2-inverted) convention."""
    if not on:
        return x
    y = x.copy()
    y[1::2] *= -1
    return y


class _PrnState:
    def __init__(self):
        self.emits = {}          # slot0 -> np.array(+-1 signs), one per distinct nav_obs emit
        self.last_obs = None     # (utc_ref, phase, br) dedup
        self.try_slot = None
        self.phase = None        # per-PRN locked pair_phase (pseudorange shifts the symbol grid)
        self.pol = {}            # slot0 -> resolved emit sign (+-1) in one consistent frame
        self.decoded = {}        # start_slot -> data bits[300] (0/1), CRC-verified
        self.msg_type = {}
        self.tow_at = {}
        self.eph10 = None
        self.eph11 = None
        self.n_decoded = 0
        self.boot = 0            # cold-bootstrap (phase, g2) rotation counter
        self.cold_fail = 0       # consecutive cold decode attempts that did NOT lock (backoff)
        self.cold_next_t = 0.0   # earliest wall time to retry a cold decode (exp backoff)


class CnavPredictor:
    """Per-PRN CNAV stitch/CRC-decode/serve. Single-threaded: call from one poll loop."""

    def __init__(self, log=None):
        self._p = {}
        self._log = log or (lambda m: None)
        # The G2-inversion convention is a LINK constant (same for every L2C PRN); the first PRN
        # to CRC locks it here and the rest skip that half of the search. pair_phase is NOT shared
        # -- pseudorange shifts each PRN's symbol grid by up to a symbol -- so it stays per-PRN.
        self._g2 = None
        self._min_gap = 1.0      # s between decode attempts across all PRNs (see ingest); 0 = off
        self._last_decode = 0.0
        self._round = set()      # PRNs decoded this round -- round-robin fairness (see ingest)

    # ---------------- ingest ----------------
    def ingest(self, prn, obs):
        """Store one combiner nav_obs (hard +-1 SYMBOLS) as an absolute-slot polarity island."""
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
        ok = True
        for s, v in m.items():
            arr[s - slot0] = v
        if (arr == 0).any():   # a gap inside the emit breaks the convolutional stream: drop it
            return
        st.emits[slot0] = arr
        if len(st.emits) > EMIT_MAX:
            for s in sorted(st.emits)[:-EMIT_MAX]:
                st.emits.pop(s, None)
        newest = hi
        # STAGGER + ROUND-ROBIN across PRNs. The broker ingests all ~12 PRNs back-to-back in one
        # poll loop; if every cold PRN ran a full decode that cycle, the loop would spend ~13 s and
        # MISS the emits arriving meanwhile -> fragmented nav_obs -> nobody decodes. So gate on wall
        # time (~one decode per poll cycle, cycle << the 4 s emit period so contiguity holds) AND
        # rotate WHICH PRN decodes: without the round set the first PRN in iteration order wins the
        # gate every cycle and the rest never decode (measured -- only PRN 1 ever attempted).
        # self._min_gap = 0 disables the gate (the single-PRN self-test). COLD BACKOFF: a cold PRN
        # (phase is None) that keeps failing its full 2^k brute is a continuous drain -- exempt it
        # after COLD_TRIES via exponential backoff (locked PRNs are cheap incremental, never backed
        # off). Without this the ~half of PRNs that never lock dropped L5 frames in the live broker.
        now = time.time()
        due = st.try_slot is None or newest - st.try_slot >= TRY_EVERY_SYMS
        cold = st.phase is None
        if (due and prn not in self._round and now - self._last_decode >= self._min_gap
                and not (cold and now < st.cold_next_t)):
            st.try_slot = newest
            self._round.add(prn)
            # a round is complete once every currently-ELIGIBLE PRN has had a turn (backed-off cold
            # PRNs are not eligible, so they don't stall the rotation among the active ones).
            eligible = sum(1 for s2 in self._p.values()
                           if s2.phase is not None or now >= s2.cold_next_t)
            if len(self._round) >= max(1, eligible):
                self._round.clear()
            self._try_decode(prn, st)
            self._last_decode = time.time()
            if cold:
                if st.phase is not None:             # locked this attempt -> full speed again
                    st.cold_fail = 0
                    st.cold_next_t = 0.0
                else:                                 # still cold -> exponential backoff
                    st.cold_fail += 1
                    over = st.cold_fail - COLD_TRIES
                    st.cold_next_t = now + (0.0 if over <= 0
                                            else min(COLD_BACKOFF_BASE * 2 ** (over - 1),
                                                     COLD_BACKOFF_MAX))

    # ---------------- decode ----------------
    # Polarity is resolved by CRC, not chaining. Each nav_obs emit's global sign is independent and
    # the 1-slot overlap between emits is too thin to chain reliably (~90%/boundary -> a 7-emit
    # window fails most of the time). Instead, every attempt decodes the freshest contiguous window
    # while brute-forcing ONLY the per-emit signs still unknown (st.pol holds the rest); pair_phase
    # (per-PRN, pseudorange-shifted) and g2 (a global link constant) lock on the first CRC. In
    # steady state only the newest 1-2 emits are free -> 2-4 Viterbi passes; a cold PRN pays a
    # one-time 2^k over the window's k emits (with the global-sign branch, since nothing is fixed).
    def _try_decode(self, prn, st):
        starts = sorted(st.emits)
        if len(starts) < 3:
            return
        # Group emits into maximal contiguous runs (each next emit abuts/overlaps the prior, i.e.
        # starts no later than one slot past its end), then take the LONGEST run -- NOT simply the
        # freshest: the newest emit is often isolated by a certification gap, and insisting on it
        # collapses the window to one emit. Ties go to the fresher run.
        runs = [[starts[0]]]
        for s in starts[1:]:
            prev = runs[-1][-1]
            if prev < s <= prev + len(st.emits[prev]):
                runs[-1].append(s)
            else:
                runs.append([s])
        best = max(runs, key=lambda r: (len(r), r[-1]))
        # COLD PRN (nothing resolved yet): use a SMALL window and a full 2^k brute -- guaranteed to
        # find the polarity (diag-proven) and bounded (2^COLD_EMITS). WARM PRN: the freshest
        # WIN_EMITS, resolved emits fixed from st.pol, only the newest few brute-forced cheaply.
        cold = not any(s in st.pol for s in best)
        chain = best[-(COLD_EMITS if cold else WIN_EMITS):]
        if len(chain) < 3:
            return
        hist, owner = {}, {}
        for s in chain:
            arr = st.emits[s]
            for i in range(len(arr)):
                hist[s + i] = int(arr[i])
                owner[s + i] = s
        lo, hi = min(hist), max(hist)
        if lo % 2:                                   # even-anchor: the pair grid is absolute (utc)
            lo += 1
        if hi - lo + 1 < MSG_SYMS:
            # SAY SO. This window can never hold a message, and returning quietly here is
            # indistinguishable from "decoding, just not yet" -- the chain simply reports
            # NO DECODE forever. It is reachable by configuration, not just by bad luck: the
            # window is COUNTED IN EMITS (WIN_EMITS / COLD_EMITS) while what matters is the
            # SYMBOL SPAN, so a combiner emitting shorter windows than L2C's 4 s silently
            # starves it -- L5-I at a 1 s emit span offers 350 symbols against the 600 needed.
            # The fix is on the PRODUCER (raise that chain's integration_length so one emit
            # spans more symbols); raising the emit COUNT is not available, because a cold
            # bootstrap is 2^COLD_EMITS Viterbis.
            if len(chain) >= (COLD_EMITS if cold else WIN_EMITS):
                self._warn_short = getattr(self, "_warn_short", set())
                if prn not in self._warn_short:
                    self._warn_short.add(prn)
                    self._log("cnav PRN %d: emit span too short to decode -- %d symbols over "
                              "%d emits, need %d. Raise this chain's integration_length "
                              "(emit span = integration_length x record period)."
                              % (prn, hi - lo + 1, len(chain), MSG_SYMS))
            return
        for x in range(lo, hi + 1):
            if x not in hist:                        # gap inside the window -- bail (rare)
                return
        raw = np.array([hist[x] for x in range(lo, hi + 1)], dtype=float)
        own = np.array([chain.index(owner[x]) for x in range(lo, hi + 1)])
        base = np.array([st.pol.get(s, 1) for s in chain])   # known signs; frees overwritten below
        free_idx = [k for k, s in enumerate(chain) if s not in st.pol]
        phases = (st.phase,) if st.phase is not None else (0, 1)
        g2s = (self._g2,) if self._g2 is not None else (False, True)
        # A COLD attempt does a full 2^k brute but over only ONE (phase, g2) per call, rotating
        # across calls -- so each attempt is 2^COLD_EMITS Viterbis (~1.5 s), staying under the 4 s
        # emit period (a longer stall misses emits and fragments the contiguity the brute needs).
        if cold:
            opts = [(ph, g2) for ph in phases for g2 in g2s]
            phases, g2s = ((opts[st.boot % len(opts)][0],), (opts[st.boot % len(opts)][1],))
            st.boot += 1
        # BOUNDED, NEVER-BLOCKING candidate set. The decode runs in the broker's single poll loop;
        # a long brute force (2^free is 5 s) would make it MISS the emits that arrive meanwhile,
        # fragmenting the very contiguity it needs (measured: heavy decode -> gappy nav_obs -> no
        # decode -> heavier decode). So no 2^free. Instead: the OVERLAP-CHAINED sign pattern (right
        # ~90%/boundary -> usually decodes as-is), plus a SINGLE-emit-flip repair per free emit
        # (fixes the one bad chain link -> ~92% of contiguous windows), plus the global negation
        # for the cold case (nothing resolved yet -> the global sign is unconstrained). Worst case
        # ~2*(free+1) candidates -> a few hundred ms, never a stall.
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
            # Full brute over the small cold window: EVERY sign pattern (covers the global sign).
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
        budget = VITERBI_BUDGET   # hard cap on Viterbi calls per attempt -> never stalls the loop
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
                    msgs = _crc_msgs(C.viterbi_decode(_g2(r[ph:], g2)))
                    if not msgs:
                        continue
                    st.phase = ph
                    if self._g2 is None:
                        self._g2 = g2
                        self._log("cnav PRN %d: locked g2_invert=%d (global)" % (prn, int(g2)))
                    # store messages; record pol ONLY for emits a decoded message actually spans
                    # (an unspanned free emit's sign was untested -> leave it for a later window).
                    spanned = set()
                    for m in msgs:
                        ss = lo + ph + 2 * m["index"]
                        st.decoded[ss] = np.array(m["bits"], dtype=np.int8)
                        st.msg_type[ss] = m["type"]
                        st.tow_at[ss] = m["tow"]
                        if m["type"] == 10:
                            st.eph10 = st.decoded[ss]
                        elif m["type"] == 11:
                            st.eph11 = st.decoded[ss]
                        st.n_decoded += 1
                        for c in chain:
                            if c + len(st.emits[c]) - 1 >= ss and c <= ss + MSG_SYMS - 1:
                                spanned.add(c)
                    for k, s in enumerate(chain):
                        if s in spanned:
                            st.pol[s] = int(signs[k])
                    if len(st.decoded) > DECODED_MAX:
                        for s in sorted(st.decoded)[:-DECODED_MAX]:
                            st.decoded.pop(s, None)
                            st.msg_type.pop(s, None)
                            st.tow_at.pop(s, None)
                    for s in list(st.pol):           # bound pol to live emits
                        if s not in st.emits:
                            st.pol.pop(s, None)
                    return

    # ---------------- serve ----------------
    def predict(self, prn, utc_now, horizon_s=4.0):
        """+-1/0 signs over [utc_now, +horizon) for slots inside a DECODED message, re-encoded
        into the RECEIVED convention (module symbols with the locked G2 toggle re-applied by
        absolute-slot parity), 0 elsewhere. Returns {"utc0","bit_s","bits"} or None (the normal
        case at a forward horizon -- see the module note: CNAV serves decoded spans, it does not
        predict the future)."""
        st = self._p.get(prn)
        if st is None or not st.decoded or self._g2 is None:
            return None
        g2 = self._g2
        slot_now = int(math.floor(utc_now / SYM_S + 1e-3))
        n = int(math.ceil(horizon_s / SYM_S)) + 2
        out = np.zeros(n, dtype=np.int8)
        for start in sorted(st.decoded):
            if start + MSG_SYMS <= slot_now or start >= slot_now + n:
                continue
            sym = C.conv_encode(st.decoded[start], flush=False)   # module (G2-inverted) frame
            pm = np.where(sym == 1, -1, 1).astype(np.int8)
            for j in range(len(pm)):
                if j < 12:                                        # leading pairs: prior-msg state
                    continue
                slot = start + j
                idx = slot - slot_now
                if 0 <= idx < n:
                    # re-apply the link's G2 convention: flip the 2nd symbol of each pair. The
                    # message starts on a pair boundary, so pair position == j % 2.
                    s = int(pm[j])
                    if g2 and (j % 2 == 1):
                        s = -s
                    out[idx] = s
        if not out.any():
            return None
        utc0 = slot_now * SYM_S
        return {"utc0": utc0, "bit_s": SYM_S, "bits": [int(x) for x in out]}

    def messages(self, prn):
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"start_slot": s, "tow": st.tow_at[s], "type": st.msg_type[s]}
                for s in sorted(st.decoded)]

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
        return {"synced": bool(st.decoded),
                "decoded": st.n_decoded, "messages": len(st.decoded),
                "eph": st.eph10 is not None and st.eph11 is not None,
                "emits": len(st.emits),
                "g2": self._g2}


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(11)
    prn = 7
    types = [10, 11, 30, 33, 10, 11, 30, 34]
    tow0 = 1000
    p10 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    p11 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    toe = np.array([(100 >> (10 - i)) & 1 for i in range(11)], dtype=np.int8)
    p10[32:43] = toe            # toe (msg10 bits 71..81) == toe2 (msg11 bits 39..49)
    p11[0:11] = toe
    msg_bits = []
    for i, mt in enumerate(types):
        payload = p10 if mt == 10 else p11 if mt == 11 else rng.randint(
            0, 2, C.DATA_BITS - 38).astype(np.int8)
        msg_bits += C.build_cnav_message(prn, mt, tow0 + i, data=payload)
    sym = C.conv_encode(msg_bits, flush=True)
    pm = np.where(sym == 1, -1, 1).astype(np.int8)       # transmitted +-1 (module convention)

    # Feed as ~201-symbol emits (the live cadence) with an INDEPENDENT random global sign each
    # (the squaring estimate's arbitrary per-emit polarity) -- the CRC search must resolve it.
    pred = CnavPredictor(log=print)
    pred._min_gap = 0.0          # the self-test feeds all emits in <1 s of wall time; no stagger
    ELEN = 201
    e = 0
    while e + ELEN <= len(pm):
        sgn = 1 - 2 * rng.randint(0, 2)
        # abutting emits: share one boundary slot (start at e, cover e..e+ELEN-1)
        obs = {"utc_ref": e * SYM_S, "rec_dt": SYM_S, "phase": 0, "br": 1,
               "bits": [[i, int(pm[e + i]) * sgn] for i in range(ELEN)]}
        pred.ingest(prn, obs)
        e += ELEN - 1        # overlap by one slot, as the live combiner does
    h = pred.health(prn)
    print("health:", h)
    assert h["synced"], "no CNAV decode"
    assert h["messages"] >= len(types) - 3, "decoded too few messages"
    assert h["eph"], "ephemeris (10+11) not assembled"
    assert h["g2"] is False, "synthetic stream is module convention -> g2 must lock False"

    eph = pred.ephemeris(prn)
    assert eph is not None
    truth = C.parse_cnav_ephemeris(list(C.build_cnav_message(prn, 10, tow0, data=p10)),
                                   list(C.build_cnav_message(prn, 11, tow0 + 1, data=p11)))
    for k in ("e", "M0", "toe", "OMEGA0"):
        assert abs(eph[k] - truth[k]) < 1e-9, "ephemeris field %s mismatch" % k

    # serve a decoded span: predict() must reproduce the true transmitted signs (polarity-free)
    starts = sorted(pred._p[prn].decoded)
    mid = starts[len(starts) // 2]
    out = pred.predict(prn, mid * SYM_S, horizon_s=24.0)
    assert out is not None, "predict returned nothing over a decoded span"
    got = np.array(out["bits"], dtype=int)
    slot0 = int(round(out["utc0"] / SYM_S))
    truth_seg = np.array([int(pm[slot0 + k]) if slot0 + k < len(pm) else 0
                          for k in range(len(got))], dtype=int)
    known = (got != 0) & (np.arange(len(got)) + slot0 < len(pm))
    ap = int((got[known] == truth_seg[known]).sum())
    an = int((got[known] == -truth_seg[known]).sum())
    pol = 1 if ap >= an else -1
    wrong = int((got[known] * pol != truth_seg[known]).sum())
    print("serve: %d known, %d wrong, %d unknown (of %d)"
          % (int(known.sum()), wrong, int((got == 0).sum()), len(got)))
    assert wrong == 0, "served signs disagree with the air"
    assert int(known.sum()) > len(got) // 2, "served span mostly unknown"
    print("PASS")


if __name__ == "__main__":
    _selftest()
