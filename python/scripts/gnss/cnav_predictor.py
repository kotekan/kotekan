#!/usr/bin/env python3
"""CNAV (L2C-CM / L5-I) decode-and-serve for the shared-knowledge broker (S3).

The CNAV analog of navbit_predictor.LnavPredictor, but the shape is deliberately different
because CNAV is a DIFFERENT problem from LNAV, and honesty about that difference is the whole
point of this module:

    combiner get_status "nav_obs"  ->  stitch symbols  ->  Viterbi + CRC-24Q  ->  store messages
                                                            (gps_cnav.py does the FEC/CRC work)

Three facts that make this NOT a copy of the LNAV predictor:

  * NOT CHEAPLY PREDICTABLE. LNAV subframes 1-3 repeat every 30 s with only a TOW increment, so
    the L1CA peel can be handed FUTURE bit signs. CNAV interleaves message TYPES (10, 11, 30-37,
    ...) on a control-segment schedule that is quasi-periodic but NOT contractually fixed, and
    each message is rate-1/2 K=7 convolutionally encoded so a symbol depends on the 6 preceding
    data bits. You cannot reliably name the symbol 4 s in the future. This module therefore does
    NOT promise a future-horizon symbol table for the CM self-wipe -- and it does not need to:
    the CL sibling chain (Mechanism A) already gives L2C a sub-floor DATALESS pilot, so the CM
    data-wipe was never CNAV's prize.

  * CNAV'S PRIZES ARE ELSEWHERE. What decoded CNAV buys is (a) a live EPHEMERIS (types 10+11)
    to cross-check -- and eventually replace -- the 2 h-latency BRDC download, and (b) an
    L5-I / L2C-CM CROSS-BAND feed (S4): decode on whichever band is strong, serve the symbols
    to the weaker one. Both consume `messages()` / `ephemeris()`, not a future table.

  * CONTINUOUS ENCODER. The transmitted symbol stream is one unbroken convolutional code -- it
    never flushes between messages. So re-encoding a decoded message to recover its per-symbol
    signs (predict(), for a consumer that wants the signs of an ALREADY-DECODED span, e.g. a
    deep window that lags real time) has to carry the encoder state across the boundary: symbol
    pair j depends on data bits [j-6 .. j], so the first 6 pairs of any maximal run of
    consecutive decoded messages are left unknown (their state came from the undecoded past).

Consumed by gps_distributed_broker.py under --nav-decoder cnav (the L2C broker). It logs decode
health + the ephemeris-vs-BRDC residual, and shadow-serves the re-encoded signs of decoded
messages as a component table; navbit_health scores anything before it ever feeds a wipe.

Self-test: python3 cnav_predictor.py  (synthetic continuous CNAV stream, chopped into
polarity-scrambled 50-symbol emits, must sync/decode, assemble the ephemeris, and re-encode the
decoded span back to the true signs).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps_cnav as C  # noqa: E402  (conv_encode / viterbi_decode / find_cnav_messages / CRC)

SYM_S = 0.020            # L2C-CM record = 20 ms = exactly ONE CNAV symbol (50 sps)
MSG_SYMS = C.MSG_BITS * 2   # 600 symbols per 300-bit message (12 s)
# Decode needs a full message at an UNKNOWN symbol-pair phase, so > 1 message + a phase of slack;
# 2 messages guarantees at least one whole message boundary lands inside the window.
SYNC_MIN_SYMS = 2 * MSG_SYMS + 40      # ~24.8 s: one whole message boundary at ANY pair phase
SCAN_MAX_SYMS = 2 * MSG_SYMS + 240     # ~26 s (~2 messages). Bounds Viterbi cost: an UNSYNCED
                                       # PRN pays 4 Viterbi passes (phase x sign) over this window
                                       # each attempt, so keep it near the sync minimum -- older
                                       # messages are already stored, nothing is lost by a shorter
                                       # decode window
TRY_EVERY_SYMS = 250                   # attempt a decode once per ~5 s of NEW symbols per PRN.
                                       # Messages arrive every 12 s and the window holds ~2, so
                                       # every message is decoded with slack; a shorter cadence
                                       # only re-Viterbis the same window (the broker's hot path)
HIST_MAX_SLOTS = 40000                 # ~13 min of symbol history
DECODED_MAX = 80                       # keep the last ~80 messages (~16 min) for serve/ephemeris
# (No polarity-chain repair search, unlike LnavPredictor: decode_cnav tries both global
# polarities and the CRC-24Q picks, so a run at one consistent polarity decodes outright.)


class _PrnState:
    def __init__(self):
        self.hist = {}           # slot -> +-1 symbol (chained polarity applied)
        self.theta = None        # last emit's theta0 (rad), for polarity chaining
        self.pol = 1             # chained polarity applied to incoming emits
        self.utc_anchor = None   # (slot, utc) freshest slot->utc mapping
        self.sym_s = SYM_S
        self.last_obs = None     # (utc_ref, phase, br) dedup (broker re-polls the same emit)
        self.try_slot = None     # newest slot at the last decode attempt (TRY_EVERY_SYMS)
        # decode state
        self.combo = None        # (pair_phase, sign) that last decoded -- tried first next time
        self.decoded = {}        # start_slot -> data bits[300] (int8 0/1), CRC-verified
        self.msg_type = {}       # start_slot -> message type
        self.tow_at = {}         # start_slot -> header TOW
        self.eph10 = None        # latest CRC-valid type-10 message bits[300]
        self.eph11 = None        # latest CRC-valid type-11 message bits[300]
        self.n_decoded = 0
        self.n_reenc_checked = 0 # re-encode agreement vs air symbols (bits compared / mismatched)
        self.n_reenc_wrong = 0


class CnavPredictor:
    """Per-PRN CNAV stitch/decode/serve. Single-threaded: call from one poll loop."""

    def __init__(self, log=None):
        self._p = {}
        self._log = log or (lambda m: None)

    # ---------------- ingest ----------------
    def ingest(self, prn, obs):
        """Fold one combiner nav_obs (hard +-1 SYMBOLS, not bits) into the PRN's history."""
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
        key = (utc_ref, phase, br)
        if key == st.last_obs:      # same emit the broker already handed us (it polls faster)
            return
        st.last_obs = key
        st.sym_s = br * rec_dt      # 0.020 at L2C (br=1); kept general
        # Polarity chaining (identical to LNAV): a ~pi jump in the reduced squaring angle between
        # emits is a global sign flip -- track it so a decode window is internally consistent.
        theta = -math.atan2(rot.imag, rot.real)
        if st.theta is not None and round((theta - st.theta) / math.pi) % 2:
            st.pol = -st.pol
        st.theta = theta
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1):
                continue
            utc_sym = utc_ref + (bi * br - phase) * rec_dt
            slot = int(round(utc_sym / st.sym_s))
            st.hist[slot] = sg * st.pol
            st.utc_anchor = (slot, utc_sym)
        if len(st.hist) > HIST_MAX_SLOTS:
            cut = max(st.hist) - HIST_MAX_SLOTS
            st.hist = {k: v for k, v in st.hist.items() if k >= cut}
        newest = max(st.hist) if st.hist else None
        if newest is not None and (st.try_slot is None
                                   or newest - st.try_slot >= TRY_EVERY_SYMS):
            st.try_slot = newest
            self._try_decode(prn, st)

    # ---------------- decode ----------------
    def _contig_run(self, st):
        """Trailing contiguous slot run (start_slot, +-1 symbols), capped at SCAN_MAX_SYMS.
        Same O(run) walk-back as LnavPredictor -- no sort over the whole history per ingest."""
        if not st.hist:
            return None
        end = max(st.hist)
        start, n = end, 1
        while n < SCAN_MAX_SYMS and (start - 1) in st.hist:
            start -= 1
            n += 1
        return start, np.array([st.hist[s] for s in range(start, end + 1)], dtype=np.int8)

    def _decode_run(self, syms_pm):
        """Viterbi+CRC decode of a +-1 symbol run. Returns (pair_phase, sign, msgs) for the
        combination that produced the most CRC-valid messages, or None. `msgs` carry a symbol
        index within the run: msg at bit `idx` starts at symbol `pair_phase + 2*idx`."""
        soft = syms_pm.astype(float)
        combos = [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)]
        # A locked satellite keeps its phase/polarity across windows: try what worked last first
        # and stop at the first combo that yields any CRC pass -- the CRC false-accept rate is
        # 2^-24, so one valid message is a real lock, not a coincidence.
        best = None
        for ph, sg in combos:
            bits = C.viterbi_decode(sg * soft[ph:])
            msgs = C.find_cnav_messages(bits)
            if msgs:
                cand = (ph, sg, msgs)
                if best is None or len(msgs) > len(best[2]):
                    best = cand
                break  # first hit wins (see above); the loop is ordered by the cached combo
        return best

    def _try_decode(self, prn, st):
        run = self._contig_run(st)
        if run is None or len(run[1]) < SYNC_MIN_SYMS:
            return
        slot0, syms_pm = run
        # Order the search so the previously-winning combo is tried first (cheap re-lock).
        if st.combo is not None:
            ph0, sg0 = st.combo
            bits = C.viterbi_decode(sg0 * syms_pm.astype(float)[ph0:])
            msgs = C.find_cnav_messages(bits)
            res = (ph0, sg0, msgs) if msgs else None
        else:
            res = None
        if res is None:
            res = self._decode_run(syms_pm)
        if res is None:
            return
        pair_phase, sign, msgs = res
        st.combo = (pair_phase, sign)
        for m in msgs:
            # symbol index (within the run) of this message's leading preamble symbol
            sym_i = pair_phase + 2 * m["index"]
            start_slot = slot0 + sym_i
            data = np.array(m["bits"], dtype=np.int8)   # 300 CRC-valid bits (0/1)
            self._reenc_score(st, start_slot, data)     # score re-encode vs air BEFORE storing
            st.decoded[start_slot] = data
            st.msg_type[start_slot] = m["type"]
            st.tow_at[start_slot] = m["tow"]
            if m["type"] == 10:
                st.eph10 = data
            elif m["type"] == 11:
                st.eph11 = data
            st.n_decoded += 1
        if len(st.decoded) > DECODED_MAX:
            for s in sorted(st.decoded)[:-DECODED_MAX]:
                st.decoded.pop(s, None)
                st.msg_type.pop(s, None)
                st.tow_at.pop(s, None)

    def _reenc_score(self, st, start_slot, data):
        """Re-encode the decoded message and compare to the air symbols at those slots
        (polarity-free). This measures how many raw symbols the Viterbi had to CORRECT -- the
        honest quality number for what a wipe would see -- and validates the encode path that
        serve()/predict() reuses. Only the interior (pairs >= 6) is scored: the leading 6 pairs
        depend on the previous message's tail, which we do not carry here."""
        sym = C.conv_encode(data, flush=False)           # 600 symbols (0/1)
        sym_pm = np.where(sym == 1, -1, 1).astype(np.int8)
        got = np.array([st.hist.get(start_slot + j, 0) for j in range(len(sym_pm))], dtype=np.int8)
        seen = (got != 0)
        seen[:12] = False                                # skip the state-dependent leading pairs
        if not seen.any():
            return
        w = int((sym_pm[seen] != got[seen]).sum())
        n = int(seen.sum())
        st.n_reenc_checked += n
        st.n_reenc_wrong += min(w, n - w)                # polarity-free: a global flip is not error

    # ---------------- serve ----------------
    def predict(self, prn, utc_now, horizon_s=4.0):
        """+-1/0 signs over [utc_now, +horizon) for slots that fall inside a DECODED message,
        re-encoded from the CRC-verified data (continuous across consecutive messages); 0 where
        no decoded message covers the slot. Returns {"utc0","bit_s","bits"} or None if the whole
        window is unknown -- which is the NORMAL case at a future horizon (see the module note:
        CNAV is not a future-prediction source, this serves already-decoded spans only)."""
        st = self._p.get(prn)
        if st is None or not st.decoded or st.utc_anchor is None:
            return None
        aslot, autc = st.utc_anchor
        slot_now = int(math.floor((utc_now - autc) / st.sym_s + 1e-3)) + aslot
        n = int(math.ceil(horizon_s / st.sym_s)) + 2
        out = np.zeros(n, dtype=np.int8)
        starts = sorted(st.decoded)
        i = 0
        while i < len(starts):
            # grow a maximal run of consecutive decoded messages (start slots MSG_SYMS apart)
            j = i
            while j + 1 < len(starts) and starts[j + 1] - starts[j] == MSG_SYMS:
                j += 1
            if starts[j] + MSG_SYMS <= slot_now or starts[i] >= slot_now + n:
                i = j + 1
                continue                                  # run does not touch the window
            concat = np.concatenate([st.decoded[s] for s in starts[i:j + 1]])
            sym = C.conv_encode(concat, flush=False)
            sym_pm = np.where(sym == 1, -1, 1).astype(np.int8)
            valid = np.ones(len(sym_pm), dtype=bool)
            valid[:12] = False                            # leading pairs: state from before the run
            run_start = starts[i]
            for k in range(len(sym_pm)):
                idx = run_start + k - slot_now
                if 0 <= idx < n and valid[k]:
                    out[idx] = sym_pm[k]
            i = j + 1
        if not out.any():
            return None
        utc0 = autc + (slot_now - aslot) * st.sym_s
        return {"utc0": utc0, "bit_s": st.sym_s, "bits": [int(x) for x in out]}

    def messages(self, prn):
        """Recently decoded messages, oldest first: [{start_slot, tow, type}]. For the type
        schedule / L5 cross-band feed (S4)."""
        st = self._p.get(prn)
        if st is None:
            return []
        return [{"start_slot": s, "tow": st.tow_at[s], "type": st.msg_type[s]}
                for s in sorted(st.decoded)]

    def ephemeris(self, prn):
        """Assembled CNAV ephemeris (SI) from the latest type-10 + type-11 pair, or None.
        Consistency-gated on toe == toe2 (same set), exactly as gps_cnav.assemble_cnav_ephemeris.
        For the BRDC cross-check: sv_position_cnav(eph, t) vs the BRDC position."""
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
        run = self._contig_run(st)
        return {"synced": st.combo is not None and bool(st.decoded),
                "decoded": st.n_decoded, "messages": len(st.decoded),
                "eph": st.eph10 is not None and st.eph11 is not None,
                "run": len(run[1]) if run else 0, "hist": len(st.hist),
                "need": SYNC_MIN_SYMS,
                "reenc_mismatch": (st.n_reenc_wrong / st.n_reenc_checked)
                if st.n_reenc_checked else None}


# ------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.RandomState(11)
    # A continuous CNAV stream: alternating types with a real 10/11 ephemeris pair, so the
    # decoder must frame-sync, CRC-check, assemble the ephemeris, AND re-encode the span back.
    prn = 7
    types = [10, 11, 30, 33, 10, 11, 30, 34]   # 8 messages, 96 s
    tow0 = 1000
    eph_payload10 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    eph_payload11 = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
    # toe (msg10 bits 71..81) and toe2 (msg11 bits 39..49) MUST match for a consistent set --
    # the real control segment guarantees it; force it here. Payload index p == message bit 38+p,
    # so toe -> payload[32:43], toe2 -> payload[0:11]; give both the same 11-bit value.
    toe_bits = np.array([(100 >> (10 - i)) & 1 for i in range(11)], dtype=np.int8)
    eph_payload10[32:43] = toe_bits
    eph_payload11[0:11] = toe_bits
    msg_bits = []
    truth = []   # (start_bit, type, 300-bit message)
    for i, mt in enumerate(types):
        if mt == 10:
            payload = eph_payload10
        elif mt == 11:
            payload = eph_payload11
        else:
            payload = rng.randint(0, 2, C.DATA_BITS - 38).astype(np.int8)
        m = C.build_cnav_message(prn, mt, tow0 + i, data=payload)
        truth.append((i * C.MSG_BITS, mt, np.array(m, dtype=np.int8)))
        msg_bits += m
    sym = C.conv_encode(msg_bits, flush=True)          # ONE continuous convolutional stream
    pm = np.where(sym == 1, -1, 1).astype(np.int8)      # transmitted +-1 signs (0->+1, 1->-1)

    # Feed as 50-symbol emits with a drifting carrier polarity, exactly the LNAV self-test model:
    # the combiner decides signs against the mod-pi reduced axis, so the emit's global sign
    # flips when the true phase crosses +-pi/2 -- which the chain must track to keep a decode
    # window internally consistent.
    pred = CnavPredictor(log=print)
    theta_true = 0.2
    for e in range(0, len(pm) - 50, 50):
        theta_true += rng.uniform(-0.05, 0.25)
        th_red = math.remainder(theta_true, math.pi)
        flip = round((theta_true - th_red) / math.pi) % 2
        rot = complex(math.cos(-th_red), math.sin(-th_red))
        sgn = 1 - 2 * flip
        obs = {"utc_ref": e * SYM_S, "rec_dt": SYM_S, "phase": 0, "br": 1,
               "rot_re": rot.real, "rot_im": rot.imag,
               "bits": [[i, int(pm[e + i]) * sgn] for i in range(50)]}
        pred.ingest(prn, obs)

    h = pred.health(prn)
    print("health:", h)
    assert h["synced"], "no CNAV sync"
    assert h["messages"] >= len(types) - 2, "decoded too few messages"
    assert h["eph"], "ephemeris (10+11) not assembled"
    # re-encode agreement: a clean synthetic stream must round-trip with ZERO mismatch
    assert h["reenc_mismatch"] == 0.0, "re-encode disagrees with the air symbols"

    # ephemeris fields recover exactly (compare a couple of the wider fields to truth)
    eph = pred.ephemeris(prn)
    assert eph is not None
    truth_eph = C.parse_cnav_ephemeris(list(truth[0][2]), list(truth[1][2]))
    for k in ("e", "M0", "toe", "OMEGA0"):
        assert abs(eph[k] - truth_eph[k]) < 1e-9, "ephemeris field %s mismatch" % k

    # serve the DECODED span: predict() at a utc inside the stream must reproduce the true signs
    # (polarity-free) for covered slots.
    t_mid = 30 * SYM_S * 2   # ~inside the 2nd/3rd message
    out = pred.predict(prn, t_mid, horizon_s=24.0)
    assert out is not None, "predict returned nothing over a decoded span"
    got = np.array(out["bits"], dtype=int)
    slot0 = int(round(out["utc0"] / SYM_S))
    truth_seg = np.array([int(pm[slot0 + k]) for k in range(len(got))], dtype=int)
    known = got != 0
    unknown = int((~known).sum())
    # polarity-free: the predictor's absolute sign is arbitrary, so pick the global sign that
    # agrees on more of the known slots, then every known slot must match under it.
    agree_pos = int((got[known] == truth_seg[known]).sum())
    agree_neg = int((got[known] == -truth_seg[known]).sum())
    pol = 1 if agree_pos >= agree_neg else -1
    ok = int((got[known] * pol == truth_seg[known]).sum())
    wrong = int(known.sum()) - ok
    print("serve: %d ok, %d wrong, %d unknown (of %d)" % (ok, wrong, unknown, len(got)))
    assert wrong == 0, "served signs disagree with the air"
    assert ok > len(got) // 2, "served span mostly unknown"
    print("PASS")


if __name__ == "__main__":
    _selftest()
