#!/usr/bin/env python3
"""S3 decode gate: does gps_cnav.decode_cnav recover CNAV from a realistically-stitched
HARD-symbol stream at our SNR? The offline record path is a dead end (records are deep emits),
so the live source is the combiner's nav_obs -- HARD +-1 signs per 20 ms symbol, no soft
Viterbi input. This proves the decoder survives (a) hard-decision symbols, (b) an unknown
start phase (2-way) and carrier polarity (2-way), (c) a leading partial message, and (d) a
realistic symbol error rate. If this passes, the CnavPredictor is plumbing; if not, we learn
the decoder needs soft symbols BEFORE building it.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gps_cnav as C

RNG = np.random.default_rng(20260728)
fails = 0


def stream(prn, tow0, n_msgs, ber, polarity, drop_lead, flip_first=False):
    """A continuous 50 sps symbol stream of n_msgs CNAV messages, hard-limited to +-1, with a
    per-symbol bit-error rate `ber`, an optional carrier polarity flip, and a partial leading
    message (drop_lead symbols) so framesync must find the preamble mid-stream."""
    bits = []
    for i in range(n_msgs):
        # types 10, 11 alternate (the ephemeris pair); TOW advances 6 s per message (12 s/msg
        # is the real cadence, but the header TOW field increments by the message count here --
        # the test only needs decode + a deterministic, recoverable TOW).
        mt = 10 if i % 2 == 0 else 11
        bits += C.build_cnav_message(prn, mt, tow0 + i)
    sym = C.conv_encode(bits, flush=True)            # 0/1 symbols
    bip = np.where(sym == 1, -1.0, 1.0) * polarity   # 0->+1, 1->-1 (decode_cnav's convention)
    if ber > 0:
        flip = RNG.random(len(bip)) < ber
        bip[flip] *= -1
    if drop_lead:
        bip = bip[drop_lead:]
    return bip


def case(name, **kw):
    global fails
    expect = kw.pop("expect", True)
    prn, tow0, n = kw.get("prn", 7), kw.get("tow0", 1000), kw.get("n_msgs", 6)
    sym = stream(prn, tow0, n, kw.get("ber", 0.0), kw.get("polarity", 1),
                 kw.get("drop_lead", 0))
    msgs = C.decode_cnav(sym)
    ok_prn = all(m["prn"] == prn for m in msgs)
    tows = sorted(m["tow"] for m in msgs)
    got = len(msgs)
    passed = (got >= max(1, n - 2)) and ok_prn if expect else (got == 0)
    fails += 0 if passed else 1
    print("%-52s decoded %d/%d  prn_ok=%s  tow=%s  %s"
          % (name, got, n, ok_prn, tows[:4], "OK" if passed else "FAIL"))


# 1. clean hard symbols, known everything
case("clean, hard +-1", ber=0.0)
# 2. carrier polarity flipped (the G2-inverted encoder makes polarity a sign -- decode_cnav
#    tries both; a live sat's polarity is unknown until the CRC picks it)
case("polarity flipped", polarity=-1)
# 3. unknown start phase: drop a partial leading message so the preamble is mid-stream
case("partial lead (framesync mid-stream)", drop_lead=137)
# 4. realistic SER: L2C-CM strong-sat per-symbol error rate. Hard-decision Viterbi K=7 fixes
#    isolated flips; a deep_snr~100 sat is ~0 SER, a marginal one maybe 1-2%.
case("1% symbol errors (marginal sat)", ber=0.01)
case("2% symbol errors", ber=0.02)
# 5. combined worst-case: flipped polarity + partial lead + 1% errors
case("polarity + partial lead + 1% errors", polarity=-1, drop_lead=213, ber=0.01)
# 6. negative control: pure noise must decode NOTHING (no false CRC pass at 2^-24)
noise = RNG.choice([-1.0, 1.0], size=6 * C.MSG_BITS * 2)
nm = C.decode_cnav(noise)
print("%-52s decoded %d (expect 0)  %s" % ("pure noise (false-accept guard)", len(nm),
                                            "OK" if len(nm) == 0 else "FAIL"))
fails += 0 if len(nm) == 0 else 1

print("\n%s (%d failures)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(1 if fails else 0)
