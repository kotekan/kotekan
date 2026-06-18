#!/usr/bin/env python3
"""GPS CNAV (L2C / L5) navigation-message decoder.

CNAV rides the L2C CM (or L5 I5) channel at 25 bps, rate-1/2 convolutionally
encoded to 50 sps. This decodes the soft symbols (the correlator's per-CM-period
prompt = peak_re of the GPS_L2C_CM records) through:

    soft symbols -> Viterbi (rate 1/2, K=7, G=171/133 oct, G2 inverted)
                 -> 25 bps bits -> frame sync (8-bit preamble) -> CRC-24Q
                 -> field parse (message type, PRN, TOW).

Distinct from LNAV (gps_navdecode.py): FEC + CRC instead of Hamming parity, and
message-type framing instead of fixed subframes. The encoder (conv_encode,
crc24q, build_cnav_message) is exposed so a synthetic message round-trips in
tests without live sky data. Ephemeris/clock field tables (message types
10/11/30..37) are a mechanical follow-on.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_beamtrack import read_records  # noqa: E402  (shared record reader)

PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]
MSG_BITS = 300
DATA_BITS = MSG_BITS - 24            # 276; CRC-24Q is computed over these
K = 7
POLY = (0o171, 0o133)                # G1, G2 (IS-GPS-200 FEC)
G2_INVERT = True
NSTATES = 1 << (K - 1)               # 64
CRC24Q_POLY = 0x1864CFB


# --------------------------------------------------------------------------
# rate-1/2 K=7 convolutional code (encoder + soft Viterbi)
# --------------------------------------------------------------------------
def _tables():
    out = np.zeros((NSTATES, 2, 2), dtype=np.int8)   # output symbols per (state, input)
    nxt = np.zeros((NSTATES, 2), dtype=np.int64)
    for s in range(NSTATES):
        for b in (0, 1):
            reg = (b << (K - 1)) | s                  # K-bit register: input at MSB, state below
            out[s, b, 0] = bin(reg & POLY[0]).count("1") & 1
            out[s, b, 1] = (bin(reg & POLY[1]).count("1") & 1) ^ (1 if G2_INVERT else 0)
            nxt[s, b] = reg >> 1                       # drop oldest -> next state
    return out, nxt


_OUT, _NXT = _tables()
_EXP = np.where(_OUT == 0, 1.0, -1.0)                  # bipolar expected symbols (0->+1, 1->-1)


def conv_encode(bits, flush=True):
    """Encode 25 bps bits to 50 sps symbols (0/1). flush returns the register to
    state 0 with K-1 trailing zeros (so a finite message decodes cleanly)."""
    seq = list(int(b) for b in bits) + ([0] * (K - 1) if flush else [])
    s, out = 0, []
    for b in seq:
        out += [int(_OUT[s, b, 0]), int(_OUT[s, b, 1])]
        s = int(_NXT[s, b])
    return np.array(out, dtype=np.int8)


def viterbi_decode(symbols, known_start=False):
    """Soft-decision Viterbi. `symbols` are bipolar-ish soft values (+ ~ 0, - ~ 1).
    known_start pins the initial state to 0 (use for a message encoded from
    state 0); otherwise all states start equal and the survivor converges."""
    n = len(symbols) // 2
    r = np.asarray(symbols[:2 * n], dtype=float).reshape(n, 2)
    NEG = -1e18
    pm = np.full(NSTATES, NEG if known_start else 0.0)
    if known_start:
        pm[0] = 0.0
    prev = np.zeros((n, NSTATES), dtype=np.int32)
    pbit = np.zeros((n, NSTATES), dtype=np.int8)
    for i in range(n):
        npm = np.full(NSTATES, NEG)
        r0, r1 = r[i, 0], r[i, 1]
        for s in range(NSTATES):
            if pm[s] <= NEG:
                continue
            base = pm[s]
            for b in (0, 1):
                m = base + _EXP[s, b, 0] * r0 + _EXP[s, b, 1] * r1
                ns = int(_NXT[s, b])
                if m > npm[ns]:
                    npm[ns] = m
                    prev[i, ns] = s
                    pbit[i, ns] = b
        pm = npm
    s = int(np.argmax(pm))
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n - 1, -1, -1):
        bits[i] = pbit[i, s]
        s = int(prev[i, s])
    return bits


# --------------------------------------------------------------------------
# CRC-24Q + message build / parse / frame sync
# --------------------------------------------------------------------------
def crc24q(bits):
    crc = 0
    for b in bits:
        crc ^= (int(b) & 1) << 23
        crc <<= 1
        if crc & 0x1000000:
            crc ^= CRC24Q_POLY
    return crc & 0xFFFFFF


def _uint(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _set(bits, start, n, value):
    for i in range(n):
        bits[start + i] = (value >> (n - 1 - i)) & 1


def build_cnav_message(prn, msg_type, tow, alert=0, data=None):
    """Assemble a 300-bit CNAV message: preamble, PRN, type, TOW, alert, payload,
    CRC-24Q (over the leading 276 bits)."""
    bits = [0] * MSG_BITS
    bits[0:8] = list(PREAMBLE)
    _set(bits, 8, 6, prn)
    _set(bits, 14, 6, msg_type)
    _set(bits, 20, 17, tow)
    bits[37] = alert & 1
    if data is not None:
        d = list(int(x) for x in data)[:DATA_BITS - 38]
        bits[38:38 + len(d)] = d
    _set(bits, DATA_BITS, 24, crc24q(bits[0:DATA_BITS]))
    return bits


def parse_cnav_message(bits):
    """Validate a 300-bit message (preamble + CRC-24Q) and parse the header."""
    if list(int(b) for b in bits[0:8]) != PREAMBLE:
        return None
    if crc24q(bits[0:DATA_BITS]) != _uint(bits[DATA_BITS:MSG_BITS]):
        return None
    return {"prn": _uint(bits[8:14]), "type": _uint(bits[14:20]),
            "tow": _uint(bits[20:37]), "alert": int(bits[37])}


def find_cnav_messages(bits):
    msgs, i, n = [], 0, len(bits)
    while i + MSG_BITS <= n:
        if list(int(b) for b in bits[i:i + 8]) == PREAMBLE:
            m = parse_cnav_message(bits[i:i + MSG_BITS])
            if m is not None:
                m["index"] = i
                msgs.append(m)
                i += MSG_BITS
                continue
        i += 1
    return msgs


def decode_cnav(symbols, known_start=False):
    """Full chain: soft symbols -> CNAV messages. Searches the two symbol-pair
    phases and the carrier polarity (the Viterbi + G2 inversion make polarity a
    sign on the input symbols); the CRC confirms the right combination."""
    sym = np.asarray(symbols, dtype=float)
    for phase in (0, 1):
        for sign in (1.0, -1.0):
            bits = viterbi_decode(sign * sym[phase:], known_start=known_start)
            msgs = find_cnav_messages(bits)
            if msgs:
                return msgs
    return []


def read_symbols(paths, prn, n_prn):
    recs = read_records(paths, n_prn=n_prn)
    sel = recs[recs["prn"].astype(int) == int(prn)]
    return sel["peak_re"].astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="GPS_L2C_CM record file, directory, or glob")
    ap.add_argument("--n-prn", type=int, required=True, help="PRNs per record frame")
    ap.add_argument("--prn", type=int, action="append", required=True, help="PRN(s) to decode")
    args = ap.parse_args(argv)

    paths = (sorted(glob.glob(os.path.join(args.path, "*.raw")))
             if os.path.isdir(args.path) else sorted(glob.glob(args.path)))
    if not paths:
        ap.error("no record files matched: %s" % args.path)

    for prn in args.prn:
        syms = read_symbols(paths, prn, args.n_prn)
        msgs = decode_cnav(syms)
        print("PRN %2d: %d symbols, %d CNAV messages" % (prn, len(syms), len(msgs)))
        for m in msgs:
            print("   type=%-2d  PRN=%-2d  TOW=%d" % (m["type"], m["prn"], m["tow"]))


if __name__ == "__main__":
    main()
