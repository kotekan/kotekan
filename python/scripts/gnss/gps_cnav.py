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
            window = [int(b) for b in bits[i:i + MSG_BITS]]
            m = parse_cnav_message(window)
            if m is not None:
                m["index"] = i
                m["bits"] = window
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


# --------------------------------------------------------------------------
# CNAV ephemeris (message types 10 + 11) -> SV position
# --------------------------------------------------------------------------
import math  # noqa: E402

GPS_PI = 3.1415926535898
MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
AREF = 26559710.0                  # CNAV reference semi-major axis, m (DELTA_A is relative)
OMEGADOT_REF = -2.6e-9 * GPS_PI    # reference rate, rad/s (DELTA_OMEGA_DOT is relative)
_P = lambda n: 2.0 ** (-n)         # 2^-n

# name -> (message type, 1-indexed start bit, length, signed, scale). From
# IS-GPS-200 Table 30-I (transcribed from GNSS-SDR's GPS_CNAV.h field table).
CNAV_EPH_FIELDS = {
    "WN":         (10, 39, 13, False, 1.0),
    "top":        (10, 55, 11, False, 300.0),
    "toe":        (10, 71, 11, False, 300.0),
    "dA":         (10, 82, 26, True, _P(9)),
    "A_dot":      (10, 108, 25, True, _P(21)),
    "dn0":        (10, 133, 17, True, _P(44) * GPS_PI),
    "dn0_dot":    (10, 150, 23, True, _P(57) * GPS_PI),
    "M0":         (10, 173, 33, True, _P(32) * GPS_PI),
    "e":          (10, 206, 33, False, _P(34)),
    "omega":      (10, 239, 33, True, _P(32) * GPS_PI),
    "toe2":       (11, 39, 11, False, 300.0),
    "OMEGA0":     (11, 50, 33, True, _P(32) * GPS_PI),
    "i0":         (11, 83, 33, True, _P(32) * GPS_PI),
    "dOMEGA_dot": (11, 116, 17, True, _P(44) * GPS_PI),
    "i0_dot":     (11, 133, 15, True, _P(44) * GPS_PI),
    "Cis":        (11, 148, 16, True, _P(30)),
    "Cic":        (11, 164, 16, True, _P(30)),
    "Crs":        (11, 180, 24, True, _P(8)),
    "Crc":        (11, 204, 24, True, _P(8)),
    "Cus":        (11, 228, 21, True, _P(30)),
    "Cuc":        (11, 249, 21, True, _P(30)),
}


def parse_cnav_ephemeris(msg10_bits, msg11_bits):
    """Decode the CNAV ephemeris from a type-10 + type-11 message pair (SI)."""
    src = {10: msg10_bits, 11: msg11_bits}
    eph = {}
    for name, (mt, start, length, signed, scale) in CNAV_EPH_FIELDS.items():
        bits = src[mt][start - 1:start - 1 + length]
        v = _uint(bits)
        if signed and bits and bits[0] == 1:
            v -= (1 << length)
        eph[name] = v * scale
    return eph


def assemble_cnav_ephemeris(msgs):
    """Pair adjacent (type 10, type 11) messages of the same PRN and parse."""
    out = []
    for i in range(len(msgs) - 1):
        a, b = msgs[i], msgs[i + 1]
        if a["type"] == 10 and b["type"] == 11 and a["prn"] == b["prn"]:
            eph = parse_cnav_ephemeris(a["bits"], b["bits"])
            if abs(eph["toe"] - eph["toe2"]) < 1e-6:   # consistent ephemeris set
                out.append(eph)
    return out


def sv_position_cnav(eph, t):
    """ECEF (x,y,z) m from CNAV ephemeris (IS-GPS-200 Table 30-II). Like LNAV
    but with a reference semi-major axis + rate (A = AREF + dA + A_dot*tk) and a
    rate-corrected mean motion."""
    A0 = AREF + eph["dA"]
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    Ak = A0 + eph["A_dot"] * tk
    n0 = math.sqrt(MU / A0 ** 3)
    n = n0 + eph["dn0"] + 0.5 * eph["dn0_dot"] * tk
    M = eph["M0"] + n * tk
    e = eph["e"]
    E = M
    for _ in range(20):
        E = M + e * math.sin(E)
    v = math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)
    phi = v + eph["omega"]
    s2, c2 = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + eph["Cus"] * s2 + eph["Cuc"] * c2
    r = Ak * (1 - e * math.cos(E)) + eph["Crs"] * s2 + eph["Crc"] * c2
    i = eph["i0"] + eph["i0_dot"] * tk + eph["Cis"] * s2 + eph["Cic"] * c2
    omdot = OMEGADOT_REF + eph["dOMEGA_dot"]
    om = eph["OMEGA0"] + (omdot - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    xp, yp = r * math.cos(u), r * math.sin(u)
    x = xp * math.cos(om) - yp * math.cos(i) * math.sin(om)
    y = xp * math.sin(om) + yp * math.cos(i) * math.cos(om)
    z = yp * math.sin(i)
    return x, y, z


def read_symbols(paths, prn, n_prn):
    """Soft CNAV symbols (one per 20 ms CM record) + their UTC stamps for a PRN.

    Auto-detects the record layout by frame stride: the CPU-chain era wrote
    11-float records (peak_re at slot 4, gps_beamtrack.read_records); the GPU
    chain (GnssGpuRecordAssemble -> combiner out_buf) writes 24-float REC_*
    records -- prompt P.re/P.im at slots 3/4, float64 UTC at slots 9-10
    (lib/stages/gnss/gnssRecord.hpp). The soft symbol is the carrier-locked
    prompt's real part, same quantity either way."""
    import struct
    buf = open(paths[0], "rb").read()
    strides = {11: 4 + n_prn * 44, 24: 4 + n_prn * 96}
    fits = [k for k, s in strides.items() if len(buf) % s == 0 and len(buf) >= s]
    if fits == [11] or (not fits and len(buf) % strides[11] < len(buf) % strides[24]):
        recs = read_records(paths, n_prn=n_prn)
        sel = recs[recs["prn"].astype(int) == int(prn)]
        return sel["utc"].astype(float), sel["peak_re"].astype(float)
    rb, stride = 96, strides[24]
    utc, sym = [], []
    for path in paths:
        buf = open(path, "rb").read()
        for off in range(0, len(buf) - stride + 1, stride):
            meta = struct.unpack_from("<I", buf, off)[0]
            ro = off + 4 + meta + prn_index(buf, off + 4 + meta, prn, n_prn, rb)
            if ro < off + 4:
                continue
            v = np.frombuffer(buf, dtype="<f4", count=6, offset=ro)
            t = struct.unpack_from("<d", buf, ro + 9 * 4)[0]
            if v[5] > 0.0 and t > 0.0:                    # P_energy > 0: a real record
                utc.append(t)
                sym.append(float(v[3]))                   # REC_P_RE
    return np.asarray(utc), np.asarray(sym)


def prn_index(buf, base, prn, n_prn, rb):
    """Byte offset (relative to base) of this PRN's record in a frame, or -1-base.
    Records carry their PRN in slot 0; the row order is fixed per chain, so scan
    once per call (cheap at 24 floats x n_prn)."""
    import struct
    for p in range(n_prn):
        if int(struct.unpack_from("<f", buf, base + p * rb)[0]) == int(prn):
            return p * rb
    return -1 - base


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
        utc, syms = read_symbols(paths, prn, args.n_prn)
        msgs = decode_cnav(syms)
        print("PRN %2d: %d symbols, %d CNAV messages" % (prn, len(syms), len(msgs)))
        for m in msgs:
            print("   type=%-2d  PRN=%-2d  TOW=%d" % (m["type"], m["prn"], m["tow"]))
        for eph in assemble_cnav_ephemeris(msgs):
            x, y, z = sv_position_cnav(eph, eph["toe"])
            print("   ephemeris WN=%d toe=%ds  ECEF=(%.1f, %.1f, %.1f) km"
                  % (eph["WN"], eph["toe"], x / 1e3, y / 1e3, z / 1e3))


if __name__ == "__main__":
    main()
