"""GPS CNAV (L2C / L5) navigation-message decoder.

CNAV rides the L2C CM channel at 25 bps and the L5 I5 channel at 50 bps, rate-1/2
convolutionally encoded to 50 and 100 sps. cnav_predictor.py decodes the combiner's
nav_obs soft symbols through:

    soft symbols -> Viterbi (rate 1/2, K=7, G=171/133 oct)
                 -> bits -> frame sync (8-bit preamble) -> CRC-24Q
                 -> field parse (message type, PRN, TOW).

Distinct from LNAV (gps_nav_decode.py): FEC + CRC instead of Hamming parity, and
message-type framing instead of fixed subframes. cnav_predictor's self-test round-trips
messages built by build_cnav_message (encoded with galileo_inav.conv_encode) without
live sky data. CNAV_EPH_FIELDS covers the ephemeris (message types 10 and 11); the
clock messages (30-37) are not parsed.
"""

# The rate-1/2 K=7 FEC (G1=171, G2=133 oct) and CRC-24Q (0x1864CFB) are the code Galileo I/NAV
# uses, except that Galileo inverts G2; galileo_inav.py holds the one copy (conv_encode,
# viterbi_decode, crc24q, _uint); cnav_predictor's g2 search bridges the G2 difference.
from galileo_inav import crc24q, _uint

PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]
MSG_BITS = 300
DATA_BITS = MSG_BITS - 24  # 276; CRC-24Q is computed over these


# --------------------------------------------------------------------------
# CNAV message build / parse
# --------------------------------------------------------------------------
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
        d = list(int(x) for x in data)[: DATA_BITS - 38]
        bits[38 : 38 + len(d)] = d
    _set(bits, DATA_BITS, 24, crc24q(bits[0:DATA_BITS]))
    return bits


def parse_cnav_message(bits):
    """Validate a 300-bit message (preamble + CRC-24Q) and parse the header."""
    if list(int(b) for b in bits[0:8]) != PREAMBLE:
        return None
    if crc24q(bits[0:DATA_BITS]) != _uint(bits[DATA_BITS:MSG_BITS]):
        return None
    return {
        "prn": _uint(bits[8:14]),
        "type": _uint(bits[14:20]),
        "tow": _uint(bits[20:37]),
        "alert": int(bits[37]),
    }


# --------------------------------------------------------------------------
# CNAV ephemeris (message types 10 + 11) -> SV position
# --------------------------------------------------------------------------
import math  # noqa: E402

GPS_PI = 3.1415926535898
MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
AREF = 26559710.0  # CNAV reference semi-major axis, m (DELTA_A is relative)
OMEGADOT_REF = -2.6e-9 * GPS_PI  # reference rate, rad/s (DELTA_OMEGA_DOT is relative)
_P = lambda n: 2.0 ** (-n)  # 2^-n

# name -> (message type, 1-indexed start bit, length, signed, scale). From
# IS-GPS-200 Table 30-I (transcribed from GNSS-SDR's GPS_CNAV.h field table).
CNAV_EPH_FIELDS = {
    "WN": (10, 39, 13, False, 1.0),
    "top": (10, 55, 11, False, 300.0),
    "toe": (10, 71, 11, False, 300.0),
    "dA": (10, 82, 26, True, _P(9)),
    "A_dot": (10, 108, 25, True, _P(21)),
    "dn0": (10, 133, 17, True, _P(44) * GPS_PI),
    "dn0_dot": (10, 150, 23, True, _P(57) * GPS_PI),
    "M0": (10, 173, 33, True, _P(32) * GPS_PI),
    "e": (10, 206, 33, False, _P(34)),
    "omega": (10, 239, 33, True, _P(32) * GPS_PI),
    "toe2": (11, 39, 11, False, 300.0),
    "OMEGA0": (11, 50, 33, True, _P(32) * GPS_PI),
    "i0": (11, 83, 33, True, _P(32) * GPS_PI),
    "dOMEGA_dot": (11, 116, 17, True, _P(44) * GPS_PI),
    "i0_dot": (11, 133, 15, True, _P(44) * GPS_PI),
    "Cis": (11, 148, 16, True, _P(30)),
    "Cic": (11, 164, 16, True, _P(30)),
    "Crs": (11, 180, 24, True, _P(8)),
    "Crc": (11, 204, 24, True, _P(8)),
    "Cus": (11, 228, 21, True, _P(30)),
    "Cuc": (11, 249, 21, True, _P(30)),
}


def parse_cnav_ephemeris(msg10_bits, msg11_bits):
    """Decode the CNAV ephemeris from a type-10 + type-11 message pair (SI)."""
    src = {10: msg10_bits, 11: msg11_bits}
    eph = {}
    for name, (mt, start, length, signed, scale) in CNAV_EPH_FIELDS.items():
        bits = src[mt][start - 1 : start - 1 + length]
        v = _uint(bits)
        if signed and bits and bits[0] == 1:
            v -= 1 << length
        eph[name] = v * scale
    return eph


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
