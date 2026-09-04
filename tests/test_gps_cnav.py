import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts"))
import gps_cnav as nc  # noqa: E402


def _soft(bits, noise=0.0, seed=0):
    """Convolutionally encode bits and map to bipolar soft symbols (+ noise)."""
    rng = np.random.default_rng(seed)
    s = (1.0 - 2.0 * nc.conv_encode(bits)).astype(float)
    if noise:
        s = s + rng.standard_normal(s.shape) * noise
    return s


# ---- rate-1/2 K=7 convolutional code ----
def test_viterbi_roundtrip_exact():
    rng = np.random.default_rng(0)
    bits = list(rng.integers(0, 2, 200))
    dec = nc.viterbi_decode(1.0 - 2.0 * nc.conv_encode(bits), known_start=True)
    assert list(dec[:200]) == bits


def test_viterbi_corrects_symbol_errors():
    rng = np.random.default_rng(1)
    bits = list(rng.integers(0, 2, 300))
    bip = (1.0 - 2.0 * nc.conv_encode(bits)).astype(float)
    for idx in (10, 100, 250, 400, 550):       # isolated symbol flips, well separated
        bip[idx] = -bip[idx]
    dec = nc.viterbi_decode(bip, known_start=True)
    assert list(dec[:300]) == bits             # FEC corrects them all


# ---- CRC-24Q ----
def test_crc24q_consistent_and_detects_error():
    rng = np.random.default_rng(2)
    msg = list(rng.integers(0, 2, nc.DATA_BITS))
    crc = nc.crc24q(msg)
    full = msg + [(crc >> (23 - i)) & 1 for i in range(24)]
    assert nc.crc24q(full[:nc.DATA_BITS]) == nc._uint(full[nc.DATA_BITS:])
    full[5] ^= 1
    assert nc.crc24q(full[:nc.DATA_BITS]) != nc._uint(full[nc.DATA_BITS:])


# ---- full CNAV message chain ----
def test_cnav_message_roundtrip():
    msgs_in = [nc.build_cnav_message(prn=5, msg_type=t, tow=100000 + k)
               for k, t in enumerate((10, 11, 30))]
    bits = sum(msgs_in, [])                     # concat -> 900 bits
    got = nc.decode_cnav(_soft(bits, noise=0.4, seed=3), known_start=True)
    assert len(got) == 3
    for g, t, k in zip(got, (10, 11, 30), range(3)):
        assert g["prn"] == 5 and g["type"] == t and g["tow"] == 100000 + k


# ---- CNAV ephemeris (types 10/11) -> SV position ----
def _to_raw(value, scale, signed, length):
    raw = int(round(value / scale))
    if signed and raw < 0:
        raw += (1 << length)
    return raw & ((1 << length) - 1)


def _build_eph_msg(prn, mtype, tow, eph):
    """Build a type-10 or -11 message carrying the ephemeris fields, with CRC."""
    bits = nc.build_cnav_message(prn, mtype, tow)         # header + zero payload + CRC
    for name, (mt, start, length, signed, scale) in nc.CNAV_EPH_FIELDS.items():
        if mt != mtype:
            continue
        nc._set(bits, start - 1, length, _to_raw(eph[name], scale, signed, length))
    nc._set(bits, nc.DATA_BITS, 24, nc.crc24q(bits[0:nc.DATA_BITS]))  # recompute CRC
    return bits


_EPH = {
    "WN": 200, "top": 14400.0, "toe": 14400.0, "dA": 120.0, "A_dot": 0.0,
    "dn0": 4.5e-9, "dn0_dot": 1e-13, "M0": 0.3, "e": 0.005, "omega": 0.5,
    "toe2": 14400.0, "OMEGA0": 1.0, "i0": 0.96, "dOMEGA_dot": 1e-10, "i0_dot": 1e-11,
    "Cis": 1e-7, "Cic": -1e-7, "Crs": 12.0, "Crc": 200.0, "Cus": 5e-6, "Cuc": 1e-6,
}


def test_cnav_ephemeris_roundtrip_and_position():
    bits = _build_eph_msg(5, 10, 100000, _EPH) + _build_eph_msg(5, 11, 100001, _EPH)
    msgs = nc.decode_cnav(_soft(bits, noise=0.3, seed=5), known_start=True)
    ephs = nc.assemble_cnav_ephemeris(msgs)
    assert len(ephs) >= 1
    eph = ephs[0]
    for name, (_mt, _s, _l, _sg, scale) in nc.CNAV_EPH_FIELDS.items():
        assert abs(eph[name] - _EPH[name]) <= abs(scale) + 1e-12, name
    x, y, z = nc.sv_position_cnav(eph, eph["toe"])
    r = (x * x + y * y + z * z) ** 0.5
    assert 25.5e6 < r < 27.5e6                            # GPS orbit radius


def test_cnav_sv_position_circular_matches_closed_form():
    """e=0, no corrections: compare to the closed-form rotation (A = AREF)."""
    eph = {name: 0.0 for name in nc.CNAV_EPH_FIELDS}
    eph.update(e=0.0, M0=math.pi / 2, i0=0.96, OMEGA0=1.0, toe=0.0)   # dA=0 -> A0=AREF
    A = nc.AREF
    x, y, z = nc.sv_position_cnav(eph, 0.0)
    assert x == pytest.approx(-A * math.cos(0.96) * math.sin(1.0), abs=1.0)
    assert y == pytest.approx(A * math.cos(0.96) * math.cos(1.0), abs=1.0)
    assert z == pytest.approx(A * math.sin(0.96), abs=1.0)


def test_cnav_recovers_under_phase_and_polarity():
    msgs_in = [nc.build_cnav_message(prn=7, msg_type=t, tow=200000 + k)
               for k, t in enumerate((30, 31, 32))]
    bits = sum(msgs_in, [])
    soft = -_soft(bits, noise=0.3, seed=4)      # inverted carrier polarity
    soft = np.concatenate([[0.05], soft])       # offset the symbol-pair phase by 1
    got = nc.decode_cnav(soft, known_start=False)
    assert len(got) >= 2                        # later messages recovered after convergence
    assert all(g["prn"] == 7 and g["type"] in (30, 31, 32) for g in got)
