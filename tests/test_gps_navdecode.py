import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts"))
import gps_navdecode as nd  # noqa: E402

SYM = nd.SYMS_PER_BIT


def _make_message(n_subframes=5, week=789, tow0=100000):   # week is 10 bits (<1024)
    """Encode a sequence of valid LNAV subframes; return (bits, expected)."""
    bits, expected = [], []
    for k in range(n_subframes):
        sfid = (k % 5) + 1
        tow = tow0 + k
        wd = [[(i * 7 + k * 3 + j) & 1 for i in range(nd.DATA_BITS)]  # arbitrary payload
              for j in range(nd.WORDS_PER_SUBFRAME)]
        bits += nd.encode_subframe(sfid, tow, wd, week=week if sfid == 1 else None)
        exp = {"sfid": sfid, "tow_count": tow}
        if sfid == 1:
            exp["week"] = week
        expected.append(exp)
    return bits, expected


def _bits_to_symbols(bits, amp=2000.0, phase=0.0, noise=0.0, prepend=0, seed=0):
    rng = np.random.default_rng(seed)
    v = 1 - 2 * np.asarray(bits)                      # bit 0->+1, bit 1->-1
    s = amp * np.repeat(v, SYM).astype(float) * np.exp(1j * phase)
    if noise:
        s = s + (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)) * noise
    if prepend:
        s = np.concatenate([(rng.standard_normal(prepend) + 1j * rng.standard_normal(prepend))
                            * amp * 0.05, s])
    return s


# ---- parity / coding ----
def test_parity_roundtrip_and_chain():
    rng = np.random.default_rng(0)
    srcs = [list(rng.integers(0, 2, nd.DATA_BITS)) for _ in range(50)]
    # encode a chain, tracking the running previous-bits
    words, e29, e30 = [], 0, 0
    for src in srcs:
        word, e29, e30 = nd.encode_word(src, e29, e30)
        words.append(word)
    # decode the chain, tracking the running previous-bits, recover each source
    c29, c30 = 0, 0
    for src, word in zip(srcs, words):
        ok, rec, c29, c30 = nd.check_word(word, c29, c30)
        assert ok and rec == src


def test_corrupted_word_fails_parity():
    word, _, _ = nd.encode_word([1, 0, 1, 1] + [0] * 20, 0, 0)
    word[5] ^= 1                                      # flip a data bit
    ok, _, _, _ = nd.check_word(word, 0, 0)
    assert not ok


# ---- frame sync + parse from bits ----
def test_subframe_sync_and_fields_from_bits():
    bits, expected = _make_message(n_subframes=5)
    subs = nd.find_subframes(bits)
    assert len(subs) == 5
    for got, exp in zip(subs, expected):
        assert got["sfid"] == exp["sfid"]
        assert got["tow_count"] == exp["tow_count"]
        if "week" in exp:
            assert got["week"] == exp["week"]


# ---- full chain from soft symbols, incl. carrier phase + polarity + noise ----
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])     # incl. inverted stream
def test_decode_from_symbols(phase):
    bits, expected = _make_message(n_subframes=4)
    syms = _bits_to_symbols(bits, phase=phase, noise=150.0, seed=1)
    subs = nd.decode_symbols(syms)
    assert len(subs) == 4
    assert [s["sfid"] for s in subs] == [e["sfid"] for e in expected]
    assert subs[0]["week"] == expected[0]["week"]


def test_bit_sync_with_offset():
    bits, expected = _make_message(n_subframes=3)
    for off in (1, 7, 13):
        syms = _bits_to_symbols(bits, prepend=off, noise=100.0, seed=off)
        subs = nd.decode_symbols(syms)
        assert [s["sfid"] for s in subs] == [e["sfid"] for e in expected]


# ---- end to end through a .raw record file ----
def _write_records(path, symbols, prn=5):
    with open(path, "wb") as f:
        for s in symbols:
            fields = np.zeros(9, dtype="<f4")
            fields[0] = prn
            fields[4] = s.real
            fields[5] = s.imag
            f.write(struct.pack("<I", 0))             # metadata length
            f.write(fields.tobytes() + struct.pack("<d", 1.70e9))  # +float64 UTC


# ---- ephemeris fields + SV position ----
def _int_to_bits(v, nbits):
    if v < 0:
        v += (1 << nbits)
    return [(v >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def _set_field(words, segments, raw):
    bits = _int_to_bits(raw, sum(s[2] for s in segments))
    p = 0
    for (w, start, length) in segments:
        for j in range(length):
            words[w - 1][start - 1 + j] = bits[p]
            p += 1


def _pack_eph(phys):
    """Pack physical ephemeris values into per-subframe word arrays (the inverse
    of parse_ephemeris, sharing nd.EPH_FIELDS). Returns (words_by_sf, raw)."""
    words = {sf: [[0] * nd.DATA_BITS for _ in range(nd.WORDS_PER_SUBFRAME)]
             for sf in (1, 2, 3)}
    raw = {}
    for name, (sf, segs, _signed, scale) in nd.EPH_FIELDS.items():
        raw[name] = int(round(phys[name] / scale))
        _set_field(words[sf], segs, raw[name])
    return words, raw


_EPH = {
    "week": 789, "IODC": 50, "TGD": -5e-9, "toc": 14400.0, "af2": 0.0,
    "af1": 1e-11, "af0": 1e-4, "IODE2": 50, "Crs": 12.0, "dn": 4.5e-9, "M0": 0.3,
    "Cuc": 1e-6, "e": 0.005, "Cus": 5e-6, "sqrtA": 5153.6, "toe": 14400.0,
    "Cic": 1e-7, "OMEGA0": 1.0, "Cis": -1e-7, "i0": 0.96, "Crc": 200.0,
    "omega": 0.5, "OMEGADOT": -8e-9, "IODE3": 50, "IDOT": 1e-10,
}


def test_ephemeris_roundtrip_and_position():
    words, raw = _pack_eph(_EPH)
    bits = []
    for sfid in (1, 2, 3):
        bits += nd.encode_subframe(sfid, 1000 + sfid, words[sfid], week=None)
    subs = nd.find_subframes(bits)
    assert len(subs) == 3
    ephs = nd.assemble_ephemeris(subs)
    assert len(ephs) == 1
    eph = ephs[0]

    for name, (_sf, _segs, _signed, scale) in nd.EPH_FIELDS.items():
        # exact through the integer quantization, and within one LSB of the input
        assert eph[name] == pytest.approx(raw[name] * scale, abs=1e-12,
                                          rel=1e-9), name
        assert abs(eph[name] - _EPH[name]) <= abs(scale) + 1e-12, name

    # position: magnitude is the GPS orbit radius (~A within e), and finite
    x, y, z = nd.sv_position(eph, eph["toe"])
    r = (x * x + y * y + z * z) ** 0.5
    A = eph["sqrtA"] ** 2
    assert abs(r - A) < A * 0.01
    assert 25.5e6 < r < 27.5e6


def test_sv_position_circular_matches_closed_form():
    """e=0, no corrections: compare to the closed-form Keplerian rotation."""
    import math
    eph = {name: 0.0 for name in nd.EPH_FIELDS}
    eph.update(sqrtA=5153.6, e=0.0, M0=math.pi / 2, i0=0.96, OMEGA0=1.0, toe=0.0)
    A = eph["sqrtA"] ** 2
    x, y, z = nd.sv_position(eph, 0.0)            # tk = 0
    # u = M0 = pi/2 -> xp=0, yp=A; Omega = OMEGA0
    assert x == pytest.approx(-A * math.cos(0.96) * math.sin(1.0), abs=1.0)
    assert y == pytest.approx(A * math.cos(0.96) * math.cos(1.0), abs=1.0)
    assert z == pytest.approx(A * math.sin(0.96), abs=1.0)


def test_ecef_to_azel_zenith():
    import math
    lat, lon, h = 43.66, -79.40, 100.0
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(la) ** 2)
    up = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    obs = ((N + h) * up[0], (N + h) * up[1], (N * (1 - e2) + h) * up[2])
    sat = tuple(obs[k] + 2.0e7 * up[k] for k in range(3))   # straight up
    az, el = nd.ecef_to_azel(sat, lat, lon, h)
    assert el > 89.5


def test_decode_from_raw_records(tmp_path):
    bits, expected = _make_message(n_subframes=3)
    syms = _bits_to_symbols(bits, phase=0.7, noise=120.0, seed=3)
    path = str(tmp_path / "gps_a_0000000.raw")
    _write_records(path, syms, prn=5)

    got = nd.read_symbols([path], prn=5, n_prn=1)
    assert got.shape[0] == len(syms)
    subs = nd.decode_symbols(got)
    assert [s["sfid"] for s in subs] == [e["sfid"] for e in expected]
    assert subs[0]["week"] == expected[0]["week"]
