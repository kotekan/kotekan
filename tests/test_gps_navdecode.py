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
