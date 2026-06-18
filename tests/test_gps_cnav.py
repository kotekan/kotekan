import os
import sys

import numpy as np

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


def test_cnav_recovers_under_phase_and_polarity():
    msgs_in = [nc.build_cnav_message(prn=7, msg_type=t, tow=200000 + k)
               for k, t in enumerate((30, 31, 32))]
    bits = sum(msgs_in, [])
    soft = -_soft(bits, noise=0.3, seed=4)      # inverted carrier polarity
    soft = np.concatenate([[0.05], soft])       # offset the symbol-pair phase by 1
    got = nc.decode_cnav(soft, known_start=False)
    assert len(got) >= 2                        # later messages recovered after convergence
    assert all(g["prn"] == 7 and g["type"] in (30, 31, 32) for g in got)
