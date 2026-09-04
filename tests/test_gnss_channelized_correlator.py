# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import struct
import tempfile

import numpy as np

from kotekan import runner

# Channelized (PFB) F-engine + measurement-tier despread end-to-end:
#   rawFileRead -> fftwEngine(num_taps>1) -> GnssChannelizedCorrelator -> rawFileWrite
# The signal is injected at a known code phase / Doppler (= 0), so the stage
# should recover the injected complex amplitude.

CHIP_RATE = 1.023e6
SP = 4                       # samples per chip -> signal fills the central half-band
SAMPLE_RATE = CHIP_RATE * SP # 4.092 MHz
N = 32                       # PFB channels (even: clean fftshift, DC at N/2)
NUM_TAPS = 4
HOPS = 127                   # ~one C/A period (1023 chips * SP / N = 127.875)
CODE_LEN = 1023
RECORD_FLOATS = 11
PRNS = [2, 5, 12]
N_PRN = len(PRNS)

G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


def ca_code(prn):
    a, b = G2_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    code = np.empty(CODE_LEN, dtype=np.float32)
    for i in range(CODE_LEN):
        g2_phase = g2[a - 1] ^ g2[b - 1]
        chip = g1[9] ^ g2_phase
        code[i] = -1.0 if chip else 1.0
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return code


def make_iq(prn, a_re, a_im, n_samp, cp_chips=0.0, fd=0.0):
    """Interleaved int16 IQ: (a_re + i a_im) * code(floor(cp + n*chip/Fs)) * carrier(fd)."""
    n = np.arange(n_samp)
    chip = np.floor(cp_chips + n * CHIP_RATE / SAMPLE_RATE).astype(np.int64) % CODE_LEN
    c = ca_code(prn)[chip]
    carrier = np.exp(2j * np.pi * fd * n / SAMPLE_RATE)
    s = (a_re + 1j * a_im) * c * carrier
    iq = np.empty(2 * n_samp, dtype=np.int16)
    iq[0::2] = np.round(s.real).astype(np.int16)
    iq[1::2] = np.round(s.imag).astype(np.int16)
    return iq


def _write_int16_frame(path_dir, name, samples):
    os.makedirs(str(path_dir), exist_ok=True)
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (name, 0))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0))  # metadata_size = 0
        f.write(samples.astype("<i2").tobytes())
    return path


def _read_one(path):
    with open(path, "rb") as f:
        meta = struct.unpack("<I", f.read(4))[0]
        if meta:
            f.read(meta)
        raw = f.read(N_PRN * RECORD_FLOATS * 4)
    recs = np.frombuffer(raw, dtype=np.float32, count=N_PRN * RECORD_FLOATS).reshape(
        N_PRN, RECORD_FLOATS)
    return {int(round(r[0])): r for r in recs}


def _run(tmpdir, iq, n_records, active_prns=None, acquire=False):
    n_complex = iq.size // 2
    work = tempfile.mkdtemp(dir=str(tmpdir))
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    _write_int16_frame(in_dir, "chin", iq)

    n_hops = n_complex // N
    buffers = {
        "in_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                   "num_frames": "buffer_depth", "frame_size": iq.size * 2},
        "chan_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                     "num_frames": "buffer_depth", "frame_size": n_hops * N * 8},
        "out_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                    "num_frames": "buffer_depth", "frame_size": N_PRN * RECORD_FLOATS * 4},
    }
    stages = {
        "read_in": {"kotekan_stage": "rawFileRead", "buf": "in_buf",
                    "base_dir": str(in_dir), "file_name": "chin", "file_ext": "raw",
                    "end_interrupt": True},
        "fengine": {"kotekan_stage": "fftwEngine", "in_buf": "in_buf", "out_buf": "chan_buf",
                    "spectrum_length": N, "input_type": "complex",
                    "num_taps": NUM_TAPS, "pfb_window": "hamming"},
        "chan_corr": {"kotekan_stage": "GnssChannelizedCorrelator",
                      "in_buf": "chan_buf", "out_buf": "out_buf",
                      "signal": "GPS_L1CA", "sample_rate": SAMPLE_RATE, "f_offset": 0.0,
                      "spectrum_length": N, "num_taps": NUM_TAPS, "pfb_window": "hamming",
                      "hops_per_record": HOPS, "prns": PRNS,
                      **({"acquire": True, "acquire_snr": 8.0, "doppler_min": -600.0,
                          "doppler_max": 600.0, "doppler_step": 200.0} if acquire else {}),
                      **({"active_prns": active_prns} if active_prns is not None else {})},
        "write_out": {"kotekan_stage": "rawFileWrite", "in_buf": "out_buf",
                      "base_dir": str(out_dir), "file_name": "rec", "file_ext": "raw",
                      "prefix_hostname": False, "num_frames_per_file": 1,
                      "exit_after_n_files": n_records},
    }
    config = {"buffer_depth": max(4, n_records + 2), "log_level": "info",
              "instrument_name": "test"}
    runner.KotekanRunner(buffers=buffers, stages=stages, config=config).run()
    return [_read_one(os.path.join(out_dir, "rec_%07d.raw" % i)) for i in range(n_records)]


def test_recovers_injected_amplitude(tmpdir):
    a_re, a_im = 2000.0, 1000.0
    iq = make_iq(5, a_re, a_im, 2 * HOPS * N)
    rec = _run(tmpdir, iq, n_records=2)[0]

    r5 = rec[5]
    assert abs(r5[4] - a_re) < 0.02 * a_re   # peak_re
    assert abs(r5[5] - a_im) < 0.02 * abs(a_im)  # peak_im
    mag = (a_re ** 2 + a_im ** 2) ** 0.5
    assert abs(r5[3] - mag) < 0.02 * mag     # peak_amp

    # Wrong PRNs see only the cross-correlation floor.
    for prn in (2, 12):
        assert rec[prn][3] < 0.15 * mag


def test_acquire_self_seeds_then_measures(tmpdir):
    a_re, a_im = 2000.0, 1000.0
    delay = 50                                  # samples; cp = 12.5 chips (off hop)
    cp = delay * CHIP_RATE / SAMPLE_RATE
    fd = 200.0                                  # on the Doppler grid
    iq = make_iq(5, a_re, a_im, 2 * HOPS * N, cp_chips=cp, fd=fd)
    rec = _run(tmpdir, iq, n_records=2, acquire=True)[0]

    r5 = rec[5]
    assert abs(r5[1] - fd) <= 100.0             # Doppler acquired (grid step 200)
    d = min((r5[2] - cp) % CODE_LEN, (cp - r5[2]) % CODE_LEN)
    assert d < 1.0                              # code phase within a chip
    mag = (a_re ** 2 + a_im ** 2) ** 0.5
    assert r5[3] > 0.6 * mag                    # self-seed good enough to measure


def test_active_prns_mask_zeroes_record(tmpdir):
    iq = make_iq(5, 2000.0, 1000.0, 2 * HOPS * N)
    rec = _run(tmpdir, iq, n_records=2, active_prns=[2, 12])[0]
    assert rec[5][3] == 0.0  # masked off -> zeroed
    assert rec[5][4] == 0.0 and rec[5][5] == 0.0
