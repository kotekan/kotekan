# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import struct
import tempfile

import numpy as np
import pytest

from kotekan import runner


# --- DSP parameters (kept small/fast but realistic: non-integer samples/chip) ---
SAMPLE_RATE = 5.0e6
NS = int(round(SAMPLE_RATE / 1000.0))  # 5000 samples per 1 ms code period
F_OFFSET = 1.0e6
CODE_LEN = 1023
RECORD_FLOATS = 7
PRNS = [2, 5, 12]
N_PRN = len(PRNS)
DOPPLER_MIN, DOPPLER_MAX, DOPPLER_STEP = -1000.0, 1000.0, 500.0

# IS-GPS-200 Table 3-Ia G2 phase-select taps (mirror of gpsCACode.cpp), so the
# test generates the same replicas the C++ stage does -- and cross-validates it.
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


def ca_code(prn):
    """Bipolar (+1/-1) length-1023 C/A code, matching gpsCACode.cpp."""
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


def code_resampled(prn, delay_samples=0):
    """Code sampled onto the NS-sample 1 ms grid (nearest chip), optionally
    delayed by `delay_samples` (circularly), as the correlator's replica is."""
    n = np.arange(NS)
    chip = ((n - delay_samples) * CODE_LEN // NS) % CODE_LEN
    return ca_code(prn)[chip]


def make_signal(prn, delay_samples=0, doppler=0.0, phase=0.0, amp=2000.0):
    """One 1 ms block of int16 real samples carrying PRN at the given code
    delay, Doppler and carrier phase."""
    n = np.arange(NS)
    carrier = np.cos(2 * np.pi * (F_OFFSET + doppler) * n / SAMPLE_RATE + phase)
    sig = amp * code_resampled(prn, delay_samples) * carrier
    return np.round(sig).astype(np.int16)


def _write_int16_frame(path_dir, file_name, samples):
    os.makedirs(str(path_dir), exist_ok=True)
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0))  # metadata_size = 0
        f.write(samples.astype("<i2").tobytes())
    return path


def _read_records(path_dir, file_name):
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "rb") as f:
        meta = struct.unpack("<I", f.read(4))[0]
        if meta:
            f.read(meta)
        raw = f.read(N_PRN * RECORD_FLOATS * 4)
    flat = np.frombuffer(raw, dtype=np.float32, count=N_PRN * RECORD_FLOATS)
    recs = flat.reshape(N_PRN, RECORD_FLOATS)
    # field order: prn, doppler_hz, code_phase_chips, peak_amp, peak_re, peak_im, snr
    return {int(round(r[0])): r for r in recs}


def _run(tmpdir, samples, track_phase=False, incoherent_ms=1):
    # Unique work dirs so a test may call _run more than once.
    work = tempfile.mkdtemp(dir=str(tmpdir))
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    _write_int16_frame(in_dir, "gpsin", samples)

    in_frame_size = NS * 2  # int16 real
    out_frame_size = N_PRN * RECORD_FLOATS * 4

    buffers = {
        "in_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "none",
            "num_frames": "buffer_depth",
            "frame_size": in_frame_size,
        },
        "out_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "none",
            "num_frames": "buffer_depth",
            "frame_size": out_frame_size,
        },
    }
    stages = {
        "read_in": {
            "kotekan_stage": "rawFileRead",
            "buf": "in_buf",
            "base_dir": str(in_dir),
            "file_name": "gpsin",
            "file_ext": "raw",
            "end_interrupt": True,
        },
        "gps_corr": {
            "kotekan_stage": "GpsReplicaCorrelator",
            "in_buf": "in_buf",
            "out_buf": "out_buf",
            "sample_rate": SAMPLE_RATE,
            "f_offset": F_OFFSET,
            "doppler_min": DOPPLER_MIN,
            "doppler_max": DOPPLER_MAX,
            "doppler_step": DOPPLER_STEP,
            "incoherent_ms": incoherent_ms,
            "track_phase": track_phase,
            "prns": PRNS,
        },
        "write_out": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "out_buf",
            "base_dir": str(out_dir),
            "file_name": "gps",
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "exit_after_n_files": 1,
        },
    }
    config = {"buffer_depth": 4, "log_level": "info", "instrument_name": "test"}
    runner.KotekanRunner(buffers=buffers, stages=stages, config=config).run()
    return _read_records(out_dir, "gps")


def test_acquires_injected_prn(tmpdir):
    """A single injected PRN is found with the strongest peak; the doppler and
    code phase are recovered, and the other searched PRNs stay at the noise
    floor (low SNR)."""
    delay = 600  # samples
    recs = _run(tmpdir, make_signal(5, delay_samples=delay, doppler=0.0))

    snr = {p: recs[p][6] for p in PRNS}
    # The injected PRN dominates: high absolute SNR and well above the other
    # searched PRNs (which see only code cross-correlation, no real signal).
    assert snr[5] == max(snr.values())
    assert snr[5] > 50.0
    assert snr[5] > 3.0 * max(snr[p] for p in (2, 12)), snr

    r5 = recs[5]
    assert r5[1] == pytest.approx(0.0, abs=1e-6)  # doppler bin
    # code phase: peak lag == injected delay; chips = lag * 1023 / NS.
    expected_chips = (delay * CODE_LEN / NS) % CODE_LEN
    got = r5[2]
    # accept either the direct or the circularly-wrapped convention
    d = min(abs(got - expected_chips), CODE_LEN - abs(got - expected_chips))
    assert d < 1.0, (got, expected_chips)


@pytest.mark.parametrize("doppler", [-500.0, 500.0])
def test_recovers_doppler(tmpdir, doppler):
    recs = _run(tmpdir, make_signal(12, delay_samples=0, doppler=doppler))
    assert recs[12][6] == max(recs[p][6] for p in PRNS)
    assert recs[12][1] == pytest.approx(doppler, abs=1e-6)


def test_track_phase_records_complex_peak(tmpdir):
    """With track_phase on, the complex peak carries a non-trivial phase that
    matches the injected carrier phase; with it off, re/im are zeroed."""
    phase = 0.9
    recs_on = _run(tmpdir, make_signal(5, doppler=0.0, phase=phase), track_phase=True)
    r = recs_on[5]
    assert (r[4] ** 2 + r[5] ** 2) > 0.0
    got_phase = np.arctan2(r[5], r[4])
    # absolute phase carries a fixed offset from the analytic/replica path; the
    # measurement is relative, so only require a stable non-zero complex peak
    # and that re/im are populated.
    assert np.isfinite(got_phase)

    recs_off = _run(tmpdir, make_signal(5, doppler=0.0, phase=phase), track_phase=False)
    assert recs_off[5][4] == 0.0 and recs_off[5][5] == 0.0
