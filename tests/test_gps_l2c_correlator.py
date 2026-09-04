# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

"""End-to-end despread of a GPS L2C CM signal through the descriptor-driven
correlator (signal=GPS_L2C_CM): a 20 ms code period (vs L1 C/A's 1 ms) and the
CM/CL chip time-multiplex. Proves the Ns-from-code_period generalization works
for a non-L1 signal."""

import os
import struct

import numpy as np

from kotekan import runner

FS = 2.5e6                         # Hz (keeps Ns small for the test)
CODE_PERIOD = 20e-3                # L2C CM period
NS = int(round(FS * CODE_PERIOD))  # 50000 samples / block
F_OFFSET = 0.5e6
CM_LEN = 10230
COMB_CHIPS = 20460                 # combined 1.023 Mcps chips over 20 ms (CM even, CL odd)
RECORD_FLOATS = 11
PRNS = [7, 12]

# L2 CM initial register states (octal), the PRNs this test uses (matches
# gpsL2CCode.cpp / PocketSDR). LFSR: R = (R>>1) ^ (0o445112474 if R&1).
L2CM_INIT = {7: 0o124510070, 12: 0o024437606}


def l2cm_code(prn):
    r = L2CM_INIT[prn]
    out = np.empty(CM_LEN, dtype=np.float32)
    for i in range(CM_LEN):
        out[i] = 1.0 - 2.0 * (r & 1)        # bit 0 -> +1, 1 -> -1
        r = (r >> 1) ^ (0o445112474 * (r & 1))
    return out


def make_l2cm_signal(prn, doppler=0.0, amp=2000.0):
    """One 20 ms block of int16 reals carrying PRN's CM on the even combined
    chips (CL slots left at zero), on a carrier at F_OFFSET+doppler."""
    cm = l2cm_code(prn)
    n = np.arange(NS)
    c = (n * COMB_CHIPS // NS) % COMB_CHIPS          # combined chip index per sample
    chip = np.where(c % 2 == 0, cm[(c // 2) % CM_LEN], 0.0).astype(np.float32)
    carrier = np.cos(2 * np.pi * (F_OFFSET + doppler) * n / FS)
    return np.round(amp * chip * carrier).astype(np.int16)


def _write_int16_frame(path_dir, file_name, samples):
    os.makedirs(str(path_dir), exist_ok=True)
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0))
        f.write(samples.astype("<i2").tobytes())


def _read_records(out_dir, file_name):
    path = os.path.join(str(out_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "rb") as f:
        meta = struct.unpack("<I", f.read(4))[0]
        if meta:
            f.read(meta)
        raw = f.read(len(PRNS) * RECORD_FLOATS * 4)
    recs = np.frombuffer(raw, dtype="<f4", count=len(PRNS) * RECORD_FLOATS).reshape(
        len(PRNS), RECORD_FLOATS)
    return {int(round(r[0])): r for r in recs}


def _run(tmpdir, samples, signal="GPS_L2C_CM", ns=NS, extra=None):
    in_dir = tmpdir.mkdir("in")
    out_dir = tmpdir.mkdir("out")
    _write_int16_frame(in_dir, "l2cin", samples)
    l2c = {"kotekan_stage": "GpsReplicaCorrelator", "in_buf": "in_buf", "out_buf": "out_buf",
           "signal": signal, "sample_rate": FS, "f_offset": F_OFFSET,
           "doppler_min": -1000.0, "doppler_max": 1000.0, "doppler_step": 1000.0, "prns": PRNS}
    l2c.update(extra or {})
    buffers = {
        "in_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                   "num_frames": "buffer_depth", "frame_size": ns * 2},
        "out_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                    "num_frames": "buffer_depth", "frame_size": len(PRNS) * RECORD_FLOATS * 4},
    }
    stages = {
        "read_in": {"kotekan_stage": "rawFileRead", "buf": "in_buf", "base_dir": str(in_dir),
                    "file_name": "l2cin", "file_ext": "raw", "end_interrupt": True},
        "l2c": l2c,
        "write_out": {"kotekan_stage": "rawFileWrite", "in_buf": "out_buf", "base_dir": str(out_dir),
                      "file_name": "l2c", "file_ext": "raw", "prefix_hostname": False,
                      "num_frames_per_file": 1, "exit_after_n_files": 1},
    }
    config = {"buffer_depth": 4, "log_level": "info", "instrument_name": "test"}
    runner.KotekanRunner(buffers=buffers, stages=stages, config=config).run()
    return _read_records(out_dir, "l2c")


def test_l2cm_despreads(tmpdir):
    recs = _run(tmpdir, make_l2cm_signal(7, doppler=0.0))
    snr7, snr12 = recs[7][6], recs[12][6]
    assert snr7 > 50.0, (snr7, snr12)          # CM PRN 7 despreads strongly
    assert snr7 > 3.0 * snr12, (snr7, snr12)   # the other PRN stays at the floor
    assert abs(recs[7][1]) < 50.0               # recovered Doppler near 0 (parabola-refined)


# ---- L2C CL pilot: time-assisted (known-phase windowed) despread ----
CL_LEN = 767250
COMB_RATE = 1.023e6                  # combined CM+CL chip rate
CL_COHERENT_MS = 10.0
CL_NS = int(round(FS * CL_COHERENT_MS * 1e-3))   # 25000 samples / window
L2CL_INIT = {7: 0o652521276, 12: 0o117776450}


def l2cl_code(prn, n_chips):
    r = L2CL_INIT[prn]
    out = np.empty(n_chips, dtype=np.float32)
    for i in range(n_chips):
        out[i] = 1.0 - 2.0 * (r & 1)
        r = (r >> 1) ^ (0o445112474 * (r & 1))
    return out


def make_l2cl_signal(prn, phi_chips=0, doppler=0.0, amp=2000.0):
    """One coherent window of CL on the odd combined chips, starting at CL
    component chip `phi_chips` (the time-assist seed the correlator is given)."""
    n = np.arange(CL_NS)
    seed_comb = 2 * phi_chips + 1                         # CL is the odd parity
    c = np.floor(seed_comb + n * COMB_RATE / FS).astype(np.int64)
    n_comp = int(c.max() // 2) + 2
    cl = l2cl_code(prn, n_comp)
    chip = np.where(c % 2 == 1, cl[c // 2], 0.0).astype(np.float32)
    carrier = np.cos(2 * np.pi * (F_OFFSET + doppler) * n / FS)
    return np.round(amp * chip * carrier).astype(np.int16)


def test_l2cl_time_assisted_despreads(tmpdir):
    phi = 0
    recs = _run(tmpdir, make_l2cl_signal(7, phi_chips=phi), signal="GPS_L2C_CL", ns=CL_NS,
                extra={"coherent_ms": CL_COHERENT_MS, "code_phase_seed_chips": float(phi)})
    snr7, snr12 = recs[7][6], recs[12][6]
    assert snr7 > 50.0, (snr7, snr12)          # CL PRN 7 pilot despreads at the seeded phase
    assert snr7 > 3.0 * snr12, (snr7, snr12)
