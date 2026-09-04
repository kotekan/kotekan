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
# 9 float32 fields + a trailing float64 UTC stamp in the last two slots.
RECORD_FLOATS = 11
UTC_SLOT = 9
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


def _read_one(path):
    with open(path, "rb") as f:
        meta = struct.unpack("<I", f.read(4))[0]
        if meta:
            f.read(meta)
        raw = f.read(N_PRN * RECORD_FLOATS * 4)
    recs = np.frombuffer(raw, dtype=np.float32, count=N_PRN * RECORD_FLOATS).reshape(
        N_PRN, RECORD_FLOATS
    )
    return {int(round(r[0])): r for r in recs}


def _read_all(out_dir, file_name, n):
    """Per-block records: list (len n) of {prn: record-row}."""
    return [
        _read_one(os.path.join(out_dir, "%s_%07d.raw" % (file_name, i))) for i in range(n)
    ]


def row_utc(row):
    """Decode the trailing float64 UTC stamp from a record's last two slots."""
    return np.frombuffer(row[UTC_SLOT:UTC_SLOT + 2].astype("<f4").tobytes(), dtype="<f8")[0]


def _run(tmpdir, samples, track_phase=False, incoherent_ms=1, lock_snr=15.0, active_prns=None,
         capture_utc0=0.0):
    """Run the pipeline over `samples` (length a multiple of NS, one record
    frame emitted per NS-sample block) and return the per-block record list."""
    assert samples.size % NS == 0
    n_blocks = samples.size // NS
    # Unique work dirs so a test may call _run more than once.
    work = tempfile.mkdtemp(dir=str(tmpdir))
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    # One input frame carrying all blocks; the correlator slices it to NS.
    _write_int16_frame(in_dir, "gpsin", samples)

    in_frame_size = samples.size * 2  # int16 real
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
            "lock_snr": lock_snr,
            "capture_utc0": capture_utc0,
            "prns": PRNS,
            **({"active_prns": active_prns} if active_prns is not None else {}),
        },
        "write_out": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "out_buf",
            "base_dir": str(out_dir),
            "file_name": "gps",
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "exit_after_n_files": n_blocks,
        },
    }
    config = {
        "buffer_depth": max(4, n_blocks + 2),
        "log_level": "info",
        "instrument_name": "test",
    }
    runner.KotekanRunner(buffers=buffers, stages=stages, config=config).run()
    return _read_all(out_dir, "gps", n_blocks)


def test_acquires_injected_prn(tmpdir):
    """A single injected PRN is found with the strongest peak; the doppler and
    code phase are recovered, and the other searched PRNs stay at the noise
    floor (low SNR)."""
    delay = 600  # samples
    recs = _run(tmpdir, make_signal(5, delay_samples=delay, doppler=0.0))[0]

    snr = {p: recs[p][6] for p in PRNS}
    # The injected PRN dominates: high absolute SNR and well above the other
    # searched PRNs (which see only code cross-correlation, no real signal).
    assert snr[5] == max(snr.values())
    assert snr[5] > 50.0
    assert snr[5] > 3.0 * max(snr[p] for p in (2, 12)), snr

    r5 = recs[5]
    assert r5[1] == pytest.approx(0.0, abs=2.0)  # doppler (parabola-refined)
    # code phase: peak lag == injected delay; chips = lag * 1023 / NS.
    expected_chips = (delay * CODE_LEN / NS) % CODE_LEN
    got = r5[2]
    # accept either the direct or the circularly-wrapped convention
    d = min(abs(got - expected_chips), CODE_LEN - abs(got - expected_chips))
    assert d < 1.0, (got, expected_chips)


@pytest.mark.parametrize("doppler", [-500.0, 500.0])
def test_recovers_doppler(tmpdir, doppler):
    # Injected exactly on a grid bin: parabolic refinement leaves it there.
    recs = _run(tmpdir, make_signal(12, delay_samples=0, doppler=doppler))[0]
    assert recs[12][6] == max(recs[p][6] for p in PRNS)
    assert recs[12][1] == pytest.approx(doppler, abs=2.0)


def test_record_has_utc_timestamp(tmpdir):
    """Each record carries a wall-clock UTC stamp (shared across PRNs in a
    block) so the post-processor can attach alt/az."""
    import time
    before = time.time()
    rec = _run(tmpdir, make_signal(5, doppler=0.0))[0]
    after = time.time()
    utc = row_utc(rec[5])
    assert before - 5.0 <= utc <= after + 5.0, (utc, before, after)
    assert row_utc(rec[2]) == utc  # all PRNs in the block share the stamp


def test_active_prns_mask_zeroes_record(tmpdir):
    """A PRN excluded from the go/no-go mask emits a zeroed record (still in the
    frame, labeled with its PRN) while active PRNs are unaffected. This is the
    same _active path the REST /set_mask endpoint drives."""
    recs = _run(tmpdir, make_signal(5, doppler=0.0), active_prns=[2, 5])[0]
    assert recs[5][6] > 50.0  # PRN 5 active -> detected (high SNR)
    assert recs[12][0] == 12  # masked slot still labeled with its PRN
    assert recs[12][3] == 0.0 and recs[12][6] == 0.0  # amp / snr zeroed


def test_capture_utc0_sample_anchored(tmpdir):
    """With capture_utc0 set, each block's record timestamp is the sample-anchored
    capture time (capture_utc0 + sample/Fs), not the wall-clock at emit -- so an
    offline replay carries the recording time, and it matches the CL seed's time
    base."""
    t0 = 1.70e9
    n_blocks = 3
    sig = np.concatenate([make_signal(5, doppler=0.0) for _ in range(n_blocks)])
    blocks = _run(tmpdir, sig, capture_utc0=t0)
    for m in range(n_blocks):
        utc = row_utc(blocks[m][5])
        assert utc == pytest.approx(t0 + m * NS / SAMPLE_RATE, abs=1e-4), (m, utc)


def test_parabolic_doppler_refines_offgrid(tmpdir):
    """A signal between grid bins is reported off-grid (parabolic interpolation),
    pulled toward the true Doppler -- even in amplitude-only mode."""
    r = _run(tmpdir, make_signal(5, doppler=200.0), track_phase=False)[0][5]
    assert 50.0 < r[1] < 400.0, r[1]  # grid bin is 0; true is 200


def test_track_phase_records_complex_peak(tmpdir):
    """With track_phase on, the complex peak is populated; with it off, zeroed."""
    phase = 0.9
    r = _run(tmpdir, make_signal(5, doppler=0.0, phase=phase), track_phase=True)[0][5]
    assert (r[4] ** 2 + r[5] ** 2) > 0.0
    assert np.isfinite(np.arctan2(r[5], r[4]))

    off = _run(tmpdir, make_signal(5, doppler=0.0, phase=phase), track_phase=False)[0][5]
    assert off[4] == 0.0 and off[5] == 0.0


# --- multi-block phase / nav-bit tracking (Stage 2) ---

def make_signal_multi(prn, n_blocks, doppler=0.0, phase0=0.0, amp=2000.0, flip_blocks=()):
    """A contiguous stream of `n_blocks` 1 ms blocks: continuous carrier across
    block boundaries, code repeating each block, and a nav-bit sign that toggles
    at each block index in `flip_blocks` (model of the 50 bps nav data)."""
    n = np.arange(n_blocks * NS)
    carrier = np.cos(2 * np.pi * (F_OFFSET + doppler) * n / SAMPLE_RATE + phase0)
    code = np.tile(code_resampled(prn, 0), n_blocks)
    sign = np.empty(n_blocks, dtype=np.float32)
    s, flips = 1.0, set(flip_blocks)
    for b in range(n_blocks):
        if b in flips:
            s = -s
        sign[b] = s
    navsig = np.repeat(sign, NS)
    return np.round(amp * navsig * code * carrier).astype(np.int16)


def test_phase_continuous_across_navbit(tmpdir):
    """Doppler 0 (constant true phase) with a nav-bit flip mid-record: nav_sign
    toggles exactly at the flip and the nav-stripped continuous phase stays
    flat (the raw peak would jump by pi)."""
    n_blocks = 40
    blocks = _run(tmpdir, make_signal_multi(5, n_blocks, doppler=0.0, phase0=0.5,
                                            flip_blocks=(20,)), track_phase=True)
    nav = np.array([blocks[i][5][7] for i in range(n_blocks)])
    pc = np.array([blocks[i][5][8] for i in range(n_blocks)])

    assert np.all(nav != 0.0)  # locked throughout (clean signal)
    # exactly one nav-sign change, at the injected flip
    changes = list(np.where(np.diff(np.sign(nav)) != 0)[0] + 1)
    assert changes == [20], changes
    # constant true phase -> stripped continuous phase is flat
    assert pc.std() < 0.1, pc

    # the RAW peak phase, by contrast, does jump ~pi at the flip
    raw = np.array([np.arctan2(blocks[i][5][5], blocks[i][5][4]) for i in range(n_blocks)])
    jump = abs(((raw[20] - raw[19] + np.pi) % (2 * np.pi)) - np.pi)
    assert jump > 2.0, jump


def test_no_phase_chatter_at_ongrid_doppler(tmpdir):
    """Regression: an on-grid, nonzero Doppler must not be read as a nav-bit
    flip every block. The per-block wipe-off phase reset (2*pi*(f_off+fd)*Ns/Fs,
    = pi/block for fd an odd multiple of the grid step) is corrected so the kept
    peak phase is continuous across blocks."""
    n_blocks = 40
    blocks = _run(tmpdir, make_signal_multi(5, n_blocks, doppler=500.0, phase0=0.3),
                  track_phase=True)
    nav = np.array([blocks[i][5][7] for i in range(n_blocks)])
    pc = np.array([blocks[i][5][8] for i in range(n_blocks)])
    assert np.all(nav != 0.0)
    assert np.sum(np.abs(np.diff(np.sign(nav))) > 0) == 0, nav  # no spurious flips
    assert pc.std() < 0.1, pc  # phase flat once the wipe is continuous


def test_phase_ramp_tracks_residual_doppler(tmpdir):
    """A residual Doppler smaller than the grid step leaves a linear phase ramp;
    nav-bit stripping keeps it linear through a mid-record bit flip."""
    n_blocks = 40
    resid = 200.0  # < half the 500 Hz step -> search picks bin 0
    blocks = _run(tmpdir, make_signal_multi(12, n_blocks, doppler=resid, phase0=0.0,
                                            flip_blocks=(20,)), track_phase=True)
    nav = np.array([blocks[i][12][7] for i in range(n_blocks)])
    pc = np.array([blocks[i][12][8] for i in range(n_blocks)])
    dop = np.array([blocks[i][12][1] for i in range(n_blocks)])
    assert np.all(nav != 0.0)

    # The FLL converges the reported Doppler onto the true residual (the search
    # only sees grid bin 0; the phase ramp supplies the rest).
    assert dop[-10:].mean() == pytest.approx(resid, abs=25.0), dop[-10:]

    x = np.arange(n_blocks)
    coef = np.polyfit(x, pc, 1)
    slope = coef[0]
    expected = 2 * np.pi * resid / 1000.0  # rad per 1 ms block
    assert abs(slope) == pytest.approx(expected, abs=0.05), (slope, expected)
    # continuous: small residual from the linear fit, no pi steps
    assert np.max(np.abs(pc - np.polyval(coef, x))) < 0.2
