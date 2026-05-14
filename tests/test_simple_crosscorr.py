# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import struct

import numpy as np
import pytest

from kotekan import runner


# Spectrum length must divide samples per frame. Keep small for fast tests.
SPECTRUM_LENGTH = 16
INTEGRATION_LENGTH = 4
SAMPLES_PER_FRAME = SPECTRUM_LENGTH * INTEGRATION_LENGTH
NUM_FRAMES = 1


def _write_raw_frames(path_dir, file_name, complex_data, num_frames):
    """Write a Python complex-float array to kotekan's rawFileWrite format.

    Each file in the format has, per frame, a 4-byte metadata_size (0 for no
    metadata) followed by frame_size bytes of payload. We pack one frame per
    file at file index 0000000.
    """
    os.makedirs(str(path_dir), exist_ok=True)
    # Pack as interleaved float32 (re, im, re, im, ...).
    flat = np.empty(complex_data.size * 2, dtype=np.float32)
    flat[0::2] = complex_data.real
    flat[1::2] = complex_data.imag
    frame_bytes = flat.tobytes()
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "wb") as f:
        for _ in range(num_frames):
            f.write(struct.pack("<I", 0))  # metadata_size = 0
            f.write(frame_bytes)
    return path


def _read_raw_frame(path_dir, file_name, frame_size):
    """Read the first frame from a rawFileWrite output file."""
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "rb") as f:
        metadata_size = struct.unpack("<I", f.read(4))[0]
        if metadata_size:
            f.read(metadata_size)
        return f.read(frame_size)


def _run_crosscorr(tmpdir, A, B):
    """Run kotekan with two complex-float inputs through SimpleCrosscorr and
    return the output frame as a raw byte string."""
    in_dir = tmpdir.mkdir("in")
    out_dir = tmpdir.mkdir("out")

    _write_raw_frames(in_dir, "inA", A, NUM_FRAMES)
    _write_raw_frames(in_dir, "inB", B, NUM_FRAMES)

    in_frame_size = SAMPLES_PER_FRAME * 2 * 4  # complex float pairs
    # Per-element layout: ``[freqs floats, 1 uint count]`` repeated once per
    # visibility stream. Same layout networkPowerStream consumes.
    out_frame_size = (SPECTRUM_LENGTH + 1) * 4 * 4

    buffers = {
        "in_bufA": {
            "kotekan_buffer": "standard",
            "metadata_pool": "main_pool",
            "num_frames": "buffer_depth",
            "frame_size": in_frame_size,
        },
        "in_bufB": {
            "kotekan_buffer": "standard",
            "metadata_pool": "main_pool",
            "num_frames": "buffer_depth",
            "frame_size": in_frame_size,
        },
        "out_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "main_pool",
            "num_frames": "buffer_depth",
            "frame_size": out_frame_size,
        },
    }
    stages = {
        "readA": {
            "kotekan_stage": "rawFileRead",
            "buf": "in_bufA",
            "base_dir": str(in_dir),
            "file_name": "inA",
            "file_ext": "raw",
            "end_interrupt": True,
        },
        "readB": {
            "kotekan_stage": "rawFileRead",
            "buf": "in_bufB",
            "base_dir": str(in_dir),
            "file_name": "inB",
            "file_ext": "raw",
        },
        "xcorr": {
            "kotekan_stage": "SimpleCrosscorr",
            "in_bufA": "in_bufA",
            "in_bufB": "in_bufB",
            "out_buf": "out_buf",
            "spectrum_length": SPECTRUM_LENGTH,
            "integration_length": INTEGRATION_LENGTH,
        },
        "write_out": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "out_buf",
            "base_dir": str(out_dir),
            "file_name": "xcorr",
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "exit_after_n_files": 1,
        },
    }
    config = {
        "buffer_depth": 4,
        "log_level": "info",
        "instrument_name": "test",
    }
    r = runner.KotekanRunner(buffers=buffers, stages=stages, config=config)
    r.run()
    return _read_raw_frame(out_dir, "xcorr", out_frame_size)


def _unpack_output(raw):
    """Unpack a SimpleCrosscorr output frame into (AA, BB, ReABstar, ImABstar, count).

    Format: ``[freqs floats, 1 uint count]`` repeated once per visibility
    stream (AA, BB, Re{AB*}, Im{AB*}). All four counts are equal -- we read
    the AA one and assert the others match.
    """
    block_floats = SPECTRUM_LENGTH + 1   # ``+1`` is the trailing count
    flat = np.frombuffer(raw, dtype=np.float32, count=block_floats * 4)
    AA   = flat[0 * block_floats : 0 * block_floats + SPECTRUM_LENGTH]
    BB   = flat[1 * block_floats : 1 * block_floats + SPECTRUM_LENGTH]
    ReAB = flat[2 * block_floats : 2 * block_floats + SPECTRUM_LENGTH]
    ImAB = flat[3 * block_floats : 3 * block_floats + SPECTRUM_LENGTH]
    counts = [
        struct.unpack_from("<I", raw, (e * block_floats + SPECTRUM_LENGTH) * 4)[0]
        for e in range(4)
    ]
    assert counts[0] == counts[1] == counts[2] == counts[3], counts
    return AA, BB, ReAB, ImAB, counts[0]


def test_simple_crosscorr_identical(tmpdir):
    """A == B: AA, BB, Re{AB*} all equal 1; Im{AB*} zero."""
    A = np.ones(SAMPLES_PER_FRAME, dtype=np.complex64)
    raw = _run_crosscorr(tmpdir, A, A.copy())
    AA, BB, ReAB, ImAB, count = _unpack_output(raw)
    np.testing.assert_allclose(AA, 1.0, atol=1e-5)
    np.testing.assert_allclose(BB, 1.0, atol=1e-5)
    np.testing.assert_allclose(ReAB, 1.0, atol=1e-5)
    np.testing.assert_allclose(ImAB, 0.0, atol=1e-5)
    assert count == INTEGRATION_LENGTH


def test_simple_crosscorr_orthogonal(tmpdir):
    """A = real-only, B = imag-only: cross sees a pure imaginary -1 in Im{AB*}."""
    A = np.full(SAMPLES_PER_FRAME, 1 + 0j, dtype=np.complex64)
    B = np.full(SAMPLES_PER_FRAME, 0 + 1j, dtype=np.complex64)
    raw = _run_crosscorr(tmpdir, A, B)
    AA, BB, ReAB, ImAB, count = _unpack_output(raw)
    np.testing.assert_allclose(AA, 1.0, atol=1e-5)
    np.testing.assert_allclose(BB, 1.0, atol=1e-5)
    np.testing.assert_allclose(ReAB, 0.0, atol=1e-5)
    np.testing.assert_allclose(ImAB, -1.0, atol=1e-5)
    assert count == INTEGRATION_LENGTH


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_simple_crosscorr_random_matches_numpy(tmpdir, seed):
    """Random complex inputs: result must match the numpy reference computation."""
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal(SAMPLES_PER_FRAME) + 1j * rng.standard_normal(SAMPLES_PER_FRAME)).astype(
        np.complex64
    )
    B = (rng.standard_normal(SAMPLES_PER_FRAME) + 1j * rng.standard_normal(SAMPLES_PER_FRAME)).astype(
        np.complex64
    )
    raw = _run_crosscorr(tmpdir, A, B)
    AA, BB, ReAB, ImAB, count = _unpack_output(raw)

    # Reference: integrate over INTEGRATION_LENGTH time samples per spectral bin.
    AA_ref = np.zeros(SPECTRUM_LENGTH, dtype=np.float32)
    BB_ref = np.zeros(SPECTRUM_LENGTH, dtype=np.float32)
    ReAB_ref = np.zeros(SPECTRUM_LENGTH, dtype=np.float32)
    ImAB_ref = np.zeros(SPECTRUM_LENGTH, dtype=np.float32)
    for t in range(INTEGRATION_LENGTH):
        a = A[t * SPECTRUM_LENGTH : (t + 1) * SPECTRUM_LENGTH]
        b = B[t * SPECTRUM_LENGTH : (t + 1) * SPECTRUM_LENGTH]
        AA_ref += (a.real ** 2 + a.imag ** 2) / INTEGRATION_LENGTH
        BB_ref += (b.real ** 2 + b.imag ** 2) / INTEGRATION_LENGTH
        ReAB_ref += (a.real * b.real + a.imag * b.imag) / INTEGRATION_LENGTH
        ImAB_ref += (a.imag * b.real - b.imag * a.real) / INTEGRATION_LENGTH

    np.testing.assert_allclose(AA, AA_ref, atol=1e-4)
    np.testing.assert_allclose(BB, BB_ref, atol=1e-4)
    np.testing.assert_allclose(ReAB, ReAB_ref, atol=1e-4)
    np.testing.assert_allclose(ImAB, ImAB_ref, atol=1e-4)
    assert count == INTEGRATION_LENGTH
