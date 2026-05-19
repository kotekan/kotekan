# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

"""SampleAutocorr is a REST-triggered on-demand stage; its compute kernel is the
single-channel reduction of SimpleCrosscorr (AA only), which is covered
end-to-end in ``test_simple_crosscorr.py``. The test here is therefore a
minimal smoke test: confirm the stage registers, consumes frames, and the
REST endpoint accepts a POST without crashing kotekan. We do not verify the
returned spectrum here because ``KotekanRunner`` discards REST responses.
"""

import os
import struct

import numpy as np

from kotekan import runner


SPECTRUM_LENGTH = 16
INTEGRATION_LENGTH = 4
SAMPLES_PER_FRAME = SPECTRUM_LENGTH * INTEGRATION_LENGTH
# Provide a generous supply of input frames so the REST request completes
# before rawFileRead runs out and interrupts kotekan.
NUM_FRAMES = 50


def _write_raw_frames(path_dir, file_name, complex_data, num_frames):
    os.makedirs(str(path_dir), exist_ok=True)
    flat = np.empty(complex_data.size * 2, dtype=np.float32)
    flat[0::2] = complex_data.real
    flat[1::2] = complex_data.imag
    frame_bytes = flat.tobytes()
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (file_name, 0))
    with open(path, "wb") as f:
        for _ in range(num_frames):
            f.write(struct.pack("<I", 0))
            f.write(frame_bytes)
    return path


def test_sample_autocorr_smoke(tmpdir):
    in_dir = tmpdir.mkdir("in")

    A = np.ones(SAMPLES_PER_FRAME, dtype=np.complex64)
    _write_raw_frames(in_dir, "in", A, NUM_FRAMES)

    in_frame_size = SAMPLES_PER_FRAME * 2 * 4

    buffers = {
        "in_buf": {
            "kotekan_buffer": "standard",
            "metadata_pool": "main_pool",
            "num_frames": "buffer_depth",
            "frame_size": in_frame_size,
        },
    }
    stages = {
        "read_in": {
            "kotekan_stage": "rawFileRead",
            "buf": "in_buf",
            "base_dir": str(in_dir),
            "file_name": "in",
            "file_ext": "raw",
            "end_interrupt": True,
        },
        "sa": {
            "kotekan_stage": "SampleAutocorr",
            "in_buf": "in_buf",
            "spectrum_length": SPECTRUM_LENGTH,
            "integration_length": INTEGRATION_LENGTH,
        },
    }
    config = {
        "buffer_depth": 4,
        "log_level": "info",
        "instrument_name": "test",
    }
    rest_commands = [
        ("post", "sa/spectrum", {"integration_length": INTEGRATION_LENGTH}),
    ]
    r = runner.KotekanRunner(
        buffers=buffers, stages=stages, config=config, rest_commands=rest_commands
    )
    r.run()
    # If we got here without an exception, kotekan exited cleanly.
