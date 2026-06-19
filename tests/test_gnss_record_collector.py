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

# GnssRecordCollector: two per-worker record buffers -> one merged stream.


def _write_frame(path_dir, name, arr):
    os.makedirs(str(path_dir), exist_ok=True)
    path = os.path.join(str(path_dir), "%s_%07d.raw" % (name, 0))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0))  # metadata_size = 0
        f.write(arr.astype("<f4").tobytes())
    return path


def test_collector_concatenates_worker_frames(tmpdir):
    # Two workers, each emitting 2 records of 11 float32 (the record layout).
    a = (np.arange(22, dtype=np.float32) + 100.0)   # worker A frame
    b = (np.arange(22, dtype=np.float32) + 200.0)   # worker B frame

    work = tempfile.mkdtemp(dir=str(tmpdir))
    da = os.path.join(work, "a")
    db = os.path.join(work, "b")
    out = os.path.join(work, "out")
    os.makedirs(out)
    _write_frame(da, "a", a)
    _write_frame(db, "b", b)

    buffers = {
        "buf_a": {"kotekan_buffer": "standard", "metadata_pool": "none",
                  "num_frames": "buffer_depth", "frame_size": a.nbytes},
        "buf_b": {"kotekan_buffer": "standard", "metadata_pool": "none",
                  "num_frames": "buffer_depth", "frame_size": b.nbytes},
        "out_buf": {"kotekan_buffer": "standard", "metadata_pool": "none",
                    "num_frames": "buffer_depth", "frame_size": a.nbytes + b.nbytes},
    }
    stages = {
        "read_a": {"kotekan_stage": "rawFileRead", "buf": "buf_a", "base_dir": da,
                   "file_name": "a", "file_ext": "raw", "end_interrupt": True},
        "read_b": {"kotekan_stage": "rawFileRead", "buf": "buf_b", "base_dir": db,
                   "file_name": "b", "file_ext": "raw", "end_interrupt": True},
        "collect": {"kotekan_stage": "GnssRecordCollector",
                    "in_bufs": ["buf_a", "buf_b"], "out_buf": "out_buf"},
        "write": {"kotekan_stage": "rawFileWrite", "in_buf": "out_buf", "base_dir": out,
                  "file_name": "merged", "file_ext": "raw", "prefix_hostname": False,
                  "num_frames_per_file": 1, "exit_after_n_files": 1},
    }
    config = {"buffer_depth": 4, "log_level": "info", "instrument_name": "test"}
    runner.KotekanRunner(buffers=buffers, stages=stages, config=config).run()

    with open(os.path.join(out, "merged_0000000.raw"), "rb") as f:
        meta = struct.unpack("<I", f.read(4))[0]
        if meta:
            f.read(meta)
        merged = np.frombuffer(f.read(), dtype="<f4")

    assert merged.size == 44
    np.testing.assert_array_equal(merged[:22], a)   # worker A first
    np.testing.assert_array_equal(merged[22:], b)   # then worker B, in order
