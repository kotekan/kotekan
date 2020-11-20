# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner


global_params = {
    "num_elements": 16,
    "num_ev": 2,
    "total_frames": 2,
    "cadence": 2048.0,
    "mode": "phase_ij",
    "freq_ids": [0, 250],
    "buffer_depth": 5,
    "vis_pool": {
        "kotekan_metadata_pool": "VisMetadata",
        "num_metadata_objects": "500 * buffer_depth",
    },
    "dataset_manager": {"use_dataset_broker": False},
    "compress": {
        "kotekan_stage": "baselineCompression",
        "in_buf": "compress_buffer",
        "out_buf": "CHANGE_ME",
        "stack_type": "diagonal",
    },
    "compress_buffer": {
        "kotekan_buffer": "vis",
        "num_frames": "buffer_depth",
        "metadata_pool": "vis_pool",
    },
}


@pytest.fixture(scope="module")
def complete_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("no_drop")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=global_params["freq_ids"], num_frames=global_params["total_frames"]
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))
    global_params.update(dump_buffer.buffer_block)
    global_params.update(dump_buffer.stage_block)
    global_params["compress"]["out_buf"] = dump_buffer.name

    valve_params = {"out_buf": "compress_buffer"}

    test = runner.KotekanStageTester(
        "Valve", valve_params, fakevis_buffer, None, global_params
    )

    test.run()

    yield dump_buffer.load()


def test_complete(complete_data):
    n_frames = len(global_params["freq_ids"]) * global_params["total_frames"]
    assert len(complete_data) == n_frames


@pytest.fixture(scope="module")
def incomplete_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("drop")

    # Make the buffer flow over:
    global_params["total_frames"] = 100

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=global_params["freq_ids"], num_frames=global_params["total_frames"]
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))
    global_params.update(dump_buffer.buffer_block)
    global_params.update(dump_buffer.stage_block)
    global_params["compress"]["out_buf"] = dump_buffer.name

    valve_params = {"out_buf": "compress_buffer"}

    test = runner.KotekanStageTester(
        "Valve", valve_params, fakevis_buffer, None, global_params
    )

    test.run()

    yield dump_buffer.load()


def test_incomplete(incomplete_data):
    n_frames = len(global_params["freq_ids"]) * global_params["total_frames"]
    assert len(incomplete_data) < n_frames


def test_metadata(incomplete_data):

    for frame in incomplete_data:

        assert frame.metadata.freq_id == 0 or frame.metadata.freq_id == 250
        assert frame.metadata.num_prod == global_params["num_elements"]
