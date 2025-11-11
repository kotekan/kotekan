# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import time

from kotekan import runner, visbuffer, visutil

root_params = {
    "log_level": "DEBUG",
    "num_elements": 16,
    "num_ev": 2,
    "num_prod": 136,
    "total_frames": 16,
    "cadence": 5.0,
    "freq_ids": [250],
    "buffer_depth": 5,
    "dataset_manager": {"use_dataset_broker": False},
}

count_params = {}

bad_feeds = [3, 4]


@pytest.fixture(scope="module")
def bad_weights():
    # Construct the weights for the frames. Where one of the inputs is a "bad feed" we
    # presume it is turned off and thus has zero variation and hence infinite weight
    num_elements = root_params["num_elements"]
    weight_square = np.ones((num_elements, num_elements), dtype=np.float32)
    for ind in bad_feeds:
        weight_square[:, ind] = np.inf
        weight_square[ind, :] = np.inf
    weight = weight_square[np.triu_indices(num_elements)]

    return weight


@pytest.fixture(scope="module")
def input_frames(bad_weights):

    counts_per_second = 390625
    start_time = time.time()

    # Generate a list of 'VisBuffer' objects.
    frame_list = []
    for ii in range(root_params["buffer_depth"]):
        frame_list.append(
            visbuffer.VisBuffer.new_from_params(
                root_params["num_elements"],
                root_params["num_prod"],
                root_params["num_ev"],
            )
        )
        frame_list[ii].metadata.fpga_seq = (
            counts_per_second * int(root_params["cadence"]) * ii
        )
        frame_list[ii].metadata.ctime.tv = int(
            start_time + int(root_params["cadence"]) * ii
        )

        frame_list[ii].vis[:] = 1.0
        frame_list[ii].weight[:] = bad_weights
        frame_list[ii].flags[:] = 1.0

    return frame_list


@pytest.fixture(scope="module")
def good_weights(bad_weights):
    return np.where(np.isinf(bad_weights), 0.0, bad_weights)


@pytest.fixture(scope="module")
def output_frames(tmpdir_factory, input_frames):

    tmpdir = tmpdir_factory.mktemp("flagbad")

    # ReadVisBuffer receives a list of frames and writes them down to disk.
    read_buffer = runner.ReadVisBuffer(str(tmpdir), input_frames)
    read_buffer.write()

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "BadInputFlag", {}, read_buffer, dump_buffer, root_params, expect_failure=True,
    )

    test.run()

    return dump_buffer.load()


def test_metadata(input_frames, output_frames):

    # The metadata should not change
    for in_frame, out_frame in zip(input_frames, output_frames):

        assert out_frame.metadata.fpga_seq == in_frame.metadata.fpga_seq
        assert out_frame.metadata.freq_id == in_frame.metadata.freq_id
        assert bytes(out_frame.metadata.dataset_id) == bytes(
            in_frame.metadata.dataset_id
        )


def test_data(input_frames, output_frames):

    for in_frame, out_frame in zip(input_frames, output_frames):

        # The data should not change
        assert np.all(out_frame.vis[:] == in_frame.vis[:])

        # The weights should stay the same
        assert np.all(out_frame.weight[:] == in_frame.weight[:])

        # But the flags should get marked with the bad feeds
        flags = in_frame.flags[:].copy()
        # TODO: at the moment we are not modifying the flags so they should not have
        # been modified. This will hopefully change.
        # flags[bad_feeds] = 0.0
        assert np.all(out_frame.flags[:] == flags)
