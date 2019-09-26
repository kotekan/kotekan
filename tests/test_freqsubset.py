# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import os
import re

from kotekan import visbuffer
from kotekan import runner


params = {
    "num_elements": 5,
    "num_ev": 0,
    "total_frames": 16,
    "cadence": 10.0,
    "mode": "default",
    "buffer_depth": 5,
    "freq_ids": list(range(50)),
    "subset_list": [0, 12, 34, 35],
    "dataset_manager": {"use_dataset_broker": False},
}


@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("freqsub")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"],
        mode=params["mode"],
        freq_ids=params["freq_ids"],
        wait=False,
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "freqSubset", {}, fakevis_buffer, dump_buffer, params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def write_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("freqsub_write")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"],
        mode=params["mode"],
        freq_ids=params["freq_ids"],
        wait=False,
    )

    write_buffer = runner.VisWriterBuffer(str(tmpdir), "raw")

    test = runner.KotekanStageTester(
        "freqSubset", {}, fakevis_buffer, write_buffer, params
    )

    test.run()

    return write_buffer.load()


def test_freqslice(vis_data):

    assert len(vis_data) == params["total_frames"] * len(params["subset_list"])

    # Count frames by frequency
    counts = [0] * len(params["subset_list"])
    for frame in vis_data:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in params["subset_list"]
        # keep track of number of frames so far
        counts[params["subset_list"].index(fid)] += 1

    assert counts == [params["total_frames"]] * len(params["subset_list"])


def test_write(write_data):

    counts = [0] * len(params["subset_list"])
    for t in range(params["total_frames"]):
        for f in range(len(params["subset_list"])):
            # get freq ids from fakeVis
            fid = int(write_data.data[t, f]["vis"][2].real)
            assert fid in params["subset_list"]
            # Check the order
            assert fid == params["subset_list"][f]
            # keep track of number of frames so far
            counts[params["subset_list"].index(fid)] += 1

    assert counts == [params["total_frames"]] * len(params["subset_list"])
