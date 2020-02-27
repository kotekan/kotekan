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
    "num_frb_total_beams": 12,
    "num_sub_freqs": 6,
    "total_frames": 16,
    "cadence": 10.0,
    "mode": "default",
    "buffer_depth": 5,
    "freq_ids": list(range(50)),
    "split_freq": 42,
    "dataset_manager": {"use_dataset_broker": False},
}


@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):

    tmpdir_l = tmpdir_factory.mktemp("freqsplit_lower")
    tmpdir_h = tmpdir_factory.mktemp("freqsplit_higher")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"],
        mode=params["mode"],
        freq_ids=params["freq_ids"],
        wait=False,
    )

    dump_buffer_l = runner.DumpVisBuffer(str(tmpdir_l))

    dump_buffer_h = runner.DumpVisBuffer(str(tmpdir_h))

    test = runner.KotekanStageTester(
        "freqSplit", {}, fakevis_buffer, [dump_buffer_l, dump_buffer_h], params
    )

    test.run()

    yield [dump_buffer_l.load(), dump_buffer_h.load()]


@pytest.fixture(scope="module")
def write_data(tmpdir_factory):

    tmpdir_l = tmpdir_factory.mktemp("freqsplit_write_lower")
    tmpdir_h = tmpdir_factory.mktemp("freqsplit_write_higher")

    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"],
        mode=params["mode"],
        freq_ids=params["freq_ids"],
        wait=False,
    )

    write_buffer_l = runner.VisWriterBuffer(str(tmpdir_l), "raw")
    write_buffer_h = runner.VisWriterBuffer(str(tmpdir_h), "raw")
    test = runner.KotekanStageTester(
        "freqSplit", {}, fakevis_buffer, [write_buffer_l, write_buffer_h], params
    )

    test.run()

    return [write_buffer_l.load(), write_buffer_h.load()]


def test_freqsplit(vis_data):
    vis_data_l = vis_data[0]
    vis_data_h = vis_data[1]

    f_lower = [f for f in params["freq_ids"] if f < params["split_freq"]]
    f_higher = [f for f in params["freq_ids"] if f >= params["split_freq"]]

    assert len(vis_data_l) == params["total_frames"] * len(f_lower)
    assert len(vis_data_h) == params["total_frames"] * len(f_higher)

    # Count frames by frequency
    counts = [0] * len(f_lower)
    for frame in vis_data_l:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in f_lower
        # keep track of number of frames so far
        counts[f_lower.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(f_lower)

    counts = [0] * len(f_higher)
    for frame in vis_data_h:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in f_higher
        # keep track of number of frames so far
        counts[f_higher.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(f_higher)


def test_write(write_data):
    write_data_l = write_data[0]
    write_data_h = write_data[1]

    f_lower = [f for f in params["freq_ids"] if f < params["split_freq"]]
    f_higher = [f for f in params["freq_ids"] if f >= params["split_freq"]]

    counts = [0] * len(f_lower)
    for t in range(params["total_frames"]):
        for f in range(len(f_lower)):
            # get freq ids from fakeVis
            fid = int(write_data_l.data[t, f]["vis"][2].real)
            assert fid in f_lower
            # Check the order
            assert fid == f_lower[f]
            # keep track of number of frames so far
            counts[f_lower.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(f_lower)

    counts = [0] * len(f_higher)
    for t in range(params["total_frames"]):
        for f in range(len(f_higher)):
            # get freq ids from fakeVis
            fid = int(write_data_h.data[t, f]["vis"][2].real)
            assert fid in f_higher
            # Check the order
            assert fid == f_higher[f]
            # keep track of number of frames so far
            counts[f_higher.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(f_higher)
