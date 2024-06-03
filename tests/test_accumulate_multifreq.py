# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner


accumulate_params = {
    "num_elements": 4,
    "samples_per_data_set": 32768,
    "int_frames": 64,
    "total_frames": 257,  # One extra sample to ensure we actually get 256
    "block_size": 2,
    "freq": 777,
    "num_freq_in_frame": 4,
}

@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "N2Accumulate",
        {},
        runner.FakeGPUBuffer(
            pattern="multifreq",
            freq=accumulate_params["freq"],
            num_frames=accumulate_params["total_frames"],
        ),
        dump_buffer,
        accumulate_params,
    )

    test.run()

    yield dump_buffer.load()


def test_structure(accumulate_data):

    n = accumulate_params["num_elements"]

    # Check that each samples is the expected shape
    for frame in accumulate_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == 0
        print(
            (
                frame.metadata.fpga_seq,
                frame.metadata.freq_id,
                frame.metadata.fpga_length,
                frame.metadata.fpga_total,
            )
        )

    # Check that we have the expected number of samples
    nsamp = accumulate_params["total_frames"] // accumulate_params["int_frames"]
    nfreq = accumulate_params["num_freq_in_frame"]
    assert len(accumulate_data) == nsamp * nfreq


def test_metadata(accumulate_data):

    nfreq = accumulate_params["num_freq_in_frame"]

    # This assumes that that the frequencies have been labelled by the decoded
    # stream_id + the index into the buffer
    for ii, frame in enumerate(accumulate_data):
        fi = ii % nfreq
        assert frame.metadata.freq_id == accumulate_params["freq"] + fi


def test_time(accumulate_data):
    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    t0 = timespec_to_float(accumulate_data[0].metadata.ctime)

    delta_samp = (
        accumulate_params["samples_per_data_set"] * accumulate_params["int_frames"]
    )

    nfreq = accumulate_params["num_freq_in_frame"]

    # Assuming that the frequencies come in a group we can figure out the times
    # in the buffer
    for ii, frame in enumerate(accumulate_data):
        print(
            (
                frame.metadata.fpga_seq,
                frame.metadata.freq_id,
                frame.metadata.fpga_length,
                frame.metadata.fpga_total,
            )
        )
        ti = ii // nfreq
        assert frame.metadata.fpga_seq == ti * delta_samp
        assert (timespec_to_float(frame.metadata.ctime) - t0) == pytest.approx(
            ti * delta_samp * 2.56e-6, abs=1e-5, rel=0
        )
        assert frame.metadata.fpga_length == delta_samp
        assert frame.metadata.fpga_total == delta_samp


def test_accumulate(accumulate_data):

    nfreq = accumulate_params["num_freq_in_frame"]
    ninput = accumulate_params["num_elements"]
    prod_ind = np.arange(ninput * (ninput + 1) // 2)

    for ii, frame in enumerate(accumulate_data):

        freq_id = accumulate_params["freq"] + (ii % nfreq)

        print((ii, freq_id, frame.metadata.fpga_seq, frame.vis))
        assert (frame.vis.real == freq_id).all()
        assert (frame.vis.imag == prod_ind).all()
        assert (frame.weight == np.inf).all()
        assert (frame.flags == 1.0).all()
        assert (frame.gain == 1.0).all()
