import pytest
import numpy as np

import kotekan_runner
import visbuffer

root_params = {
    'num_elements': 16,
    'num_prod': 136,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5
}

count_params = {}


@pytest.fixture(scope="module")
def kotekan_output(tmpdir_factory):

    counts_per_second = 390625
    tol = 3600 * counts_per_second

    tmpdir = tmpdir_factory.mktemp("countcheck")

    # Generate a list of 'VisBuffer' objects.
    frame_list = []
    for ii in range(root_params['buffer_depth']):
        frame_list.append(visbuffer.VisBuffer.new_from_params(
                                        root_params['num_elements'],
                                        root_params['num_prod'],
                                        root_params['num_ev']))
        frame_list[ii].metadata.fpga_seq = (
                        2 * tol + int(root_params['cadence']) * ii)

    # Set last frame to be more than 'tol' before the previous one:
    frame_list[-1].metadata.fpga_seq = (
                        frame_list[-2].metadata.fpga_seq - (tol + 10))

    # ReadVisBuffer receives a list of frames and writes them down to disk.
    read_buffer = kotekan_runner.ReadVisBuffer(str(tmpdir), frame_list)
    read_buffer.write()

    test = kotekan_runner.KotekanProcessTester(
        'countCheck', count_params,
        read_buffer,
        None,  # buffers_out is None
        root_params
    )

    test.run()

    yield test.output


def test_countcheck(kotekan_output):
    # Kotekan returns 0 whether 'countcheck' raises SIGINT or not.
    # So I can't check the return code to test 'countcheck'.
    # For now, I parse the output for specific messages from 'countcheck'
    # and for a SIGINT message. Not very elegant...
    msg = 'Current frame has FPGA count more than 1 hour behind previous one'
    assert ((msg in kotekan_output) and ('SIGINT' in kotekan_output))