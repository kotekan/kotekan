import pytest
import numpy as np

import kotekan_runner
import visbuffer

subset_params = {
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
def subset_data(tmpdir_factory):

    counts_per_second = 390625
    tol = 3600 * counts_per_second

    tmpdir = tmpdir_factory.mktemp("subset")

    # Generate a list of 'VisBuffer' objects.
    frame_list = []
    for ii in range(subset_params['buffer_depth']):
        frame_list.append(visbuffer.VisBuffer.new_from_params(
                                        subset_params['num_elements'],
                                        subset_params['num_prod'],
                                        subset_params['num_ev']))
        frame_list[ii].metadata.fpga_seq = (
                        2 * tol + int(subset_params['cadence']) * ii)

    # Set last frame to be more than 'tol' before the previous one:
    frame_list[-1].metadata.fpga_seq = (
                        frame_list[-2].metadata.fpga_seq - (tol + 10))

    # ReadVisBuffer receives a list of frames and writes them down to disk.
    read_buffer = kotekan_runner.ReadVisBuffer(str(tmpdir), frame_list)
    read_buffer.write()

    test = kotekan_runner.KotekanProcessTester(
        'countCheck', count_params,
        read_buffer,
        None, # buffers_out is None
        #dump_buffer,
        subset_params
    )

    test.run()

    yield 1
#    yield dump_buffer.load()


def test_subset(subset_data):

#    for frame in subset_data:
#        print frame.metadata.freq_id, frame.metadata.fpga_seq
    pass