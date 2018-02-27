
import pytest
import numpy as np

import kotekan_runner


merge_params = {
    'num_elements': 4,
    'num_eigenvectors': 0,
    'total_frames': 16,
    'cadence': 5.0,
    'fill_ij': True,
    'freq': list(range(3)),
    'buffer_depth': 5
}


@pytest.fixture(scope="module")
def merge_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("merge")

    fakevis_buffers = [
        kotekan_runner.FakeVisBuffer(
            freq=[f],
            num_frames=merge_params['total_frames']
        ) for f in merge_params['freq']
    ]

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visMerge', {},
        fakevis_buffers,
        dump_buffer,
        merge_params
    )

    test.run()

    yield dump_buffer.load()


def test_metadata(merge_data):

    freqs = [dump.metadata.freq_id for dump in merge_data]
    fpga_seq = [dump.metadata.fpga_seq for dump in merge_data]

    # Assert that all the frquencies appeared
    assert (np.bincount(freqs) == merge_params['total_frames']).all()

    # Check that they appeared in the right order (i.e. the timestamps occur
    # in blocks of num freq
    assert (np.ptp(np.array(fpga_seq).reshape(
                merge_params['total_frames'], -1
            ), axis=1) == 0.0).all()


def test_data(merge_data):

    rows, cols = np.triu_indices(merge_params['num_elements'])

    test_pattern = (rows + 1.0J * cols).astype(np.complex64)

    for dump in merge_data:
        assert (dump.vis == test_pattern).all()
