
import pytest
import numpy as np

import kotekan_runner


merge_params = {
    'num_elements': 4,
    'num_ev': 0,
    'total_frames': 16,
    'cadence': 5.0,
    'mode': 'fill_ij',
    'freq': list(range(3)),
    'buffer_depth': 20,
    'timeout': 0.1
}


@pytest.fixture(scope="module")
def merge_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("merge")

    fakevis_buffers = [
        kotekan_runner.FakeVisBuffer(
            freq_ids=[f],
            num_frames=merge_params['total_frames']
        ) for f in merge_params['freq']
    ]

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'mergeBuffer', {},
        fakevis_buffers,
        dump_buffer,
        merge_params
    )

    test.run()

    yield dump_buffer.load()

# This test case is designed to test deadlock issues when frames are arriving
# at different rates. If the buffer deadlock occurs then we should only
# receiver 3 frames, otherwise we'll get 6.
@pytest.fixture(scope="module")
def mergewait_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("mergewait")

    fakevis_fast = kotekan_runner.FakeVisBuffer(
        freq_ids=[0],
        cadence=0.3,
        wait=True,
        num_frames=5
    )

    fakevis_slow = kotekan_runner.FakeVisBuffer(
        freq_ids=[1],
        cadence=5.0,
        wait=True,
        num_frames=10
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'mergeBuffer', {},
        [fakevis_fast, fakevis_slow],
        dump_buffer,
        merge_params
    )

    test.run()

    yield dump_buffer.load()


def test_metadata(merge_data):

    freqs = [frame.metadata.freq_id for frame in merge_data]
    fpga_seq = [frame.metadata.fpga_seq for frame in merge_data]

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

    for frame in merge_data:
        assert (frame.vis == test_pattern).all()


# Combine two streams where one produced data far more slowly that the other,
# this test is designed to check that the buffer wait function doesn't get
# stuck
def test_deadlock(mergewait_data):

    assert len(mergewait_data) == 6
