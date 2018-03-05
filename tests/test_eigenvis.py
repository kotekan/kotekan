import pytest
import numpy as np

import kotekan_runner

params = {
    'num_elements': 4,
    'num_eigenvectors': 1,
    'total_frames': 16,
    'cadence': 5.0,
    'mode': 'phase_ij',
    'freq': [0],
    'buffer_depth': 5,
    'num_diagonals_filled': 0
     }


@pytest.fixture(scope="module")
def eigen_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("eigenvis")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
            freq_ids=params['freq'],
            num_frames=params['total_frames'],
            mode=params['mode'],
            )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'eigenVis', {},
        fakevis_buffer,
        dump_buffer,
        params
    )

    test.run()
    yield dump_buffer.load()


def test_data(eigen_data):

    rows, cols = np.triu_indices(params['num_elements'])

    # test_pattern = (rows + 1.0J * cols).astype(np.complex64)

    for frame in eigen_data:
        print(frame.evals)
        print(frame.evecs)
        print(frame.vis)
        # assert (frame.vis == test_pattern).all()
