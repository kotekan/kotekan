import pytest
import numpy as np

import kotekan_runner

merge_params = {
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
def merge_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("merge")

    fakevis_buffers = [
        kotekan_runner.FakeVisBuffer(
            freq_ids=[f],
            num_frames=merge_params['total_frames']
	    #mode = merge_params['mode']
        ) for f in merge_params['freq']
    ]

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'eigenVis', {},
        fakevis_buffers[0],
        dump_buffer,
        merge_params
    )


    test.run()
    yield dump_buffer.load()

def test_data(merge_data):

    rows, cols = np.triu_indices(merge_params['num_elements'])

    test_pattern = (rows + 1.0J * cols).astype(np.complex64)

    for frame in merge_data:
        print(frame.evals)
        print(frame.evecs)
	print(frame.vis)
	#assert (frame.vis == test_pattern).all()


