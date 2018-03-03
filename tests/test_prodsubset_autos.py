
import pytest
import numpy as np

import kotekan_runner


replace_params = {
    'num_elements': 16,
    'num_prod': 120,
    'num_eigenvectors': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode':'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5
}

vis_params = {
    'type' : 'autos'
}
@pytest.fixture(scope="module")
def replace_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("replace")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq=replace_params['freq_ids'],
        num_frames=replace_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        dump_buffer,
        replace_params
    )

    test.run()

    yield dump_buffer.load()


def test_replace(replace_data):

    for frame in replace_data:
        print frame.metadata.freq_id, frame.metadata.fpga_seq
        print

    for frame in replace_data:
        # With fill_ij, vis_ij = i+j*(1j)
        assert (frame.vis.real == frame.vis.imag).all()    
        assert (frame.evals == np.arange(
                replace_params['num_eigenvectors'])).all()
        evecs = (np.arange(replace_params['num_eigenvectors'],
                    dtype=complex)[:,None]*replace_params['num_elements']
            +np.arange(replace_params['num_elements'])[None,:]).flatten()
        assert (frame.evecs == evecs).all()
        assert (frame.rms == 1.)

