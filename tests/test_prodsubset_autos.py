
import pytest
import numpy as np
import h5py
import glob

import kotekan_runner


subset_params = {
    'num_elements': 16,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode':'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5,
    'prod_subset_type': 'autos'
}

vis_params = {}

@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=subset_params['freq_ids'],
        num_frames=subset_params['total_frames'],
        use_dataset_manager=True
    )

    write_buffer = kotekan_runner.VisWriterBuffer(
        str(tmpdir), "raw",
        subset_params['freq_ids'],
        extra_config={'use_dataset_manager': True})

    test = kotekan_runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        write_buffer,
        subset_params
    )

    test.run()

    return write_buffer.load()


def test_subset(subset_data):

    for t in range(subset_params['total_frames']):
        for f in range(len(subset_params['freq_ids'])):
            frame = subset_data[t][f]
            # With fill_ij, vis_ij = i+j*(1j)
            assert np.all(frame.vis[:].real
                          == np.arange(subset_params['num_elements']))
            assert np.all(frame.vis[:].imag
                          == np.arange(subset_params['num_elements']))
            assert (frame.vis.real == frame.vis.imag).all()
            assert (frame.eval == np.arange(
                    subset_params['num_ev'])).all()
            evecs = (np.arange(subset_params['num_ev'])[:, None] +
                     1.0J * np.arange(subset_params['num_elements'])[None, :]).flatten()
            assert (frame.evec == evecs).all()
            assert (frame.erms == 1.)