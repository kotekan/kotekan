
import pytest
import numpy as np
import h5py
import glob

from kotekan import runner


subset_params = {
    'num_elements': 16,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'mode':'fill_ij',
    'freq_ids': [250],
    'buffer_depth': 5,
    'prod_subset_type': 'autos',
    'use_dataset_manager': False
}

vis_params = {}

@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("subset")

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=subset_params['freq_ids'],
        num_frames=subset_params['total_frames'],
        use_dataset_manager=False
    )

    write_buffer = runner.VisWriterBuffer(
        str(tmpdir), "raw",
        subset_params['freq_ids'],
        extra_config={'use_dataset_manager': False})

    test = runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        write_buffer,
        subset_params
    )

    test.run()

    return write_buffer.load()


def test_subset(subset_data):

    vis = (np.arange(subset_params['num_elements']) +
           1.0J * np.arange(subset_params['num_elements']))
    evecs = (np.arange(subset_params['num_ev'])[:, None] +
             1.0J * np.arange(subset_params['num_elements'])[None, :]).flatten()

    assert (subset_data.data['vis'] == np.array(vis)).all()
    assert (subset_data.data['eval'] == np.arange(
            subset_params['num_ev'])).all()
    assert (subset_data.data['evec'] == evecs).all()
    assert (subset_data.data['erms'] == 1.0).all()
