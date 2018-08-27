
import pytest
import numpy as np

import kotekan_runner
import visutil

subset_params = {
    'num_elements': 16,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 2048.0,
    'mode': 'phase_ij',
    'freq_ids': [0, 250],
    'buffer_depth': 5,
    'use_dataset_manager': True
}

vis_params = {
    'stack_type': 'diagonal'
}

@pytest.fixture(scope="module")
def diagonal_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("diagonal")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=subset_params['freq_ids'],
        num_frames=subset_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'baselineCompression', vis_params,
        fakevis_buffer,
        dump_buffer,
        subset_params
    )

    test.run()

    yield dump_buffer.load()


def test_metadata(diagonal_data):

    freq_ids = np.array([frame.metadata.freq_id for frame in diagonal_data])
    fpga_seqs = np.array([frame.metadata.fpga_seq for frame in diagonal_data])
    dset_ids = np.array([frame.metadata.dataset_id for frame in diagonal_data])
    nprod = np.array([frame.metadata.num_prod for frame in diagonal_data])

    assert (freq_ids.reshape((-1, 2)) == np.array([[0, 250]])).all()
    assert (dset_ids == 0).all()
    assert ((fpga_seqs.reshape((-1, 2)) / 800e6) ==
            (np.arange(subset_params['total_frames']))[:, np.newaxis]).all()
    assert (nprod == subset_params['num_elements']).all()


def test_diagonal(diagonal_data):

    vis_diag = np.exp(-1.0J * np.arange(subset_params['num_elements']))

    for frame in diagonal_data:
        assert np.allclose(frame.vis, vis_diag.astype(np.complex64))
