
import pytest
import numpy as np
import h5py

from kotekan import runner

writer_params = {
    'num_elements': 4,
    'num_ev': 2,
    'cadence': 5.0,
    'total_frames': 10,
    'freq': [3, 777, 554],
    'write_freq': [3, 777]
}


def written_data_base(outdir, process_extra=None, root_extra=None):

    fakevis_buffer = runner.FakeVisBuffer(
        freq_ids=writer_params['freq'],
        num_frames=writer_params['total_frames'],
        cadence=writer_params['cadence']
    )

    root_params = writer_params.copy()
    root_params['root_path'] = outdir

    if root_extra is not None:
        root_params.update(root_extra)

    process_params = {
        'freq_ids': writer_params['write_freq'],
        'node_mode': False
    }

    if process_extra is not None:
        process_params.update(process_extra)

    test = runner.KotekanProcessTester(
        'visWriter', process_params,
        fakevis_buffer,
        None,
        root_params
    )

    test.run()

    import glob

    files = sorted(glob.glob(outdir + '/20??????T??????Z_*_corr/*.h5'))

    return [h5py.File(fname, 'r') for fname in files]


@pytest.fixture(scope="module")
def written_data_ev(request, tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("writer_ev")

    fhlist = written_data_base(str(tmpdir))

    yield fhlist

    for fh in fhlist:
        fh.close()


@pytest.fixture(scope="module")
def written_data_dm(request, tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("writer_dm")

    fhlist = written_data_base(str(tmpdir),
                               root_extra={'use_dataset_manager': True})

    yield fhlist

    for fh in fhlist:
        fh.close()


def test_vis(written_data_ev):

    nt = writer_params['total_frames']

    for fh in written_data_ev:
        vis = fh['vis'][:nt]

        # Check the diagonals are correct
        pi = 0
        for ii in range(writer_params['num_elements']):
            assert (vis[:, :, pi].imag == ii).all()
            pi += writer_params['num_elements'] - ii

        # Check the times are correct
        ftime = fh['index_map/time']['fpga_count'][:nt].astype(np.float32)
        ctime = fh['index_map/time']['ctime'][:nt].astype(np.float32)
        assert (vis[:, :, 0].real == ftime[:, np.newaxis]).all()
        assert (vis[:, :, 1].real == ctime[:, np.newaxis]).all()

        # Check the frequencies are correct
        freq = fh['index_map/freq']['centre']
        vfreq = (800.0 - 400.0 * vis[:, :, 2].real / 1024)
        assert (vfreq == freq[np.newaxis, :]).all()


def test_metadata(written_data_ev):

    nt = writer_params['total_frames']

    for fh in written_data_ev:

        # Check the number of samples has been written correctly
        assert fh.attrs['num_time'] == nt

        # Check that number of stacks is not there (unstacked data)
        assert 'num_stack' not in fh.attrs

        # Check the times
        ctime = fh['index_map/time']['ctime'][:nt]
        assert np.allclose(np.diff(ctime), writer_params['cadence'])

        # Check the frequencies
        freq = fh['index_map/freq']['centre']
        wfreq = 800.0 - 400.0 * np.array(writer_params['write_freq']) / 1024
        assert (freq == wfreq).all()

        # Check the products
        ia, ib = np.triu_indices(writer_params['num_elements'])
        assert (fh['index_map/prod']['input_a'] == ia).all()
        assert (fh['index_map/prod']['input_b'] == ib).all()


def test_eigenvectors(written_data_ev):

    for fh in written_data_ev:
        nt = writer_params['total_frames']
        nf = len(writer_params['write_freq'])
        ne = writer_params['num_ev']
        ni = writer_params['num_elements']

        evals = fh['eval'][:nt]
        evecs = fh['evec'][:nt]
        erms = fh['erms'][:nt]

        # Check datasets are present
        assert evals.shape == (nt, nf, ne)
        assert evecs.shape == (nt, nf, ne, ni)
        assert erms.shape == (nt, nf)

        # Check that the index map is there correctly
        assert (fh['index_map/ev'][:] == np.arange(ne)).all()

        # Check that the datasets have the correct values
        assert (evals == np.arange(ne)[np.newaxis, np.newaxis, :]).all()
        assert (evecs.real == np.arange(ne)[np.newaxis, np.newaxis, :, np.newaxis]).all()
        assert (evecs.imag == np.arange(ni)[np.newaxis, np.newaxis, np.newaxis, :]).all()
        assert (erms == 1.0).all()


def test_unwritten(written_data_ev):

    nt = writer_params['total_frames']

    for fh in written_data_ev:

        assert (fh['vis'][nt:] == 0.0).all()
        assert (fh['flags/vis_weight'][nt:] == 0.0).all()
        assert (fh['index_map/time'][nt:]['ctime'] == 0.0).all()
        assert (fh['index_map/time'][nt:]['fpga_count'] == 0).all()


def test_metadata_dataset_manager(written_data_dm):

    nt = writer_params['total_frames']

    for fh in written_data_dm:

        # Check the number of samples has been written correctly
        assert fh.attrs['num_time'] == nt

        # Check the times
        ctime = fh['index_map/time']['ctime'][:nt]
        assert np.allclose(np.diff(ctime), writer_params['cadence'])

        # Check the frequencies, this should have all the frequencies, not just
        # those in write_freq like in the default case
        freq = fh['index_map/freq']['centre']
        wfreq = 800.0 - 400.0 * np.array(writer_params['freq']) / 1024
        assert (freq == wfreq).all()

        # Check the products
        ia, ib = np.triu_indices(writer_params['num_elements'])
        assert (fh['index_map/prod']['input_a'] == ia).all()
        assert (fh['index_map/prod']['input_b'] == ib).all()

        # Check that the input names were fetched and written correctly from the datasetState
        inp = np.array(["dm_input_%i" % i for i in range(writer_params['num_elements'])])
        assert (fh['index_map/input']['correlator_input'] == inp).all()