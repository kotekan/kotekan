
import pytest
import numpy as np
import h5py

import kotekan_runner

writer_params = {
    'num_elements': 4,
    'num_eigenvectors': 2,
    'cadence': 5.0,
    'total_frames': 10,  # One extra sample to ensure we actually get 256
    'freq': [3, 777, 554],
    'write_freq':[3, 777]
}

@pytest.fixture(scope="module")
def written_data(request, tmpdir_factory):

    write_eigen, = request.param

    tmpdir = tmpdir_factory.mktemp("writer")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=writer_params['freq'],
        num_frames=writer_params['total_frames'],
        cadence=writer_params['cadence']
    )

    params = writer_params.copy()
    params['root_path'] = str(tmpdir)

    test = kotekan_runner.KotekanProcessTester(
        'visWriter', {'freq_ids': params['write_freq'], 'node_mode': False,
        'write_eigen': write_eigen},
        fakevis_buffer,
        None,
        params
    )

    test.run()

    import glob

    files = sorted(glob.glob(str(tmpdir) + '/20??????T??????Z_*_corr/*.h5'))

    fhlist = [h5py.File(fname, 'r') for fname in files]

    yield fhlist

    for fh in fhlist:
        fh.close()


@pytest.mark.parametrize('written_data', [(False,)], indirect=True)
def test_vis(written_data):

    for fh in written_data:
        vis = fh['vis'][:]

        # Check the diagonals are correct
        pi = 0
        for ii in range(writer_params['num_elements']):
            assert (vis[:, :, pi].imag == ii).all()
            pi += writer_params['num_elements'] - ii

        # Check the times are correct
        ftime = fh['index_map/time']['fpga_count'].astype(np.float32)
        ctime = fh['index_map/time']['ctime'].astype(np.float32)
        assert (fh['vis'][:, :, 0].real == ftime[:, np.newaxis]).all()
        assert (fh['vis'][:, :, 1].real == ctime[:, np.newaxis]).all()

        # Check the frequencies are correct
        freq = fh['index_map/freq']['centre']
        vfreq = (800.0 - 400.0 * fh['vis'][:, :, 2].real / 1024)
        assert (vfreq == freq[np.newaxis, :]).all()


@pytest.mark.parametrize('written_data', [(False,)], indirect=True)
def test_metadata(written_data):

    for fh in written_data:

        # Check the times
        ctime = fh['index_map/time']['ctime']
        assert np.allclose(np.diff(ctime), writer_params['cadence'])

        # Check the frequencies
        freq = fh['index_map/freq']['centre']
        wfreq = 800.0 - 400.0 * np.array(writer_params['write_freq']) / 1024
        assert (freq == wfreq).all()

        # Check the products
        ia, ib = np.triu_indices(writer_params['num_elements'])
        assert (fh['index_map/prod']['input_a'] == ia).all()
        assert (fh['index_map/prod']['input_b'] == ib).all()


@pytest.mark.parametrize('written_data', [(False,)], indirect=True)
def test_no_eigenvectors(written_data):

    for fh in written_data:
        assert 'eigenvalues' not in fh
        assert 'eigenvectors' not in fh
        assert 'eigen_rms' not in fh


@pytest.mark.parametrize('written_data', [(True,)], indirect=True)
def test_eigenvectors(written_data):
    
    for fh in written_data:
        nt = writer_params['total_frames']
        nf = len(writer_params['write_freq'])
        ne = writer_params['num_eigenvectors']
        ni = writer_params['num_elements']

        assert fh['eigenvalues'].shape == (nt, nf, ne)
        assert fh['eigenvectors'].shape == (nt, nf, ne, ni)
        assert fh['eigen_rms'].shape == (nt, nf)
