
import pytest
import numpy as np
import h5py

import visbuffer
import kotekan_runner

writer_params = {
    'num_elements': 4,
    'num_ev': 2,
    'cadence': 5.0,
    'total_frames': 10,  # One extra sample to ensure we actually get 256
    'freq': [3, 777, 554],
    'write_freq': [3, 777]
}

@pytest.fixture(scope="module")
def written_data(tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp("writer"))

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=writer_params['freq'],
        num_frames=writer_params['total_frames'],
        cadence=writer_params['cadence']
    )

    params = writer_params.copy()
    params['root_path'] = tmpdir

    test = kotekan_runner.KotekanProcessTester(
        'visWriter', {'freq_ids': params['write_freq'], 'node_mode': False,
        'write_ev': True, 'file_type': 'raw'},
        fakevis_buffer,
        None,
        params
    )

    test.run()

    import glob

    files = sorted(glob.glob(tmpdir + '/20??????T??????Z_*_corr/*.meta'))

    yield [visbuffer.VisRaw(fname) for fname in files]


def test_vis(written_data):

    nt = writer_params['total_frames']

    for vr in written_data:

        # Construct vis array
        vis_data = [[fr.vis for fr in freq_data] for freq_data in vr.data]
        vis = np.array(vis_data)

        # Extract metadata
        ftime = np.array([t['fpga_count'] for t in vr.metadata['index_map']['time']])
        ctime = np.array([t['ctime'] for t in vr.metadata['index_map']['time']])
        freq = np.array([f['centre'] for f in vr.metadata['index_map']['freq']])

        # Check the diagonals are correct
        pi = 0
        for ii in range(writer_params['num_elements']):
            assert (vis[:, :, pi].imag == ii).all()
            pi += writer_params['num_elements'] - ii

        # Check the times are correct
        assert (vis[:, :, 0].real == ftime[:, np.newaxis].astype(np.float32)).all()
        assert (vis[:, :, 1].real == ctime[:, np.newaxis].astype(np.float32)).all()

        # Check the frequencies are correct
        vfreq = (800.0 - 400.0 * vis[:, :, 2].real / 1024)
        assert (vfreq == freq[np.newaxis, :]).all()


def test_metadata(written_data):

    nt = writer_params['total_frames']

    for vr in written_data:

        # Extract metadata
        ftime = np.array([t['fpga_count'] for t in vr.metadata['index_map']['time']])
        ctime = np.array([t['ctime'] for t in vr.metadata['index_map']['time']])
        freq = np.array([f['centre'] for f in vr.metadata['index_map']['freq']])
        input_a = np.array([p[0] for p in vr.metadata['index_map']['prod']])
        input_b = np.array([p[1] for p in vr.metadata['index_map']['prod']])
    
        # Check the number of samples has been written correctly
        assert vr.metadata['structure']['ntime'] == nt

        # Check the times
        assert np.allclose(np.diff(ctime), writer_params['cadence'])

        # Check the frequencies
        wfreq = 800.0 - 400.0 * np.array(writer_params['write_freq']) / 1024
        assert (freq == wfreq).all()

        # Check the products
        ia, ib = np.triu_indices(writer_params['num_elements'])
        assert (input_a == ia).all()
        assert (input_b == ib).all()


def test_eigenvectors(written_data):
    
    for vr in written_data:

        nt = writer_params['total_frames']
        nf = len(writer_params['write_freq'])
        ne = writer_params['num_ev']
        ni = writer_params['num_elements']

        evals = np.array([[fr.eval for fr in freq_data] for freq_data in vr.data])
        evecs = np.array([[fr.evec for fr in freq_data] for freq_data in vr.data])
        erms = np.array([[fr.erms for fr in freq_data] for freq_data in vr.data])

        # Check datasets are present
        assert evals.shape == (nt, nf, ne)
        assert evecs.shape == (nt, nf, ne * ni)
        assert erms.shape == (nt, nf, 1)

        evecs = evecs.reshape(nt, nf, ne, ni) 

        im_ev = np.array(vr.metadata['index_map']['ev'])

        print im_ev, ne

        # Check that the index map is there correctly
        assert (im_ev == np.arange(ne)).all()

        # Check that the datasets have the correct values
        assert (evals == np.arange(ne)[np.newaxis, np.newaxis, :]).all()
        assert (evecs.real == np.arange(ne)[np.newaxis, np.newaxis, :, np.newaxis]).all()
        assert (evecs.imag == np.arange(ni)[np.newaxis, np.newaxis, np.newaxis, :]).all()
        assert (erms == 1.0).all()
