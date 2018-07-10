
import pytest
import numpy as np
import h5py
import glob
import os

import kotekan_runner

writer_params = {
    'num_elements': 4,
    'num_ev': 2,
    'cadence': 5.0,
    'total_frames': 10,  # One extra sample to ensure we actually get 256
    'freq': [3, 777, 554],
    'chunk_size': [2, 6, 5]
}

@pytest.fixture(scope="module")
def transposed_data(request, tmpdir_factory):

    # Write fake data in raw format
    tmpdir = str(tmpdir_factory.mktemp("writer"))

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=writer_params['freq'],
        num_frames=writer_params['total_frames'],
        cadence=writer_params['cadence'],
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(output_dir=str(tmpdir),
            in_buf = fakevis_buffer.name)

    params = writer_params.copy()
    params['root_path'] = tmpdir

    writer = kotekan_runner.KotekanProcessTester(
        'visWriter',
        {'freq_ids': params['freq'], 'node_mode': False, 'write_ev': True, 'file_type': 'raw',
            'file_length': writer_params['total_frames']},
        fakevis_buffer,
        None,
        params,
        dump_buffer
    )

    writer.run()

    files = sorted(glob.glob(tmpdir + '/20??????T??????Z_*_corr/*.meta'))
    assert len(files) == 1
    infile = os.path.splitext(files[0])[0]

    # Tranpose and write data
    raw_buf = kotekan_runner.ReadRawBuffer(infile, writer_params['chunk_size'])
    outfile = tmpdir + "/transposed"
    transposer = kotekan_runner.KotekanProcessTester(
        'visTranspose',
        {'outfile': outfile, 'infile': infile,
            'chunk_size': writer_params['chunk_size']},
        raw_buf,
        None,
        params
    )

    transposer.run()

    fh = h5py.File(outfile + '.h5', 'r')

    yield (fh, dump_buffer.load())

    fh.close()


def test_transpose(transposed_data):

    # The transposed file
    f_tr = transposed_data[0]
    f = transposed_data[1]

    n_t = writer_params['total_frames']
    n_f = len(writer_params['freq'])
    n_elems = writer_params['num_elements']
    n_prod = n_elems * (n_elems + 1) / 2
    n_ev = writer_params['num_ev'];

    assert f_tr['index_map/time'].shape[0] == n_t
    assert f_tr['index_map/freq'].shape[0] == n_f
    assert f_tr['vis'].shape == (n_f, n_prod, n_t)
    assert f_tr['flags/vis_weight'].shape == (n_f, n_prod, n_t)
    assert f_tr['eval'].shape == (n_f, writer_params['num_ev'], n_t)
    assert f_tr['evec'].shape == (n_f, writer_params['num_ev'], writer_params['num_elements'], n_t)
    assert f_tr['erms'].shape == (n_f, n_t)

    i = 0
    i_f = 0
    for frame in f:
        # compare vis
        assert frame.vis.shape == (n_prod,)
        m_in = frame.vis
        m_tr = f_tr['vis'].value[:,:,i_f]
        assert np.all(m_tr[i,:] == m_in)

        # compare weight
        assert np.all(frame.weight == f_tr['flags/vis_weight'].value[:,:,i_f])

        #compare eigenvalues
        assert np.all(frame.eval == f_tr['eval'].value[:,:,i_f])

        # compare eigenvectors
        for i_ev in range(0, n_ev):
            ev_in = f_tr['evec'].value[i,i_ev,:,i_f]
            ev_out = frame.evec[i_ev * n_elems : i_ev * n_elems + n_elems]
            assert np.all(ev_in == ev_out)

        # compare erms
        assert np.all(frame.erms == f_tr['erms'].value)

        i = i + 1
        if i == n_f:
            i_f  = i_f + 1
            i = 0
