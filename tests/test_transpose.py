
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
    'chunk_size': [2, 6, 5],
    'mode': 'gaussian_random',
    'file_type': 'hdf5fast'
}

@pytest.fixture(scope="module")
def transpose(tmpdir_factory):

    writer_params['file_length'] = writer_params['total_frames']

    # Write fake data in raw format
    tmpdir = str(tmpdir_factory.mktemp("writer"))
    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=writer_params['freq'],
        num_frames=writer_params['total_frames'],
        cadence=writer_params['cadence'],
        mode=writer_params['mode'],
    )

    # Write fake data in hdf5 format
    tmpdir_h5 = str(tmpdir_factory.mktemp("dump_h5"))
    dumph5_conf = writer_params.copy()
    dumph5_conf['freq_ids'] = dumph5_conf['freq']
    dumph5_conf['root_path'] = str(tmpdir_h5)
    dumph5_conf['file_name'] = 'dumph5'
    dumph5_conf['write_ev'] = True
    dumph5_conf['node_mode'] = False

    params = writer_params.copy()
    params['root_path'] = tmpdir

    writer = kotekan_runner.KotekanProcessTester(
        'visWriter',
        {'freq_ids': params['freq'], 'node_mode': False, 'write_ev': True, 'file_type': 'raw'},
        fakevis_buffer,
        None,
        params,
        parallel_process_type = 'visWriter',
        parallel_process_config = dumph5_conf
    )

    writer.run()

    # get raw infile
    files = sorted(glob.glob(tmpdir + '/20??????T??????Z_*_corr/*.meta'))
    assert len(files) == 1
    infile = os.path.splitext(files[0])[0]

    # get hdf5 infile
    files = sorted(glob.glob(tmpdir_h5 + '/20??????T??????Z_*_corr/*.h5'))
    assert len(files) == 1
    infile_h5 = os.path.splitext(files[0])[0]

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

    fh = h5py.File(infile_h5 + '.h5', 'r')
    fh_t = h5py.File(outfile + '.h5', 'r')

    yield (fh_t, fh)

    fh.close()
    fh_t.close()


def test_transpose(transpose):

    # The transposed and untransposed files
    f_tr = transpose[0]
    f = transpose[1]

    # some useful params
    n_t = writer_params['total_frames']
    n_f = len(writer_params['freq'])
    n_elems = writer_params['num_elements']
    n_prod = n_elems * (n_elems + 1) / 2
    n_ev = writer_params['num_ev'];

    # get all the data
    vis = f['vis'].value
    vis_tr = f_tr['vis'].value

    weight = f['flags/vis_weight'].value
    weight_tr = f_tr['flags/vis_weight'].value

    eigenval = f['eval'].value
    eigenval_tr = f_tr['eval'].value

    evec = f['evec'].value
    evec_tr = f_tr['evec'].value

    erms = f['erms'].value
    erms_tr = f_tr['erms'].value

    # check if shapes are correct
    assert f_tr['index_map/time'].shape[0] == n_t
    assert f_tr['index_map/freq'].shape[0] == n_f
    assert f_tr['index_map/prod'].shape[0] == n_prod
    assert f_tr['index_map/input'].shape[0] == n_elems
    assert f_tr['vis'].shape == (n_f, n_prod, n_t)
    assert f_tr['flags/vis_weight'].shape == (n_f, n_prod, n_t)
    assert f_tr['eval'].shape == (n_f, writer_params['num_ev'], n_t)
    assert f_tr['evec'].shape == (n_f, writer_params['num_ev'], writer_params['num_elements'], n_t)
    assert f_tr['erms'].shape == (n_f, n_t)
    assert f_tr['gain'].shape == (n_f, n_elems, n_t)
    assert f_tr['flags/inputs'].shape == (n_elems, n_t)
    assert f_tr['flags/frac_lost'].shape == (n_f, n_t)

    # transpose with numpy and see if data is the same
    dsets = ['vis', 'flags/vis_weight',
             'eval', 'evec', 'erms']
    for d in dsets:
        assert np.all(f_tr[d][:] == np.moveaxis(f[d], 0, -1))