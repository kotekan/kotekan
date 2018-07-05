
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

    params = writer_params.copy()
    params['root_path'] = tmpdir

    writer = kotekan_runner.KotekanProcessTester(
        'visWriter',
        {'freq_ids': params['freq'], 'node_mode': False, 'write_ev': True, 'file_type': 'raw',
            'file_length': writer_params['total_frames']},
        fakevis_buffer,
        None,
        params
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

    yield fh

    fh.close()


def test_transpose(transposed_data):

    # The transposed file
    f_tr = transposed_data

    nt = writer_params['total_frames']

    assert f_tr['index_map/time'].shape[0] == nt
