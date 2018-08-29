
import pytest
import numpy as np

import kotekan_runner


diag_global_params = {
    'num_elements': 16,
    'num_ev': 2,
    'total_frames': 128,
    'cadence': 2048.0,
    'mode': 'phase_ij',
    'freq_ids': [0, 250],
    'buffer_depth': 5,
    'use_dataset_manager': True
}

diag_process_params = {
    'stack_type': 'diagonal'
}

chime_global_params = {
    'num_elements': 2048,
    'num_ev': 2,
    'total_frames': 10,
    'wait': True,
    'cadence': 2.0,
    'mode': 'chime',
    'freq_ids': [0, 250, 500],
    'buffer_depth': 100,
    'use_dataset_manager': True
}

chime_process_params = {
    'stack_type': 'chime_in_cyl'
}

@pytest.fixture(scope="module")
def diagonal_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("diagonal")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=diag_global_params['freq_ids'],
        num_frames=diag_global_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'baselineCompression', diag_process_params,
        fakevis_buffer,
        dump_buffer,
        diag_global_params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def chime_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("chime")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq_ids=chime_global_params['freq_ids'],
        num_frames=chime_global_params['total_frames'],
        wait=True
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'baselineCompression', chime_process_params,
        fakevis_buffer,
        dump_buffer,
        chime_global_params
    )

    test.run()

    yield dump_buffer.load()


def test_metadata(diagonal_data):

    freq_ids = np.array([frame.metadata.freq_id for frame in diagonal_data])
    fpga_seqs = np.array([frame.metadata.fpga_seq for frame in diagonal_data])
    dset_ids = np.array([frame.metadata.dataset_id for frame in diagonal_data])
    nprod = np.array([frame.metadata.num_prod for frame in diagonal_data])

    assert (freq_ids.reshape((-1, 2)) == np.array([[0, 250]])).all()
    assert (dset_ids == 1).all()
    assert ((fpga_seqs.reshape((-1, 2)) / 800e6) ==
            (np.arange(diag_global_params['total_frames']))[:, np.newaxis]).all()
    assert (nprod == diag_global_params['num_elements']).all()


def test_chime(chime_data):

    nvis_chime = 4 * (4 * 256 - 1) + 6 * 4 * 511

    # This is the typical number of entries per polarisation (for XX, XY and YY, not YX)
    np1 = 4 * 256 + 6 * 511

    print "hello", chime_data

    for frame in chime_data:
        assert frame.vis.shape[0] == nvis_chime

        # Check that the entries in XX and XY are the same
        assert (frame.vis[:np1] == frame.vis[np1:(2 * np1)]).all()

        v1 = frame.vis[:np1]
        w1 = frame.weight[:np1]

        # Loop over all pairs of cylinders for XX
        for ci in range(4):
            for cj in range(ci, 4):

                # These numbers depend if we are within a cyl or not
                nv = 256 if ci == cj else 511  # Number of entries to compare
                lb = 0 if ci == cj else -255  # The most negative separation

                # A list of the feed separations in the NS dir
                d = np.arange(lb, 256)

                assert (v1[:nv] == (cj - ci + 1.0j * d)).all()
                assert (w1[:nv] == (256 - np.abs(d))).all()

                v1 = v1[nv:]
                w1 = w1[nv:]
