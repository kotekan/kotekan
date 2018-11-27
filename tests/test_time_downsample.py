import pytest
import numpy as np

import kotekan_runner

downsamp_params = {
    'num_elements': 4,
    'num_ev': 4,
    'num_samples': 2,
    'total_frames': 11,
    'fakevis_mode': 'fill_ij',
    'cadence': 2.
}

@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("vis_data")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'timeDownsample', downsamp_params,
        kotekan_runner.FakeVisBuffer(
            num_frames=downsamp_params['total_frames'],
            mode=downsamp_params['fakevis_mode'],
            cadence=downsamp_params['cadence']
        ),
        dump_buffer,
        downsamp_params
    )

    test.run()

    yield dump_buffer.load()


def test_structure(vis_data):

    n = downsamp_params['num_elements']

    # Check that each samples is the expected shape
    for frame in vis_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) / 2)
        assert (frame.metadata.num_ev == downsamp_params['num_ev'])

    # Check that we have the expected number of samples
    nsamp = downsamp_params['total_frames'] / downsamp_params['num_samples']
    assert len(vis_data) == nsamp


def test_metadata(vis_data):

    input_frame_length = int(800e6 / 2048 * downsamp_params['cadence']) 
    frame_length = input_frame_length * downsamp_params['num_samples']

    for frame in vis_data:
        assert frame.metadata.freq_id == 0
        assert frame.metadata.fpga_length == frame_length
        assert frame.metadata.fpga_total == frame_length


def test_time(vis_data):

    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    ctime = np.array([ timespec_to_float(v.metadata.ctime) for v in vis_data ])

    # Check downsampled cadence
    assert np.all(np.diff(ctime) == downsamp_params['cadence'] * downsamp_params['num_samples'])


def test_contents(vis_data):

    n = downsamp_params['num_elements']
    n_ev = downsamp_params['num_ev']

    # Reproduce expected fakeVis output
    model_vis = np.zeros(n * (n+1) / 2, dtype=np.complex64)
    ind = 0
    for i in range(n):
        for j in range(i, n):
            model_vis[ind] = i + j * 1j
            ind += 1
    model_evec = np.zeros(n_ev * n, dtype=np.complex64)
    model_eval = np.zeros(n_ev, dtype=np.float32)
    for i in range(n_ev):
        model_eval[i] = i
        for j in range(n):
            model_evec[i * n + j] = i + 1j * j

    # Averaging shouldn't change vis, eigenstuff
    for frame in vis_data:
        assert np.all(frame.vis == model_vis)
        assert np.all(frame.evec == model_evec)
        assert np.all(frame.eval == model_eval)
        assert frame.erms == 1.

    # weights get an extra factor of nsamp
    for frame in vis_data:
        assert np.all(frame.weight == downsamp_params['num_samples'])
