import pytest
import numpy as np

import kotekan_runner

trunc_params = {
    'fakevis_mode': 'phase_ij',
    'cadence': 2.,
    'total_frames': 10,
    'err_sq_lim': 0.003,
    'fixed_precision': 0.001,
    'num_ev': 4,
    'num_elements': 4
}

@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("vis_data")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTruncate', trunc_params,
        kotekan_runner.FakeVisBuffer(
            num_frames=trunc_params['total_frames'],
            mode=trunc_params['fakevis_mode'],
            cadence=trunc_params['cadence']
        ),
        dump_buffer,
        trunc_params
    )

    test.run()

    yield dump_buffer.load()

def test_truncation(vis_data):
    n = trunc_params['num_elements']

    # Reproduce expected fakeVis output
    model_vis = np.zeros(n * (n+1) / 2, dtype=np.complex64)
    ind = 0
    for i in range(n):
        for j in range(i, n):
            phase = i - j
            model_vis[ind] = np.cos(phase) + np.sin(phase) * 1j
            ind += 1

    for frame in vis_data:
        assert np.any(frame.vis != model_vis)
        assert np.all(np.abs(frame.vis - model_vis)
                      <= np.sqrt(trunc_params['err_sq_lim']))
        assert np.any(frame.weight != 1.)
        assert np.all(np.abs(frame.weight - 1.) <= trunc_params['fixed_precision'])
