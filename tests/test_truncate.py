import pytest
import numpy as np

import kotekan_runner

trunc_params = {
    'fakevis_mode': 'gaussian',
    'cadence': 2.,
    'total_frames': 10,
    'err_sq_lim': 0.003,
    'weight_fixed_precision': 0.001,
    'data_fixed_precision': 0.0001,
    'num_ev': 4,
    'num_elements': 4
}

@pytest.fixture(scope="module")
def vis_data_t(tmpdir_factory):
    """ Truncated visibilities """

    tmpdir = tmpdir_factory.mktemp("vis_data_t")

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

@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):
    """ Raw visibilities. """

    tmpdir = tmpdir_factory.mktemp("vis_data")
    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    process_config = { 'num_frames': trunc_params['total_frames'],
            'freq_ids': [0], 'mode': trunc_params['fakevis_mode'] }
    process_config.update(trunc_params)
    test = kotekan_runner.KotekanProcessTester(
        'fakeVis', process_config,
        buffers_in=None,
        buffers_out=dump_buffer,
        global_config=trunc_params
    )

    test.run()

    yield dump_buffer.load()

def test_truncation(vis_data_t, vis_data):
    n = trunc_params['num_elements']

    for i, frame in enumerate(vis_data_t):
        assert np.any(frame.vis != vis_data[i].vis)
        assert np.all(np.abs(frame.vis - vis_data[i].vis)
                      <= np.sqrt(trunc_params['err_sq_lim'] / vis_data[i].weight))
        assert np.any(frame.weight != vis_data[i].weight)
        assert np.all(np.abs(frame.weight - vis_data[i].weight) 
                      <= vis_data[i].weight * trunc_params['weight_fixed_precision'])
        assert np.all(np.abs(frame.evec.real - vis_data[i].evec.real) 
                      <= vis_data[i].evec.real * trunc_params['data_fixed_precision'])
        assert np.all(np.abs(frame.evec.imag - vis_data[i].evec.imag) 
                      <= np.abs(vis_data[i].evec.imag) * trunc_params['data_fixed_precision'])
