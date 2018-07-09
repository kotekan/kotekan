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
    'num_elements': 4,
    'out_file': '/tmp/out.csv'
}

@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):
    """ Truncated visibilities """

    tmpdir = tmpdir_factory.mktemp("vis_data_t")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
            num_frames=trunc_params['total_frames'],
            mode=trunc_params['fakevis_mode'],
            cadence=trunc_params['cadence']);

    in_dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir),
            in_buf=fakevis_buffer.name)

    out_dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTruncate', trunc_params,
        buffers_in = fakevis_buffer,
        buffers_out = out_dump_buffer,
        global_config = trunc_params,
        buffers_extra = in_dump_buffer
    )

    test.run()

    yield (out_dump_buffer.load(), in_dump_buffer.load())

def test_truncation(vis_data):
    n = trunc_params['num_elements']

    for frame_t, frame in zip(vis_data[0], vis_data[1]):
        assert np.any(frame.vis != frame_t.vis)
        assert np.all(np.abs(frame.vis - frame_t.vis)
                      <= np.sqrt(trunc_params['err_sq_lim'] / frame.weight))
        assert np.any(frame.weight != frame_t.weight)
        assert np.all(np.abs(frame.weight - frame_t.weight)
                      <= frame.weight * trunc_params['weight_fixed_precision'])
        assert np.all(np.abs(frame.evec.real - frame_t.evec.real)
                      <= frame.evec.real * trunc_params['data_fixed_precision'])
        assert np.all(np.abs(frame.evec.imag - frame_t.evec.imag)
                      <= np.abs(frame.evec.imag) * trunc_params['data_fixed_precision'])
