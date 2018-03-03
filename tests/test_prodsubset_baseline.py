
import pytest
import numpy as np

import kotekan_runner


replace_params = {
    'num_elements': 16,
    'num_prod': 120,
    'num_eigenvectors': 2,
    'total_frames': 128,
    'cadence': 5.0,
    'fill_ij': True,
    'freq_ids': [250],
    'buffer_depth': 5
}

vis_params = {
    'type' : 'autos'
}
@pytest.fixture(scope="module")
def replace_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("replace")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        freq=replace_params['freq_ids'],
        num_frames=replace_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'prodSubset', vis_params,
        fakevis_buffer,
        dump_buffer,
        replace_params
    )

    test.run()

    yield dump_buffer.load()


def test_replace(replace_data):

    for frame in replace_data:
        print frame.metadata.freq_id, frame.metadata.fpga_seq
        print frame.vis
        print frame.evals

    for frame in replace_data:
        # With fill_ij, vis_ij = i+j*(1j)
        assert (frame.vis.real == frame.vis.imag).all()
        
    #    assert (frame.vis.real[1::2] ==
    #            np.array(frame.metadata.fpga_seq).astype(np.float32)).all()
    #    assert (frame.vis.imag == np.arange(frame.metadata.num_prod)).all()
