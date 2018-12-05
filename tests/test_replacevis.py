
import pytest
import numpy as np

from kotekan import runner


replace_params = {
    'num_elements': 16,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 5.0,
    'fill_ij': True,
    'freq': list(range(16)),
    'buffer_depth': 5,
    'dataset_manager': {
        'use_dataset_broker': False
    },
}


@pytest.fixture(scope="module")
def replace_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("replace")

    fakevis_buffer = runner.FakeVisBuffer(
        freq=replace_params['freq'],
        num_frames=replace_params['total_frames']
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanProcessTester(
        'replaceVis', {},
        fakevis_buffer,
        dump_buffer,
        replace_params
    )

    test.run()

    yield dump_buffer.load()


def test_replace(replace_data):

    # for frame in replace_data:
    #     print frame.metadata.freq_id, frame.metadata.fpga_seq
    #     print frame.vis

    for frame in replace_data:
        assert (frame.vis.real[0::2] == frame.metadata.freq_id).all()
        assert (frame.vis.real[1::2] ==
                np.array(frame.metadata.fpga_seq).astype(np.float32)).all()
        assert (frame.vis.imag == np.arange(frame.metadata.num_prod)).all()
