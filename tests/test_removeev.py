
import pytest
import numpy as np

from kotekan import runner


remove_params = {
    'num_elements': 16,
    'num_ev': 4,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'default',
    'freq': list(range(16)),
    'buffer_depth': 5
}


@pytest.fixture(scope="module")
def remove_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("remove")

    fakevis_buffer = runner.FakeVisBuffer(
        freq=remove_params['freq'],
        num_frames=remove_params['total_frames']
    )

    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    test = runner.KotekanProcessTester(
        'removeEv', {},
        fakevis_buffer,
        dump_buffer,
        remove_params
    )

    test.run()

    yield dump_buffer.load()


def test_structure(remove_data):

    ne = remove_params['num_elements']

    for frame in remove_data:
        assert frame.metadata.num_elements == ne
        assert frame.metadata.num_prod == ne * (ne + 1) / 2
        assert frame.metadata.num_ev == 0


def test_vis(remove_data):

    for frame in remove_data:
        vis = frame.vis[:]

        # Check the diagonals are correct
        pi = 0
        for ii in range(remove_params['num_elements']):
            assert (vis.imag[pi] == ii).all()
            pi += remove_params['num_elements'] - ii

        # Check the times are correct
        ftime = frame.metadata.fpga_seq
        assert (vis.real[0] == np.array(ftime, dtype=np.float32)).all()

        # Check the frequencies are correct
        freq = frame.metadata.freq_id
        vfreq = vis[2].real
        assert (vfreq == freq).all()

        assert (frame.weight == 1.0).all()
        assert (frame.flags == 1.0).all()
        assert (frame.gain == 1.0).all()

