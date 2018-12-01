import pytest
import numpy as np

import kotekan_runner


accumulate_params = {
    'num_elements': 4,
    'num_ev': 4,
    'samples_per_data_set': 32768,
    'int_frames': 64,
    'total_frames': 257,  # One extra sample to ensure we actually get 256
    'block_size': 2,
    'freq': 777
}

gaussian_params = accumulate_params.copy()
gaussian_params.update({
    'samples_per_data_set': 10000,
    'num_gpu_frames': 100,
    'total_frames': 200000
})

time_params = accumulate_params.copy()
time_params.update({'integration_time': 5.0})

pulsar_params = gaussian_params.copy()
pulsar_params.update({
    'mode': 'pulsar',
    'gaussian_bgnd': False,
    'wait': True,
    'samples_per_data_set': 3906,  # 10. ms frames
    'num_frames': 200,
    'integration_time': 0.5,
    'coeff': [0., 0.],
    'dm': 0.,
    't_ref': 58000.,
    'phase_ref': 0.,
    'rot_freq': 0.025e3,  # one period spans 4 frames
    'pulse_width': 1e-3,
})


@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', {'num_ev': 4},
        kotekan_runner.FakeGPUBuffer(
            mode='accumulate',
            freq=accumulate_params['freq'],
            num_frames=accumulate_params['total_frames']
        ),
        dump_buffer,
        accumulate_params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def gaussian_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("gaussian")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', {'num_ev': 4},
        kotekan_runner.FakeGPUBuffer(
            mode='gaussian',
            freq=gaussian_params['freq'],
            num_frames=gaussian_params['total_frames']
        ),
        dump_buffer,
        gaussian_params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def lostsamples_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("lostsamples")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', {'num_ev': 4},
        kotekan_runner.FakeGPUBuffer(
            mode='lostsamples',
            freq=accumulate_params['freq'],
            num_frames=accumulate_params['total_frames']
        ),
        dump_buffer,
        accumulate_params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def time_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("time")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', {'num_ev': 4},
        kotekan_runner.FakeGPUBuffer(
            mode='accumulate',
            freq=time_params['freq'],
            num_frames=time_params['total_frames']
        ),
        dump_buffer,
        time_params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def pulsar_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("pulsar")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))
    dump_buffer_gated = kotekan_runner.DumpVisBuffer(str(tmpdir))
    # Insert an extra buffer for gated stream
    dump_buffer.buffer_block.update(dump_buffer_gated.buffer_block)
    dump_buffer.process_block.update(dump_buffer_gated.process_block)

    acc_par = pulsar_params.copy()
    acc_par.update({
        'gating': {
            'psr0': {
                'mode': 'pulsar',
                'buf': dump_buffer_gated.name
            },
        },
        'updatable_config': {
            'psr0': "/updatable_config/psr0_config"
        },
        'num_gpu_frames': 16,
    })
    updatable_params = pulsar_params.copy()
    updatable_params.update({
        'updatable_config': {
            'psr0_config': {
                'kotekan_update_endpoint': 'json',
                'enabled': True,
                'pulsar_name': 'fakepsr',
                'segment': 100.,
                'coeff': [pulsar_params['coeff']],
                'dm': pulsar_params['dm'],
                't_ref': [pulsar_params['t_ref']],
                'phase_ref': [pulsar_params['phase_ref']],
                'rot_freq': pulsar_params['rot_freq'],  # one period spans 3 frames
                'pulse_width': pulsar_params['pulse_width'],
            }
        }
    })

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', acc_par,
        kotekan_runner.FakeGPUBuffer(**pulsar_params),
        dump_buffer,
        updatable_params
    )

    test.run()

    yield dump_buffer_gated.load()


def test_structure(accumulate_data):

    n = accumulate_params['num_elements']

    # Check that each samples is the expected shape
    for frame in accumulate_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) / 2)
        assert (frame.metadata.num_ev ==
                accumulate_params['num_ev'])

    # Check that we have the expected number of samples
    nsamp = accumulate_params['total_frames'] / accumulate_params['int_frames']
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for frame in accumulate_data:
        assert frame.metadata.freq_id == accumulate_params['freq']


def test_time(accumulate_data):

    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    t0 = timespec_to_float(accumulate_data[0].metadata.ctime)

    delta_samp = (accumulate_params['samples_per_data_set'] *
                  accumulate_params['int_frames'])

    for ii, frame in enumerate(accumulate_data):
        assert frame.metadata.fpga_seq == ii * delta_samp
        assert ((timespec_to_float(frame.metadata.ctime) - t0) ==
                pytest.approx(ii * delta_samp * 2.56e-6, abs=1e-5, rel=0))
        assert frame.metadata.fpga_length == delta_samp
        assert frame.metadata.fpga_total == delta_samp


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(accumulate_params['num_elements'])

    pat = (row + 1.0J * col).astype(np.complex64)

    for frame in accumulate_data:

        assert (frame.vis == pat).all()
        assert (frame.weight == 8.0).all()
        assert (frame.flags == 1.0).all()
        assert (frame.gain == 1.0).all()


# Test the the statistics are being calculated correctly
def test_gaussian(gaussian_data):

    vis_set = np.array([frame.vis for frame in gaussian_data])
    weight_set = np.array([frame.weight for frame in gaussian_data])

    assert np.allclose(vis_set.var(axis=0), 1e-6, rtol=1e-1, atol=0)
    assert np.allclose((1.0 / weight_set).mean(axis=0),
                       1e-6, rtol=1e-1, atol=0)
    assert np.allclose(vis_set.mean(axis=0),
                       np.identity(4)[np.triu_indices(4)],
                       atol=1e-4, rtol=0)


def test_int_time(time_data):

    time_per_frame = 2.56e-6 * time_params['samples_per_data_set']
    frames_per_int = (int(time_params['integration_time'] /
                          time_per_frame) / 2) * 2
    delta_samp = time_params['samples_per_data_set'] * frames_per_int
    fpga0 = time_data[0].metadata.fpga_seq

    for ii, frame in enumerate(time_data):
        assert (frame.metadata.fpga_seq - fpga0 == ii * delta_samp)
        assert frame.metadata.fpga_length == delta_samp
        assert frame.metadata.fpga_total == delta_samp


# Test that we are correctly normalising for lost packets
def test_lostsamples(lostsamples_data):

    row, col = np.triu_indices(accumulate_params['num_elements'])

    pat = (row + 1.0J * col).astype(np.complex64)

    for frame in lostsamples_data:

        assert np.allclose(frame.vis, pat, rtol=1e-7, atol=1e-8)


def test_pulsar(pulsar_data):

    assert len(pulsar_data) != 0
    print len(pulsar_data)
    for frame in pulsar_data:
        assert np.allclose(
            frame.vis,
            10.,
            rtol=1e-7, atol=1e-8
        )
