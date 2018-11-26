import pytest
import numpy as np

import kotekan_runner


params = {
    'num_elements': 4,
    'num_ev': 4,
    'samples_per_data_set': 12,
    'total_frames': 7,
    'block_size': 1,
    'freq': 777,
    'num_gpu_buffers': 1
}

@pytest.fixture(scope="module")
def transform_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("transform")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTransform', {'num_ev': 4},
        [kotekan_runner.FakeGPUBuffer(
            mode='accumulate',
            freq=params['freq'],
            num_frames=params['total_frames']
        ) for g in range(params['num_gpu_buffers'])],
        dump_buffer,
        params
    )

    test.run()

    yield dump_buffer.load()


@pytest.fixture(scope="module")
def lostsamples_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("lostsamples")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTransform', {'num_ev': 4},
        [kotekan_runner.FakeGPUBuffer(
            mode='lostsamples',
            freq=params['freq'],
            num_frames=params['total_frames']
        ) for g in range(params['num_gpu_buffers'])],
        dump_buffer,
        params
    )

    test.run()

    yield dump_buffer.load()


def test_structure(transform_data):

    n = params['num_elements']

    # Check that each samples is the expected shape
    for frame in transform_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) / 2)
        assert (frame.metadata.num_ev ==
                params['num_ev'])

    # Check that we have the expected number of samples
    nsamp = (params['total_frames'] + 1) * params['num_gpu_buffers']
    assert len(transform_data) == nsamp


def test_metadata(transform_data):

    for frame in transform_data:
        assert frame.metadata.freq_id == params['freq']


def test_time(transform_data):

    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    t0 = timespec_to_float(transform_data[0].metadata.ctime)

    delta_samp = params['samples_per_data_set']

    for ii, frame in enumerate(transform_data):
        assert frame.metadata.fpga_seq == ii * delta_samp
        assert ((timespec_to_float(frame.metadata.ctime) - t0) ==
                pytest.approx(ii * delta_samp * 2.56e-6, abs=1e-5, rel=0))