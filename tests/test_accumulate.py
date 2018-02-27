import pytest
import numpy as np

import kotekan_runner


accumulate_params = {
    'num_elements': 4,
    'num_eigenvectors': 4,
    'samples': 32768,
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

@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visAccumulate', {'num_eigenvectors': 4},
        kotekan_runner.FakeGPUBuffer(
            pattern='accumulate',
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
        'visAccumulate', {'num_eigenvectors': 4},
        kotekan_runner.FakeGPUBuffer(
            pattern='gaussian',
            freq=gaussian_params['freq'],
            num_frames=gaussian_params['total_frames']
        ),
        dump_buffer,
        gaussian_params
    )

    test.run()

    yield dump_buffer.load()


def test_structure(accumulate_data):

    n = accumulate_params['num_elements']

    # Check that each samples is the expected shape
    for dump in accumulate_data:
        assert dump.metadata.num_elements == n
        assert dump.metadata.num_prod == (n * (n + 1) / 2)
        assert dump.metadata.num_eigenvectors == accumulate_params['num_eigenvectors']

    # Check that we have the expected number of samples
    nsamp = accumulate_params['total_frames'] / accumulate_params['int_frames']
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for dump in accumulate_data:
        assert dump.metadata.freq_id == accumulate_params['freq']
        assert dump.metadata.dataset_id == 0


def test_time(accumulate_data):

    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    t0 = timespec_to_float(accumulate_data[0].metadata.ctime)

    delta_samp = accumulate_params['samples'] * accumulate_params['int_frames']

    for ii, dump in enumerate(accumulate_data):
        assert dump.metadata.fpga_seq == ii * delta_samp
        assert ((timespec_to_float(dump.metadata.ctime) - t0) ==
                pytest.approx(ii * delta_samp * 2.56e-6, abs=1e-5, rel=0))


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(accumulate_params['num_elements'])

    pat = (row + 1.0J * col).astype(np.complex64)

    for dump in accumulate_data:

        assert (dump.vis == pat).all()
        assert (dump.weight == 8.0).all()


# Test the the statistics are being calculated correctly
def test_gaussian(gaussian_data):

    vis_set = np.array([dump.vis for dump in gaussian_data])
    weight_set = np.array([dump.weight for dump in gaussian_data])

    assert np.allclose(vis_set.var(axis=0), 1e-6, rtol=1e-1, atol=0)
    assert np.allclose((1.0 / weight_set).mean(axis=0), 1e-6, rtol=1e-1, atol=0)
    assert np.allclose(vis_set.mean(axis=0), np.identity(4)[np.triu_indices(4)],
                       atol=1e-4, rtol=0)