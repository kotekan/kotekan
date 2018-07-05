import pytest
import numpy as np
import csv
import math

import kotekan_runner


params = {
    'num_elements': 200,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 10.0,
    'mode': 'test_pattern',
    'buffer_depth': 5,
    'tolerance': 0.001,
    'report_freq': 1000,
    'expected_val_real': 1.0,
    'expected_val_imag': 0.0,
    'out_file': '/tmp/out.csv'
}


@pytest.fixture(scope="module")
def test_pattern(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(output_dir=str(tmpdir),
            in_buf=fakevis_buffer.name)

    test = kotekan_runner.KotekanProcessTester(
        'visCheckTestPattern', {},
        fakevis_buffer,
        None,
        params
    )

    test.run()

    out_data = []
    with open(params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)

    yield out_data


def test_no_noise(test_pattern):

    # for frame in replace_data:
    #     print frame.metadata.freq_id, frame.metadata.fpga_seq
    #     print frame.vis

    for row in test_pattern:
        assert (float(row['avg_err']) == 0.0)
        assert (float(row['min_err']) == 0.0)
        assert (float(row['max_err']) == 0.0)

noise_params = {
    'num_elements': 200,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'test_pattern_noise',
    'buffer_depth': 2,
    'tolerance': 0.001,
    'report_freq': 1000,
    'expected_val_real': 1.0,
    'expected_val_imag': 0.0,
    'out_file': '/tmp/out.csv'
}

@pytest.fixture(scope="module")
def test_pattern_noise(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern_noise")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=noise_params['total_frames']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(output_dir=str(tmpdir),
            in_buf=fakevis_buffer.name)

    test = kotekan_runner.KotekanProcessTester(
        'visCheckTestPattern', {},
        fakevis_buffer,
        None,
        noise_params
    )

    test.run()

    out_data = []
    with open(noise_params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)


    yield (out_data, dump_buffer.load())


def test_noise(test_pattern_noise):

    report = test_pattern_noise[0]
    vis_in = test_pattern_noise[1]

    for frame,row in zip(vis_in,report):
        _errors = frame.vis - np.complex64(noise_params['expected_val_real'] +
            noise_params['expected_val_imag'] * 1j)
        _errors = np.absolute(_errors)

        num_bad = 0
        avg_error = 0.0
        errors = []
        for e in _errors:
            if e <= noise_params['tolerance']:
                e = 0.0
            else:
                avg_error += e
                num_bad += 1
                errors.append(e)
        max_error = max(errors)
        min_error = min(errors)
        avg_error /= num_bad

        assert (float(row['avg_err']) == pytest.approx(avg_error, abs=1e-5))
        assert (float(row['min_err']) == pytest.approx(min_error, abs=1e-5))
        assert (float(row['max_err']) == pytest.approx(max_error, abs=1e-5))
        assert (int(row['num_bad']) == num_bad)
