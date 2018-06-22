
import pytest
import numpy as np
import csv
import math

import kotekan_runner


params = {
    'num_elements': 16,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 5.0,
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

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

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
        assert (float(row['min_err']) == 0)
        assert (float(row['max_err']) == 0)

noise_params = {
    'num_elements': 16,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'test_pattern_noise',
    'buffer_depth': 5,
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

    dump_buffer = kotekan_runner.DumpVisBuffer(str(tmpdir))

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

    yield out_data


def test_noise(test_pattern_noise):

    # for frame in replace_data:
    #     print frame.metadata.freq_id, frame.metadata.fpga_seq
    #     print frame.vis

    # norm of 0.01 + j0.01
    max_noise = math.sqrt(pow(0.01,2)+pow(0.01,2))

    for row in test_pattern_noise:
        assert (float(row['avg_err']) <= max_noise)
        assert (float(row['avg_err']) >= 0)
        assert (float(row['min_err']) <= max_noise)
        assert (float(row['min_err']) >= 0)
        assert (float(row['max_err']) <= max_noise)
        assert (float(row['max_err']) >= 0)
        assert (float(row['max_err']) >= float(row['min_err']))

