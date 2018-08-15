import pytest
import numpy as np
import csv
import math

import visbuffer
import kotekan_runner


params = {
    'num_elements': 200,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 10.0,
    'mode': 'gaussian',
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
        num_frames=params['total_frames'],
        vis_mean_real=1.0,
        vis_mean_imag=0.0,
        vis_std=0.0
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visCheckTestPattern', {},
        fakevis_buffer,
        dump_buffer,
        params
    )

    test.run()

    out_data = []
    with open(params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)

    yield (out_data, dump_buffer.load())


def test_no_noise(test_pattern):
    assert (len(test_pattern[0]) == 0)
    assert (len(test_pattern[1]) == 0)

noise_params = {
    'num_elements': 200,
    'num_ev': 0,
    'total_frames': 128,
    'cadence': 5.0,
    'mode': 'gaussian_random',
    'buffer_depth': 2,
    'tolerance': 0.01,
    'report_freq': 1000,
    'expected_val_real': 1.0,
    'expected_val_imag': 0.0,
    'out_file': '/tmp/out.csv'
}

@pytest.fixture(scope="module")
def test_pattern_noise(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern_noise")

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=noise_params['total_frames'],
        vis_mean_real=1.0,
        vis_mean_imag=0.0,
        vis_std=0.01
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    fakevis_dump_conf = noise_params.copy()
    fakevis_dump_conf['file_name'] = 'fakevis_dump'
    fakevis_dump_conf['file_ext'] = 'dump'
    fakevis_dump_conf['base_dir'] = str(tmpdir)

    test = kotekan_runner.KotekanProcessTester(
        'visCheckTestPattern', {},
        buffers_in = fakevis_buffer,
        buffers_out = dump_buffer,
        global_config = noise_params,
        parallel_process_type = 'rawFileWrite',
        parallel_process_config = fakevis_dump_conf
    )

    test.run()

    out_data = []
    with open(noise_params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)


    yield [out_data, visbuffer.VisBuffer.load_files("%s/*fakevis_dump*.dump"
        % str(tmpdir)), dump_buffer.load()]


def test_noise(test_pattern_noise):

    report = test_pattern_noise[0]
    vis_in = test_pattern_noise[1]
    vis_out = test_pattern_noise[2]

    for frame, row in zip(vis_in, report):
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

        if num_bad:
            # if the frame was bad, it should also be in the output buffer
            assert np.all(frame.vis == vis_out.pop(0).vis)

        max_error = max(errors)
        min_error = min(errors)
        avg_error /= num_bad

        assert (float(row['avg_err']) == pytest.approx(avg_error, abs=1e-5))
        assert (float(row['min_err']) == pytest.approx(min_error, abs=1e-5))
        assert (float(row['max_err']) == pytest.approx(max_error, abs=1e-5))
        assert (int(row['num_bad']) == num_bad)

    # we should have popped all frames from the output buffer now
    assert (len(vis_out) == 0)
