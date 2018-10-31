import pytest
import numpy as np
import csv
import math
import random

import visbuffer
import kotekan_runner


params = {
    'num_elements': 20,
    'num_ev': 0,
    'total_frames': 12,
    'cadence': 1.0,
    'mode': 'test_pattern_simple',
    'buffer_depth': 5,
    'tolerance': 0.001,
    'report_freq': 60,
    'log_level': 'DEBUG'
}

@pytest.fixture(scope="module")
def test_pattern(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern")
    params['out_file'] = str(tmpdir) + '/report.csv'

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=params['total_frames'],
        mode='test_pattern_simple'
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTestPattern', {},
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

@pytest.fixture(scope="module")
def test_pattern_noise(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern_noise")
    params['out_file'] = str(tmpdir) + '/report.csv'

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=params['total_frames'],
        mode='test_pattern_simple',
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    fakevis_dump_conf = params.copy()
    fakevis_dump_conf['file_name'] = 'fakevis_dump'
    fakevis_dump_conf['file_ext'] = 'dump'
    fakevis_dump_conf['base_dir'] = str(tmpdir)

    test = kotekan_runner.KotekanProcessTester(
        'visTestPattern', {},
        buffers_in = fakevis_buffer,
        buffers_out = dump_buffer,
        global_config = params,
        parallel_process_type = 'rawFileWrite',
        parallel_process_config = fakevis_dump_conf,
        noise = True
    )

    test.run()

    out_data = []
    with open(params['out_file']) as csvfile:
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
        _errors = frame.vis - np.complex64(1 + 0j)
        _errors = np.absolute(_errors)

        num_bad = 0
        avg_error = 0.0
        errors = []
        for e in _errors:
            if e <= params['tolerance']:
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

freq_params = params.copy()

@pytest.fixture(scope="module")
def test_pattern_no_noise_freq(tmpdir_factory):
    freq_params['mode'] = "test_pattern_freq"
    freq_params['frequencies'] = [0,1,2,3,5,8,13,21,34,55]
    freq_params['freq_values'] = [
      {"real": 0, "imag":0},
      {"real": 1, "imag":-1},
      {"real": 2, "imag":-2},
      {"real": 3, "imag":-3},
      {"real": 5, "imag":-5},
      {"real": 8, "imag":-8},
      {"real": 13, "imag":-13},
      {"real": 21, "imag":-21},
      {"real": 34, "imag":-34},
      {"real": 55, "imag":-55}]
    freq_params['freq_ids'] = range(0,30)
    freq_params['total_frames'] = 2

    random.seed()
    for i in range(10):
        freq_params['freq_values'][i]['real'] = random.random() * 128
        freq_params['freq_values'][i]['imag'] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_freq_no_noise")
    freq_params['out_file'] = str(tmpdir) + '/report.csv'

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=freq_params['total_frames'],
        mode='test_pattern_freq',
        freq_ids = freq_params['freq_ids']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    test = kotekan_runner.KotekanProcessTester(
        'visTestPattern', {"num_freq": 30},
        buffers_in = fakevis_buffer,
        buffers_out = dump_buffer,
        global_config = freq_params,
    )

    test.run()

    out_data = []
    with open(freq_params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)


    yield (out_data, dump_buffer.load())

def test_no_noise_freq(test_pattern_no_noise_freq):
    assert (len(test_pattern_no_noise_freq[0]) == 0)
    assert (len(test_pattern_no_noise_freq[1]) == 0)

@pytest.fixture(scope="module")
def test_pattern_noise_freq(tmpdir_factory):
    freq_params['mode'] = "test_pattern_freq"
    freq_params['frequencies'] = [0,1,2,3,5,8,13,21,34,55]
    freq_params['freq_values'] = [
      {"real":  0, "imag":  0},
      {"real":  1, "imag": -1},
      {"real":  2, "imag": -2},
      {"real":  3, "imag": -3},
      {"real":  5, "imag": -5},
      {"real":  8, "imag": -8},
      {"real": 13, "imag":-13},
      {"real": 21, "imag":-21},
      {"real": 34, "imag":-34},
      {"real": 55, "imag":-55}]
    freq_params['freq_ids'] = range(0,30)
    freq_params['total_frames'] = 2
    freq_params['num_freq'] = 30

    random.seed()
    for i in range(10):
        freq_params['freq_values'][i]['real'] = random.random() * 128
        freq_params['freq_values'][i]['imag'] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_freq_noise")
    freq_params['out_file'] = str(tmpdir) + '/report.csv'

    fakevis_buffer = kotekan_runner.FakeVisBuffer(
        num_frames=freq_params['total_frames'],
        mode='test_pattern_freq',
        freq_ids = freq_params['freq_ids']
    )

    dump_buffer = kotekan_runner.DumpVisBuffer(
            str(tmpdir))

    fakevis_dump_conf = freq_params.copy()
    fakevis_dump_conf['file_name'] = 'fakevis_dump'
    fakevis_dump_conf['file_ext'] = 'dump'
    fakevis_dump_conf['base_dir'] = str(tmpdir)

    test = kotekan_runner.KotekanProcessTester(
        'visTestPattern', freq_params,
        buffers_in = fakevis_buffer,
        buffers_out = dump_buffer,
        global_config = freq_params,
        parallel_process_type = 'rawFileWrite',
        parallel_process_config = fakevis_dump_conf,
        noise = True
    )

    test.run()

    out_data = []
    with open(freq_params['out_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out_data.append(row)


    yield [out_data, visbuffer.VisBuffer.load_files("%s/*fakevis_dump*.dump"
        % str(tmpdir)), dump_buffer.load()]


def test_noise_freq(test_pattern_noise_freq):

    report = test_pattern_noise_freq[0]
    vis_in = test_pattern_noise_freq[1]
    vis_out = test_pattern_noise_freq[2]

    i_report = 0
    for i in range(0, len(vis_in)):
        frame = vis_in[i]

        freq_id = frame.metadata.freq_id
        if freq_id in freq_params['frequencies']:
            i = freq_params['frequencies'].index(freq_id)
            expected = np.complex64(freq_params['freq_values'][i]['real'] +
                                    freq_params['freq_values'][i]['imag'] *(1j))

        else:
            expected = np.complex64(128 + 0j)
        _errors = frame.vis - expected
        _errors = np.absolute(_errors)

        num_bad = 0
        avg_error = 0.0
        errors = []
        for e in _errors:
            if e <= freq_params['tolerance']:
                e = 0.0
            else:
                avg_error += e
                num_bad += 1
                errors.append(e)

        if num_bad:
            row = report[i_report]

            # if the frame was bad, it should also be in the output buffer
            assert np.all(frame.vis == vis_out.pop(0).vis)

            max_error = max(errors)
            min_error = min(errors)
            avg_error /= num_bad

            assert (float(row['avg_err']) == pytest.approx(avg_error, abs=1e-3))
            assert (float(row['min_err']) == pytest.approx(min_error, abs=1e-3))
            assert (float(row['max_err']) == pytest.approx(max_error, abs=1e-3))
            assert (int(row['num_bad']) == num_bad)
            i_report += 1

    # we should have popped all frames from the output buffer now
    assert (len(vis_out) == 0)
