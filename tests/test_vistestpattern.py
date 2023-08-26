# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import csv
import random
import copy
import sys

from kotekan import visbuffer
from kotekan import runner

ENDPOINT_NAME = "run_test"
REPORT_PRECISION = 1e-2

params = {
    "num_elements": 2,
    "num_ev": 0,
    "total_frames": 12,
    "freq_ids": [0, 1, 2],
    "cadence": 1.0,
    "mode": "test_pattern_simple",
    "buffer_depth": 5,
    "tolerance": 0.001,
    "report_freq": 60,
    "dataset_manager": {"use_dataset_broker": False},
    ENDPOINT_NAME: {
        "name": "none",
        "test_pattern": {"F44332211": [[1, 0], [1, 0]]},
        "num_frames": 0,
    },
    "log_level": "debug",
}

STAGE_NAME = "testpattern_stage"


def command_test_pattern(name, num_frames, test_pattern):
    return [
        (
            "post",
            ENDPOINT_NAME,
            {
                "reply_host": "none",
                "reply_port": 0,
                "reply_path": "none",
                "num_frames": num_frames,
                "test_pattern": test_pattern,
                "name": name,
            },
        )
    ]


def run_test(
    write_dir,
    rest_commands=None,
    params=params,
    noise=False,
    name="simple",
    expect_failure=False,
):
    params["write_dir"] = write_dir

    fakevis_buffer = runner.FakeVisBuffer(
        stage_name=STAGE_NAME,
        num_frames=params["total_frames"],
        mode=params["mode"],
        freq_ids=params["freq_ids"],
        wait=True,
    )

    dump_buffer = runner.DumpVisBuffer(write_dir)

    if noise:
        fakevis_dump_conf = params.copy()
        fakevis_dump_conf["file_name"] = "fakevis_dump"
        fakevis_dump_conf["file_ext"] = "dump"
        fakevis_dump_conf["base_dir"] = write_dir

        test = runner.KotekanStageTester(
            "visTestPattern",
            params,
            buffers_in=fakevis_buffer,
            buffers_out=dump_buffer,
            global_config=params,
            parallel_stage_type="rawFileWrite",
            parallel_stage_config=fakevis_dump_conf,
            noise=True,
            rest_commands=rest_commands,
            expect_failure=expect_failure,
        )
    else:
        test = runner.KotekanStageTester(
            "visTestPattern",
            params,
            fakevis_buffer,
            dump_buffer,
            params,
            rest_commands=rest_commands,
            expect_failure=expect_failure,
        )

    test.run()

    out_data = []
    out_file = params["write_dir"] + "/" + name + ".csv"
    try:
        with open(out_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                out_data.append(row)
    except:
        out_data = None

    if noise:
        in_data = visbuffer.VisBuffer.load_files("%s/*fakevis_dump*.dump" % write_dir)
    else:
        in_data = None

    return out_data, in_data, dump_buffer.load()


@pytest.fixture(scope="module")
def no_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("test_pattern")
    write_dir = str(tmpdir)
    yield run_test(write_dir)


def test_idle(no_data):
    # No command was sent to the endpoint, so no dump file should be created.
    assert no_data[0] == None
    assert no_data[1] == None
    assert len(no_data[2]) == 0


@pytest.fixture(scope="module")
def test_pattern(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("test_pattern")
    write_dir = str(tmpdir)

    simple_test_pattern = {
        "dm_input_0": [[1, 0], [1, 0], [1, 0]],
        "dm_input_1": [[1, 0], [1, 0], [1, 0]],
    }

    # Test 3 frames, to get one of each frequency
    rest_commands = command_test_pattern("simple", 3, simple_test_pattern)

    yield run_test(write_dir, rest_commands, expect_failure=False)


def test_no_noise(test_pattern):
    # A test was started by sending a command to the endpoint, so the files should exist.
    # But they should be empty, since there was no noise added to the test pattern data.
    assert len(test_pattern[0]) == 0
    assert test_pattern[1] == None
    assert len(test_pattern[2]) == 0


@pytest.fixture(scope="module")
def test_pattern_noise(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("test_pattern_noise")
    write_dir = str(tmpdir)

    simple_test_pattern = {
        "dm_input_0": [[1, 0], [1, 0], [1, 0]],
        "dm_input_1": [[1, 0], [1, 0], [1, 0]],
    }

    rest_commands = command_test_pattern("simple", 3, simple_test_pattern)

    yield run_test(
        write_dir, rest_commands=rest_commands, noise=True, expect_failure=False
    )


def test_noise(test_pattern_noise):

    report = test_pattern_noise[0]
    vis_in = test_pattern_noise[1]
    vis_out = test_pattern_noise[2]

    assert len(vis_out) <= 3
    assert len(vis_out) == len(report)

    i_report = 0

    vis_out_copy = copy.copy(vis_out)
    for frame_out in vis_out_copy:

        for f in vis_in:
            if (
                f.metadata.fpga_seq == frame_out.metadata.fpga_seq
                and f.metadata.freq_id == frame_out.metadata.freq_id
            ):
                frame = f
        assert frame is not None

        _errors = frame.vis - np.complex64(1 + 0j)
        _errors = np.absolute(_errors)

        num_bad = 0
        avg_error = 0.0
        errors = []
        for e in _errors:
            if e <= params["tolerance"]:
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

        row = report[i_report]
        i_report += 1
        assert float(row["avg_err"]) == pytest.approx(avg_error, abs=REPORT_PRECISION)
        assert float(row["min_err"]) == pytest.approx(min_error, abs=REPORT_PRECISION)
        assert float(row["max_err"]) == pytest.approx(max_error, abs=REPORT_PRECISION)
        assert int(row["num_bad"]) == num_bad

    # we should have popped all frames from the output buffer now
    assert len(vis_out) == 0


freq_params = params.copy()


@pytest.fixture(scope="module")
def test_pattern_no_noise_freq(tmpdir_factory):
    freq_params["mode"] = "test_pattern_freq"
    freq_params["frequencies"] = [0, 1, 2]
    freq_params["freq_values"] = [[0, 0], [1, 1], [2, -2]]
    freq_params["freq_ids"] = [0, 1, 2, 3]
    freq_params["total_frames"] = 12

    random.seed()
    for i in range(2):
        freq_params["freq_values"][i][0] = random.random() * 128
        freq_params["freq_values"][i][1] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_freq_no_noise")
    write_dir = str(tmpdir)

    # a complex value for each frequency (4) and each input channel (3)
    freq_test_pattern = {
        "dm_input_0": freq_params["freq_values"] + [[128, 0]],
        "dm_input_1": freq_params["freq_values"] + [[128, 0]],
    }
    rest_commands = command_test_pattern("freq", 4, freq_test_pattern)

    yield run_test(
        write_dir=write_dir,
        params=freq_params,
        rest_commands=rest_commands,
        noise=False,
        name="freq",
        expect_failure=False,
    )

@pytest.mark.skipif(sys.version_info > (3,7),
                    reason="Test currently only building on old python3.7")
def test_no_noise_freq(test_pattern_no_noise_freq):
    # A test was started by sending a command to the endpoint, so the files should exist.
    # But they should be empty, since there was no noise added to the test pattern data.
    assert len(test_pattern_no_noise_freq[0]) == 0
    assert test_pattern_no_noise_freq[1] == None
    assert len(test_pattern_no_noise_freq[2]) == 0


@pytest.fixture(scope="module")
def test_pattern_noise_freq(tmpdir_factory):
    freq_params["mode"] = "test_pattern_freq"
    freq_params["frequencies"] = [0, 1, 2, 3, 5, 8, 13, 21, 22, 28]
    freq_params["freq_values"] = [
        [0, 0],
        [1, -1],
        [2, -2],
        [3, -3],
        [5, -5],
        [8, -8],
        [13, -13],
        [21, -21],
        [34, -34],
        [55, -55],
    ]
    freq_params["freq_ids"] = list(range(0, 30))
    freq_params["total_frames"] = 12

    random.seed()
    for i in range(10):
        freq_params["freq_values"][i][0] = random.random() * 128
        freq_params["freq_values"][i][1] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_freq_noise")
    write_dir = str(tmpdir)

    freq_test_pattern = [[128, 0]] * 30

    for i in range(len(freq_params["freq_values"])):
        freq_test_pattern[freq_params["frequencies"][i]] = freq_params["freq_values"][i]

    # a complex value for each frequency (30) and each input channel (3)
    freq_test_pattern = {
        "dm_input_0": freq_test_pattern,
        "dm_input_1": freq_test_pattern,
    }
    rest_commands = command_test_pattern("freq", 3, freq_test_pattern)

    yield run_test(
        write_dir=write_dir,
        params=freq_params,
        rest_commands=rest_commands,
        noise=True,
        name="freq",
        expect_failure=False,
    )


def test_noise_freq(test_pattern_noise_freq):

    report = test_pattern_noise_freq[0]
    vis_in = test_pattern_noise_freq[1]
    vis_out = test_pattern_noise_freq[2]

    assert len(vis_out) <= 3
    assert len(vis_out) == len(report)

    i_report = 0
    num_to_compare = 0

    # Find the first frame that was tested.
    for i in range(0, len(vis_in)):
        frame = vis_in[i]

        if (
            frame.metadata.fpga_seq == vis_out[0].metadata.fpga_seq
            and frame.metadata.freq_id == vis_out[0].metadata.freq_id
            and num_to_compare == 0
        ):
            num_to_compare = 3

        if num_to_compare > 0:
            num_to_compare -= 1

            freq_id = frame.metadata.freq_id
            if freq_id in freq_params["frequencies"]:
                i = freq_params["frequencies"].index(freq_id)
                expected = np.complex64(
                    freq_params["freq_values"][i][0]
                    + freq_params["freq_values"][i][1] * (1j)
                )
            else:
                expected = np.complex64(128 + 0j)

            expected *= expected.conj()

            _errors = frame.vis - expected
            _errors = np.absolute(_errors)

            num_bad = 0
            avg_error = 0.0
            errors = []
            for e in _errors:
                if e <= freq_params["tolerance"]:
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

                assert int(row["num_bad"]) == num_bad
                assert float(row["avg_err"]) == pytest.approx(
                    avg_error, abs=REPORT_PRECISION
                )
                assert float(row["min_err"]) == pytest.approx(
                    min_error, abs=REPORT_PRECISION
                )
                assert float(row["max_err"]) == pytest.approx(
                    max_error, abs=REPORT_PRECISION
                )
                i_report += 1
            if num_to_compare == 0:
                break

    # we should have popped all frames from the output buffer now
    assert len(vis_out) == 0


input_params = freq_params.copy()


@pytest.fixture(scope="module")
def test_pattern_no_noise_inputs(tmpdir_factory):
    input_params["mode"] = "test_pattern_inputs"
    input_params["input_values"] = [[0, 0], [1, 1]]
    input_params["total_frames"] = 12

    random.seed()
    for i in range(2):
        input_params["input_values"][i][0] = random.random() * 128
        input_params["input_values"][i][1] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_input_no_noise")
    write_dir = str(tmpdir)

    # a complex value for each frequency (3) and each input channel (2)
    inputs_test_pattern = {
        "dm_input_0": [
            input_params["input_values"][0],
            input_params["input_values"][0],
            input_params["input_values"][0],
        ],
        "dm_input_1": [
            input_params["input_values"][1],
            input_params["input_values"][1],
            input_params["input_values"][1],
        ],
    }
    rest_commands = command_test_pattern("inputs", 2, inputs_test_pattern)

    yield run_test(
        write_dir=write_dir,
        params=input_params,
        rest_commands=rest_commands,
        noise=False,
        name="inputs",
        expect_failure=False,
    )


def test_no_noise_inputs(test_pattern_no_noise_inputs):
    # A test was started by sending a command to the endpoint, so the files should exist.
    # But they should be empty, since there was no noise added to the test pattern data.
    assert len(test_pattern_no_noise_inputs[0]) == 0
    assert test_pattern_no_noise_inputs[1] == None
    assert len(test_pattern_no_noise_inputs[2]) == 0


@pytest.fixture(scope="module")
def test_pattern_noise_inputs(tmpdir_factory):
    input_params["mode"] = "test_pattern_inputs"
    input_params["input_values"] = [[0, 0], [1, 1]]
    input_params["total_frames"] = 8

    random.seed()
    for i in range(2):
        input_params["input_values"][i][0] = random.random() * 128
        input_params["input_values"][i][1] = random.random() * 128

    tmpdir = tmpdir_factory.mktemp("test_pattern_input_no_noise")
    write_dir = str(tmpdir)

    # a complex value for each frequency (4) and each input channel (2)
    inputs_test_pattern = {
        "dm_input_0": [
            input_params["input_values"][0],
            input_params["input_values"][0],
            input_params["input_values"][0],
        ],
        "dm_input_1": [
            input_params["input_values"][1],
            input_params["input_values"][1],
            input_params["input_values"][1],
        ],
    }
    rest_commands = command_test_pattern("inputs", 2, inputs_test_pattern)

    yield run_test(
        write_dir=write_dir,
        params=input_params,
        rest_commands=rest_commands,
        noise=True,
        name="inputs",
        expect_failure=False,
    )


def test_noise_inputs(test_pattern_noise_inputs):

    report = test_pattern_noise_inputs[0]
    vis_in = test_pattern_noise_inputs[1]
    vis_out = test_pattern_noise_inputs[2]

    assert len(vis_out) <= 2
    assert len(vis_out) == len(report)

    i_report = 0

    vis_out_copy = copy.copy(vis_out)
    for frame_out in vis_out_copy:

        for f in vis_in:
            if (
                f.metadata.fpga_seq == frame_out.metadata.fpga_seq
                and f.metadata.freq_id == frame_out.metadata.freq_id
            ):
                frame = f
        assert frame is not None

        _errors = list()
        ind = 0
        for i in range(2):
            for j in range(i, 2):
                expected = np.complex64(
                    input_params["input_values"][i][0]
                    + input_params["input_values"][i][1] * 1j
                )
                expected *= np.complex64(
                    input_params["input_values"][j][0]
                    + input_params["input_values"][j][1] * 1j
                ).conj()
                _errors.append(frame.vis[ind] - expected)
                ind += 1
        _errors = np.absolute(_errors)

        num_bad = 0
        avg_error = 0.0
        errors = []
        for e in _errors:
            if e <= params["tolerance"]:
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

        row = report[i_report]
        i_report += 1
        assert float(row["avg_err"]) == pytest.approx(avg_error, abs=REPORT_PRECISION)
        assert float(row["min_err"]) == pytest.approx(min_error, abs=REPORT_PRECISION)
        assert float(row["max_err"]) == pytest.approx(max_error, abs=REPORT_PRECISION)
        assert int(row["num_bad"]) == num_bad

    # we should have popped all frames from the output buffer now
    assert len(vis_out) == 0
