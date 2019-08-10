import glob
import random
import pytest

import numpy as np
import h5py

from kotekan import runner


default_params = {
    'max_dump_samples': 3500,
    'num_elements': 256,
    'total_frames': 60,
    'stream_id': 0,
    'buffer_depth': 20,
    'num_frames_buffer': 18,
    'type': 'tpluse',
    'value': 153,
    'samples_per_data_set': 1024,
    'rest_mode': 'step',
     }


DATAGEN_PNAME = 'fakenetwork'


def command_rest_frames(num_frames):
    return ('post', DATAGEN_PNAME + '/generate_test_data', {'num_frames': num_frames})


def command_trigger(start, length, event_id=123456, file_path="", dm=0, dm_error=0):
    if start < 0:
        start_unix_seconds = start
        start_unix_nano = 0
    else:
        start_unix_seconds = 0
        start_unix_nano = start * 2560

    data = {
            "event_id": event_id,
            "start_unix_seconds": start_unix_seconds,
            "start_unix_nano": start_unix_nano,
            "duration_nano": length*2560,
            "dm": dm,
            "dm_error": dm_error,
            "file_path": file_path,
            }
    return ('post', 'baseband', data)


def wait(wait_time):
    return ('wait', wait_time, None)


def run_baseband(tdir_factory, params=None, rest_commands=None, expect_a_failure=False):

    p = dict(default_params)
    tmpdir = tdir_factory.mktemp("baseband")
    p['base_dir'] = str(tmpdir) + '/'

    if params:
        p.update(params)

    fake_buffer = runner.FakeNetworkBuffer(
            stage_name=DATAGEN_PNAME,
            num_frames=p['total_frames'],
            type=p['type'],
            )

    test = runner.KotekanStageTester(
        'basebandReadout', {},
        fake_buffer,
        None,
        p,
        rest_commands=rest_commands,
        expect_failure = expect_a_failure
    )

    test.run()

    dump_files = glob.glob(str(tmpdir) + '/*.h5')
    return dump_files


def test_fails_nonwritable(tmpdir_factory):

    params = {
            "base_dir": "/not/an/actual/directory",
            "rest_mode": "none",
            }

    import subprocess
    run_baseband(tmpdir_factory, params, expect_a_failure = True)


def test_io_errors_and_max_samples(tmpdir_factory):

    rest_commands = [
            command_rest_frames(1),
            wait(0.5),
            command_trigger(1437, 1839, 10, "doesnt_exist"),
            command_trigger(10457, 3237, 31),
            wait(0.1),
            command_rest_frames(25),
            # Give it some time to write the capture before shutdown.
            wait(1.),
            command_rest_frames(5),
            ]
    params = {
            'total_frames': 30,
            'max_dump_samples': 2123,
            }
    dump_files = run_baseband(tmpdir_factory, params, rest_commands)
    assert len(dump_files) == 1
    f = h5py.File(dump_files[0], 'r')
    assert f['baseband'].shape == (params['max_dump_samples'],
                                   default_params['num_elements'])


def test_negative_start_time(tmpdir_factory):
    """Test using the 'save whatever you have' mode of the baseband dump

    Using -1 as the trigger start point initiates the dump using the oldest
    frame available in the buffers.
    """

    rest_commands = [
            command_rest_frames(1),
            wait(0.5),
            command_trigger(-1, 3237, 31),
            wait(0.1),
            command_rest_frames(25),
            # Give it some time to write the capture before shutdown.
            wait(1.),
            command_rest_frames(5),
            ]
    params = {
            'total_frames': 30,
            }
    dump_files = run_baseband(tmpdir_factory, params, rest_commands)
    assert len(dump_files) == 1
    f = h5py.File(dump_files[0], 'r')
    assert f['baseband'].shape == (3237, default_params['num_elements'])


def test_basic(tmpdir_factory):

    rest_commands = [
            command_rest_frames(1),
            wait(0.5),
            command_trigger(1437, 1839, 10),
            command_trigger(20457, 3237, 17),
            command_trigger(41039, 2091, 31),
            wait(0.1),
            command_rest_frames(60),
            ]
    dump_files = run_baseband(tmpdir_factory, {}, rest_commands)
    assert len(dump_files) == 3

    num_elements = default_params['num_elements']
    for ii, f in enumerate(sorted(dump_files)):
        f = h5py.File(f, 'r')
        shape = f['baseband'].shape
        assert f.attrs['time0_fpga_count'] * 2560 == rest_commands[2 + ii][2]['start_unix_nano']
        assert f.attrs['event_id'] == rest_commands[2 + ii][2]['event_id']
        assert f.attrs['freq_id'] == 0
        assert shape == (rest_commands[2 + ii][2]['duration_nano']/2560, num_elements)
        assert np.all(f['index_map/input'][:]['chan_id']
                      == np.arange(num_elements))
        edata = f.attrs['time0_fpga_count'] + np.arange(shape[0], dtype=int)
        edata = edata[:, None] + np.arange(shape[1], dtype=int)
        edata = edata % 256
        assert np.all(f['baseband'][:] == edata)


def test_missed(tmpdir_factory):

    good_trigger = (2437, 3123)
    rest_commands = [
            command_rest_frames(21),
            wait(.5),
            command_trigger(*good_trigger),  # Catch part of this one.
            command_rest_frames(30),
            command_trigger(100, 100),       # Miss this one.
            command_trigger(1002, 112),      # Miss this one.
            command_trigger(1001, 300),      # Miss this one.
            command_trigger(81039, 7091),    # This one never arrives.
            command_rest_frames(10),
            ]
    dump_files = run_baseband(tmpdir_factory, {}, rest_commands)

    files = sorted(dump_files)
    assert len(files) == 1
    f = h5py.File(files[0], 'r')
    stime = f.attrs['time0_fpga_count']
    etime = good_trigger[0] + good_trigger[1]
    assert stime > good_trigger[0]
    assert stime < good_trigger[0] + good_trigger[1]
    assert f['baseband'].shape[0] == etime - stime


def test_bigdump(tmpdir_factory):

    rest_commands = [
            command_rest_frames(1),
            wait(0.5),
            command_trigger(1000, 25423),    # Bigger than ring buffer.
            command_rest_frames(60),
            ]
    dump_files = run_baseband(tmpdir_factory, {}, rest_commands)
    assert len(dump_files) == 1
    f = h5py.File(dump_files[0], 'r')
    assert f['baseband'].shape == (default_params['max_dump_samples'],
                                   default_params['num_elements'])


def test_overload_no_crash(tmpdir_factory):

    params = dict(default_params)
    params.update({
            'write_throttle': 2.,
            'samples_per_data_set': 16384,
            'num_elements': 128,
            'total_frames': 60,
            })

    rest_commands = [
            command_rest_frames(1),
            wait(0.5),
            ]
    random.seed()
    tf = params['total_frames']
    spd = params['samples_per_data_set']
    n = 30
    for ii in range(n):
        start = random.randrange(1, (ii * tf / n + 20) * spd)
        length = random.randrange(1, spd * 5)
        rest_commands += [command_trigger(start, length, (ii+1))]
    rest_commands += [command_rest_frames(params['total_frames'])]

    run_baseband(tmpdir_factory, params, rest_commands)
