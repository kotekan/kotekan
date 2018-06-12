import glob
import random

import numpy as np
import h5py

import kotekan_runner


default_params = {
    'num_elements': 256,
    'total_frames': 60,
    'stream_id': 0,
    'buffer_depth': 20,
    'num_frames_buffer': 18,
    'type': 'tpluse',
    'value': 153,
    'file_ext': '.h5',
    'samples_per_data_set': 1024,
    'rest_mode': 'step',
     }


def command_rest_frames(num_frames):
    return ('post', 'testdata_gen', {'num_frames': num_frames})


def command_trigger(start, length, freq_id=0):
    data = {
            "start": start,
            "length": length,
            "freq_id": freq_id,
            }
    return ('post', 'baseband', data)


def wait(wait_time):
    return ('wait', wait_time, None)


def run_baseband(tdir_factory, params=None, rest_commands=None):

    p = dict(default_params)
    if params:
        p.update(params)

    tmpdir = tdir_factory.mktemp("baseband")
    p['base_dir'] = str(tmpdir) + '/'

    fakevis_buffer = kotekan_runner.FakeNetworkBuffer(
            num_frames=p['total_frames'],
            type=p['type'],
            )

    test = kotekan_runner.KotekanProcessTester(
        'basebandReadout', {},
        fakevis_buffer,
        None,
        p,
        rest_commands,
    )

    test.run()

    dump_files = glob.glob(str(tmpdir) + '/*.h5')
    return dump_files


def test_basic(tmpdir_factory):

    rest_commands = [
            command_rest_frames(1),
            command_trigger(1437, 1839),
            command_trigger(40457, 3237),
            command_trigger(51039, 2091),
            command_rest_frames(60),
            ]
    dump_files = run_baseband(tmpdir_factory, {}, rest_commands)
    trigger_starts = [rest_commands[i][2]['start'] for i in range(1, 4)]

    num_elements = default_params['num_elements']
    for ii, f in enumerate(sorted(dump_files)):
        f = h5py.File(f, 'r')
        shape = f['baseband'].shape
        assert shape == (rest_commands[1 + ii][2]['length'], num_elements)
        assert np.all(f['index_map/input'][:]['chan_id']
                      == np.arange(num_elements))
        edata = f.attrs['time0_fpga_count'] + np.arange(shape[0], dtype=int)
        edata = edata[:, None] + np.arange(shape[1], dtype=int)
        edata = edata % 256
        assert np.all(f['baseband'][:] == edata)
        assert f.attrs['time0_fpga_count'] in trigger_starts


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
            command_trigger(5437, 25423),    # Bigger than ring buffer.
            command_rest_frames(60),
            ]
    dump_files = run_baseband(tmpdir_factory, {}, rest_commands)
    assert len(dump_files) == 0


def test_overload_no_crash(tmpdir_factory):

    params = dict(default_params)
    params.update({
            'write_throttle': 2.,
            'samples_per_data_set': 16384,
            'num_elements': 128,
            'total_frames': 60,
            })

    rest_commands = [command_rest_frames(1)]
    random.seed()
    tf = params['total_frames']
    spd = params['samples_per_data_set']
    n = 30
    for ii in range(n):
        start = random.randrange(1, (ii * tf / n + 20) * spd)
        lenth = random.randrange(1, spd * 5)
        rest_commands += [command_trigger(start, lenth)]
    rest_commands += [command_rest_frames(params['total_frames'])]

    dump_files = run_baseband(tmpdir_factory, params, rest_commands)
    print dump_files
