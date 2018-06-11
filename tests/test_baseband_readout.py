import glob


import numpy as np
import h5py

import kotekan_runner


default_params = {
    'num_elements': 256,
    'total_frames': 60,
    'stream_id': 0,
    'buffer_depth': 20,
    'num_frames_buffer': 18,
    'type': 'const',
    'value': 153,
    'file_ext': '.h5',
    'samples_per_data_set': 4096,
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

    params = {'rest_mode': 'step'}

    rest_commands = [
            command_rest_frames(1),
            command_trigger(5000, 3000),
            command_trigger(15000, 11000),
            command_trigger(200000, 7000),
            command_rest_frames(60),
            ]
    dump_files = run_baseband(tmpdir_factory, params, rest_commands)

    num_elements = default_params['num_elements']
    for ii, f in enumerate(sorted(dump_files)):
        f = h5py.File(f, 'r')
        print dict(f.attrs)
        assert f['baseband'].shape == (rest_commands[1 + ii][2]['length'], num_elements)
        assert np.all(f['index_map/input'][:]['chan_id']
                      == np.arange(num_elements))
