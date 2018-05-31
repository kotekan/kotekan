import kotekan_runner


default_params = {
    'num_elements': 200,
    'total_frames': 10,
    'stream_id': 0,
    'buffer_depth': 20,
    'num_frames_buffer': 18,
    'type': 'const',
    'value': 153,
    'file_ext': '.h5',
     }


def run_baseband(tdir_factory, params=None, rest_commands=None):

    if not params:
        params = default_params

    tmpdir = tdir_factory.mktemp("baseband")
    params['base_dir'] = str(tmpdir)

    fakevis_buffer = kotekan_runner.FakeNetworkBuffer(
            num_frames=params['total_frames'],
            type=params['type'],
            )

    test = kotekan_runner.KotekanProcessTester(
        'basebandReadout', {},
        fakevis_buffer,
        None,
        params,
        rest_commands,
    )

    test.run()


def test_basic(tmpdir_factory):

    params = dict(default_params)
    params['rest_mode'] = 'start'
    rest_commands = [
            ('post', 'testdata_gen', {'num_frames': 110}),
            ]
    run_baseband(tmpdir_factory, params, rest_commands)
