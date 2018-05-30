import kotekan_runner


default_params = {
    'num_elements': 200,
    'total_frames': 100,
    'stream_id': 0,
    'buffer_depth': 20,
    'num_frames_buffer': 18,
    'type': 'const',
    'value': 153,
     }


def run_baseband(tdir_factory, params=None):

    if not params:
        params = default_params

    # tmpdir = tdir_factory.mktemp("baseband")

    fakevis_buffer = kotekan_runner.FakeNetworkBuffer(
            num_frames=params['total_frames'],
            mode=params['mode'],
            )

    test = kotekan_runner.KotekanProcessTester(
        'basebandReadout', {},
        fakevis_buffer,
        None,
        params
    )

    test.run()


def test_basic(tmpdir_factory):

    params = dict(default_params)
    run_baseband(tmpdir_factory, params)
