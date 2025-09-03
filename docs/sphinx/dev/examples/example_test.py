import pytest
from kotekan import runner

# this is the equivalent of the pipeline config file for kotekan to run your test
params = {
    "num_elements": 7,
    "num_ev": 0,
    "total_frames": 128,
    "cadence": 10.0,
    "mode": "default",
    "variable_my_stage_needs": -1,
}

# this runs kotekan and yields the data you want to inspect
@pytest.fixture(scope="module")
def data(tmpdir_factory):

    # keep all the data this test produces in a tmp directory
    tmpdir = tmpdir_factory.mktemp("name_of_the_test_case")

    # you can use FakeVisBuffer as a mock producer stage
    # to produce a Kotekan buffer with fake data
    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"], mode=params["mode"]
    )

    # DumpVisBuffer can be used to dump data
    # afterwards for asserting expectations
    dump_buffer = runner.DumpVisBuffer(str(tmpdir))

    # KotekanStageTester is used to run your
    # kotekan stage with your config
    # Replace stage_type with the name of your stage
    test = runner.KotekanStageTester(
        stage_type="stageUnderTest",
        stage_config={},
        buffers_in=fakevis_buffer,
        buffers_out=dump_buffer,
        global_config=params,
    )

    test.run()

    # here the data that the stage under test
    # put out is passed on to test the stage
    yield dump_buffer.load()


# this is the actual test (give a name to it)
def test_<name>(data):

    for frame in data:
        assert frame.vis == {1, 0}
