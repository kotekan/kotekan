import pytest
import os
from subprocess import Popen
import time
import shutil
import signal

from kotekan import runner

from test_dataset_broker import has_redis

params = {
    "num_elements": 5,
    "num_ev": 0,
    "total_frames": 4,
    "cadence": 10.0,
    "mode": "default",
    "buffer_depth": 5,
    "freq_ids": list(range(500, 550)),
    "subset_list": [500, 506, 507, 508, 511, 512, 513, 514, 535, 549],
    "use_dataset_manager": True,
    "log_level": "Debug",
}

params["dataset_manager"] = {
    "use_dataset_broker": True,
    "ds_broker_port": 12050,
    "ds_broker_host": "127.0.0.1",
}

params_fakevis = {
    "num_frames": params["total_frames"],
    "mode": params["mode"],
    "freq_ids": params["freq_ids"],
    "use_dataset_manager": True,
    "wait": False,
}


@pytest.fixture(scope="module")
def subset_data(tmpdir_factory):
    broker_path = shutil.which("comet")
    if not broker_path:
        pytest.skip(
            "Make sure PYTHONPATH is set to where the comet dataset broker is installed."
        )
    if not has_redis():
        pytest.skip("Redis is not available and so comet will fail")


    # run the dataset broker
    broker = Popen([broker_path, "--recover", "False"])
    time.sleep(1.5)

    try:
        tmpdir = tmpdir_factory.mktemp("freqsub_broker_fakevis")

        dump_buffer_gen = runner.DumpVisBuffer(str(tmpdir))

        test = runner.KotekanStageTester(
            "FakeVis", params_fakevis, None, dump_buffer_gen, params
        )

        test.run()

        data_gen = dump_buffer_gen.load()

        ### freqSubset ###
        tmpdir = tmpdir_factory.mktemp("freqsub_broker")

        ds_id = data_gen[0].metadata.dataset_id
        fakevis_buffer_subset = runner.FakeVisBuffer(
            num_frames=params["total_frames"],
            mode=params["mode"],
            freq_ids=params["freq_ids"],
            use_dataset_manager=True,
            wait=False,
            dataset_id="{:016x}{:016x}".format(ds_id[1], ds_id[0]),
        )

        dump_buffer_subset = runner.DumpVisBuffer(str(tmpdir))

        test = runner.KotekanStageTester(
            "freqSubset", {}, fakevis_buffer_subset, dump_buffer_subset, params
        )

        test.run()

        data_subset = dump_buffer_subset.load()

        ### freqSplit ###
        time.sleep(10)
        tmpdir = tmpdir_factory.mktemp("freqsplit_broker")

        ds_id = data_subset[0].metadata.dataset_id
        fakevis_buffer_split = runner.FakeVisBuffer(
            num_frames=params["total_frames"],
            mode=params["mode"],
            freq_ids=params["subset_list"],
            use_dataset_manager=True,
            wait=False,
            dataset_id="{:016x}{:016x}".format(ds_id[1], ds_id[0]),
        )

        dump_buffer_split_lower = runner.DumpVisBuffer(str(tmpdir))
        dump_buffer_split_higher = runner.DumpVisBuffer(str(tmpdir))

        test = runner.KotekanStageTester(
            "freqSplit",
            {},
            fakevis_buffer_split,
            (dump_buffer_split_lower, dump_buffer_split_higher),
            params,
        )

        test.run()

        data_split_lower = dump_buffer_split_lower.load()
        data_split_higher = dump_buffer_split_higher.load()

        ### 2 visWriter processes ###

        tmpdir = tmpdir_factory.mktemp("freqsub_write_lower")

        params_fakevis_write_lower = params_fakevis.copy()

        ds_id = data_split_lower[0].metadata.dataset_id

        params_fakevis_write_lower["dataset_id"] = "{:016x}{:016x}".format(
            ds_id[1], ds_id[0]
        )

        # the writer is not given the subset list, it get's it through the broker
        write_buffer_lower = runner.VisWriterBuffer(
            str(tmpdir), "raw", None, extra_config={"use_dataset_manager": True}
        )

        test = runner.KotekanStageTester(
            "FakeVis", params_fakevis_write_lower, None, write_buffer_lower, params
        )

        test.run()

        tmpdir = tmpdir_factory.mktemp("freqsub_write_higher")

        params_fakevis_write_higher = params_fakevis.copy()
        ds_id = data_split_higher[0].metadata.dataset_id

        params_fakevis_write_higher["dataset_id"] = "{:016x}{:016x}".format(
            ds_id[1], ds_id[0]
        )

        # the writer is not given the subset list, it get's it through the broker
        write_buffer_higher = runner.VisWriterBuffer(
            str(tmpdir), "raw", None, extra_config={"use_dataset_manager": True}
        )

        test = runner.KotekanStageTester(
            "FakeVis", params_fakevis_write_higher, None, write_buffer_higher, params
        )

        test.run()

        yield [
            data_gen,
            data_subset,
            data_split_lower,
            data_split_higher,
            write_buffer_lower.load(),
            write_buffer_higher.load(),
        ]
    finally:
        pid = broker.pid
        os.kill(pid, signal.SIGINT)
        broker.terminate()


@pytest.mark.serial
def test_subset_broker(subset_data):
    data_gen = subset_data[0]
    data_subset = subset_data[1]
    data_split_lower = subset_data[2]
    data_split_higher = subset_data[3]
    data_write_lower = subset_data[4]
    data_write_higher = subset_data[5]

    # basic checks on the data dumped from fakeVis and freqSubset
    subset_lower = [x for x in params["subset_list"] if x < 512]
    subset_higher = [x for x in params["subset_list"] if x >= 512]

    assert len(data_gen) == params["total_frames"] * len(params["freq_ids"])
    assert len(data_subset) == params["total_frames"] * len(params["subset_list"])
    assert len(data_split_lower) == params["total_frames"] * len(subset_lower)
    assert len(data_split_higher) == params["total_frames"] * len(subset_higher)

    # Count frames by frequency
    counts = [0] * len(params["subset_list"])
    for frame in data_subset:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in params["subset_list"]
        # keep track of number of frames so far
        counts[params["subset_list"].index(fid)] += 1

    assert counts == [params["total_frames"]] * len(params["subset_list"])

    # split lower
    counts = [0] * len(subset_lower)
    for frame in data_split_lower:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in subset_lower
        # keep track of number of frames so far
        counts[subset_lower.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(subset_lower)

    # split higher
    counts = [0] * len(subset_higher)
    for frame in data_split_higher:
        # get freq ids from fakeVis
        fid = int(frame.vis[2].real)
        assert fid in subset_higher
        # keep track of number of frames so far
        counts[subset_higher.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(subset_higher)

    # this is what it's really about:
    # check the data the visWriter wrote
    assert len(data_write_lower.valid_frames) == params["total_frames"]
    assert len(data_write_higher.valid_frames) == params["total_frames"]

    # split < 512
    assert data_write_lower.data.shape[1] == len(subset_lower)

    counts = [0] * len(subset_lower)
    for t in range(params["total_frames"]):
        for f in range(len(subset_lower)):
            # get freq ids from fakeVis
            fid = int(data_write_lower.metadata["freq_id"][t][f])
            assert fid in subset_lower
            # Check the order
            assert fid == subset_lower[f]
            # keep track of number of frames so far
            counts[subset_lower.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(subset_lower)

    # split > 512
    assert data_write_higher.data.shape[1] == len(subset_higher)

    counts = [0] * len(subset_higher)
    for t in range(params["total_frames"]):
        for f in range(len(subset_higher)):
            # get freq ids from fakeVis
            fid = int(data_write_higher.metadata["freq_id"][t][f])
            assert fid in subset_higher
            # Check the order
            assert fid == subset_higher[f]
            # keep track of number of frames so far
            counts[subset_higher.index(fid)] += 1

    assert counts == [params["total_frames"]] * len(subset_higher)
