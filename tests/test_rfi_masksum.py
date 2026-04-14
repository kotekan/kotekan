import pytest
import numpy as np

from kotekan import runner

global_config = {
    "buffer_depth": 5,
    "samples_per_data_set": 16384,
    "num_local_freq": 3,
    "sub_integration_ntime": 8192,
    "rfi_downsampling_factor": 256,
}


def generate_rfimask(val, seq_num, num_times, num_freq):

    data = np.empty((num_times // 1024, num_freq, 128), dtype=np.uint8)
    meta = runner.chordbuffer.get_metadata(
        "RFImask", "uint1x8", ("T8hi128", "F", "T8lo128")
    )
    meta["fpga_seq_num"] = seq_num
    meta["time_downsampling_fpga"] = 1024

    data.flat[:] = val

    print(data)

    return runner.chordbuffer.ChordBuffer(data, meta)


@pytest.fixture(scope="module")
def rfimasksum_data(tmpdir_factory):

    num_frames = 2 * global_config["buffer_depth"]

    rfimasks = [
        generate_rfimask(
            0,
            seq_num,
            global_config["samples_per_data_set"],
            global_config["num_local_freq"],
        )
        for seq_num in np.arange(
            0,
            num_frames * global_config["samples_per_data_set"],
            global_config["samples_per_data_set"],
        )
    ]

    tmpdir = tmpdir_factory.mktemp("rfimasksum")

    input_buffer = runner.ReadChordBuffer(str(tmpdir), rfimasks)
    input_buffer.write()

    dump_buffer = runner.DumpChordBuffer(
        str(tmpdir),
        shape=(
            global_config["samples_per_data_set"]
            // global_config["sub_integration_ntime"],
            global_config["num_local_freq"],
        ),
        dtype=np.int32,
        max_frames=num_frames,
    )

    test = runner.KotekanStageTester(
        "RfiMaskSum", {}, input_buffer, dump_buffer, global_config,
    )

    test.run()

    yield dump_buffer.load()


def test_meta(rfimasksum_data):

    for idx, frame in enumerate(rfimasksum_data):

        assert frame.metadata["name"] == "RFImask_count"
        assert (frame.metadata["dim_names"] == ["Tc", "F"]).all()
        assert (
            frame.metadata["time_downsampling_fpga"]
            == global_config["sub_integration_ntime"]
        )
        assert (
            frame.metadata["fpga_seq_num"]
            == idx * global_config["samples_per_data_set"]
        )


def test_structure(rfimasksum_data):

    for idx, frame in enumerate(rfimasksum_data):

        assert frame.data.shape == (
            global_config["samples_per_data_set"]
            // global_config["sub_integration_ntime"],
            global_config["num_local_freq"],
        )
        assert frame.data.dtype == np.int32


def test_count(rfimasksum_data):

    for idx, frame in enumerate(rfimasksum_data):

        print(frame.data)

        count = 1 * (global_config["sub_integration_ntime"] // 8)

        assert (frame.data == count).all()
