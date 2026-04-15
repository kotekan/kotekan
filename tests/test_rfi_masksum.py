import pytest
import numpy as np

from kotekan import runner

prod_config = {
    "buffer_depth": 5,
    "samples_per_data_set": 16384,
    "num_local_freq": 3,
    "sub_integration_ntime": 8192,
    "rfi_downsampling_factor": 256,
}

small_config = {
    "buffer_depth": 3,
    "samples_per_data_set": 1024,
    "num_local_freq": 3,
    "sub_integration_ntime": 32,
    "rfi_downsampling_factor": 1,
}

very_small_config = {
    "buffer_depth": 2,
    "samples_per_data_set": 1024,
    "num_local_freq": 1,
    "sub_integration_ntime": 1,
    "rfi_downsampling_factor": 1,
}


def generate_rfimask(vals, seq_num, num_times, num_freq, rfi_downsampling):
    """
    Generate an RFImask frame in a ChordBuffer

    Parameters
    ----------
    vals : List[List[0,1]]
        A list-of-lists containing 0 or 1. The outer list indexes frequency, the inner time. Each 0 or 1 is applied to `rfi_downsampling` number of time samples.
        Both frequency and time axes are cycled through if the list is shorter than the data array.
    seq_num : int
        FPGA sequence number at start of frame.
    num_times : int
        Number of times in this frame
    num_freq : int
        Number of frequencies in this frame
    rfi_downsampling : int
        RFI Downsampling factor: RFImask elements are repeated this many times
    """

    data = np.empty((num_times // 1024, num_freq, 128), dtype=np.uint8)
    meta = runner.chordbuffer.get_metadata(
        "RFImask", "uint1x8", ("T8hi128", "F", "T8lo128")
    )
    meta["fpga_seq_num"] = seq_num
    meta["time_downsampling_fpga"] = 1024

    for f in range(num_freq):
        val = vals[f % len(vals)]
        time_series = []
        for trfi in range(num_times // rfi_downsampling):
            idx = (trfi + seq_num // rfi_downsampling) % len(val)
            time_series.extend([val[idx]] * rfi_downsampling)
        rfimask_t = np.packbits(time_series, bitorder="little")

        data[:, f, :].flat = rfimask_t

    return runner.chordbuffer.ChordBuffer(data, meta)


def count_rfi(frame_idx, config, vals):
    """
    Count the number of bad samples that would be in the RFI mask made with `vals`.
    """

    # Number of unique RFI samples per frame
    nrfi = config["samples_per_data_set"] // config["rfi_downsampling_factor"]

    # Number of integrations per frame
    nint = config["samples_per_data_set"] // config["sub_integration_ntime"]

    # initialize count array to 0
    counts = np.zeros((nint, config["num_local_freq"]), dtype=int)

    # the RFI-time index this frame starts at
    trfi0 = frame_idx * nrfi

    # loop over freq
    for f in range(config["num_local_freq"]):
        # grab the vals for this freq
        val_f = vals[f % len(vals)]
        # loop over all RFI times in this frame
        for trfi in range(trfi0, trfi0 + nrfi):
            # the global t (counts samples)
            t = trfi * config["rfi_downsampling_factor"]
            # The integration t (counts correlator outputs)
            tint = t // config["sub_integration_ntime"]

            # val_f[idx] is the RFImask value at this time
            idx = trfi % len(val_f)

            # integration index
            int_idx = tint % nint

            # accumulate the counts
            counts[int_idx, f] += (1 - val_f[idx]) * config["rfi_downsampling_factor"]

    return counts


@pytest.fixture(
    scope="module",
    params=[
        (very_small_config, [[1, 0, 1, 1, 0]]),
        (small_config, [[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
        (
            prod_config,
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [1, 1, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ],
        ),
    ],
)
def setup(request):
    config, vals = request.param
    return config, vals


@pytest.fixture(scope="module")
def rfimasksum_data(tmpdir_factory, setup):

    config, vals = setup

    num_frames = 2 * config["buffer_depth"]

    rfimasks = [
        generate_rfimask(
            vals,
            seq_num,
            config["samples_per_data_set"],
            config["num_local_freq"],
            config["rfi_downsampling_factor"],
        )
        for seq_num in config["samples_per_data_set"] * np.arange(num_frames)
    ]
    for rfimask in rfimasks:
        print(rfimask.data.shape)

    tmpdir = tmpdir_factory.mktemp("rfimasksum")

    input_buffer = runner.ReadChordBuffer(str(tmpdir), rfimasks)
    input_buffer.write()

    dump_buffer = runner.DumpChordBuffer(
        str(tmpdir),
        shape=(
            config["samples_per_data_set"] // config["sub_integration_ntime"],
            config["num_local_freq"],
        ),
        dtype=np.int32,
        max_frames=num_frames,
    )

    test = runner.KotekanStageTester(
        "RfiMaskSum", {}, input_buffer, dump_buffer, config,
    )

    test.run()

    yield dump_buffer.load()


# @pytest.mark.parametrize("setup", [(very_small_config, [[1]]),
#                                   (small_config, [[1]]),
#                                   (prod_config, [[1]])], indirect=True)
def test_meta(rfimasksum_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfimasksum_data):

        assert frame.metadata["name"] == "RFImask_count"
        assert (frame.metadata["dim_names"] == ["Tc", "F"]).all()
        assert (
            frame.metadata["time_downsampling_fpga"] == config["sub_integration_ntime"]
        )
        assert frame.metadata["fpga_seq_num"] == idx * config["samples_per_data_set"]


# @pytest.mark.parametrize("setup", [(very_small_config, [[1]]),
#                                   (small_config, [[1]]),
#                                   (prod_config, [[1]])], indirect=True)
def test_structure(rfimasksum_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfimasksum_data):

        assert frame.data.shape == (
            config["samples_per_data_set"] // config["sub_integration_ntime"],
            config["num_local_freq"],
        )
        assert frame.data.dtype == np.int32


def test_count(rfimasksum_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfimasksum_data):

        count = count_rfi(idx, config, vals)

        assert (frame.data == count).all()
