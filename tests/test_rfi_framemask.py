import pytest
import numpy as np

from kotekan import runner

prod_config = {
    "buffer_depth": 5,
    "samples_per_data_set": 16384,
    "num_local_freq": 3,
    "sub_integration_ntime": 8192,
    "rfi_downsampling_factor": 256,
    "updatable_config": {
        "rfi_enabled": {
            "kotekan_update_endpoint": "json",
            "enabled": True,
            "valid_from_time_ns": 0,
        },
        "rfi_thresholds": {
            "kotekan_update_endpoint": "json",
            "thresholds": [{"threshold": 1.0, "fraction": 0.5},],
            "valid_from_time_ns": 0,
        },
    },
}

small_config = {
    "buffer_depth": 3,
    "samples_per_data_set": 1024,
    "num_local_freq": 3,
    "sub_integration_ntime": 32,
    "rfi_downsampling_factor": 1,
    "updatable_config": {
        "rfi_enabled": {
            "kotekan_update_endpoint": "json",
            "enabled": True,
            "valid_from_time_ns": 0,
        },
        "rfi_thresholds": {
            "kotekan_update_endpoint": "json",
            "thresholds": [{"threshold": 1.0, "fraction": 0.5},],
            "valid_from_time_ns": 0,
        },
    },
}

very_small_config = {
    "buffer_depth": 2,
    "samples_per_data_set": 1024,
    "num_local_freq": 1,
    "sub_integration_ntime": 1,
    "rfi_downsampling_factor": 1,
    "updatable_config": {
        "rfi_enabled": {
            "kotekan_update_endpoint": "json",
            "enabled": True,
            "valid_from_time_ns": 0,
        },
        "rfi_thresholds": {
            "kotekan_update_endpoint": "json",
            "thresholds": [{"threshold": 1.0, "fraction": 0.5},],
            "valid_from_time_ns": 0,
        },
    },
}


def generate_sktilde(vals, seq_num, num_times, num_freq, rfi_downsampling):
    """
    Generate an SKtilde frame in a ChordBuffer

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

    rfi_num_times = num_times // rfi_downsampling

    data = np.empty((rfi_num_times, num_freq, 3), dtype=np.float32)
    meta = runner.chordbuffer.get_metadata("SKtilde", "float32", ("Trfi", "F", "SK"))
    meta["fpga_seq_num"] = seq_num
    meta["time_downsampling_fpga"] = rfi_downsampling

    data[:, :, :] = 0.0

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
def rfiframemask_data(tmpdir_factory, setup):

    config, vals = setup

    num_frames = 2 * config["buffer_depth"]

    sktildes = [
        generate_sktilde(
            vals,
            seq_num,
            config["samples_per_data_set"],
            config["num_local_freq"],
            config["rfi_downsampling_factor"],
        )
        for seq_num in config["samples_per_data_set"] * np.arange(num_frames)
    ]

    tmpdir = tmpdir_factory.mktemp("rfiframemask")

    input_buffer = runner.ReadChordBuffer(str(tmpdir), sktildes)
    input_buffer.write()

    dump_buffer = runner.DumpChordBuffer(
        str(tmpdir),
        shape=(
            config["samples_per_data_set"] // config["sub_integration_ntime"],
            config["num_local_freq"],
        ),
        dtype=np.uint8,
        max_frames=num_frames,
    )

    test = runner.KotekanStageTester(
        "RfiFrameMask",
        {
            "enabled_updatable_config": "/updatable_config/rfi_enabled",
            "thresholds_updatable_config": "/updatable_config/rfi_thresholds",
        },
        input_buffer,
        dump_buffer,
        config,
    )

    test.run()

    yield dump_buffer.load()


"""
def test_meta(rfimasksum_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfimasksum_data):

        assert frame.metadata["name"] == "RFImask_count"
        assert (frame.metadata["dim_names"] == ["Tc", "F"]).all()
        assert (
            frame.metadata["time_downsampling_fpga"] == config["sub_integration_ntime"]
        )
        assert frame.metadata["fpga_seq_num"] == idx * config["samples_per_data_set"]
"""


def test_structure(rfiframemask_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfiframemask_data):

        assert frame.data.shape == (
            config["samples_per_data_set"] // config["sub_integration_ntime"],
            config["num_local_freq"],
        )
        assert frame.data.dtype == np.uint8


"""
def test_count(rfimasksum_data, setup):

    config, vals = setup

    for idx, frame in enumerate(rfimasksum_data):

        count = count_rfi(idx, config, vals)

        assert (frame.data == count).all()
"""
