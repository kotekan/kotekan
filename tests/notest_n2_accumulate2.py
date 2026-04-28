import pytest
import numpy as np

from kotekan import runner
from kotekan.chordbuffer import ChordBuffer

prod_config = {
    "buffer_depth": 5,
    "samples_per_data_set": 16384,
    "num_local_freq": 12,
    "sub_integration_ntime": 8192,
    "num_polarizations": 2,
    "num_dishes": 512,
    "num_n2k_samples_to_accumulate": 2,
    "variance_mode": "EvenOddPosDef",
    "num_ev": 0,
}


@pytest.fixture(
    scope="module", params=[{"config": prod_config, "fail": False,},],
)
def setup(request):
    """
    Generate the base config and setup parameters to run a test. This (and the other fixtures) should provide *everything* needed to test a stage.

    Parameters
    ----------
    request : pytest magic
        Provides the parameterized values.

    Returns
    -------
    tuple
        The base config, as well as parameters to generate input PL mask data
    """
    config = request.param["config"]
    request.param["config"]["num_elements"] = (
        config["num_polarizations"] * config["num_dishes"]
    )
    request.param["num_frames"] = 2 * config["buffer_depth"]
    return request.param


@pytest.fixture(scope="module")
def corr_data(setup):
    """
    Generate a list of input PLmask frames in ChordBuffers

    Parameters
    ----------
    setup : fixture
        Includes the base config as well as PL mask generation parameters

    Returns
    -------
    List of ChordBuffers each containing a PL mask frame.
    """

    config = setup["config"]
    num_frames = setup["num_frames"]

    num_el = config["num_elements"]

    blocksize = 16

    num_blocks_lin = num_el // blocksize

    num_blocks = (num_blocks_lin * (num_blocks_lin + 1)) // 2

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    bufs = []

    shape = ((num_int, num_freq, num_blocks, blocksize, blocksize, 2),)

    for idx in range(num_frames):

        seq_num = idx * config["samples_per_data_set"]

        data = np.empty(*shape, dtype=np.int32)

        meta = runner.chordbuffer.get_metadata(
            "n2k_correlation", "int32", ("Tc", "F", "DPhi", "DPlo1", "DPlo2", "C")
        )
        meta["fpga_seq_num"] = seq_num
        meta["time_downsampling_fpga"] = config["sub_integration_ntime"]

        data[:, :, :, :, :, :] = 0.0

        bufs.append(ChordBuffer(data, meta))

    return bufs


@pytest.fixture(scope="module")
def count_data(setup):
    """
    Generate a list of input PLmask frames in ChordBuffers

    Parameters
    ----------
    setup : fixture
        Includes the base config as well as PL mask generation parameters

    Returns
    -------
    List of ChordBuffers each containing a PL mask frame.
    """

    config = setup["config"]
    num_frames = setup["num_frames"]

    num_el = config["num_elements"]

    blocksize = 8

    num_blocks_lin = (num_el // 8) // blocksize

    num_blocks = (num_blocks_lin * (num_blocks_lin + 1)) // 2

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    bufs = []

    shape = ((num_int, num_freq, num_blocks, blocksize, blocksize),)

    for idx in range(num_frames):

        seq_num = idx * config["samples_per_data_set"]

        data = np.empty(*shape, dtype=np.int32)

        meta = runner.chordbuffer.get_metadata(
            "n2k_counts", "int32", ("Tc", "F", "D8Phi", "D8Plo1", "D8Plo2")
        )
        meta["fpga_seq_num"] = seq_num
        meta["time_downsampling_fpga"] = config["sub_integration_ntime"]

        data[:, :, :, :, :] = config["sub_integration_ntime"]

        bufs.append(ChordBuffer(data, meta))

    return bufs


@pytest.fixture(scope="module")
def rficount_data(setup):
    """
    Generate a list of input PLmask frames in ChordBuffers

    Parameters
    ----------
    setup : fixture
        Includes the base config as well as PL mask generation parameters

    Returns
    -------
    List of ChordBuffers each containing a PL mask frame.
    """

    config = setup["config"]
    num_frames = setup["num_frames"]

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    bufs = []

    shape = ((num_int, num_freq),)

    for idx in range(num_frames):

        seq_num = idx * config["samples_per_data_set"]

        data = np.empty(*shape, dtype=np.int32)

        meta = runner.chordbuffer.get_metadata("RFImask_count", "int32", ("Tc", "F"))

        meta["fpga_seq_num"] = seq_num
        meta["time_downsampling_fpga"] = config["sub_integration_ntime"]

        data[:, :] = 0

        bufs.append(ChordBuffer(data, meta))

    return bufs


@pytest.fixture(scope="module")
def plcount_data(setup):
    """
    Generate a list of input PLmask frames in ChordBuffers

    Parameters
    ----------
    setup : fixture
        Includes the base config as well as PL mask generation parameters

    Returns
    -------
    List of ChordBuffers each containing a PL mask frame.
    """

    config = setup["config"]
    num_frames = setup["num_frames"]

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    bufs = []

    shape = ((num_int, num_freq),)

    for idx in range(num_frames):

        seq_num = idx * config["samples_per_data_set"]

        data = np.empty(*shape, dtype=np.int32)

        meta = runner.chordbuffer.get_metadata(
            "pl_lost_counts_scalar", "int32", ("Tc", "F")
        )

        meta["fpga_seq_num"] = seq_num
        meta["time_downsampling_fpga"] = config["sub_integration_ntime"]

        data[:, :] = 0

        bufs.append(ChordBuffer(data, meta))

    return bufs


@pytest.fixture(scope="module")
def rfiframemask_data(setup):
    """
    Generate a list of input PLmask frames in ChordBuffers

    Parameters
    ----------
    setup : fixture
        Includes the base config as well as PL mask generation parameters

    Returns
    -------
    List of ChordBuffers each containing a PL mask frame.
    """

    config = setup["config"]
    num_frames = setup["num_frames"]

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    bufs = []

    shape = ((num_int, num_freq),)

    for idx in range(num_frames):

        seq_num = idx * config["samples_per_data_set"]

        data = np.empty(*shape, dtype=np.uint8)

        meta = runner.chordbuffer.get_metadata("RFIFrameMask", "uint8", ("Tc", "F"))
        meta["fpga_seq_num"] = seq_num
        meta["time_downsampling_fpga"] = config["sub_integration_ntime"]

        data[:, :] = 1

        bufs.append(ChordBuffer(data, meta))

    return bufs


@pytest.fixture(scope="module")
def accum_data(
    tmpdir_factory,
    setup,
    corr_data,
    count_data,
    rficount_data,
    plcount_data,
    rfiframemask_data,
):
    """
    Run the CountLostPLSamplesScalar stage on the given input and yield the output buffers.

    Parameters
    ----------
    tmpdir_factory : pytest magic
        Generates temporary directories
    setup : fixture
        Includes the base config
    plmask_data : fixture
        A List of ChordBuffers for input

    Yields
    ------
    List of ChordBuffers each containing an PL Lost Counts frame generated by CountLostPLSamples.
    """

    config = setup["config"]

    tmpdir = tmpdir_factory.mktemp("n2accum")

    # Make input buffer and write the files for it to read.
    corr_buffer = runner.ReadChordBuffer(str(tmpdir), corr_data)
    corr_buffer.write()
    count_buffer = runner.ReadChordBuffer(str(tmpdir), count_data)
    count_buffer.write()
    rficount_buffer = runner.ReadChordBuffer(str(tmpdir), rficount_data)
    rficount_buffer.write()
    plcount_buffer = runner.ReadChordBuffer(str(tmpdir), plcount_data)
    plcount_buffer.write()
    rfiframemask_buffer = runner.ReadChordBuffer(str(tmpdir), rfiframemask_data)
    rfiframemask_buffer.write()

    # Make the output buffer we'll read from
    dump_buffer = runner.DumpN2Buffer(
        str(tmpdir),
        exit_after_n_files=setup["num_frames"],
        num_elements=config["num_elements"],
        num_ev=config["num_ev"],
    )

    # The test stage!
    test = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": config["num_local_freq"],
            "packet_loss_is_scalar": True,
        },
        {
            "in_buf": corr_buffer,
            "in_counts_buf": count_buffer,
            "in_rficounts_buf": rficount_buffer,
            "in_plcounts_buf": plcount_buffer,
            "in_rfiframemask_buf": rfiframemask_buffer,
        },
        dump_buffer,
        config,
        expect_failure=setup["fail"],
    )

    test.run()

    yield dump_buffer.load()


def count_lost_samples(plmask, config):
    """
    Count the lost (plmask = 0) samples in the PL mask, assuming values are the same for each element.

    Parameters
    ----------
    plmask : ChordBuffer
        The input PL mask
    config : dict
        The base config for the run

    Returns
    -------
    ndarray (num_integrations, num_frequency), int32
        The count of lost (0'd) samples in the PL mask, expanded to the full frequency range, downsampled by sub_integration_ntime in time.
    """

    # Number of integrations per frame
    nint = config["samples_per_data_set"] // config["sub_integration_ntime"]

    nf = config["num_local_freq"]

    # Allocate data array
    lost_counts = np.empty((nint, nf), dtype=np.int32)

    # Loop over integrations
    for tint in range(nint):

        # Start and stop of sub integration in global time
        t0 = tint * config["sub_integration_ntime"]
        t1 = (tint + 1) * config["sub_integration_ntime"]

        # t0 and t1 in the "slow" PL mask time.
        tpl0 = t0 // 128
        tpl1 = t1 // 128

        # grab the pl slice for our integtation, count the 1's in each u8, cast to int32s, then
        # sum over remaining time indices.
        lost_counts_f4 = config["sub_integration_ntime"] - 2 * np.bitwise_count(
            plmask.data[tpl0:tpl1, :, 0, 0, :]
        ).astype(np.int32).sum(axis=(0, 2))

        # place the counts into appropriate frequency bins.
        for f in range(config["num_local_freq"]):
            lost_counts[tint, f] = lost_counts_f4[f // 4]

    return lost_counts


def test_meta(accum_data, setup):
    # Test the metadata

    config = setup["config"]

    for idx, frame in enumerate(accum_data):

        assert frame.metadata["name"] == "pl_lost_counts_scalar"
        assert (frame.metadata["dim_names"] == ["Tc", "F"]).all()
        assert (
            frame.metadata["time_downsampling_fpga"] == config["sub_integration_ntime"]
        )
        assert frame.metadata["fpga_seq_num"] == idx * config["samples_per_data_set"]


"""
def test_pl_lost_counts(pllostcounts_data, setup, plmask_data):
    # Test the actual lost sample count.

    config = setup["config"]

    # Should have same number of frames!
    if not setup["fail"]:
        assert len(pllostcounts_data) == len(plmask_data)

    for idx, frame in enumerate(pllostcounts_data):

        # Check output is the right shape and type
        assert frame.data.shape == (
            config["samples_per_data_set"] // config["sub_integration_ntime"],
            config["num_local_freq"],
        )
        assert frame.data.dtype == np.int32

        # Generate the correct counts
        pl_lost_counts = count_lost_samples(plmask_data[idx], config)

        # Confirm the Kotekan and Python counts have same shape and type
        assert frame.data.shape == pl_lost_counts.shape
        assert frame.data.dtype == pl_lost_counts.dtype

        # Check the counts are identical
        assert (frame.data == pl_lost_counts).all()
"""
