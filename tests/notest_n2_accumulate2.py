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
    "num_dishes": 64,
    "num_n2k_samples_to_accumulate": 2,
    "variance_mode": "EvenOddPosDef",
    "num_ev": 0,
    "num_workers": 1,
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


def make_zeroed_chord_buffer(
    name,
    dtype,
    typename,
    shape,
    dim_names,
    seq0,
    dseq,
    num_frames,
    freq_ids=None,
    time_downsampling=None,
    extra_meta=None,
):

    bufs = []

    for idx in range(num_frames):
        seq = seq0 + idx * dseq

        data = np.zeros(shape, dtype=dtype)

        meta = runner.chordbuffer.get_metadata(name, typename, dim_names)
        meta["fpga_seq_num"] = seq
        if freq_ids is not None:
            meta["coarse_freq"] = freq_ids
        if time_downsampling is not None:
            meta["time_downsampling_fpga"] = time_downsampling

        if extra_meta is not None:
            for key, val in extra_meta.items():
                meta[key] = val

        bufs.append(ChordBuffer(data, meta))

    return bufs


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

    freq_ids = np.arange(num_freq, dtype=np.int32)

    shape = (num_int, num_freq, num_blocks, blocksize, blocksize, 2)

    bufs = make_zeroed_chord_buffer(
        "n2k_correlation",
        np.int32,
        "int32",
        shape,
        ("Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"),
        0,
        config["samples_per_data_set"],
        setup["num_frames"],
        freq_ids=freq_ids,
        time_downsampling=config["sub_integration_ntime"],
    )

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

    shape = (num_int, num_freq, num_blocks, blocksize, blocksize)

    bufs = make_zeroed_chord_buffer(
        "n2k_counts",
        np.int32,
        "int32",
        shape,
        ("Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"),
        0,
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

    for buf in bufs:
        buf.data[...] = config["sub_integration_ntime"]

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

    shape = (num_int, num_freq)

    bufs = make_zeroed_chord_buffer(
        "RFImask_count",
        np.int32,
        "int32",
        shape,
        ("Tc", "F"),
        0,
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

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

    shape = (num_int, num_freq)

    bufs = make_zeroed_chord_buffer(
        "pl_lost_counts_scalar",
        np.int32,
        "int32",
        shape,
        ("Tc", "F"),
        0,
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

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

    shape = (num_int, num_freq)

    bufs = make_zeroed_chord_buffer(
        "RFIFrameMask",
        np.uint8,
        "uint8",
        shape,
        ("Tc", "F"),
        0,
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
        extra_meta={
            "rfi_frame_excision_enabled": False,
            "rfi_frame_excision_thresholds": np.empty((0, 0), dtype=np.float32),
        },
    )

    for buf in bufs:
        buf.data[...] = 1

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


def test_meta(accum_data, setup):
    # Test the metadata

    config = setup["config"]

    for idx, frame in enumerate(accum_data):

        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        assert frame.metadata.freq_id == f
        assert frame.metadata.abs_time_idx == t


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
