import pytest
import numpy as np

from kotekan import runner
from kotekan.chordbuffer import ChordBuffer
import kotekan.telescope as tel

prod_config = {
    "buffer_depth": 5,
    "samples_per_data_set": 16384,
    "num_local_freq": 12,
    "sub_integration_ntime": 8192,
    "first_frame_index": 1,
    "num_polarizations": 2,
    "num_dishes": 64,
    "num_n2k_samples_to_accumulate": 2,
    "variance_mode": "EvenOddPosDef",
    "num_ev": 0,
    "freq_ids": [0, 8191, 300, 1000, 4000],
}

chord_tel = {
    "name": "CHORDTelescope",
    "sampling_rate_MHz": 3.2e3,
    "fft_length": 16384,
    "nyquist_zone": 1,
    "num_dishes_x": 12,
    "num_dishes_y": 6,
    "dish_inputs": [],
}

chime_tel = {
    "name": "CHIMETelescope",
    "sampling_rate_MHz": 800.0,
    "fft_length": 2048,
    "nyquist_zone": 2,
    "require_gps": False,
}

ERA_TOL = 4e-12  # ~1 ns of Earth Rotation
PM_TOL = 1e-14  # Just a big bigger than roundoff error


@pytest.fixture(
    scope="module",
    params=[
        {
            "config": prod_config,
            "num_frames": 48,
            "fail": False,
            "tel": chord_tel,
            "set_eop": True,
            "corr": {"type": "random"},
            "count": {"type": "random"},
            "rficount": {"type": "random", "use_counts": True},
            "plcount": {"type": "random", "use_counts": True},
            "rfiframemask": {
                "type": "random",
                "enabled": True,
                "thresholds": np.array([[3.0, 0.1]], dtype=np.float32),
            },
            "accum": {
                "bin_in_ERA": False,
                "num_n2k_samples_to_accumulate": 6,
                "variance_mode": "EvenOddPosDef",
                "do_fringestop": False,
            },
        },
        {
            "config": prod_config,
            "num_frames": 48,
            "fail": False,
            "tel": chime_tel,
            "set_eop": False,
            "corr": {"type": "random"},
            "count": {"type": "random"},
            "rficount": {"type": "random", "use_counts": True},
            "plcount": {"type": "random", "use_counts": True},
            "rfiframemask": {
                "type": "random",
                "enabled": True,
                "thresholds": np.array([[3.0, 0.1]], dtype=np.float32),
            },
            "accum": {
                "bin_in_ERA": False,
                "num_n2k_samples_to_accumulate": 6,
                "variance_mode": "EvenOddPosDef",
                "do_fringestop": False,
            },
        },
    ],
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
    if "num_frames" not in request.param:
        request.param["num_frames"] = 2 * config["buffer_depth"]

    config["num_frames"] = request.param["num_frames"]

    current_time = tel.get_unix_time_ns("now")

    request.param["tel"]["frame0_nano"] = current_time
    config["gps_time"] = {"frame0_nano": current_time}

    if request.param["set_eop"]:
        config["eop"] = {
            "kotekan_update_endpoint": "json",
            "earth_orientation_parameter_table": tel.get_EOP_table(current_time, 3),
        }
        request.param["tel"]["eop_updatable_config"] = "/eop"

    config["telescope"] = request.param["tel"]

    request.param["rng"] = np.random.default_rng()

    yield request.param


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
    corr_setup = setup["corr"]
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
        config["first_frame_index"] * config["samples_per_data_set"],
        config["samples_per_data_set"],
        setup["num_frames"],
        freq_ids=freq_ids,
        time_downsampling=config["sub_integration_ntime"],
    )

    if corr_setup["type"] == "random":

        max_val = 49 * config["sub_integration_ntime"]
        rng = setup["rng"]

        for buf in bufs:
            buf.data[...] = rng.integers(
                -max_val, max_val, size=shape, dtype=np.int32, endpoint=True
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
    count_setup = setup["count"]
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
        config["first_frame_index"] * config["samples_per_data_set"],
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

    if count_setup["type"] == "random":

        max_val = config["sub_integration_ntime"]
        scalar_shape = shape[:2]
        rng = setup["rng"]

        for buf in bufs:
            buf.data[...] = rng.integers(
                0, max_val, size=scalar_shape, dtype=np.int32, endpoint=True
            )[:, :, None, None, None]
    else:
        for buf in bufs:
            buf.data[...] = config["sub_integration_ntime"]

    return bufs


@pytest.fixture(scope="module")
def rficount_data(setup, count_data):
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
    rficount_setup = setup["rficount"]
    num_frames = setup["num_frames"]

    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]

    shape = (num_int, num_freq)

    bufs = make_zeroed_chord_buffer(
        "RFImask_counts",
        np.int32,
        "int32",
        shape,
        ("Tc", "F"),
        config["first_frame_index"] * config["samples_per_data_set"],
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

    if rficount_setup["type"] == "random":

        max_val = config["sub_integration_ntime"]
        rng = setup["rng"]

        for i, buf in enumerate(bufs):
            if rficount_setup["use_counts"]:
                max_val = (
                    config["sub_integration_ntime"] - count_data[i].data[:, :, 0, 0, 0]
                )
            buf.data[...] = rng.integers(
                0, max_val, size=shape, dtype=np.int32, endpoint=True
            )

    return bufs


@pytest.fixture(scope="module")
def plcount_data(setup, count_data):
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
    plcount_setup = setup["plcount"]
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
        config["first_frame_index"] * config["samples_per_data_set"],
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
    )

    if plcount_setup["type"] == "random":

        max_val = config["sub_integration_ntime"]
        rng = setup["rng"]

        for i, buf in enumerate(bufs):
            if plcount_setup["use_counts"]:
                max_val = (
                    config["sub_integration_ntime"] - count_data[i].data[:, :, 0, 0, 0]
                )
            buf.data[...] = rng.integers(
                0, max_val, size=shape, dtype=np.int32, endpoint=True
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
    rfiframemask_setup = setup["rfiframemask"]
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
        config["first_frame_index"] * config["samples_per_data_set"],
        config["samples_per_data_set"],
        setup["num_frames"],
        time_downsampling=config["sub_integration_ntime"],
        extra_meta={
            "rfi_frame_excision_enabled": rfiframemask_setup["enabled"],
            "rfi_frame_excision_thresholds": np.array(
                rfiframemask_setup["thresholds"], dtype=np.float32
            ),
        },
    )

    if rfiframemask_setup["type"] == "random":

        rng = setup["rng"]

        for i, buf in enumerate(bufs):
            buf.data[...] = rng.integers(
                0, 1, size=shape, dtype=np.uint8, endpoint=True
            )
    else:
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

    accum_config = {
        "num_freq_per_n2k_frame": config["num_local_freq"],
        "packet_loss_is_scalar": True,
    }
    accum_config.update(setup["accum"])

    # The test stage!
    test = runner.KotekanStageTester(
        "N2Accumulate",
        accum_config,
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


@pytest.fixture(scope="module")
def accum_list(setup):

    config = setup["config"]
    accum = setup["accum"]

    seq0 = config["first_frame_index"] * config["samples_per_data_set"]
    num_sub_per_frame = (
        config["samples_per_data_set"] // config["sub_integration_ntime"]
    )

    subs = np.arange(num_sub_per_frame * config["num_frames"])
    seqs = seq0 + config["sub_integration_ntime"] * subs

    if accum["bin_in_ERA"]:
        pass
    else:
        subs_per_accum = accum["num_n2k_samples_to_accumulate"]
        seq_per_accum = subs_per_accum * config["sub_integration_ntime"]

        accum_idx_sub = seqs // seq_per_accum

        first_sub_in_accum = subs[seqs % seq_per_accum == 0][0]

    accum_bin_idx = np.unique(accum_idx_sub[first_sub_in_accum:])

    accum_subs = []
    accum_seqs = []

    accums = []

    for idx in accum_bin_idx:

        mask = accum_idx_sub == idx

        accums.append(
            {
                "bin_idx": idx,
                "sub_idx": subs[mask],
                "seq": seqs[mask][0],
                "seq_len": config["sub_integration_ntime"] * len(subs[mask]),
            }
        )

    return accums


def test_simple_meta(accum_data, setup, accum_list):
    # Test the metadata

    config = setup["config"]

    for idx, frame in enumerate(accum_data):

        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        expected_seq = accum_list[t]["seq"]

        assert frame.metadata.freq_id == f
        assert frame.metadata.freq_MHz == tel.get_freq_MHz(f, setup["tel"])
        assert frame.metadata.abs_time_idx == accum_list[t]["bin_idx"]
        assert frame.metadata.fpga_start_tick == expected_seq
        assert frame.metadata.frame_length_fpga_ticks == accum_list[t]["seq_len"]
        assert frame.metadata.frame_start_time_ns == tel.get_t_inst(
            expected_seq, setup["tel"]
        )

        num_thresh = frame.metadata.rfi_frame_excision_num
        data_threshold = np.array(frame.metadata.rfi_frame_excision_threshold)[
            :num_thresh
        ]
        data_fraction = np.array(frame.metadata.rfi_frame_excision_fraction)[
            :num_thresh
        ]
        assert (
            frame.metadata.rfi_frame_excision_enabled
            == setup["rfiframemask"]["enabled"]
        )
        assert frame.metadata.rfi_frame_excision_num == len(
            setup["rfiframemask"]["thresholds"]
        )
        assert (data_threshold == setup["rfiframemask"]["thresholds"][:, 0]).all()
        assert (data_fraction == setup["rfiframemask"]["thresholds"][:, 1]).all()


def test_eop_meta(accum_data, setup, accum_list):
    # Test the metadata

    config = setup["config"]

    for idx, frame in enumerate(accum_data):

        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        seq_t_cen = accum_list[t]["seq"] + accum_list[t]["seq_len"] // 2
        t_inst_cen = tel.get_t_inst(seq_t_cen, setup["tel"])

        eop = tel.get_EOP_at_t_inst_ns(t_inst_cen, setup["tel"], setup["set_eop"])

        assert frame.metadata.time_center_eop.t_inst_ns == eop.t_inst_ns
        assert frame.metadata.time_center_eop.t_ut1_ns == eop.t_ut1_ns
        assert frame.metadata.time_center_eop.delta_UT1_inst == eop.delta_UT1_inst
        assert abs(frame.metadata.time_center_eop.ERA_deg - eop.ERA_deg) <= ERA_TOL
        assert abs(frame.metadata.time_center_eop.xp_as - eop.xp_as) <= PM_TOL
        assert abs(frame.metadata.time_center_eop.yp_as - eop.yp_as) <= PM_TOL


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
