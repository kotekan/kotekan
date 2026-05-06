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
    "origin_itrs_lat_deg": 50.0,
    "origin_itrs_lon_deg": -120.0,
}

chime_tel = {
    "name": "CHIMETelescope",
    "sampling_rate_MHz": 800.0,
    "fft_length": 2048,
    "nyquist_zone": 2,
    "require_gps": False,
    "origin_itrs_lat_deg": 50.0,
    "origin_itrs_lon_deg": -120.0,
}

NS_TOL = 5  # Allowing 5 ns of slop in eop
ERA_TOL = 2e-12 * NS_TOL  # There are ~2e-12 deg of Earth Rotation per ns
DUT1_TOL = 1e-14  # Just a big bigger than roundoff error
PM_TOL = 1e-14  # Just a big bigger than roundoff error


@pytest.fixture(
    scope="module",
    ids=["by-frame-pos-weights", "by-frame-chime-weights", "by-ERA-pos-weights"],
    params=[
        {
            "bin_in_ERA": False,
            "num_n2k_samples_to_accumulate": 6,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": False,
        },
        {
            "bin_in_ERA": False,
            "num_n2k_samples_to_accumulate": 8,
            "variance_mode": "CHIMEv1",
            "do_fringestop": False,
        },
        {
            "bin_in_ERA": True,
            "num_bins_per_rotation": 3 * 86400,
            "variance_mode": "CHIMEv1",
            "do_fringestop": False,
        },
    ],
)
def accum_setup(request):
    return request.param


@pytest.fixture(
    scope="module",
    ids=["chord-prod", "chime-prod"],
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
        },
    ],
)
def setup(request, accum_setup):
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

    if accum_setup["bin_in_ERA"] and request.param["tel"]["name"] != "CHORDTelescope":
        request.param["fail"] = True

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
    accum_setup,
    corr_data,
    count_data,
    rficount_data,
    plcount_data,
    rfiframemask_data,
    accum_list,
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
        exit_after_n_files=(config["num_local_freq"] * len(accum_list)),
        num_elements=config["num_elements"],
        num_ev=config["num_ev"],
    )

    accum_config = {
        "num_freq_per_n2k_frame": config["num_local_freq"],
        "packet_loss_is_scalar": True,
    }
    accum_config.update(accum_setup)

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

    print(config)
    print(accum_config)

    test.run()

    yield dump_buffer.load()


def get_era_nrot_idx_of_seq(seq, tel_config, use_eop, num_bins_per_rotation):
    t = tel.get_t_inst_ns(seq, tel_config)
    era = tel.get_ERA_at_t_inst_ns(t, tel_config, use_eop)
    nrot = tel.get_nrot_at_t_inst_ns(t, tel_config, use_eop)

    era_idx = np.floor((era / 360.0) * num_bins_per_rotation).astype(int)
    era_nrot_idx = era_idx + nrot * num_bins_per_rotation
    return era_nrot_idx


@pytest.fixture(scope="module")
def accum_list(setup, accum_setup):

    config = setup["config"]

    seq0 = config["first_frame_index"] * config["samples_per_data_set"]
    num_subs_per_frame = (
        config["samples_per_data_set"] // config["sub_integration_ntime"]
    )

    num_subs = num_subs_per_frame * config["num_frames"]
    subs = np.arange(num_subs)
    seqs_per_sub = config["sub_integration_ntime"]
    seqs = seq0 + seqs_per_sub * subs

    seqs_ext = np.concatenate(
        ([seqs[0] - seqs_per_sub], seqs, [seqs[-1] + seqs_per_sub])
    )

    if accum_setup["bin_in_ERA"]:
        seq_pair_cen = np.empty_like(seqs_ext)

        sub_idx = seqs_ext // seqs_per_sub
        even = sub_idx % 2 == 0
        seq_pair_cen[even] = seqs_ext[even] + seqs_per_sub
        seq_pair_cen[~even] = seqs_ext[~even]

        accum_sub_idx_ext = get_era_nrot_idx_of_seq(
            seq_pair_cen,
            setup["tel"],
            setup["set_eop"],
            accum_setup["num_bins_per_rotation"],
        )
    else:
        subs_per_accum = accum_setup["num_n2k_samples_to_accumulate"]
        seq_per_accum = subs_per_accum * seqs_per_sub

        accum_sub_idx_ext = seqs_ext // seq_per_accum

    accum_sub_idx = accum_sub_idx_ext[1:-1]

    idx_diff = np.diff(accum_sub_idx_ext)
    accum_edges = np.nonzero(idx_diff)[0]
    first_sub_in_accum = accum_edges[0]
    last_accum_full = accum_edges[-1] == num_subs

    accum_bin_idx = np.unique(accum_sub_idx[first_sub_in_accum:])

    accums = []

    for idx in accum_bin_idx:

        mask = accum_sub_idx == idx

        accums.append(
            {
                "bin_idx": idx,
                "sub_idx": subs[mask],
                "seq": seqs[mask][0],
                "seq_len": config["sub_integration_ntime"] * len(subs[mask]),
            }
        )

    if not last_accum_full:
        accums = accums[:-1]

    for accum in accums:
        if accum_setup["bin_in_ERA"]:

            t_cen = tel.get_t_inst_ns(accum['seq'] + accum['seq_len'], setup["tel"])
            nrot = tel.get_nrot_at_t_inst_ns(t_cen, setup["tel"], setup["set_eop"])

            dera = 360.0 / accum_setup["num_bins_per_rotation"]
            era_start = (accum["bin_idx"] % accum_setup["num_bins_per_rotation"]) * dera
            era_end = ((accum["bin_idx"] + 1) % accum_setup["num_bins_per_rotation"]) * dera
        
            t_bin_start = tel.get_t_inst_ns_at_ERA_nrot(era_start, nrot, setup["tel"], setup["set_eop"])
            t_bin_cen = tel.get_t_inst_ns_at_ERA_nrot(era_start, nrot, setup["tel"], setup["set_eop"])
            t_bin_end = tel.get_t_inst_ns_at_ERA_nrot(era_start + dera, nrot, setup["tel"], setup["set_eop"])

            accum["t_bin_cen"] = t_bin_cen
            accum["bin_start_ERA_deg"] = era_start
            accum["bin_end_ERA_deg"] = era_end 
            accum["bin_start_ERAL_deg"] = tel.get_local_ERA_at_t_inst_ns(t_bin_start, setup["tel"], setup["set_eop"]) 
            accum["bin_end_ERAL_deg"] = tel.get_local_ERA_at_t_inst_ns(t_bin_end, setup["tel"], setup["set_eop"])
        else:
            seq_start = accum["seq"]
            seq_end = accum["seq"] + accum["seq_len"]
            seq_cen = accum["seq"] + accum["seq_len"] // 2
            t_start = tel.get_t_inst_ns(seq_start, setup["tel"])
            t_end = tel.get_t_inst_ns(seq_end, setup["tel"])
            accum["t_bin_cen"] = tel.get_t_inst_ns(seq_cen, setup["tel"])
            accum["bin_start_ERA_deg"] = tel.get_ERA_at_t_inst_ns(
                t_start, setup["tel"], setup["set_eop"]
            )
            accum["bin_end_ERA_deg"] = tel.get_ERA_at_t_inst_ns(
                t_end, setup["tel"], setup["set_eop"]
            )
            accum["bin_start_ERAL_deg"] = -1.0
            accum["bin_end_ERAL_deg"] = -1.0
            # accum['bin_start_ERAL_deg'] = tel.get_local_ERA_at_t_inst_ns(t_start, setup['tel'], setup['set_eop'])
            # accum['bin_end_ERAL_deg'] = tel.get_local_ERA_at_t_inst_ns(t_end, setup['tel'], setup['set_eop'])

    return accums


def safe_invert(x, dtype=float):

    mask = x != 0
    inv_x = np.zeros(x.shape, dtype=dtype)
    inv_x[mask] = 1.0 / x[mask]

    return inv_x


@pytest.fixture(scope="module")
def expected_accum(
    setup,
    accum_setup,
    corr_data,
    count_data,
    rficount_data,
    plcount_data,
    rfiframemask_data,
    accum_list,
):

    config = setup["config"]

    # sizes
    num_int = config["samples_per_data_set"] // config["sub_integration_ntime"]
    num_freq = config["num_local_freq"]
    num_el = config["num_polarizations"] * config["num_dishes"]
    num_accum = len(accum_list)
    num_n2_prod = (num_el * (num_el + 1)) // 2
    blocksize = 16

    # indices to correlation array for each N2 entry
    # correlation format: blocked lower triangular
    # n2 format: upper triangular
    corr_idx_b = np.empty(num_n2_prod, dtype=int)
    corr_idx_i = np.empty(num_n2_prod, dtype=int)
    corr_idx_j = np.empty(num_n2_prod, dtype=int)

    # build the index maps
    idx = 0
    for n2_i in range(num_el):
        for n2_j in range(n2_i, num_el):
            corr_i = n2_j
            corr_j = n2_i

            bi = corr_i // blocksize
            bj = corr_j // blocksize

            b = (bi * (bi + 1)) // 2 + bj
            sbi = corr_i % blocksize
            sbj = corr_j % blocksize

            corr_idx_b[idx] = b
            corr_idx_i[idx] = sbi
            corr_idx_j[idx] = sbj
            idx += 1

    # accumulation arrays
    corr_shape = (num_accum, num_freq) + corr_data[0].data.shape[2:]
    count_shape = (num_accum, num_freq)  # Assuming counts are scalar

    accum_corr = np.zeros(corr_shape, dtype=np.int32)
    accum_var = np.zeros(corr_shape[:-1], dtype=float)
    accum_bias = np.zeros(corr_shape[:-1], dtype=float)
    accum_count = np.zeros(count_shape, dtype=np.int32)
    accum_plcount = np.zeros(count_shape, dtype=np.int32)
    accum_rficount = np.zeros(count_shape, dtype=np.int32)
    accum_n2 = np.zeros((num_accum, num_freq, num_n2_prod), dtype=np.complex128)
    accum_n2_var = np.zeros((num_accum, num_freq, num_n2_prod), dtype=float)

    # Run the actual accumulation
    for i, accum in enumerate(accum_list):

        # Accumulate subframes in pairs for weights and rfi rejection
        for tsub in accum["sub_idx"][::2]:
            # grab and frame and subintegration (correlation) indices for each pair
            tf1 = tsub // num_int
            tc1 = tsub % num_int
            tf2 = (tsub + 1) // num_int
            tc2 = (tsub + 1) % num_int

            # grab the RFI Frame Mask for the pair (these are still arrays over frequency)
            mask = (
                rfiframemask_data[tf1].data[tc1] & rfiframemask_data[tf2].data[tc2]
            ).astype(np.int32)
            corr_mask = mask[:, None, None, None, None]

            corr1 = corr_data[tf1].data[tc1]
            corr2 = corr_data[tf2].data[tc2]
            N1 = count_data[tf1].data[tc1, :, 0, 0, 0]
            N2 = count_data[tf2].data[tc2, :, 0, 0, 0]

            # apply the RFI frame mask and accumulate!
            accum_corr[i] += corr_mask * (corr1 + corr2)
            accum_count[i] += mask * (N1 + N2)
            accum_plcount[i] += mask * (
                plcount_data[tf1].data[tc1] + plcount_data[tf2].data[tc2]
            )
            accum_rficount[i] += mask * (
                rficount_data[tf1].data[tc1] + rficount_data[tf2].data[tc2]
            )

            # Accumulate the weights too.
            if accum_setup["variance_mode"] == "CHIMEv1":
                dcorr = (corr2 - corr1).astype(
                    float
                )  # int32s will overflow when squaring
                dcorr2 = dcorr[..., 0] ** 2 + dcorr[..., 1] ** 2
                accum_var[i] += mask[:, None, None, None] * dcorr2
                accum_bias[i] += (mask * ((N2 - N1) ** 2))[:, None, None, None]
            elif accum_setup["variance_mode"] == "EvenOddPosDef":
                inv_N1 = safe_invert(N1, float)
                inv_N2 = safe_invert(N2, float)
                inv_var = safe_invert(inv_N1 + inv_N2, float)
                vis1 = corr1 * inv_N1[:, None, None, None, None]
                vis2 = corr2 * inv_N2[:, None, None, None, None]
                dvis = vis2 - vis1
                dvis2 = dvis[..., 0] ** 2 + dvis[..., 1] ** 2
                accum_var[i] += (mask * inv_var)[:, None, None, None] * dvis2

        # Remap the accumulated correlation array to the N2 array
        for f in range(num_freq):
            # We're storing Upper Triangular entries now so take the complex conjugate.
            accum_n2[i, f, :] = (
                accum_corr[i, f, corr_idx_b, corr_idx_i, corr_idx_j, 0]
                - 1.0j * accum_corr[i, f, corr_idx_b, corr_idx_i, corr_idx_j, 1]
            )
            inv_N = safe_invert(accum_count[i, f], float)

            if accum_setup["variance_mode"] == "CHIMEv1":
                vis = accum_corr[i, f] * inv_N
                bias = accum_bias[i, f] * (vis[..., 0] ** 2 + vis[..., 1] ** 2)
                accum_n2_var[i, f, :] = (accum_var[i, f] - bias)[
                    corr_idx_b, corr_idx_i, corr_idx_j
                ] * (inv_N ** 2)

            elif accum_setup["variance_mode"] == "EvenOddPosDef":
                M = len(accum["sub_idx"])
                norm = 2 * inv_N / M
                accum_n2_var[i, f, :] = (
                    accum_var[i, f, corr_idx_b, corr_idx_i, corr_idx_j] * norm
                )

    # Normalize N2 by the valid counts, if counts are 0, then 0 the vis
    inv_count = safe_invert(accum_count, float)

    vis = accum_n2 * inv_count[:, :, None]

    w = safe_invert(accum_n2_var, float)

    return vis, w, accum_count, accum_rficount, accum_plcount


def test_structure(accum_data, accum_list, setup):

    config = setup["config"]
    num_el = config["num_elements"]
    num_freq = config["num_local_freq"]
    num_ev = 0
    num_n2_prod = (num_el * (num_el + 1)) // 2

    if setup["fail"]:
        assert len(accum_data) == 0
        return

    assert len(accum_data) == (len(accum_list) * num_freq)

    for frame in accum_data:
        assert frame.vis.shape == (num_n2_prod,)
        assert frame.vis.dtype == np.complex64

        assert frame.weight.shape == (num_n2_prod,)
        assert frame.weight.dtype == np.float32

        assert frame.flags.shape == (num_el,)
        assert frame.flags.dtype == np.float32

        assert frame.eval.shape == (num_ev,)
        assert frame.eval.dtype == np.float32

        assert frame.evec.shape == (num_ev * num_el,)
        assert frame.evec.dtype == np.complex64

        assert frame.emethod.shape == (1,)
        assert frame.emethod.dtype == np.int32

        assert frame.erms.shape == (1,)
        assert frame.erms.dtype == np.float32

        assert frame.radiometer_chi2.shape == (3,)
        assert frame.radiometer_chi2.dtype == np.float32

        assert frame.gain.shape == (num_el,)
        assert frame.gain.dtype == np.complex64

        assert frame.mask.shape == (num_el,)
        assert frame.mask.dtype == np.uint8


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
        assert frame.metadata.frame_start_time_ns == tel.get_t_inst_ns(
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


def test_eop_meta(setup, accum_setup, accum_data, accum_list):
    # Test the metadata

    config = setup["config"]

    t_inst_cen = np.empty(len(accum_list), dtype=int)
    t_inst_bin = np.empty(len(accum_list), dtype=int)

    # First we run through the frames to get the list of times we'll need EOP for.
    for idx, frame in enumerate(accum_data):

        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        if f == 0:
            seq_t_cen = accum_list[t]["seq"] + accum_list[t]["seq_len"] // 2
            t_inst_cen[t] = tel.get_t_inst_ns(seq_t_cen, setup["tel"])
            t_inst_bin[t] = accum_list[t]["t_bin_cen"]

    # Now make the expected EOP (this is slow to do one at a time)
    eop_cen = tel.get_EOP_at_t_inst_ns(t_inst_cen, setup["tel"], setup["set_eop"])
    eop_bin = tel.get_EOP_at_t_inst_ns(t_inst_bin, setup["tel"], setup["set_eop"])

    # Now check everything has the correct values
    for idx, frame in enumerate(accum_data):
        t = idx // config["num_local_freq"]

        eop = eop_cen[t]
        assert abs(frame.metadata.time_center_eop.t_inst_ns - eop.t_inst_ns) <= NS_TOL
        assert abs(frame.metadata.time_center_eop.t_ut1_ns - eop.t_ut1_ns) <= NS_TOL
        assert (
            abs(frame.metadata.time_center_eop.delta_UT1_inst - eop.delta_UT1_inst)
            <= DUT1_TOL
        )
        assert abs(frame.metadata.time_center_eop.ERA_deg - eop.ERA_deg) <= ERA_TOL
        assert abs(frame.metadata.time_center_eop.xp_as - eop.xp_as) <= PM_TOL
        assert abs(frame.metadata.time_center_eop.yp_as - eop.yp_as) <= PM_TOL

        eop = eop_bin[t]
        assert abs(frame.metadata.bin_eop.t_inst_ns - eop.t_inst_ns) <= NS_TOL
        assert abs(frame.metadata.bin_eop.t_ut1_ns - eop.t_ut1_ns) <= NS_TOL
        assert (
            abs(frame.metadata.bin_eop.delta_UT1_inst - eop.delta_UT1_inst) <= DUT1_TOL
        )
        assert abs(frame.metadata.bin_eop.ERA_deg - eop.ERA_deg) <= ERA_TOL
        assert abs(frame.metadata.bin_eop.xp_as - eop.xp_as) <= PM_TOL
        assert abs(frame.metadata.bin_eop.yp_as - eop.yp_as) <= PM_TOL

        assert (
            abs(frame.metadata.bin_start_ERA_deg - accum_list[t]["bin_start_ERA_deg"])
            <= ERA_TOL
        )
        assert (
            abs(frame.metadata.bin_end_ERA_deg - accum_list[t]["bin_end_ERA_deg"])
            <= ERA_TOL
        )
        assert (
            abs(frame.metadata.bin_start_ERAL - accum_list[t]["bin_start_ERAL_deg"])
            <= ERA_TOL
        )
        assert (
            abs(frame.metadata.bin_end_ERAL - accum_list[t]["bin_end_ERAL_deg"])
            <= ERA_TOL
        )


def test_counts(accum_data, expected_accum, accum_list, setup):

    config = setup["config"]

    _, _, exp_cnts, exp_rficnts, exp_plcnts = expected_accum

    for idx, frame in enumerate(accum_data):
        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        frame.metadata.n_valid_fpga_ticks = exp_cnts[t, f]
        frame.metadata.n_rfi_fpga_ticks = exp_rficnts[t, f]
        frame.metadata.n_pl_fpga_ticks = exp_plcnts[t, f]
        frame.metadata.n_rfi_only_fpga_ticks = accum_list[t]["seq_len"] - (
            exp_cnts[t, f] + exp_plcnts[t, f]
        )


def test_vis(accum_data, expected_accum, accum_list, setup):

    config = setup["config"]

    exp_vis, _, _, _, _ = expected_accum

    for idx, frame in enumerate(accum_data):
        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        assert frame.vis.shape == exp_vis[t, f].shape

        # vis_diff = frame.vis - exp_vis[t, f]
        # tol =  1.0e-6 + 1.0e-6 * 0.5*np.abs(frame.vis + exp_vis[t, f])
        np.testing.assert_allclose(frame.vis, exp_vis[t, f], 1.0e-6, 1.0e-6)


def test_weight(accum_data, expected_accum, accum_list, setup, accum_setup):

    config = setup["config"]

    _, exp_w, _, _, _ = expected_accum

    for idx, frame in enumerate(accum_data):
        t = idx // config["num_local_freq"]
        f = idx % config["num_local_freq"]

        assert frame.weight.shape == exp_w[t, f].shape

        var_data = safe_invert(frame.weight, float)
        var_exp = safe_invert(exp_w[t, f], float)

        if accum_setup["variance_mode"] == "CHIMEv1":
            # The bias subtraction can introduce a LOT of truncation error on random
            # data, so need to set the tolerances wide (in lieu of replicating the truncation
            # error in this test)
            rtol = 1.0e-2
            atol = 1.0e-2
        elif accum_setup["variance_mode"] == "EvenOddPosDef":
            rtol = 1.0e-4
            atol = 1.0e-5

        bad = np.abs(var_data - var_exp) > atol + rtol * np.abs(var_exp)
        msg = ""
        if bad.any():
            msg = "First bad value - actual: {:e} expected {:e}".format(
                var_data[bad][0], var_exp[bad][0]
            )
        np.testing.assert_allclose(var_data, var_exp, rtol, atol, err_msg=msg)
