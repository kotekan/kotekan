"""Round-trip kotekan buffers through hdf5FileWrite and hdf5FileRead.

Session A ("siphon") generates deterministic frames and writes them with
hdf5FileWrite, either one file per frame or a single file
(`create_single_file: true`). Python then inspects the files with h5py.
Session B ("replay") reads the files back with hdf5FileRead, regenerates the
identical frames with the same generator config, and compares payload and
chordMetadata with a testDataCheck* stage.

The two sessions are separate kotekan processes on purpose: hdf5FileRead
requires its input to exist when it starts, and a single file is only
complete once the writer has closed it.

A single file concatenates all frames along axis 0 and stores the metadata
only once, so hdf5FileWrite accepts it only for a contiguous stream: axis 0
must be a time axis, `dim_scaling[0]` must equal `time_downsampling_fpga`,
`fpga_seq_num` must advance by `dim[0] * time_downsampling_fpga` per frame,
and frame decimation is forbidden. The negative tests below check that each
violation aborts the writer, that the same data round-trips fine as per-frame
files (which have none of those constraints), and that hdf5FileRead rejects a
single file that violates the same rules on disk.
"""
import glob
import os
import shutil

import h5py
import numpy as np
import pytest

try:
    import hdf5plugin  # noqa: F401  registers the bitshuffle filter for h5py
except ImportError:  # HDF5_PLUGIN_PATH may still be set in the environment
    pass

from kotekan import runner

if not runner.has_hdf5():
    pytest.fail("HDF5 support not available; unable to run tests!")

NUM_FRAMES = 6
FILE_NAME = "siphon"


def _available_stages():
    return set(runner.KotekanRunner.kotekan_config().get("available_stages", []))


def _ramp(extents, dtype):
    return (np.arange(int(np.prod(extents))) % 256).astype(dtype).reshape(extents)


# generator stage + config, matching ndarray buffer declaration, checker,
# numpy dtype of the payload, optional expected payload of every frame.
CASES = {
    "int8_random": dict(
        gen_stage="testDataGen",
        gen=dict(
            type="random8",
            seed=12345,
            samples_per_data_set=256,
            num_local_freq=4,
            name="E",
            array_shape=[256, 4, 8],
            dim_name=["T", "F", "D"],
            dim_scaling=[1, 1, 1],
        ),
        buf=dict(
            value_type="int8",
            quantity_name="E",
            extents=[256, 4, 8],
            dimnames=["T", "F", "D"],
            dimscalings=[1, 1, 1],
        ),
        checker="testDataCheckUchar",
        dtype=np.int8,
        expected=None,
    ),
    "uint8_ramp": dict(
        gen_stage="testDataGen",
        gen=dict(
            type="ramp",
            value=1,
            samples_per_data_set=256,
            num_local_freq=4,
            name="E",
            array_shape=[256, 4, 8],
            dim_name=["T", "F", "D"],
            dim_scaling=[1, 1, 1],
        ),
        buf=dict(
            value_type="uint8",
            quantity_name="E",
            extents=[256, 4, 8],
            dimnames=["T", "F", "D"],
            dimscalings=[1, 1, 1],
        ),
        checker="testDataCheckUchar",
        dtype=np.uint8,
        expected=_ramp([256, 4, 8], np.uint8),
    ),
    # 4-byte elements + time downsampling: fpga_seq_num advances by 256 per
    # frame = extents[0] (16) * time_downsampling_fpga (16).
    "int32_downsampled": dict(
        gen_stage="testDataGen",
        gen=dict(
            type="random32",
            seed=777,
            samples_per_data_set=256,
            meta_time_downsample_factor=16,
            num_local_freq=4,
            name="N",
            array_shape=[16, 4, 8],
            dim_name=["Tc", "F", "D"],
            dim_scaling=[16, 1, 1],
        ),
        buf=dict(
            value_type="int32",
            quantity_name="N",
            extents=[16, 4, 8],
            dimnames=["Tc", "F", "D"],
            dimscalings=[16, 1, 1],
        ),
        checker="testDataCheckInt",
        dtype=np.int32,
        expected=None,
    ),
    # FRB-beam-like: float16, 4 axes, zeros in the payload.
    "float16_beams": dict(
        gen_stage="testDataGenFloat",
        gen=dict(
            type="ramp",
            value=1.0,
            value_type="float16",
            samples_per_data_set=64,
            num_local_freq=4,
            name="I",
            array_shape=[64, 4, 4, 2],
            dim_name=["Ttilde", "Fbar", "beamQ", "beamP"],
            dim_scaling=[1, 1, 1, 1],
        ),
        buf=dict(
            value_type="float16",
            quantity_name="I",
            extents=[64, 4, 4, 2],
            dimnames=["Ttilde", "Fbar", "beamQ", "beamP"],
            dimscalings=[1, 1, 1, 1],
        ),
        checker="testDataCheckFloat16",
        dtype=np.float16,
        expected=_ramp([64, 4, 4, 2], np.float16),
    ),
}

# CHIME baseband voltages for hdf5FileReadSingleFile: rank 4, [T, F, P, D],
# int4x2_swapped_withoffset. `random_signed_offset` puts 1..15 into both
# nibbles, so no byte is ever 0x00 -- that is the poison value the reader
# checks for.
VOLTAGE_CASE = dict(
    gen_stage="testDataGen",
    gen=dict(
        type="random_signed_offset",
        seed=4242,
        samples_per_data_set=256,
        num_local_freq=4,
        name="E",
        array_shape=[256, 4, 2, 8],
        dim_name=["T", "F", "P", "D"],
        dim_scaling=[1, 1, 1, 1],
    ),
    buf=dict(
        value_type="int4x2_swapped_withoffset",
        quantity_name="E",
        extents=[256, 4, 2, 8],
        dimnames=["T", "F", "P", "D"],
        dimscalings=[1, 1, 1, 1],
    ),
    checker="testDataCheckUchar",
    dtype=np.uint8,
    expected=None,
)


def _variant(case, gen=None, buf=None):
    """A copy of `case` with some generator and/or buffer settings replaced."""
    variant = dict(case)
    if gen is not None:
        variant["gen"] = dict(case["gen"], **gen)
    if buf is not None:
        variant["buf"] = dict(case["buf"], **buf)
    return variant


def _buffer(name, buf):
    return {
        name: dict(
            kotekan_buffer="ndarray",
            metadata_pool="main_pool",
            num_frames="buffer_depth",
            **buf
        )
    }


def _generator(case, out_buf):
    gen = dict(
        case["gen"], kotekan_stage=case["gen_stage"], num_frames=NUM_FRAMES, wait=False
    )
    key = "network_out_buf" if case["gen_stage"] == "testDataGenFloat" else "out_buf"
    gen[key] = out_buf
    return gen


def _siphon(tmpdir, case, single_file, write_extra=None, expect_failure=False):
    """Session A: generate frames and write them. Returns the runner."""
    write = dict(
        kotekan_stage="hdf5FileWrite",
        in_buf="siphon_buf",
        base_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        max_frames=NUM_FRAMES,
        create_single_file=single_file,
    )
    write.update(write_extra or {})
    r = runner.KotekanRunner(
        _buffer("siphon_buf", case["buf"]),
        {"gen": _generator(case, "siphon_buf"), "write": write},
        {},
        expect_failure=expect_failure,
    )
    r.run()
    return r


def _check_files(tmpdir, case, single_file):
    extents = case["buf"]["extents"]
    indexed = sorted(glob.glob(os.path.join(str(tmpdir), FILE_NAME + ".*.h5")))
    if single_file:
        assert indexed == []
        files = [os.path.join(str(tmpdir), FILE_NAME + ".h5")]
        frames_per_file = NUM_FRAMES
    else:
        assert [os.path.basename(f) for f in indexed] == [
            "%s.%08d.h5" % (FILE_NAME, i) for i in range(NUM_FRAMES)
        ]
        assert not os.path.exists(os.path.join(str(tmpdir), FILE_NAME + ".h5"))
        files = indexed
        frames_per_file = 1
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            assert list(f.keys()) == [FILE_NAME]
            ds = f[FILE_NAME]
            assert ds.dtype == np.dtype(case["dtype"])
            assert list(ds.shape) == [frames_per_file * extents[0]] + extents[1:]
            a = ds.attrs
            assert list(a["chord_metadata_version"]) == [2, 0]
            assert a["name"] == case["buf"]["quantity_name"]
            assert a["type"] == case["buf"]["value_type"]
            assert list(a["dim_names"]) == case["buf"]["dimnames"]
            assert list(a["dim_scalings"]) == case["buf"]["dimscalings"]
            assert a["fpga_seq_num"] == i * case["gen"]["samples_per_data_set"]
            assert list(a["coarse_freq"]) == list(range(case["gen"]["num_local_freq"]))
            for key in ("telescope_name", "num_dishes", "num_polarizations"):
                assert key in a
            if case["expected"] is not None:
                data = ds[...]
                for k in range(frames_per_file):
                    np.testing.assert_array_equal(
                        data[k * extents[0] : (k + 1) * extents[0]], case["expected"]
                    )


def _replay(tmpdir, case, single_file, read_extra=None, check_extra=None):
    """Session B: read the files back and compare against the generator."""
    read = dict(
        kotekan_stage="hdf5FileRead",
        out_buf="read_buf",
        input_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        read_single_file=single_file,
    )
    read.update(read_extra or {})
    check = dict(
        kotekan_stage=case["checker"],
        first_buf="regen_buf",
        second_buf="read_buf",
        num_frames_to_test=NUM_FRAMES,
        check_metadata=True,
        epsilon=0.0,
    )
    check.update(check_extra or {})
    buffers = {**_buffer("regen_buf", case["buf"]), **_buffer("read_buf", case["buf"])}
    r = runner.KotekanRunner(
        buffers,
        {"regen": _generator(case, "regen_buf"), "read": read, "check": check},
        {},
    )
    r.run()
    return r


def _read_only(tmpdir, buf, read_extra=None):
    """A reader-only session that is expected to abort. Returns the runner."""
    read = dict(
        kotekan_stage="hdf5FileRead",
        out_buf="read_buf",
        input_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        read_single_file=True,
    )
    read.update(read_extra or {})
    r = runner.KotekanRunner(
        _buffer("read_buf", buf), {"read": read}, {}, expect_failure=True
    )
    r.run()
    return r


@pytest.mark.parametrize("single_file", [False, True], ids=["perframe", "singlefile"])
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_roundtrip(tmpdir_factory, case_name, single_file):
    case = CASES[case_name]
    if case["checker"] not in _available_stages():
        pytest.skip("%s not built" % case["checker"])
    tmpdir = tmpdir_factory.mktemp("hdf5_roundtrip")
    _siphon(tmpdir, case, single_file)
    _check_files(tmpdir, case, single_file)
    _replay(tmpdir, case, single_file)


def test_singlefile_missing_input_is_fatal(tmpdir_factory):
    case = CASES["int8_random"]
    tmpdir = tmpdir_factory.mktemp("hdf5_roundtrip_missing")
    r = _read_only(tmpdir, case["buf"])
    assert r.return_code != 0


def test_perframe_end_of_input_is_info(tmpdir_factory):
    """Running out of per-frame files is a normal end of input, not an error."""
    case = CASES["int8_random"]
    tmpdir = tmpdir_factory.mktemp("hdf5_perframe_eoi")
    _siphon(tmpdir, case, False)
    r = _replay(tmpdir, case, False)
    assert "terminating reader" in r.output
    # No ERROR from the reader stage, apart from the benign thread-affinity
    # complaints every stage logs in this environment.
    read_errors = [
        line
        for line in r.output.splitlines()
        if line.startswith("ERROR: /read")
        and "thread affinity" not in line
        and "thread name" not in line
    ]
    assert read_errors == []


def test_singlefile_do_once(tmpdir_factory):
    """`do_once` reads frame 0 and then idles instead of ending the pipeline."""
    case = CASES["int8_random"]
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_do_once")
    _siphon(tmpdir, case, True)
    r = _replay(
        tmpdir,
        case,
        True,
        read_extra=dict(do_once=True),
        check_extra=dict(num_frames_to_test=1),
    )
    assert "do_once is set" in r.output
    assert "Done reading" not in r.output


def test_singlefile_rejects_frame_decimation(tmpdir_factory):
    case = CASES["int8_random"]
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_decimation")
    r = _siphon(
        tmpdir,
        case,
        True,
        write_extra=dict(write_x_frames=0, per_y_frames=2),
        expect_failure=True,
    )
    assert r.return_code != 0
    assert "cannot be combined with create_single_file" in r.output


def test_singlefile_rejects_non_time_axis0(tmpdir_factory):
    """Axis 0 of a single file has to be a time axis (a name starting with T)."""
    case = _variant(
        CASES["int8_random"],
        gen=dict(dim_name=["F", "T", "D"]),
        buf=dict(dimnames=["F", "T", "D"]),
    )
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_axis0")
    r = _siphon(tmpdir, case, True, expect_failure=True)
    assert r.return_code != 0
    assert (
        "create_single_file requires a time axis as dimension 0, but dimension 0 is"
        in r.output
    )


def test_perframe_allows_non_time_axis0(tmpdir_factory):
    """The same non-time-major buffer round-trips fine as per-frame files."""
    case = _variant(
        CASES["int8_random"],
        gen=dict(dim_name=["F", "T", "D"]),
        buf=dict(dimnames=["F", "T", "D"]),
    )
    if case["checker"] not in _available_stages():
        pytest.skip("%s not built" % case["checker"])
    tmpdir = tmpdir_factory.mktemp("hdf5_perframe_axis0")
    _siphon(tmpdir, case, False)
    _check_files(tmpdir, case, False)
    _replay(tmpdir, case, False)


def test_singlefile_rejects_dim_scaling_tds_mismatch(tmpdir_factory):
    """dim_scaling[0] must equal time_downsampling_fpga in single-file mode."""
    case = _variant(CASES["int8_random"], gen=dict(meta_time_downsample_factor=16),)
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_scaling")
    r = _siphon(tmpdir, case, True, expect_failure=True)
    assert r.return_code != 0
    assert (
        "create_single_file requires dim_scaling[0] (1) == time_downsampling_fpga (16)"
        in r.output
    )


def test_singlefile_rejects_gap_in_stream(tmpdir_factory):
    """An fpga_seq_num jump (simulated FPGA restart) breaks the single stream."""
    case = _variant(CASES["int8_random"], gen=dict(simulate_fpga_restart_at_frame=3))
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_gap")
    r = _siphon(tmpdir, case, True, expect_failure=True)
    assert r.return_code != 0
    assert "create_single_file requires a contiguous stream" in r.output


def test_perframe_allows_gap_in_stream(tmpdir_factory):
    """Per-frame files carry their own fpga_seq_num, so a gap is fine there."""
    case = _variant(CASES["int8_random"], gen=dict(simulate_fpga_restart_at_frame=3))
    if case["checker"] not in _available_stages():
        pytest.skip("%s not built" % case["checker"])
    tmpdir = tmpdir_factory.mktemp("hdf5_perframe_gap")
    _siphon(tmpdir, case, False)
    _replay(tmpdir, case, False)


@pytest.fixture(scope="module")
def good_single_file(tmpdir_factory):
    """One valid single-file dump of the int8 case, written once.

    The reader-negative tests below copy it and corrupt the copy with h5py.
    """
    tmpdir = tmpdir_factory.mktemp("hdf5_singlefile_source")
    _siphon(tmpdir, CASES["int8_random"], True)
    path = os.path.join(str(tmpdir), FILE_NAME + ".h5")
    assert os.path.exists(path)
    return path


def _corrupt_copy(tmpdir_factory, good_single_file, name, corrupt):
    """Copy the good single file into a fresh directory and corrupt the copy."""
    tmpdir = tmpdir_factory.mktemp(name)
    path = os.path.join(str(tmpdir), FILE_NAME + ".h5")
    shutil.copy(good_single_file, path)
    with h5py.File(path, "r+") as f:
        corrupt(f[FILE_NAME])
    return tmpdir


def test_singlefile_read_rejects_truncated_file(tmpdir_factory, good_single_file):
    """Fewer samples along axis 0 than one frame is a read error."""
    case = CASES["int8_random"]
    extents = case["buf"]["extents"]

    def corrupt(ds):
        ds.resize((extents[0] - 1, *extents[1:]))

    tmpdir = _corrupt_copy(
        tmpdir_factory, good_single_file, "hdf5_read_truncated", corrupt
    )
    r = _read_only(tmpdir, case["buf"])
    assert r.return_code != 0
    assert "fewer than one frame" in r.output


def test_singlefile_read_rejects_storage_type_mismatch(
    tmpdir_factory, good_single_file
):
    """The HDF5 storage type has to agree with the declared `type` attribute."""
    case = CASES["int8_random"]

    def corrupt(ds):
        ds.attrs["type"] = "int32"

    tmpdir = _corrupt_copy(
        tmpdir_factory, good_single_file, "hdf5_read_storage_type", corrupt
    )
    r = _read_only(tmpdir, case["buf"])
    assert r.return_code != 0
    assert "storage type" in r.output


def test_singlefile_read_rejects_missing_attribute(tmpdir_factory, good_single_file):
    """A missing mandatory attribute is fatal, and the message names it."""
    case = CASES["int8_random"]

    def corrupt(ds):
        del ds.attrs["dim_names"]

    tmpdir = _corrupt_copy(
        tmpdir_factory, good_single_file, "hdf5_read_missing_attr", corrupt
    )
    r = _read_only(tmpdir, case["buf"])
    assert r.return_code != 0
    assert 'lacks the required attribute "dim_names"' in r.output


def test_singlefile_read_rejects_non_time_axis0(tmpdir_factory, good_single_file):
    """The reader enforces the time axis rule on what is actually in the file."""
    case = CASES["int8_random"]

    def corrupt(ds):
        ds.attrs["dim_names"] = ["X", "F", "D"]

    tmpdir = _corrupt_copy(tmpdir_factory, good_single_file, "hdf5_read_axis0", corrupt)
    # Leave `dimnames` out of the buffer declaration so the labels come from
    # the file; a declared "T" would be reported as a label conflict first.
    buf = {k: v for k, v in case["buf"].items() if k != "dimnames"}
    r = _read_only(tmpdir, buf)
    assert r.return_code != 0
    assert "read_single_file requires a time axis as dimension 0" in r.output


def test_singlefile_read_rejects_dim_scaling_tds_mismatch(
    tmpdir_factory, good_single_file
):
    """dim_scalings[0] != time_downsampling_fpga makes the frames unsplittable."""
    case = CASES["int8_random"]

    def corrupt(ds):
        ds.attrs["time_downsampling_fpga"] = np.int32(2)

    tmpdir = _corrupt_copy(
        tmpdir_factory, good_single_file, "hdf5_read_scaling", corrupt
    )
    r = _read_only(tmpdir, case["buf"])
    assert r.return_code != 0
    assert "dim_scalings[0] (1) != time_downsampling_fpga (2)" in r.output


def test_voltage_reader_roundtrip(tmpdir_factory):
    """hdf5FileReadSingleFile replays a single-file voltage dump.

    hdf5FileWrite writes the CHIME-style rank-4 voltage buffer, and the
    special-purpose reader hands it back one frame of `num_times` samples at a
    time. That stage never ends the pipeline; the checker does.
    """
    case = VOLTAGE_CASE
    stages = _available_stages()
    for stage in ("hdf5FileReadSingleFile", case["checker"]):
        if stage not in stages:
            pytest.skip("%s not built" % stage)
    tmpdir = tmpdir_factory.mktemp("hdf5_voltage")
    _siphon(tmpdir, case, True)
    _check_files(tmpdir, case, True)
    read = dict(
        kotekan_stage="hdf5FileReadSingleFile",
        out_buf="read_buf",
        input_dir=str(tmpdir),
        file_name=FILE_NAME,
        frequency_channels=list(range(case["gen"]["num_local_freq"])),
        num_times=case["buf"]["extents"][0],
    )
    check = dict(
        kotekan_stage=case["checker"],
        first_buf="regen_buf",
        second_buf="read_buf",
        num_frames_to_test=NUM_FRAMES,
        check_metadata=True,
        epsilon=0.0,
    )
    buffers = {**_buffer("regen_buf", case["buf"]), **_buffer("read_buf", case["buf"])}
    runner.KotekanRunner(
        buffers,
        {"regen": _generator(case, "regen_buf"), "read": read, "check": check},
        {},
    ).run()
