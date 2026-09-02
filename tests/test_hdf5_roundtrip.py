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
"""
import glob
import os

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


def _siphon(tmpdir, case, single_file):
    write = dict(
        kotekan_stage="hdf5FileWrite",
        in_buf="siphon_buf",
        base_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        max_frames=NUM_FRAMES,
        create_single_file=single_file,
    )
    runner.KotekanRunner(
        _buffer("siphon_buf", case["buf"]),
        {"gen": _generator(case, "siphon_buf"), "write": write},
        {},
    ).run()


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


def _replay(tmpdir, case, single_file):
    read = dict(
        kotekan_stage="hdf5FileRead",
        out_buf="read_buf",
        input_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        read_single_file=single_file,
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
    read = dict(
        kotekan_stage="hdf5FileRead",
        out_buf="read_buf",
        input_dir=str(tmpdir),
        file_name=FILE_NAME,
        prefix_hostname=False,
        read_single_file=True,
    )
    r = runner.KotekanRunner(
        _buffer("read_buf", case["buf"]), {"read": read}, {}, expect_failure=True
    )
    r.run()
    assert r.return_code != 0
