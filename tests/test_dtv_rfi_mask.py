"""Test DTV mask routing and correlator counts with synthetic inputs."""
import copy
import struct

import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from test_pilotproxy_integration import Pipeline, ROOT, reference, runtime

pytestmark = pytest.mark.serial
T, F, D, FRAMES = 8192, 4, 64, 8


def buffer(size):
    return dict(
        kotekan_buffer="standard",
        num_frames=4,
        frame_size=size,
        metadata_pool="main_pool",
    )


def dump(name):
    return dict(
        kotekan_stage="rawFileWrite",
        in_buf=f"host_{name}_buffer",
        file_name=name,
        file_ext="raw",
        prefix_hostname=False,
        num_frames_per_file=1,
        allow_ndarray=True,
        exit_after_n_files=FRAMES,
    )


def send(name, size):
    return {
        "gpu_0": dict(
            kotekan_stage="cudaProcess",
            gpu_id=0,
            in_buffers={f"host_{name}_buffer": f"host_{name}_buffer"},
            out_buffers={f"host_{name}_ringbuffer": f"host_{name}_ringbuffer"},
            commands=[
                dict(
                    name="cudaCopyToRingbuffer",
                    input_size=size,
                    ring_buffer_size=4 * size,
                    in_buf=f"host_{name}_buffer",
                    signal_buf=f"host_{name}_ringbuffer",
                    gpu_mem_output=f"{name}_buffer",
                )
            ],
        )
    }


def raw_payload(path, shape):
    with path.open("rb") as stream:
        offset = struct.unpack("<I", stream.read(4))[0] + 4
    return np.memmap(path, mode="r+", dtype=np.uint8, offset=offset, shape=shape)


def setup_pipeline(tmp_path, runtime, *, real_detector=False, fault=None):
    pipeline = Pipeline(tmp_path, runtime)
    pipeline.calibrate()
    cfg = pipeline.config
    cfg.update(num_dishes=D, num_frequencies=F)
    pipeline.generator["array_shape"] = [T, F, 2, D]
    # Generate metadata-bearing input files first, then introduce reproducible
    # packet loss and masks and replay through the actual production stages.
    for name, size, shape, dims, scales, kind, quantity, period in (
        (
            "rfi_RFImask",
            T * F // 8,
            [8, F, 128],
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
            "const1x8",
            "RFImask",
            4096,
        ),
        (
            "pl_expanded_mask",
            T * F * 2 * D // 64,
            [128, F, 2, 8, 8],
            ["Thi64", "F", "P", "D8", "Tlo64"],
            [64, 1, 1, 8, 8],
            "const1x8",
            "pl_mask_exp",
            256,
        ),
        ("dtv_mask", F, [F], ["F"], [1], "const8", "dtv_mask", T * 4),
    ):
        cfg[f"host_{name}_buffer"] = buffer(size)
        cfg[f"gen_{name}"] = dict(
            kotekan_stage="testDataGen",
            out_buf=f"host_{name}_buffer",
            type=kind,
            value=0,
            name=quantity,
            array_shape=shape,
            dim_name=dims,
            dim_scaling=scales,
            first_frame_index=17,
            samples_per_data_set=T * 4,
            num_frames=FRAMES,
            wait=False,
            manual_freq_ids=[2408, 1600, 2623, 4000],
            meta_time_downsample_factor=period,
        )
        cfg[f"dump_{name}"] = dump(name)
    # Frequency mismatch must fail even when the byte layout is identical.
    if fault == "frequency":
        cfg["gen_dtv_mask"]["manual_freq_ids"] = [1600, 2408, 2623, 4000]
    full = copy.deepcopy(cfg)
    for key in (
        "run_send_voltage",
        "run_dtv_detector",
        "dump_dtv_powers",
        "host_voltage_ringbuffer",
        "host_dtv_powers_buffer",
        "frame_arrival_period",
        "pilot_profiles_path",
        "weights_path",
        "samples_per_detector_frame",
        "num_frequencies",
    ):
        cfg.pop(key, None)
    code, log = pipeline.run()
    assert code == 0, log[-10000:]
    (tmp_path / "input").mkdir()
    for path in (tmp_path / "out").glob("*.raw"):
        path.rename(tmp_path / "input" / path.name)
    keep_masks, pl_masks, decisions = [], [], []
    for index in range(FRAMES):

        def source(name):
            return sorted((tmp_path / "input").glob(f"{name}_*.raw"))[index]

        # Nonuniform in both time and frequency, with accepted and rejected
        # samples on both sides of every 1024-sample layout boundary.
        t, f = np.indices((T, F))
        keep = (t + 3 * f + index) % 11 > 1
        keep[1023:1025, index % F] = False
        raw_payload(source("rfi_RFImask"), (8, F, 128))[:] = np.packbits(
            keep.reshape(8, 1024, F).transpose(0, 2, 1), axis=-1, bitorder="little"
        )
        t, f, group = np.indices((T, F, 16))
        pl = (t // 32 + 5 * f + group + index) % 7 != 0
        # Include wholly absent input groups and a wholly absent time block.
        pl[:, 0, 3] = False
        pl[4096:4160] = False
        raw_payload(source("pl_expanded_mask"), (128, F, 2, 8, 8))[:] = np.packbits(
            pl.reshape(128, 64, F, 2, 8).transpose(0, 2, 3, 4, 1),
            axis=-1,
            bitorder="little",
        )
        voltage = raw_payload(source("voltage"), (T, F, 128))
        voltage[~np.repeat(pl, 8, axis=2)] = 0x88  # offset-binary complex zero
        decision = np.array([(index + f) % 3 == 0 for f in range(F)], dtype=np.uint8)
        if index == 0:
            decision[:] = 0
        if index == 1:
            decision[:] = 1
        raw_payload(source("dtv_mask"), (F,))[:] = decision
        if index == 4 and fault in ("late", "missing", "period"):
            path = source("dtv_mask")
            data = bytearray(path.read_bytes())
            if fault in ("late", "missing"):
                seq = struct.unpack_from("<q", data, 20)[0]
                struct.pack_into(
                    "<q", data, 20, seq + (-1 if fault == "late" else T * 4)
                )
            else:
                struct.pack_into("<i", data, 28, T * 8)
            path.write_bytes(data)
        keep_masks.append(keep)
        pl_masks.append(pl)
        decisions.append(decision)
    pipeline.config = cfg = full
    for name in ("voltage", "rfi_RFImask", "pl_expanded_mask", "dtv_mask"):
        cfg.pop(f"gen_{name}", None)
        cfg.pop(f"dump_{name}", None)
        cfg[f"read_{name}"] = dict(
            kotekan_stage="rawFileRead",
            buf=f"host_{name}_buffer",
            base_dir="input",
            file_name=name,
            file_ext="raw",
            prefix_hostname=False,
            end_interrupt=False,
        )
    cfg.pop("samples_per_data_set")
    cfg.pop("num_gen_frames")
    for key in ("dump_dtv_mask", "dump_dtv_powers"):
        if key in cfg:
            cfg[key]["exit_after_n_files"] = FRAMES
    if real_detector:
        cfg.pop("read_dtv_mask")
        cfg["dump_dtv_mask"] = dump("dtv_mask")
    else:
        cfg.pop("run_dtv_detector")
        cfg.pop("dump_dtv_powers")
        cfg.pop("host_dtv_powers_buffer")
        for key in (
            "pilot_profiles_path",
            "weights_path",
            "samples_per_detector_frame",
        ):
            cfg.pop(key)
    cfg["host_dtv_RFImask_buffer"] = buffer(T * F // 8)
    cfg["combine"] = dict(
        kotekan_stage="DtvRfiMask",
        num_local_freq=F,
        rfi_buf="host_rfi_RFImask_buffer",
        dtv_buf="host_dtv_mask_buffer",
        out_buf="host_dtv_RFImask_buffer",
    )
    for name, dtype, extents, quantity, dims, scales in (
        (
            "rfi_RFImask",
            "uint1x8",
            [8, F, 128],
            "RFImask",
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
        ),
        (
            "dtv_RFImask",
            "uint1x8",
            [8, F, 128],
            "RFImask",
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
        ),
        ("dtv_mask", "int8", [F], "dtv_mask", ["F"], [1]),
    ):
        cfg[f"host_{name}_buffer"] = dict(
            kotekan_buffer="ndarray",
            num_frames=4,
            value_type=dtype,
            extents=extents,
            quantity_name=quantity,
            dimnames=dims,
            dimscalings=scales,
            metadata_pool="main_pool",
        )
    cfg["dump_combined"] = dump("dtv_RFImask")
    cfg["host_rficounts_buffer"] = dict(
        kotekan_buffer="ndarray",
        num_frames=4,
        value_type="int32",
        extents=[1, F],
        quantity_name="RFImask_counts",
        dimnames=["Tc", "F"],
        dimscalings=[T, 1],
        metadata_pool="main_pool",
    )
    cfg["count_rfi"] = dict(
        kotekan_stage="RfiMaskSum",
        num_local_freq=F,
        samples_per_data_set=T,
        sub_integration_ntime=T,
        rfi_downsampling_factor=1,
        in_buf="host_dtv_RFImask_buffer",
        out_buf="host_rficounts_buffer",
    )
    cfg["dump_rficounts"] = dump("rficounts")
    for name, size in (
        ("dtv_RFImask", T * F // 8),
        ("pl_expanded_mask", T * F * 2 * D // 64),
    ):
        cfg[f"host_{name}_ringbuffer"] = dict(
            kotekan_buffer="ring", ring_buffer_size=4 * size, metadata_pool="main_pool"
        )
        cfg[f"send_{name}"] = send(name, size)
    for product, command, input_name, size in (
        ("n2k_correlation", "cudaCorrelator", "voltage", F * 36 * 16 * 16 * 2 * 4),
        ("n2k_counts", "cudaPL1bitCorrelator", "pl_expanded_mask", F * 3 * 8 * 8 * 4),
    ):
        cfg[f"host_{product}_buffer"] = buffer(size)
        cmd = dict(
            name=command,
            num_elements=128,
            num_local_freq=F,
            sub_integration_ntime=T,
            rfi_RFImask_name="dtv_RFImask",
            **{f"{input_name}_name": input_name, f"{product}_name": product},
        )
        if command == "cudaPL1bitCorrelator":
            cmd.pop("num_elements")
            cmd.pop("num_local_freq")
        cfg[f"run_{product}"] = {
            "gpu_0": dict(
                kotekan_stage="cudaProcess",
                gpu_id=0,
                in_buffers={
                    f"host_{input_name}_ringbuffer": f"host_{input_name}_ringbuffer",
                    "host_dtv_RFImask_ringbuffer": "host_dtv_RFImask_ringbuffer",
                },
                out_buffers={f"host_{product}_buffer": f"host_{product}_buffer"},
                commands=[
                    cmd,
                    dict(name="cudaSyncOutput"),
                    dict(
                        name="cudaOutputData",
                        gpu_mem=f"{product}_buffer",
                        out_buf=f"host_{product}_buffer",
                    ),
                ],
            )
        }
        cfg[f"dump_{product}"] = dump(product)
    return pipeline, keep_masks, pl_masks, decisions


def triangles(matrix, block):
    n = matrix.shape[-1] // block
    return np.stack(
        [
            matrix[..., i * block : (i + 1) * block, j * block : (j + 1) * block]
            for i in range(n)
            for j in range(i + 1)
        ],
        axis=-3,
    )


@pytest.mark.parametrize("real_detector", [False, True])
def test_mask_visibility_and_counts_across_ring_wrap(tmp_path, runtime, real_detector):
    pipeline, masks, packet_masks, decisions = setup_pipeline(
        tmp_path, runtime, real_detector=real_detector
    )
    code, log = pipeline.run()
    assert code == 0, log[-12000:]

    def read(folder, name, size):
        result = reference.read_raw_frames(
            str(tmp_path / folder / f"{name}_*.raw"), size
        )
        assert len(result) == FRAMES
        return result

    voltage = read("input", "voltage", T * F * 128)
    combined = read("out", "dtv_RFImask", T * F // 8)
    vis = read("out", "n2k_correlation", F * 36 * 16 * 16 * 2 * 4)
    counts = read("out", "n2k_counts", F * 3 * 8 * 8 * 4)
    rficounts = read("out", "rficounts", F * 4)
    if real_detector:
        decisions = [r.payload for r in read("out", "dtv_mask", F)]
        assert set(np.concatenate(decisions)) == {0, 1}
    for index in range(FRAMES):
        good = masks[index] & ~decisions[index].astype(bool)[None, :]
        packed = np.packbits(
            good.reshape(8, 1024, F).transpose(0, 2, 1), axis=-1, bitorder="little"
        )
        np.testing.assert_array_equal(
            combined[index].payload.reshape(8, F, 128), packed
        )
        np.testing.assert_array_equal(
            rficounts[index].payload.view("<i4"), (~good).sum(axis=0)
        )
        for product, period in ((combined, 4096), (vis, T * 4), (counts, T * 4)):
            assert product[index].fpga_seq_num == (17 + index) * T * 4
            assert product[index].time_downsampling_fpga == period
        raw = voltage[index].payload.reshape(T, F, 128).astype(np.int16)
        z = ((raw >> 4) - 8 + 1j * ((raw & 15) - 8)).astype(np.complex64)
        z *= good[:, :, None]
        expected_vis = np.stack([z[:, f].T @ z[:, f].conj() for f in range(F)])
        got_vis = vis[index].payload.view("<i4").reshape(F, 36, 16, 16, 2)
        want_vis = triangles(expected_vis, 16)
        np.testing.assert_array_equal(got_vis[..., 0], want_vis.real)
        np.testing.assert_array_equal(got_vis[..., 1], want_vis.imag)
        p = (packet_masks[index] & good[:, :, None]).astype(np.float64)
        expected_counts = np.stack([p[:, f].T @ p[:, f] for f in range(F)])
        got_counts = counts[index].payload.view("<i4").reshape(F, 3, 8, 8)
        np.testing.assert_array_equal(got_counts, triangles(expected_counts, 8))
        # Products without accepted samples have no measured visibility.
        assert np.any(expected_counts == 0)
        if not real_detector and index == 1:
            assert not got_counts.any() and not got_vis.any()


@pytest.mark.parametrize("fault", ["frequency", "late", "missing", "period"])
def test_refuse_wrong_or_missing_decision_identity(tmp_path, runtime, fault):
    pipeline, *_ = setup_pipeline(tmp_path, runtime, fault=fault)
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "DtvRfiMask" in log and ("mismatch" in log), log[-12000:]
    assert len(list((tmp_path / "out").glob("dtv_RFImask_*.raw"))) < FRAMES


@pytest.mark.parametrize("apply", [False, True])
def test_template_routes_both_correlators_and_rfi_counts(apply):
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/fengine")))
    source = env.get_template("chord.j2").render(dtv_apply_mask=apply)
    cfg = yaml.safe_load(source)
    name = "dtv_RFImask" if apply else "rfi_RFImask"
    for stage in ("run_n2k_correlation", "run_n2k_counts"):
        gpu = cfg[stage]["gpu_0"]
        assert gpu["commands"][0]["rfi_RFImask_name"] == name
        assert f"host_{name}_ringbuffer" in gpu["in_buffers"]
    assert cfg["run_n2_counting"]["count_rfi_mask"]["in_buf"] == f"host_{name}_buffer"
    assert ("run_combine_dtv_mask" in cfg) is apply
