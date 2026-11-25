"""Test the CUDA pipeline using synthetic calibration and malformed bundles.

The binary and a matching runtime bundle are given by PILOTPROXY_TEST_BINARY
and PILOTPROXY_TEST_BUNDLE. PILOTPROXY_SOAK_FRAMES (at least 8) also enables
the full-size CHORD and CHIME cases described in dev_pilotproxy_testing.rst.

Those cases repeat four voltage frames with two polarizations and full-rate
fine support. Every mask, power, and support output is checked against its
CPU reference. The timeout of each pipeline is 1800 seconds unless
PILOTPROXY_SOAK_TIMEOUT_SECONDS is set. The calibration is synthetic and applies
only to these tests.
"""

import importlib.util
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import time

import numpy as np
import pytest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pilotproxy_integration_reference", ROOT / "tests/verify_pilotproxy_pipeline.py"
)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
pytestmark = pytest.mark.serial


@pytest.fixture(scope="module")
def runtime():
    # The binary is taken from the usual build tree (KOTEKAN_BUILD_DIRNAME) unless
    # overridden. The bundle is an extra input, and the CUDA tests skip without it.
    binary = os.environ.get("PILOTPROXY_TEST_BINARY")
    if not binary:
        from kotekan import runner

        binary = runner.KotekanRunner.kotekan_binary()
    bundle = os.environ.get("PILOTPROXY_TEST_BUNDLE")
    if not bundle:
        pytest.skip("set PILOTPROXY_TEST_BUNDLE for CUDA tests")
    binary, bundle = Path(binary).resolve(), Path(bundle).resolve()
    assert binary.is_file(), binary
    assert (bundle / "pilot_profiles.json").is_file(), bundle
    assert (bundle / "weights.bin").is_file(), bundle
    # A requested GPU run must fail if its Python reference is unavailable.
    from pilot_proxy.fxfft import fine_power_fx

    assert callable(fine_power_fx)
    return binary, bundle


class Pipeline:
    def __init__(self, directory, runtime):
        self.directory = directory
        self.binary, source_bundle = runtime
        shutil.copytree(source_bundle, directory / "bundle")
        (directory / "out").mkdir()
        self.bundle = json.loads((directory / "bundle/pilot_profiles.json").read_text())
        self.config = yaml.safe_load(
            (ROOT / "config/tests/verify_pilotproxy_pipeline.yaml").read_text()
        )
        self.config.update(
            base_dir="out",
            pilot_profiles_path="bundle/pilot_profiles.json",
            weights_path="bundle/weights.bin",
        )
        self.config["gen_voltage"]["wait"] = False
        # Six frames still wrap the ring of buffer_depth (4) frames; the random voltage fill
        # takes most of a run.
        self.config["num_gen_frames"] = 6
        for block in self.config.values():
            if isinstance(block, dict) and "exit_after_n_files" in block:
                block["exit_after_n_files"] = 6
        self.generator = self.config["gen_voltage"]
        self.detector = self.config["run_dtv_detector"]["gpu_0"]["commands"][0]
        window_samples = self.bundle["profiles"][0]["weight_bank_nbytes"] // 3
        self.detector["detector_window_samples"] = window_samples
        self.block_samples = window_samples * 128
        self.config["num_times"] = self.block_samples
        self.generator["array_shape"][0] = self.block_samples
        namespace = self.bundle.get("receiver_channel_id_namespace", "chord_freq_id")
        self.detector["receiver_channel_id_namespace"] = namespace
        if namespace == "chime_freq_id":
            self.config["frame_arrival_period"] = "2.56e-6 * num_times"
            ids = [
                row["receiver_channel_id"]
                for row in self.bundle["profiles"]
                if row.get("receiver_channel_id") is not None
            ]
            absent = [i for i in range(1024) if i not in ids]
            self.generator["manual_freq_ids"] = [ids[0], absent[0], ids[1], absent[1]]
        self.elapsed = 0.0

    def calibrate(self):
        reference.synthetic_calibration(self.bundle)

    def run(self, timeout=60, seal=True):
        if seal:
            reference.seal_bundle(self.directory / "bundle", self.bundle)
        (self.directory / "pipeline.yaml").write_text(yaml.safe_dump(self.config))
        start = time.monotonic()
        with (self.directory / "pipeline.log").open("w") as log:
            result = subprocess.run(
                [str(self.binary), "-b", "127.0.0.1:0", "--config", "pipeline.yaml"],
                cwd=self.directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        self.elapsed = time.monotonic() - start
        output = (self.directory / "pipeline.log").read_text()
        return result.returncode, output

    def verify(self, *, repeated_seeds=False, fine_support=False):
        config = self.config
        block_samples = self.block_samples
        freq_ids = self.generator["manual_freq_ids"]
        num_freq, num_dishes = len(freq_ids), config["num_dishes"]
        num_frames = config["num_gen_frames"]
        num_times = config["num_times"]
        num_blocks = num_frames * num_times // block_samples
        seed_count = (
            min(config["buffer_depth"], num_frames) if repeated_seeds else num_frames
        )
        voltage = reference.read_raw_frames(
            str(self.directory / "out/voltage_*.raw"),
            num_times * num_freq * 2 * num_dishes,
        )
        masks = reference.read_raw_frames(
            str(self.directory / "out/dtv_mask_*.raw"), num_freq
        )
        powers = reference.read_raw_frames(
            str(self.directory / "out/dtv_powers_*.raw"), num_freq * 3 * 8
        )
        assert (len(voltage), len(masks), len(powers)) == (
            seed_count,
            num_blocks,
            num_blocks,
        )
        downsample = self.generator["meta_time_downsample_factor"]
        start = self.generator["first_frame_index"] * num_times * downsample
        products = [
            (voltage, downsample, num_times),
            (masks, block_samples * downsample, block_samples),
            (powers, block_samples * downsample, block_samples),
        ]
        if fine_support:
            support = reference.read_raw_frames(
                str(self.directory / "out/dtv_fine_support_*.raw"), num_freq * 2 * 4
            )
            assert len(support) == num_blocks
            products.append((support, block_samples * downsample, block_samples))
        for frames, factor, stride in products:
            for index, frame in enumerate(frames):
                assert frame.fpga_seq_num == start + index * stride * downsample
                assert frame.time_downsampling_fpga == factor
        weights = (self.directory / "bundle/weights.bin").read_bytes()
        # Avoid another large copy in the production geometry test.
        if num_times == block_samples:
            blocks = [frame.payload for frame in voltage]
        else:
            blocks = np.concatenate([frame.payload for frame in voltage]).reshape(
                -1, block_samples * num_freq * 2 * num_dishes
            )
        expected = [
            reference.expected_products(
                block, freq_ids, self.bundle, weights, num_dishes=num_dishes
            )
            for block in blocks
        ]
        for index, (mask, power) in enumerate(zip(masks, powers)):
            want = expected[index % len(expected)]
            want_mask, want_power = want[:2]
            np.testing.assert_array_equal(
                mask.payload, want_mask, err_msg=f"block {index} mask"
            )
            got_power = power.payload.view("<u8").reshape(num_freq, 3)
            np.testing.assert_array_equal(
                got_power, want_power, err_msg=f"block {index} powers"
            )
            if fine_support:
                got_support = support[index].payload.view("<i4").reshape(num_freq, 2)
                np.testing.assert_array_equal(
                    got_support, want[2], err_msg=f"block {index} fine support"
                )
        return np.stack([pair[0] for pair in expected])


@pytest.fixture
def pipeline(tmp_path, runtime):
    return Pipeline(tmp_path, runtime)


def add_output(pipeline, product, frame_size):
    """Add an optional detector output and a raw writer for it."""
    config = pipeline.config
    name = f"dtv_{product}"
    config[f"host_{name}_buffer"] = copy.deepcopy(config["host_dtv_powers_buffer"])
    config[f"host_{name}_buffer"]["frame_size"] = frame_size
    gpu = config["run_dtv_detector"]["gpu_0"]
    gpu["out_buffers"][f"host_{name}"] = f"host_{name}_buffer"
    pipeline.detector[f"{name}_name"] = name
    gpu["commands"].append(
        dict(
            name="cudaOutputData",
            cuda_stream=4,
            gpu_mem=f"{name}_buffer",
            out_buf=f"host_{name}",
        )
    )
    config[f"dump_{name}"] = copy.deepcopy(config["dump_dtv_mask"])
    config[f"dump_{name}"].update(in_buf=f"host_{name}_buffer", file_name=name)


def bound_indices(pipeline):
    ids = pipeline.generator["manual_freq_ids"]
    bound = {p["receiver_channel_id"] for p in pipeline.bundle["profiles"]}
    return [f for f, freq in enumerate(ids) if freq in bound]


# The production setting: the control is recorded and not masked
@pytest.mark.parametrize("mask_control", [False])
def test_control_channel_is_recorded_and_masked_only_with_mask_control(
    pipeline, mask_control
):
    pipeline.calibrate()
    pipeline.detector["mask_control"] = mask_control
    profiles = pipeline.bundle["profiles"]
    for profile in profiles:
        profile["fine_calibration"]["cfar_multiplier_q16"] = 1
    for profile in profiles[1:]:
        # a bundle with its own control row gives that row the channel of profile 0
        if profile["physical_channel"] == 37:
            profile["physical_channel"] = profiles[0]["physical_channel"]
            profile["fine_calibration"]["status"] = "calibrated"
    profiles[0]["fine_calibration"]["status"] = "control"
    profiles[0]["physical_channel"] = 37
    ids = pipeline.generator["manual_freq_ids"]
    num_freq = len(ids)
    add_output(pipeline, "fine_support", num_freq * 2 * 4)
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    out = pipeline.directory / "out"
    masks = reference.read_raw_frames(str(out / "dtv_mask_*.raw"), num_freq)
    powers = reference.read_raw_frames(str(out / "dtv_powers_*.raw"), num_freq * 24)
    supports = reference.read_raw_frames(
        str(out / "dtv_fine_support_*.raw"), num_freq * 8
    )
    mask = np.array([frame.payload.view("i1") for frame in masks])
    power = np.array([frame.payload.view("<u8") for frame in powers])
    power = power.reshape(len(powers), num_freq, 3)
    support = np.array([frame.payload.view("<i4") for frame in supports])
    support = support.reshape(len(supports), num_freq, 2)
    control = ids.index(profiles[0]["receiver_channel_id"])
    others = [
        ids.index(p["receiver_channel_id"])
        for p in profiles[1:]
        if p["receiver_channel_id"] in ids
    ]
    assert power[:, control].any()
    if mask_control:
        # at multiplier 1 the control fires and gets rank support, as the pilots do
        assert mask[:, control].any()
        assert (support[:, control, 0] >= 0).all()
    else:
        assert not mask[:, control].any()
        # DtvRfiMask clears a frequency with support (0, n), so the control row has none
        assert (support[:, control] == -1).all()
    assert (support[:, others, 0] >= 0).all()
    assert mask[:, others].any()


@pytest.mark.parametrize("lifetime_blocks", [2])
def test_bad_feed_mask_zeroes_flagged_inputs(pipeline, lifetime_blocks):
    pipeline.calibrate()
    config = pipeline.config
    num_dishes = config["num_dishes"]
    lifetime = lifetime_blocks * pipeline.block_samples
    downsample = pipeline.generator["meta_time_downsample_factor"]
    # both streams must start at the same sequence number
    pipeline.generator["first_frame_index"] = 16
    config["bf_mask_lifetime_in_samples"] = lifetime
    config["host_bf_mask_buffer"] = {
        "kotekan_buffer": "ndarray",
        "num_frames": "buffer_depth",
        "value_type": "int8",
        "extents": [1, "num_polarizations", "num_dishes"],
        "quantity_name": "bf_mask",
        "dimnames": ["Tbf", "P", "D"],
        "dimscalings": ["bf_mask_lifetime_in_samples", 1, 1],
        "metadata_pool": "main_pool",
    }
    config["host_bf_mask_ringbuffer"] = {
        "kotekan_buffer": "ring",
        "ring_buffer_size": "buffer_depth * num_polarizations * num_dishes",
        "metadata_pool": "main_pool",
    }
    # elements alternate between all good and all bad, so a stale element shows in the powers
    config["gen_bf_mask"] = {
        "kotekan_stage": "testDataGen",
        "out_buf": "host_bf_mask_buffer",
        "type": "const8",
        "values": [1, 0],
        "wait": False,
        "samples_per_data_set": lifetime * downsample,
        "first_frame_index": 16 // lifetime_blocks,
        "num_frames": config["num_gen_frames"] // lifetime_blocks,
        "name": "bf_mask",
        "array_shape": [1, 2, num_dishes],
        "dim_name": ["Tbf", "P", "D"],
        "dim_scaling": [lifetime, 1, 1],
        "manual_freq_ids": pipeline.generator["manual_freq_ids"],
        "meta_time_downsample_factor": downsample,
    }
    config["dump_bf_mask"] = dict(
        config["dump_voltage"], in_buf="host_bf_mask_buffer", file_name="bf_mask"
    )
    del config["dump_bf_mask"]["exit_after_n_files"]
    config["run_send_bf_mask"] = {
        "gpu_0": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": 0,
            "in_buffers": {"host_bf_mask_buffer_in": "host_bf_mask_buffer"},
            "out_buffers": {"host_bf_mask_ringbuffer": "host_bf_mask_ringbuffer"},
            "commands": [
                {
                    "name": "cudaCopyToRingbuffer",
                    "input_size": "num_polarizations * num_dishes",
                    "ring_buffer_size": "buffer_depth * input_size",
                    "in_buf": "host_bf_mask_buffer_in",
                    "signal_buf": "host_bf_mask_ringbuffer",
                    "gpu_mem_output": "bf_mask_buffer",
                }
            ],
        }
    }
    stage = config["run_dtv_detector"]["gpu_0"]
    stage["in_buffers"]["host_bf_mask_ringbuffer"] = "host_bf_mask_ringbuffer"
    pipeline.detector["bf_mask_name"] = "bf_mask"
    inputs = 2 * num_dishes
    for profile in pipeline.bundle["profiles"]:
        profile["fine_calibration"]["provenance"]["summed_inputs"] = inputs
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    # the all-bad elements are logged against the calibrated count, and the all-good ones clear it
    assert f"0 inputs are summed but the bundle was calibrated with {inputs}" in output
    assert (
        f"{inputs} inputs are summed, within 10% of the calibrated {inputs}" in output
    )

    masks = reference.read_raw_frames(
        str(pipeline.directory / "out/bf_mask_*.raw"), 2 * num_dishes
    )
    assert len(masks) == config["num_gen_frames"] // lifetime_blocks
    assert len({frame.payload.tobytes() for frame in masks}) > 1

    # the stage must match a reference that sees the flagged inputs as zero
    paths = sorted((pipeline.directory / "out").glob("voltage_*.raw"))
    for block, path in enumerate(paths):
        with path.open("rb") as stream:
            metadata_size = struct.unpack("<I", stream.read(4))[0]
        data = np.memmap(
            path,
            dtype=np.uint8,
            mode="r+",
            offset=metadata_size + 4,
            shape=(pipeline.block_samples, 4, 2 * num_dishes),
        )
        bad = masks[block // lifetime_blocks].payload.view(np.int8) == 0
        data[:, :, bad] = 0x88
        data.flush()
        del data
    pipeline.verify()


# One field for each integer type the loader reads
@pytest.mark.parametrize(
    "namespace,num_dishes,num_freq,pilot_start",
    [
        ("chord_freq_id", 64, 384, 0),
        ("chord_freq_id", 512, 48, 0),
        ("chime_freq_id", 1024, 16, 0),
        ("chime_freq_id", 1024, 16, 15),
    ],
    ids=["pathfinder-padded", "full-chord", "chime-pilots-14-28", "chime-pilots-29-36"],
)
def test_production_geometry_soak(
    pipeline, record_property, namespace, num_dishes, num_freq, pilot_start
):
    requested = os.environ.get("PILOTPROXY_SOAK_FRAMES")
    if requested is None:
        pytest.skip("set PILOTPROXY_SOAK_FRAMES to run the production geometry soak")
    frames = int(requested)
    assert frames >= 8, "use at least eight frames to wrap the ring buffer"
    if pipeline.detector["receiver_channel_id_namespace"] != namespace:
        pytest.skip(f"geometry requires a {namespace} bundle")
    window_samples = 128 if namespace == "chime_freq_id" else 64
    assert pipeline.block_samples == window_samples * 128
    pipeline.calibrate()
    pilot_ids = [
        row.get("receiver_channel_id", row.get("chord_channel_id"))
        for row in pipeline.bundle["profiles"]
    ]
    # CHIME has 16 channels per GPU, so the pilots are covered in two batches with
    # at least one unbound channel in each. CHORD fits all 23 in either geometry.
    ids = pilot_ids[pilot_start : pilot_start + num_freq - 1]
    bound_profiles = len(ids)
    assert bound_profiles > 0
    unbound = range(1024) if namespace == "chime_freq_id" else range(1600, 2400)
    ids += [freq for freq in unbound if freq not in pilot_ids][: num_freq - len(ids)]
    assert len(ids) == num_freq and len(set(ids)) == num_freq
    pipeline.config.update(
        num_dishes=num_dishes,
        num_local_freq=num_freq,
        num_gen_frames=frames,
        samples_per_data_set="num_times",
    )
    pipeline.config["gen_voltage"].update(
        array_shape=[pipeline.block_samples, num_freq, 2, num_dishes],
        manual_freq_ids=ids,
        reuse_random=True,
        meta_time_downsample_factor=1,
    )
    pipeline.config["dump_voltage"]["exit_after_n_files"] = 4
    for product in ("mask", "powers"):
        pipeline.config[f"dump_dtv_{product}"]["exit_after_n_files"] = frames
    config = pipeline.config
    config["host_dtv_fine_support_buffer"] = copy.deepcopy(
        config["host_dtv_powers_buffer"]
    )
    config["host_dtv_fine_support_buffer"]["frame_size"] = num_freq * 2 * 4
    gpu = config["run_dtv_detector"]["gpu_0"]
    gpu["out_buffers"]["host_dtv_fine_support"] = "host_dtv_fine_support_buffer"
    pipeline.detector["dtv_fine_support_name"] = "dtv_fine_support"
    gpu["commands"].append(
        dict(
            name="cudaOutputData",
            cuda_stream=4,
            gpu_mem="dtv_fine_support_buffer",
            out_buf="host_dtv_fine_support",
        )
    )
    config["dump_dtv_fine_support"] = copy.deepcopy(config["dump_dtv_mask"])
    config["dump_dtv_fine_support"].update(
        in_buf="host_dtv_fine_support_buffer", file_name="dtv_fine_support"
    )
    timeout = float(os.environ.get("PILOTPROXY_SOAK_TIMEOUT_SECONDS", "1800"))
    assert np.isfinite(timeout) and timeout > 0
    code, output = pipeline.run(timeout=timeout)
    assert code == 0, output[-12000:]
    verification_start = time.monotonic()
    pipeline.verify(repeated_seeds=True, fine_support=True)
    record_property("frames", frames)
    record_property("pipeline_seconds", pipeline.elapsed)
    record_property("verification_seconds", time.monotonic() - verification_start)
    record_property("fine_support_rows_checked", frames * num_freq)
    record_property("bound_fine_profiles", bound_profiles)
    record_property("receiver_channel_id_namespace", namespace)
    record_property("block_samples", pipeline.block_samples)
    record_property("num_dishes", num_dishes)
    record_property("num_frequencies", num_freq)
    record_property("pipeline_timeout_seconds", timeout)
    record_property(
        "voltage_gib",
        frames * pipeline.block_samples * num_freq * 2 * num_dishes / 2**30,
    )
    print(
        f"Verified {frames} production-size blocks; pipeline took {pipeline.elapsed:.2f}s"
    )


# Time-reversed windows, and detector blocks that span two input frames
@pytest.mark.parametrize("reverse,frame_blocks", [(True, 0.5)])
def test_receiver_geometry_against_cpu(pipeline, reverse, frame_blocks):
    """Check either compiled geometry against independent integer references, with ring reuse."""
    pipeline.calibrate()
    pipeline.bundle["input_preprocessing"][
        "time_reverse_detector_windows_before_kernel"
    ] = reverse
    samples = int(pipeline.block_samples * frame_blocks)
    config = pipeline.config
    config["num_times"] = samples
    pipeline.generator["array_shape"][0] = samples
    blocks = config["num_gen_frames"] * samples // pipeline.block_samples
    for product in ("mask", "powers"):
        config[f"dump_dtv_{product}"]["exit_after_n_files"] = blocks
    config["host_dtv_fine_support_buffer"] = copy.deepcopy(
        config["host_dtv_powers_buffer"]
    )
    config["host_dtv_fine_support_buffer"]["frame_size"] = config["num_local_freq"] * 8
    gpu = config["run_dtv_detector"]["gpu_0"]
    gpu["out_buffers"]["host_dtv_fine_support"] = "host_dtv_fine_support_buffer"
    pipeline.detector["dtv_fine_support_name"] = "dtv_fine_support"
    gpu["commands"].append(
        dict(
            name="cudaOutputData",
            cuda_stream=4,
            gpu_mem="dtv_fine_support_buffer",
            out_buf="host_dtv_fine_support",
        )
    )
    config["dump_dtv_fine_support"] = copy.deepcopy(config["dump_dtv_mask"])
    config["dump_dtv_fine_support"].update(
        in_buf="host_dtv_fine_support_buffer", file_name="dtv_fine_support"
    )
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.verify(fine_support=True)


def test_v3_statistic_matches_the_reference(pipeline):
    """A fine_decision_v3 bundle writes [n_peak, d_rho] and leaves the decision to DtvRfiMask."""
    from pilot_proxy.fine_decision import fine_stat_v3

    pipeline.calibrate()
    for profile in pipeline.bundle["profiles"]:
        calibration = profile["fine_calibration"]
        calibration["decision_version"] = "fine_decision_v3"
        calibration.pop("bulk_mask_words_hex", None)
        calibration.pop("cfar_rank", None)
        calibration.pop("anchor_bin", None)
        calibration.pop("designated_half_width", None)
    num_freq = len(pipeline.generator["manual_freq_ids"])
    bound = bound_indices(pipeline)
    rows = len(bound)
    add_output(pipeline, "stat", num_freq * 2 * 8)
    add_output(pipeline, "fine_support", num_freq * 2 * 4)
    add_output(pipeline, "fine_powers", rows * 3 * 256 * 8)
    pipeline.detector["dtv_fine_powers_rows"] = rows
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    out = pipeline.directory / "out"
    stats = reference.read_raw_frames(str(out / "dtv_stat_*.raw"), num_freq * 2 * 8)
    fines = reference.read_raw_frames(
        str(out / "dtv_fine_powers_*.raw"), rows * 3 * 256 * 8
    )
    masks = reference.read_raw_frames(str(out / "dtv_mask_*.raw"), num_freq)
    supports = reference.read_raw_frames(
        str(out / "dtv_fine_support_*.raw"), num_freq * 2 * 4
    )
    assert len(stats) == len(fines) == len(masks) == len(supports) > 0
    for stat, fine, mask, support in zip(stats, fines, masks, supports):
        pairs = stat.payload.view("<u8").reshape(num_freq, 2)
        fine = fine.payload.view("<u8").reshape(rows, 3, 256)
        for f in range(num_freq):
            if f not in bound:
                # not evaluated: every word keeps it
                assert pairs[f].tolist() == [0, 2**64 - 1], f
                continue
            want = fine_stat_v3(np.ascontiguousarray(fine[bound.index(f)]))
            assert pairs[f].tolist() == [want.n_peak, want.d_rho], f
        # v3 makes no decision in the stage
        assert not mask.payload.any()
        assert (support.payload.view("<i4") == -1).all()
