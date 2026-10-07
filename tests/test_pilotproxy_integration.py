"""Test the CUDA pipeline with synthetic calibration and malformed bundles.

Set PILOTPROXY_TEST_BINARY and PILOTPROXY_TEST_BUNDLE to run these tests.
The bundle must match the compiled core. PILOTPROXY_SOAK_FRAMES (at least 8)
also enables 64-dish/384-channel and 512-dish/48-channel runs, each with two
polarizations and full-rate fine support. Four voltage frames repeat; every
mask, power, and support output is compared with its CPU reference. Each
pipeline has a 1,800-second timeout, overridable with
PILOTPROXY_SOAK_TIMEOUT_SECONDS. Synthetic calibration is for these tests only.
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

from kotekan import runner
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pilotproxy_integration_reference", ROOT / "tools/verify_pilotproxy_pipeline.py"
)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
pytestmark = pytest.mark.serial


def expected_fine_products_with_support(block, freq_ids, bundle, weights, num_dishes):
    """Calculate all three outputs once from the independent CPU references."""
    from pilot_proxy.fine_decision import fine_mask_decision
    from pilot_proxy.fxfft import fine_power_fx

    profiles = {row["chord_channel_id"]: row for row in bundle["profiles"]}
    reverse = bundle["input_preprocessing"][
        "time_reverse_detector_windows_before_kernel"
    ]
    masks = np.zeros(len(freq_ids), dtype=np.uint8)
    powers = np.zeros((len(freq_ids), 3), dtype=np.uint64)
    support = np.full((len(freq_ids), 2), -1, dtype=np.int32)
    for index, freq_id in enumerate(freq_ids):
        profile = profiles.get(freq_id)
        if profile is None:
            continue
        calibration = profile["fine_calibration"]
        assert calibration["status"] == "calibrated"
        offset = profile["weight_bank_offset_bytes"]
        bank = weights[offset : offset + profile["weight_bank_nbytes"]]
        real, imag = reference.detector_rows_from_voltage(
            block, index, reverse, num_freq=len(freq_ids), num_dishes=num_dishes
        )
        projections = reference.expected_row_sums(real, imag, bank)
        powers[index] = (projections * projections).sum(axis=(1, 2), dtype=np.uint64)
        fine = fine_power_fx(projections, num_streams=2 * num_dishes)
        decision = fine_mask_decision(
            fine,
            anchor_bin=calibration["anchor_bin"],
            designated_half_width=calibration["designated_half_width"],
            bulk_mask=[int(word, 16) for word in calibration["bulk_mask_words_hex"]],
            cfar_rank=calibration["cfar_rank"],
            multiplier_q16=calibration["cfar_multiplier_q16"],
        )
        masks[index] = decision.mask
        support[index] = (int(decision.valid), decision.n_bulk)
    return masks, powers, support


@pytest.fixture(scope="module")
def runtime():
    # The binary comes from the usual build tree (KOTEKAN_BUILD_DIRNAME) unless overridden;
    # the bundle is a separate input, so without it the CUDA tests skip.
    binary = (
        os.environ.get("PILOTPROXY_TEST_BINARY")
        or runner.KotekanRunner.kotekan_binary()
    )
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
        self.generator = self.config["gen_voltage"]
        self.detector = self.config["run_dtv_detector"]["gpu_0"]["commands"][0]
        self.elapsed = 0.0

    def calibrate(self):
        reference.synthetic_calibration(self.bundle)

    def run(self, timeout=60):
        (self.directory / "bundle/pilot_profiles.json").write_text(
            json.dumps(self.bundle)
        )
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
        freq_ids = self.generator["manual_freq_ids"]
        num_freq, num_dishes = len(freq_ids), config["num_dishes"]
        num_frames = config["num_gen_frames"]
        num_times = config["num_times"]
        num_blocks = num_frames * num_times // 8192
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
            (masks, 8192 * downsample, 8192),
            (powers, 8192 * downsample, 8192),
        ]
        if fine_support:
            support = reference.read_raw_frames(
                str(self.directory / "out/dtv_fine_support_*.raw"), num_freq * 2 * 4
            )
            assert len(support) == num_blocks
            products.append((support, 8192 * downsample, 8192))
        for frames, factor, stride in products:
            for index, frame in enumerate(frames):
                assert frame.fpga_seq_num == start + index * stride * downsample
                assert frame.time_downsampling_fpga == factor
        weights = (self.directory / "bundle/weights.bin").read_bytes()
        # Avoid another large copy in the production geometry test.
        if num_times == 8192:
            blocks = [frame.payload for frame in voltage]
        else:
            blocks = np.concatenate([frame.payload for frame in voltage]).reshape(
                -1, 8192 * num_freq * 2 * num_dishes
            )
        if fine_support:
            expected = [
                expected_fine_products_with_support(
                    block, freq_ids, self.bundle, weights, num_dishes
                )
                for block in blocks
            ]
        else:
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


def test_independent_detectors_share_gpu(pipeline):
    pipeline.calibrate()
    other_bundle = copy.deepcopy(pipeline.bundle)
    other_bundle["profiles"] = other_bundle["profiles"][:1]
    other_weights = (pipeline.directory / "bundle/weights.bin").read_bytes()[
        : other_bundle["profiles"][0]["weight_bank_nbytes"]
    ]
    (pipeline.directory / "other_profiles.json").write_text(json.dumps(other_bundle))
    (pipeline.directory / "other_weights.bin").write_bytes(other_weights)
    config = pipeline.config
    other = copy.deepcopy(config["run_dtv_detector"])
    command = other["gpu_0"]["commands"][0]
    command.update(
        pilot_profiles_path="other_profiles.json", weights_path="other_weights.bin"
    )
    for product in ("mask", "powers"):
        name = f"dtv_{product}"
        config[f"host_{name}_other_buffer"] = copy.deepcopy(
            config[f"host_{name}_buffer"]
        )
        other["gpu_0"]["out_buffers"][f"host_{name}"] = f"host_{name}_other_buffer"
        command[f"{name}_name"] = f"{name}_other"
        for output in other["gpu_0"]["commands"][2:]:
            if output["out_buf"] == f"host_{name}":
                output["gpu_mem"] = f"{name}_other_buffer"
        writer = copy.deepcopy(config[f"dump_{name}"])
        writer.update(in_buf=f"host_{name}_other_buffer", file_name=f"other_{name}")
        config[f"dump_{name}_other"] = writer
    config["run_other_detector"] = other
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.verify()
    num_freq = len(pipeline.generator["manual_freq_ids"])
    masks = reference.read_raw_frames(
        str(pipeline.directory / "out/other_dtv_mask_*.raw"), num_freq
    )
    powers = reference.read_raw_frames(
        str(pipeline.directory / "out/other_dtv_powers_*.raw"), num_freq * 24
    )
    voltages = reference.read_raw_frames(
        str(pipeline.directory / "out/voltage_*.raw"),
        config["num_times"] * num_freq * 2 * config["num_dishes"],
    )
    assert len(masks) == len(powers) == len(voltages) == config["num_gen_frames"]
    for voltage, mask, power in zip(voltages, masks, powers):
        for frame in (mask, power):
            assert frame.fpga_seq_num == voltage.fpga_seq_num
            assert frame.time_downsampling_fpga == 8192 * voltage.time_downsampling_fpga
        expected_mask, expected_power = reference.expected_products(
            voltage.payload,
            pipeline.generator["manual_freq_ids"],
            other_bundle,
            other_weights,
            num_dishes=config["num_dishes"],
        )
        np.testing.assert_array_equal(mask.payload, expected_mask)
        np.testing.assert_array_equal(
            power.payload.view("<u8").reshape(num_freq, 3), expected_power
        )


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("mode", ["fine", "disabled"])
def test_pipeline_decisions(pipeline, reverse, mode):
    pipeline.calibrate()
    pipeline.bundle["input_preprocessing"][
        "time_reverse_detector_windows_before_kernel"
    ] = reverse
    if mode == "disabled":
        for row in pipeline.bundle["profiles"]:
            row["fine_calibration"]["status"] = "pending_campaign"
        pipeline.config["gen_voltage"]["manual_freq_ids"] = [1600, 4000, 1601, 4001]
    elif not reverse:
        pipeline.config["gen_voltage"]["manual_freq_ids"] = [2623, 4000, 2408, 1600]
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    masks = pipeline.verify()
    if mode == "fine":
        ids = pipeline.config["gen_voltage"]["manual_freq_ids"]
        assert set(masks[:, [ids.index(2408), ids.index(2623)]].flat) == {0, 1}


@pytest.mark.parametrize("num_times", [4096, 16384])
def test_detector_blocks_cross_input_frame_boundaries(pipeline, num_times):
    pipeline.calibrate()
    pipeline.config["num_times"] = num_times
    pipeline.config["gen_voltage"]["array_shape"][0] = num_times
    blocks = pipeline.config["num_gen_frames"] * num_times // 8192
    for product in ("mask", "powers"):
        pipeline.config[f"dump_dtv_{product}"]["exit_after_n_files"] = blocks
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.verify()


@pytest.mark.parametrize("active_inputs", [0, 16, 32])
def test_pathfinder_sparse_padded_inputs(pipeline, active_inputs):
    pipeline.calibrate()
    pipeline.config["num_dishes"] = 64
    pipeline.generator["array_shape"][3] = 64
    full_config = pipeline.config
    pipeline.config = copy.deepcopy(full_config)
    # Produce CHORD metadata with the in-tree generator, then replay edited voltages.
    for key in (
        "run_send_voltage",
        "run_dtv_detector",
        "dump_dtv_mask",
        "dump_dtv_powers",
        "host_voltage_ringbuffer",
        "host_dtv_mask_buffer",
        "host_dtv_powers_buffer",
        "frame_arrival_period",
        "pilot_profiles_path",
        "samples_per_detector_frame",
        "weights_path",
    ):
        del pipeline.config[key]
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.config = full_config
    shutil.copyfile(
        pipeline.directory / "pipeline.log", pipeline.directory / "input.log"
    )
    source = pipeline.directory / "input"
    source.mkdir()
    for path in sorted((pipeline.directory / "out").glob("voltage_*.raw")):
        with path.open("rb") as stream:
            metadata_size = struct.unpack("<I", stream.read(4))[0]
        data = np.memmap(
            path,
            dtype=np.uint8,
            mode="r+",
            offset=metadata_size + 4,
            shape=(8192, 4, 2, 64),
        )
        # Spread live dishes across both polarizations and the padded dish dimension.
        active_dishes = np.linspace(0, 63, active_inputs // 2, dtype=int)
        absent = np.ones(64, dtype=bool)
        absent[active_dishes] = False
        data[..., absent] = 0x88  # Offset-binary complex zero.
        data.flush()
        del data
        path.rename(source / path.name)
    del pipeline.config["gen_voltage"]
    del pipeline.config["samples_per_data_set"]
    for stage in ("dump_voltage", "dump_dtv_mask", "dump_dtv_powers"):
        pipeline.config[stage]["exit_after_n_files"] = "num_gen_frames"
    pipeline.config["replay_voltage"] = {
        "kotekan_stage": "rawFileRead",
        "buf": "host_voltage_buffer",
        "base_dir": "input",
        "file_name": "voltage",
        "file_ext": "raw",
        "prefix_hostname": False,
        "end_interrupt": False,
    }
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    masks = pipeline.verify()
    if active_inputs == 0:
        assert not masks.any()
    else:
        assert set(masks[:, [0, 2]].flat) == {0, 1}


@pytest.mark.parametrize(
    "field",
    [
        "physical_channel",
        "chord_channel_id",
        "weight_bank_offset_bytes",
        "weight_bank_nbytes",
        "fine_calibration.anchor_bin",
        "fine_calibration.designated_half_width",
        "fine_calibration.cfar_rank",
        "fine_calibration.cfar_multiplier_q16",
    ],
)
@pytest.mark.parametrize("value", [1.5, True, 1 << 65])
def test_integer_fields_reject_coercion(pipeline, field, value):
    pipeline.calibrate()
    node = pipeline.bundle["profiles"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    code, output = pipeline.run()
    assert code != 0, f"{field} accepted {value!r}"
    assert parts[-1] in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


def test_uint64_fine_multiplier_limit(pipeline):
    pipeline.calibrate()
    for row in pipeline.bundle["profiles"]:
        row["fine_calibration"]["cfar_multiplier_q16"] = (1 << 64) - 1
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.verify()


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("schema_version", "bad", "schema_version"),
        ("profiles", [], "non-empty array"),
        ("profiles.0.physical_channel", 13, "outside ATSC"),
        ("profiles.1.physical_channel", 14, "duplicate physical channel"),
        ("profiles.1.chord_channel_id", 2408, "duplicate chord_channel_id"),
        ("profiles.0.chord_channel_id", -1, "chord_channel_id"),
        ("profiles.0.pilot_frequency_hz", 0, "pilot_frequency_hz"),
        ("profiles.0.weight_bank_nbytes", 64, "compiled kernel contract"),
        ("profiles.0.weight_bank_offset_bytes", -1, "weight bank range"),
        ("profiles.0.weight_bank_offset_bytes", (1 << 63) - 1, "weight bank range"),
        ("profiles.0.weight_bank_offset_bytes", 1, "expected contiguous offset"),
        ("profiles.0.fine_calibration", None, "missing fine_calibration"),
        (
            "profiles.0.fine_calibration.decision_version",
            "bad",
            "fine decision version",
        ),
        ("profiles.0.fine_calibration.status", "bad", "calibration status"),
        (
            "profiles.0.fine_calibration.status",
            "pending_campaign",
            "fine calibration is not deployable",
        ),
        ("profiles.0.fine_calibration.anchor_bin", 256, "anchor_bin"),
        (
            "profiles.0.fine_calibration.designated_half_width",
            128,
            "designated_half_width",
        ),
        ("profiles.0.fine_calibration.cfar_rank", -1, "cfar_rank"),
        ("profiles.0.fine_calibration.cfar_rank", 255, "bulk population"),
        ("profiles.0.fine_calibration.cfar_multiplier_q16", 0, "cfar_multiplier_q16"),
        ("profiles.0.fine_calibration.cfar_multiplier_q16", -1, "cfar_multiplier_q16"),
        ("profiles.0.fine_calibration.cfar_multiplier_q16", 1.5, "cfar_multiplier_q16"),
        ("profiles.0.fine_calibration.bulk_mask_words_hex", ["0x0"], "4 hex words"),
        ("profiles.0.fine_calibration.bulk_mask_words_hex.0", "0xg", "invalid digit"),
        (
            "profiles.0.fine_calibration.bulk_mask_words_hex.0",
            "0x10000000000000000",
            "uint64 hex",
        ),
        (
            "profiles.0.fine_calibration.bulk_mask_words_hex.0",
            "0xffffffffffffffff",
            "designated/guard",
        ),
    ],
)
def test_rejects_invalid_bundle(pipeline, field, value, diagnostic):
    pipeline.calibrate()
    node = pipeline.bundle
    parts = field.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
    code, output = pipeline.run()
    assert code != 0, "invalid bundle was accepted"
    assert diagnostic in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("missing_weights", "cannot open weights_path"),
        ("empty_weights", "weights.bin is empty"),
        ("truncated_weights", "weight bank range"),
        ("extra_weights", "profile table accounts for"),
        ("missing_channel_ids", "no chord_channel_id"),
        ("window_alignment", "multiple"),
        ("fine_geometry", "128"),
        ("zero_block", "positive"),
        ("negative_block", "positive"),
        ("metadata_length", "coarse_freq"),
    ],
)
def test_rejects_invalid_runtime(pipeline, case, diagnostic):
    pipeline.calibrate()
    weights = pipeline.directory / "bundle/weights.bin"
    if case == "missing_weights":
        weights.unlink()
    elif case == "empty_weights":
        weights.write_bytes(b"")
    elif case == "truncated_weights":
        weights.write_bytes(weights.read_bytes()[:-1])
    elif case == "extra_weights":
        weights.write_bytes(weights.read_bytes() + b"\0")
    elif case == "missing_channel_ids":
        for row in pipeline.bundle["profiles"]:
            row["chord_channel_id"] = None
    elif case == "window_alignment":
        pipeline.config["samples_per_detector_frame"] = 8191
    elif case == "fine_geometry":
        pipeline.config["samples_per_detector_frame"] = 4096
    elif case in ("zero_block", "negative_block"):
        pipeline.config["samples_per_detector_frame"] = (
            0 if case == "zero_block" else -64
        )
    elif case == "metadata_length":
        pipeline.config["gen_voltage"]["num_local_freq"] = 2
    code, output = pipeline.run()
    assert code != 0, f"invalid {case} was accepted"
    assert diagnostic in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


@pytest.mark.parametrize(
    "num_dishes,num_freq",
    [(64, 384), (512, 48)],
    ids=["pathfinder-padded", "full-chord"],
)
def test_production_geometry_soak(pipeline, record_property, num_dishes, num_freq):
    requested = os.environ.get("PILOTPROXY_SOAK_FRAMES")
    if requested is None:
        pytest.skip("set PILOTPROXY_SOAK_FRAMES to run the production geometry soak")
    frames = int(requested)
    assert frames >= 8, "use at least eight frames to wrap the ring buffer"
    pipeline.calibrate()
    # Stress all 23 pilots simultaneously, more than a normal CHORD node carries.
    ids = [row["chord_channel_id"] for row in pipeline.bundle["profiles"]]
    ids += [freq for freq in range(1600, 2400) if freq not in ids][
        : num_freq - len(ids)
    ]
    assert len(ids) == num_freq and len(set(ids)) == num_freq
    pipeline.config.update(
        num_dishes=num_dishes,
        num_local_freq=num_freq,
        num_gen_frames=frames,
        samples_per_data_set="num_times",
    )
    pipeline.config["gen_voltage"].update(
        array_shape=[8192, num_freq, 2, num_dishes],
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
    record_property("bound_fine_profiles", len(pipeline.bundle["profiles"]))
    record_property("pipeline_timeout_seconds", timeout)
    record_property("voltage_gib", frames * 8192 * num_freq * 2 * num_dishes / 2 ** 30)
    print(
        f"Verified {frames} production-size blocks; pipeline took {pipeline.elapsed:.2f}s"
    )
