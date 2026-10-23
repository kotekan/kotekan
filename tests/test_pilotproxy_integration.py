"""Test the CUDA pipeline with synthetic calibration and malformed bundles.

Set PILOTPROXY_TEST_BINARY and PILOTPROXY_TEST_BUNDLE to run these tests.
The bundle must match the compiled core. PILOTPROXY_SOAK_FRAMES (at least 8)
also enables full-size CHORD and CHIME runs with two polarizations and
full-rate fine support; see dev_pilotproxy_testing.rst for their geometry.
Four voltage frames repeat; every
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
    # The binary comes from the usual build tree (KOTEKAN_BUILD_DIRNAME) unless overridden;
    # the bundle is a separate input, so without it the CUDA tests skip.
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


def test_independent_detectors_share_gpu(pipeline):
    pipeline.calibrate()
    other_dir = pipeline.directory / "other"
    shutil.copytree(pipeline.directory / "bundle", other_dir)
    other_bundle = copy.deepcopy(pipeline.bundle)
    other_bundle["profiles"] = other_bundle["profiles"][:1]
    other_weights = (other_dir / "weights.bin").read_bytes()[
        : other_bundle["profiles"][0]["weight_bank_nbytes"]
    ]
    (other_dir / "weights.bin").write_bytes(other_weights)
    reference.seal_bundle(other_dir, other_bundle)
    config = pipeline.config
    other = copy.deepcopy(config["run_dtv_detector"])
    command = other["gpu_0"]["commands"][0]
    command.update(
        pilot_profiles_path="other/pilot_profiles.json",
        weights_path="other/weights.bin",
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
        expected_mask, expected_power, _ = reference.expected_products(
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
    if field == "profiles.1.chord_channel_id" and "receiver_channel_id" in node:
        node["receiver_channel_id"] = value
    code, output = pipeline.run()
    assert code != 0, "invalid bundle was accepted"
    assert diagnostic in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


def test_exported_bundle_loads_unchanged(pipeline):
    # The bundle as exported, with no pilot on the node, passes the digest checks.
    pipeline.config["gen_voltage"]["manual_freq_ids"] = [1600, 4000, 1601, 4001]
    code, output = pipeline.run(seal=False)
    assert code == 0, output[-12000:]
    assert "detector DISABLED" in output, output[-12000:]
    pipeline.verify()


@pytest.mark.parametrize(
    "case",
    [
        "flipped_weight_byte",
        "pilot_profiles.json",
        "weights.manifest.json",
        "dropped_listed_file",
        "manifest_not_json",
        "weights_outside_bundle",
        "missing_pilot_frequency",
    ],
)
def test_rejects_unverified_bundle(pipeline, case):
    pipeline.calibrate()
    bundle = pipeline.directory / "bundle"
    row = pipeline.bundle["profiles"][0]
    if case == "missing_pilot_frequency":
        del row["pilot_frequency_hz"]
    reference.seal_bundle(bundle, pipeline.bundle)
    if case == "flipped_weight_byte":
        listed = reference.file_sha256(bundle / "weights.bin")
        weights = bytearray((bundle / "weights.bin").read_bytes())
        weights[len(weights) // 2] ^= 1
        (bundle / "weights.bin").write_bytes(weights)
        diagnostic = (
            f"sha256 mismatch for {bundle.name}/weights.bin: the file has "
            f"{reference.file_sha256(bundle / 'weights.bin')}, sha256sums.txt lists {listed}"
        )
    elif case.endswith(".json"):
        payload = json.loads((bundle / case).read_text())
        payload["weights_sha256"] = "0" * 64
        (bundle / case).write_text(json.dumps(payload))
        reference.write_sha256sums(bundle)
        diagnostic = (
            f"weights_sha256 in {case} is \"{'0' * 64}\", but {bundle.name}/weights.bin has "
            f"{reference.file_sha256(bundle / 'weights.bin')}"
        )
    elif case == "dropped_listed_file":
        (bundle / "detector_contract.json").unlink()
        diagnostic = f"lists {bundle.name}/detector_contract.json, which cannot be read"
    elif case == "manifest_not_json":
        (bundle / "weights.manifest.json").write_text("{not json")
        reference.write_sha256sums(bundle)
        diagnostic = f"cannot parse {bundle.name}/weights.manifest.json"
    elif case == "weights_outside_bundle":
        # Both weights_sha256 name a weights_path outside the bundle directory.
        weights = bytearray((bundle / "weights.bin").read_bytes())
        weights[len(weights) // 2] ^= 1
        (pipeline.directory / "weights.bin").write_bytes(weights)
        outside = reference.file_sha256(pipeline.directory / "weights.bin")
        for name in ("pilot_profiles.json", "weights.manifest.json"):
            payload = json.loads((bundle / name).read_text())
            payload["weights_sha256"] = outside
            (bundle / name).write_text(json.dumps(payload))
        reference.write_sha256sums(bundle)
        command = pipeline.config["run_dtv_detector"]["gpu_0"]["commands"][0]
        command["weights_path"] = "weights.bin"
        diagnostic = (
            f"sha256 mismatch for weights.bin: the file has {outside}, sha256sums.txt "
            f"lists {reference.file_sha256(bundle / 'weights.bin')}"
        )
    else:
        diagnostic = (
            f"pilot_profiles entry for physical channel {row['physical_channel']} "
            "has no pilot_frequency_hz"
        )
    code, output = pipeline.run(seal=False)
    assert code != 0, f"{case} was accepted"
    assert diagnostic in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


@pytest.mark.parametrize("index", [0, 1], ids=["even-anchor", "odd-anchor"])
@pytest.mark.parametrize("half_width", [0, 2])
@pytest.mark.parametrize("side", [-1, 1])
def test_bulk_mask_excludes_anchor_guard(pipeline, index, half_width, side):
    # The bulk must exclude max(half_width, 2) fine bins on each side of the anchor.
    pipeline.calibrate()
    calibration = pipeline.bundle["profiles"][index]["fine_calibration"]
    calibration["designated_half_width"] = half_width
    guard_bin = (calibration["anchor_bin"] + 2 * side) % 256
    words = [int(word, 16) for word in calibration["bulk_mask_words_hex"]]
    words[guard_bin // 64] |= 1 << (guard_bin % 64)
    calibration["bulk_mask_words_hex"] = [f"0x{word:016x}" for word in words]
    code, output = pipeline.run()
    assert code != 0, "guard bin in the bulk was accepted"
    assert "designated/guard" in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("missing_weights", "cannot open weights_path"),
        ("empty_weights", "weights.bin is empty"),
        ("truncated_weights", "weight bank range"),
        ("extra_weights", "profile table accounts for"),
        ("missing_channel_ids", "no receiver channel IDs"),
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
            if "receiver_channel_id" in row:
                row["receiver_channel_id"] = None
    elif case == "metadata_length":
        pipeline.config["gen_voltage"]["num_local_freq"] = 2
    code, output = pipeline.run()
    assert code != 0, f"invalid {case} was accepted"
    assert diagnostic in output, output[-12000:]
    assert not list((pipeline.directory / "out").glob("dtv_*.raw"))


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
    # CHIME has 16 channels per GPU; cover all pilots in two batches and retain
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
        frames * pipeline.block_samples * num_freq * 2 * num_dishes / 2 ** 30,
    )
    print(
        f"Verified {frames} production-size blocks; pipeline took {pipeline.elapsed:.2f}s"
    )


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("frame_blocks", [0.5, 1, 2])
def test_receiver_geometry_against_cpu(pipeline, reverse, frame_blocks):
    """Use either compiled geometry with independent integer references and ring reuse."""
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


@pytest.mark.parametrize(
    "fault", ["config", "row", "alias", "missing_top", "generic_float", "generic_bool"]
)
def test_receiver_namespace_refusal(pipeline, fault):
    pipeline.calibrate()
    namespace = pipeline.detector["receiver_channel_id_namespace"]
    alias = "chime_channel_id" if namespace == "chime_freq_id" else "chord_channel_id"
    other = "chord_freq_id" if namespace == "chime_freq_id" else "chime_freq_id"
    pipeline.bundle["receiver_channel_id_namespace"] = namespace
    for row in pipeline.bundle["profiles"]:
        row["receiver_channel_id_namespace"] = namespace
        row["receiver_channel_id"] = row[alias]
    row = pipeline.bundle["profiles"][0]
    if fault == "config":
        pipeline.detector["receiver_channel_id_namespace"] = other
    elif fault == "row":
        row["receiver_channel_id_namespace"] = other
    elif fault == "alias":
        row[alias] += 1
    elif fault == "missing_top":
        pipeline.bundle.pop("receiver_channel_id_namespace")
    else:
        # Omit the optional alias to exercise generic-ID validation itself.
        row.pop(alias)
        row["receiver_channel_id"] = 1.5 if fault == "generic_float" else True
    code, output = pipeline.run()
    assert code != 0, output[-12000:]
    assert "cudaPilotProxyDetector" in output
    assert not list((pipeline.directory / "out").glob("dtv_mask_*.raw"))


def test_archive_review_thresholds_refused_for_active_evaluation(pipeline):
    pipeline.calibrate()
    for row in pipeline.bundle["profiles"]:
        row["fine_calibration"]["provenance"]["threshold_selection"] = {
            "method": "operating_curve_candidate_v1",
            "qualification": "archive_review_only",
            "selection_granularity": "detector_frame",
        }
    code, output = pipeline.run()
    assert code != 0, output[-12000:]
    assert "archive-review-only thresholds" in output
    assert not list((pipeline.directory / "out").glob("dtv_mask_*.raw"))


def test_legacy_chord_namespace_default(pipeline):
    if pipeline.detector["receiver_channel_id_namespace"] != "chord_freq_id":
        pytest.skip("legacy bundle compatibility applies to CHORD")
    pipeline.calibrate()
    pipeline.detector.pop("receiver_channel_id_namespace")
    pipeline.bundle.pop("receiver_channel_id_namespace", None)
    for row in pipeline.bundle["profiles"]:
        row.pop("receiver_channel_id_namespace", None)
        row.pop("receiver_channel_id", None)
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    pipeline.verify()


def test_archive_review_profiles_allow_permanent_reject_limit(pipeline):
    pipeline.calibrate()
    ids = []
    for row in pipeline.bundle["profiles"]:
        row["fine_calibration"]["provenance"]["threshold_selection"] = {
            "qualification": "archive_review_only"
        }
        value = row.get("receiver_channel_id", row.get("chord_channel_id"))
        if value in pipeline.generator["manual_freq_ids"]:
            ids.append(value)
    pipeline.detector["permanent_mask_freq_ids"] = ids
    code, output = pipeline.run()
    assert code == 0, output[-12000:]
    masks = reference.read_raw_frames(str(pipeline.directory / "out/dtv_mask_*.raw"), 4)
    assert len(masks) == pipeline.config["num_gen_frames"]
    assert all(not frame.payload.any() for frame in masks)
