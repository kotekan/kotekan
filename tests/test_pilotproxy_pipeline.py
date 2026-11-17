"""Checks for complete, correctly timestamped PilotProxy captures."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_pilotproxy_pipeline", ROOT / "tests/verify_pilotproxy_pipeline.py"
)
verifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verifier
spec.loader.exec_module(verifier)


def capture():
    start, downsampling = 557056, 4
    voltage, masks, powers = [], [], []
    for i in range(8):
        sequence = start + i * verifier.NUM_TIMES * downsampling
        for frames, factor in (
            (voltage, downsampling),
            (masks, downsampling * verifier.BLOCK_SAMPLES),
            (powers, downsampling * verifier.BLOCK_SAMPLES),
        ):
            frames.append(verifier.RawFrame(np.zeros(1, np.uint8), sequence, factor))
    return voltage, masks, powers


def test_complete_capture_preserves_epoch_and_downsampling():
    assert verifier.validate_frames(*capture(), 8) == 8


@pytest.mark.parametrize(
    "counts", [(8, 1, 1), (1, 1, 1), (8, 7, 8), (8, 8, 7), (9, 8, 8), (0, 0, 0)]
)
def test_incomplete_or_extra_frames_are_rejected(counts):
    frames = capture()
    selected = [
        items[:count] if count <= 8 else items + items[:1]
        for items, count in zip(frames, counts)
    ]
    with pytest.raises(ValueError, match="expected 8 voltage"):
        verifier.validate_frames(*selected, 8)


@pytest.mark.parametrize("product", [0, 1, 2])
@pytest.mark.parametrize("field", ["fpga_seq_num", "time_downsampling_fpga"])
def test_incorrect_frame_timing_is_rejected(product, field):
    frames = capture()
    setattr(frames[product][2], field, 0)
    with pytest.raises(ValueError, match="FPGA timing"):
        verifier.validate_frames(*frames, 8)


def raw_frame(sequence=557056, downsampling=4, payload=b"abcd"):
    # layout of the chordMetadata.cpp serializer, with the complete metadata tail
    metadata = bytearray(148888)
    struct.pack_into("<7i", metadata, 0, 10, 24, 24, 12288, 128, 8, 17)
    struct.pack_into("<q", metadata, 32, sequence)
    struct.pack_into("<q", metadata, 40, downsampling)
    return struct.pack("<I", len(metadata)) + metadata + payload


def test_reads_timing_for_every_frame_in_a_file(tmp_path):
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(raw_frame() + raw_frame(589824))
    frames = verifier.read_raw_frames(str(path), 4)
    assert [f.fpga_seq_num for f in frames] == [557056, 589824]
    assert [f.time_downsampling_fpga for f in frames] == [4, 4]
    assert frames[0].payload.tobytes() == b"abcd"


def test_reads_full_width_sequence(tmp_path):
    path = tmp_path / "voltage_0000000.raw"
    sequence = (1 << 40) + 557056
    path.write_bytes(raw_frame(sequence, 32768))
    frame = verifier.read_raw_frames(str(path), 4)[0]
    assert frame.fpga_seq_num == sequence
    assert frame.time_downsampling_fpga == 32768


@pytest.mark.parametrize("field", range(6))
def test_incompatible_metadata_limits_are_rejected(tmp_path, field):
    data = bytearray(raw_frame())
    struct.pack_into("<i", data, 4 + 4 * field, -1)
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="unsupported CHORD metadata layout"):
        verifier.read_raw_frames(str(path), 4)


@pytest.mark.parametrize("size", [28, 44, 148887, 148889])
def test_incompatible_metadata_size_is_rejected(tmp_path, size):
    prefix = raw_frame()[4:48]
    metadata = (prefix + bytes(size))[:size]
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(struct.pack("<I", size) + metadata + b"abcd")
    with pytest.raises(ValueError, match="unsupported CHORD metadata size"):
        verifier.read_raw_frames(str(path), 4)


@pytest.mark.parametrize(
    "data",
    [
        b"x",
        struct.pack("<I", 148888),
        raw_frame()[:-10],
        raw_frame()[:-1],
        raw_frame(-1),
        raw_frame(downsampling=0),
    ],
    ids=[
        "one-byte",
        "size-only",
        "truncated-metadata",
        "truncated-payload",
        "negative-seq",
        "zero-downsampling",
    ],
)
def test_truncated_frames_or_missing_timing_are_rejected(tmp_path, data):
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        verifier.read_raw_frames(str(path), 4)


@pytest.mark.parametrize("channel_key", ["chord_channel_id", "receiver_channel_id"])
@pytest.mark.parametrize(
    "fault",
    [None, "mask", "powers", "timing", "missing", "uncalibrated", "no_receiver"],
)
def test_verifier_cli_checks_capture_products(
    tmp_path, monkeypatch, capsys, channel_key, fault
):
    # The GPU integration tests check the reference arithmetic. Keep CLI checks
    # independent of the optional PilotProxy package used by those tests.
    reference = Mock(
        return_value=(
            np.zeros(verifier.NUM_FREQ, dtype=np.uint8),
            np.zeros((verifier.NUM_FREQ, 3), dtype=np.uint64),
            np.zeros((verifier.NUM_FREQ, 2), dtype=np.int32),
        )
    )
    if fault == "uncalibrated":
        reference.side_effect = ValueError("physical channel 14 is not calibrated")
    monkeypatch.setattr(verifier, "expected_products", reference)
    bundle = {
        "input_preprocessing": {"time_reverse_detector_windows_before_kernel": False},
        "profiles": [
            {
                channel_key: verifier.FREQ_IDS[0],
                "physical_channel": 14,
                "weight_bank_offset_bytes": 0,
                "weight_bank_nbytes": 3 * verifier.K,
            }
        ],
    }
    verifier.synthetic_calibration(bundle)
    if fault == "uncalibrated":
        bundle["profiles"][0]["fine_calibration"]["status"] = "pending_campaign"
    elif fault == "no_receiver":
        bundle["profiles"][0][channel_key] = None
    (tmp_path / "pilot_profiles.json").write_text(json.dumps(bundle))
    (tmp_path / "weights.bin").write_bytes(bytes([0x11]) * 3 * verifier.K)
    voltage_size = (
        verifier.NUM_TIMES * verifier.NUM_FREQ * verifier.NUM_POL * verifier.NUM_DISHES
    )
    (tmp_path / "voltage_0.raw").write_bytes(
        raw_frame(payload=bytes([0x88]) * voltage_size)
    )
    masks = bytearray(verifier.NUM_FREQ)
    powers = bytearray(verifier.NUM_FREQ * 3 * 8)
    if fault == "mask":
        masks[0] = 1
    if fault == "powers":
        powers[0] = 1
    (tmp_path / "dtv_mask_0.raw").write_bytes(
        raw_frame(downsampling=4 * verifier.BLOCK_SAMPLES, payload=masks)
    )
    if fault != "missing":
        (tmp_path / "dtv_powers_0.raw").write_bytes(
            raw_frame(
                sequence=0 if fault == "timing" else 557056,
                downsampling=4 * verifier.BLOCK_SAMPLES,
                payload=powers,
            )
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_pilotproxy_pipeline.py",
            "--bundle-dir",
            str(tmp_path),
            "--dump-dir",
            str(tmp_path),
            "--num-input-frames",
            "1",
        ],
    )
    assert verifier.main() == (0 if fault is None else 1)
    assert ("PASS:" if fault is None else "FAIL:") in capsys.readouterr().out


def test_verifier_cli_calibration_restamps_all_bundle_digests(tmp_path, monkeypatch):
    bundle = {"profiles": [{"physical_channel": 14}, {"physical_channel": 15}]}
    (tmp_path / "pilot_profiles.json").write_text(json.dumps(bundle))
    (tmp_path / "weights.manifest.json").write_text("{}")
    (tmp_path / "detector_contract.json").write_text("{}")
    (tmp_path / "weights.bin").write_bytes(bytes([0x11]) * 3 * verifier.K)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_pilotproxy_pipeline.py", "--calibrate", "--bundle-dir", str(tmp_path)],
    )
    assert verifier.main() == 0
    profiles = json.loads((tmp_path / "pilot_profiles.json").read_text())
    manifest = json.loads((tmp_path / "weights.manifest.json").read_text())
    digest = verifier.file_sha256(tmp_path / "weights.bin")
    assert profiles["weights_sha256"] == manifest["weights_sha256"] == digest
    assert all(
        row["fine_calibration"]["status"] == "calibrated"
        for row in profiles["profiles"]
    )
    lines = (tmp_path / "sha256sums.txt").read_text().splitlines()
    assert len(lines) == len(verifier.BUNDLE_FILES)
    for line in lines:
        digest, name = line.split()
        assert digest == verifier.file_sha256(tmp_path / name)


def test_default_chord_template_leaves_detector_disabled():
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/fengine"))
    config = yaml.safe_load(env.get_template("chord.j2").render())
    assert "run_dtv_detector" not in config
    assert "host_dtv_mask_buffer" not in config
    assert "host_dtv_powers_buffer" not in config


@pytest.mark.parametrize("calibration", [False, True])
def test_pathfinder_sends_what_recv_expects(calibration):
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/chord"))
    config = yaml.safe_load(
        env.get_template("pathfinder.j2").render(
            dtv_write_enabled=True, dtv_calibration_mode=calibration
        )
    )
    recv = yaml.safe_load(
        env.get_template("recv.j2").render(dtv_calibration_mode=calibration)
    )
    products = ["mask"]
    if calibration:
        products += ["powers", "fine_support", "fine_powers"]
    assert sorted(recv["buffer_recv_dtv"]) == sorted(products)
    assert sorted(recv["write_dtv"]) == sorted(products)
    assert sorted(k[: -len("_0")] for k in config["send_dtv"] if k.endswith("_0")) == (
        sorted(products)
    )
    detector = config["run_dtv_detector"]["gpu_0"]["commands"][0]
    assert ("dtv_fine_powers_name" in detector) is calibration
    # DtvRfiMask reads the fine support in both modes.
    assert "host_dtv_fine_support_buffer_0" in config
    for product in products:
        sent = config[f"host_dtv_{product}_send_buffer_0"]["extents"]
        sent = [config[e] if isinstance(e, str) else e for e in sent]
        assert recv[f"dtv_{product}_buffer"]["extents"] == sent
        assert recv["buffer_recv_dtv"][product]["listen_port"] == (
            config["send_dtv"][f"{product}_0"]["server_port"]
        )


@pytest.mark.parametrize(
    "apply_mask, write", [(False, False), (True, False), (False, True), (True, True)]
)
def test_pathfinder_dtv_switches(apply_mask, write):
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/chord"))
    config = yaml.safe_load(
        env.get_template("pathfinder.j2").render(
            dtv_apply_mask=apply_mask, dtv_write_enabled=write
        )
    )
    # The pipeline is the same, and the switches set the endpoints' starting values.
    assert "run_dtv_detector" in config
    assert config["dtv_mask_enabled"]["enabled"] is apply_mask
    assert config["dtv_write_enabled"]["enabled"] is write
    merge = config["merge_dtv_mask"]["half_0"]
    assert merge["enabled_updatable_config"] == "/dtv_mask_enabled"
    accumulate = config["n2_accumulate"]
    assert accumulate["dtv_mask_enabled_updatable_config"] == "/dtv_mask_enabled"


@pytest.mark.parametrize("apply_mask", [False, True])
def test_dtv_diagnostics_preserve_per_frame_metadata(apply_mask):
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/fengine"))
    config = yaml.safe_load(
        env.get_template("chord.j2").render(
            dtv_enabled=not apply_mask, dtv_apply_mask=apply_mask
        )
    )
    for name in ("write_dtv_mask", "write_dtv_powers"):
        assert config["write_data"][name]["create_single_file"] is False


@pytest.mark.parametrize("apply_mask", [False, True])
def test_detector_runs_on_its_own_cuda_streams(apply_mask):
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/fengine"))
    off = yaml.safe_load(env.get_template("chord.j2").render())
    on = yaml.safe_load(
        env.get_template("chord.j2").render(
            dtv_enabled=not apply_mask, dtv_apply_mask=apply_mask
        )
    )
    detector = on["run_dtv_detector"]["gpu_0"]
    assert detector["num_cuda_streams"] == 5
    commands = [(c["name"], c["cuda_stream"]) for c in detector["commands"]]
    assert commands[:2] == [("cudaPilotProxyDetector", 3), ("cudaSyncOutput", 4)]
    assert commands[2:] == [("cudaOutputData", 4)] * 3

    # Every other stage keeps the default streams 0 to 2, so 3 and 4 are free.
    def stream_keys(node, path=""):
        if isinstance(node, list):
            node = dict(enumerate(node))
        if not isinstance(node, dict):
            return []
        found = [f"{path}/{key}" for key in node if "cuda_stream" in str(key)]
        for key, value in node.items():
            if path or key != "run_dtv_detector":
                found += stream_keys(value, f"{path}/{key}")
        return found

    assert stream_keys(on) == stream_keys(off) == []
