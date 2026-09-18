"""Checks for complete, correctly timestamped PilotProxy captures."""

import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_pilotproxy_pipeline", ROOT / "tools/verify_pilotproxy_pipeline.py"
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


def raw_frame(sequence=557056, downsampling=4):
    metadata = struct.pack("<4iqi", 10, 24, 12288, 17, sequence, downsampling)
    return struct.pack("<I", len(metadata)) + metadata + b"abcd"


def test_reads_timing_for_every_frame_in_a_file(tmp_path):
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(raw_frame() + raw_frame(589824))
    frames = verifier.read_raw_frames(str(path), 4)
    assert [f.fpga_seq_num for f in frames] == [557056, 589824]
    assert [f.time_downsampling_fpga for f in frames] == [4, 4]
    assert frames[0].payload.tobytes() == b"abcd"


@pytest.mark.parametrize(
    "data",
    [
        b"x",
        struct.pack("<I", 28),
        raw_frame()[:-1],
        raw_frame(-1),
        raw_frame(downsampling=0),
    ],
)
def test_truncated_frames_or_missing_timing_are_rejected(tmp_path, data):
    path = tmp_path / "voltage_0000000.raw"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        verifier.read_raw_frames(str(path), 4)


def test_default_chord_template_leaves_detector_disabled():
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(ROOT / "config/fengine"))
    config = yaml.safe_load(env.get_template("chord.j2").render())
    assert "run_dtv_detector" not in config
    assert "host_dtv_mask_buffer" not in config
    assert "host_dtv_powers_buffer" not in config
