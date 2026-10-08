"""Test raw file round trips and malformed records."""
import ctypes
import struct

import pytest
from kotekan.n2buffer import N2Buffer, N2Metadata

from kotekan import runner


def specification(kind):
    if kind == "standard":
        return (
            64,
            {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "frame_size": 64,
                "num_frames": 3,
            },
        )
    size = N2Buffer.calculate_layout(4, 10, 0)["size"]
    return (
        size,
        {
            "kotekan_buffer": "N2",
            "metadata_pool": "N2_pool",
            "n2_layout": "FullUpperTri",
            "num_frames": 3,
        },
    )


def records(kind, count):
    size, _ = specification(kind)
    result = []
    for i in range(count):
        payload = bytes((k * 11 + i * 37) % 256 for k in range(size))
        metadata = b""
        if kind != "standard":
            meta = N2Metadata()
            meta.abs_time_idx = i + 100
            meta.freq_id = i + 600
            meta.fpga_start_tick = i * 128
            meta.frame_length_fpga_ticks = 128
            metadata = bytes(meta)
            assert len(metadata) == ctypes.sizeof(N2Metadata)
        result.append(struct.pack("<I", len(metadata)) + metadata + payload)
    return result


def transfer(
    source, destination, kind, *, frames_per_file, output_files, failure=False
):
    destination.mkdir(exist_ok=True)
    _, buffer = specification(kind)
    stages = {
        "read": {
            "kotekan_stage": "rawFileRead",
            "buf": "transport",
            "base_dir": str(source),
            "file_name": "record",
            "file_ext": "raw",
            "prefix_hostname": False,
            "end_interrupt": True,
        },
        "write": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "transport",
            "base_dir": str(destination),
            "file_name": "record",
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": frames_per_file,
            "exit_after_n_files": output_files,
        },
    }
    config = {
        "buffer_depth": 3,
        "num_elements": 4,
        "num_dishes": 2,
        "num_polarizations": 2,
        "num_ev": 0,
        "log_level": "INFO",
    }
    task = runner.KotekanRunner(
        buffers={"transport": buffer},
        stages=stages,
        config=config,
        expect_failure=failure,
        timeout=15,
    )
    task.run()
    return task


@pytest.mark.parametrize("kind", ["standard", "n2-scalar"])
@pytest.mark.parametrize("count", [1, 3])
def test_raw_file_roundtrip(tmp_path, kind, count):
    source = tmp_path / "source"
    source.mkdir()
    expected = records(kind, count)
    for i, record in enumerate(expected):
        (source / f"record_{i:07d}.raw").write_bytes(record)
    packed = tmp_path / "packed"
    transfer(source, packed, kind, frames_per_file=count, output_files=1)
    written = (packed / "record_0000000.raw").read_bytes()
    assert written == b"".join(expected)
    unpacked = tmp_path / "unpacked"
    transfer(packed, unpacked, kind, frames_per_file=1, output_files=count)
    actual = [p.read_bytes() for p in sorted(unpacked.glob("record_*.raw"))]
    assert actual == expected


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("short-header", "Failed to read file"),
        ("changed-header", "metadata size changed between records"),
        ("metadata-type-size", "serialized metadata size does not match"),
    ],
)
def test_invalid_raw_records(tmp_path, fault, diagnostic):
    source = tmp_path / "source"
    source.mkdir()
    size, _ = specification("n2-scalar")
    kind = "n2-scalar"
    packed = b"".join(records(kind, 3))
    if fault == "short-header":
        packed = b"\0\0\0"
    elif fault == "changed-header":
        first = len(records(kind, 1)[0])
        data = bytearray(packed)
        struct.pack_into("<I", data, first, ctypes.sizeof(N2Metadata) + 1)
        packed = bytes(data)
    elif fault == "metadata-type-size":
        packed = struct.pack("<I", 1) + b"x" + bytes(size)
    (source / "record_0000000.raw").write_bytes(packed)
    task = transfer(
        source,
        tmp_path / "output",
        kind,
        frames_per_file=1,
        output_files=3,
        failure=True,
    )
    assert task.return_code != 0
    assert diagnostic in task.output


@pytest.mark.parametrize("fault,whole_frames", [("truncated", 2), ("trailing", 3)])
def test_trailing_bytes_are_ignored(tmp_path, fault, whole_frames):
    source = tmp_path / "source"
    source.mkdir()
    kind = "n2-scalar"
    expected = records(kind, 3)
    packed = b"".join(expected)
    packed = packed[:-1] if fault == "truncated" else packed + b"x"
    (source / "record_0000000.raw").write_bytes(packed)
    unpacked = tmp_path / "unpacked"
    task = transfer(
        source, unpacked, kind, frames_per_file=1, output_files=whole_frames
    )
    assert task.return_code == 0
    assert "trailing bytes" in task.output
    actual = [p.read_bytes() for p in sorted(unpacked.glob("record_*.raw"))]
    assert actual == expected[:whole_frames]


def test_large_file_size_is_not_truncated(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    packed = struct.pack("<I", 0) + bytes(64)
    file = source / "record_0000000.raw"
    file.write_bytes(packed)
    # Sparse file, to check sizes above 4 GiB are not truncated to 32 bits.
    with file.open("r+b") as f:
        f.truncate((1 << 32) + len(packed))
    size = file.stat().st_size
    assert size > (1 << 32)
    task = transfer(
        source, tmp_path / "output", "standard", frames_per_file=1, output_files=1
    )
    assert task.return_code == 0
    assert "File size: {:d} bytes".format(size) in task.output
