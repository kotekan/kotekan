#!/usr/bin/env python3
"""Replay globally aligned native CHIME voltages through the detector stage.

This is an offline numerical test, not a calibration exporter. The original
archive-review bundle is never modified. A temporary test profile carries its
literal numerical parameters, and the only stages read files, run the detector,
and write its products. No mask consumer or observing pipeline is constructed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import time

import h5py
import numpy as np
import yaml

import verify_pilotproxy_pipeline as reference

FRAME_SAMPLES = 16384
SAMPLE_SECONDS = 2.56e-6
SCHEMA = "kotekan-native-offline-replay-v2"


class MetadataPrefix(ctypes.LittleEndianStructure):
    """Pinned chordMetadataFormat fields through the receiver identifiers."""

    _fields_ = [
        ("limits", ctypes.c_int32 * 6),
        ("frame", ctypes.c_int32),
        ("sequence", ctypes.c_int64),
        ("period", ctypes.c_int64),
        ("name", ctypes.c_char * 24),
        ("type", ctypes.c_int32),
        ("dims", ctypes.c_int32),
        ("shape", ctypes.c_int32 * 10),
        ("dim_names", (ctypes.c_char * 24) * 10),
        ("scaling", ctypes.c_int64 * 10),
        ("stride", ctypes.c_int64 * 10),
        ("offset", ctypes.c_int64),
        ("nfreq", ctypes.c_int32),
        ("frequencies", ctypes.c_int32 * 12288),
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_candidate_files(directory):
    names = {
        "detector_contract.json",
        "pilot_profiles.json",
        "weights.bin",
        "weights.manifest.json",
    }
    checked = {}
    for line in (directory / "sha256sums.txt").read_text().splitlines():
        expected, name = line.split()
        if name not in names or name in checked:
            raise ValueError(
                "candidate digest manifest has unexpected or repeated files"
            )
        actual = sha256(directory / name)
        if actual != expected:
            raise ValueError(f"candidate digest mismatch: {name}")
        checked[name] = actual
    if set(checked) != names:
        raise ValueError("candidate digest manifest is incomplete")
    checked["sha256sums.txt"] = sha256(directory / "sha256sums.txt")
    return checked


def check_metadata(path, freq_id, shape):
    with path.open("rb") as stream:
        size = struct.unpack("<I", stream.read(4))[0]
        data = stream.read(size)
    if size != reference.CHORD_METADATA_SIZE:
        raise ValueError("unsupported output metadata ABI")
    header = MetadataPrefix.from_buffer_copy(data)
    if tuple(header.limits) != reference.CHORD_METADATA_LIMITS:
        raise ValueError("metadata layout differs")
    if header.nfreq != 1 or header.frequencies[0] != freq_id:
        raise ValueError("receiver identity differs in stage metadata")
    if header.dims != len(shape) or list(header.shape)[: header.dims] != shape:
        raise ValueError(
            f"stage metadata shape differs: {list(header.shape)[:header.dims]} != {shape}"
        )


def aligned_offsets(first_sample, length, limit):
    """Drop only partial detector frames; never reset the absolute pair phase."""
    if type(first_sample) is not int or first_sample < 0:
        raise ValueError("a nonnegative integer FPGA start is required")
    if length < 0 or limit <= 0:
        raise ValueError("length and frame limit must be valid")
    begin = (-first_sample) % FRAME_SAMPLES
    count = min(max(0, (length - begin) // FRAME_SAMPLES), limit)
    return [begin + i * FRAME_SAMPLES for i in range(count)]


def paired_selection(sequences, keep):
    """Return complete zero-origin pairs and their common accepted-frame mask."""
    sequences = [int(value) for value in sequences]
    if any(value < 0 or value % FRAME_SAMPLES for value in sequences):
        raise ValueError("pair selection requires globally aligned frame starts")
    if len(sequences) != len(set(sequences)) or len(sequences) != len(keep):
        raise ValueError("frame identities must be unique and match the decisions")
    lookup = {value: index for index, value in enumerate(sequences)}
    complete = np.zeros(len(sequences), dtype=bool)
    paired_keep = np.zeros(len(sequences), dtype=bool)
    for index, sequence in enumerate(sequences):
        even_start = sequence - sequence % (2 * FRAME_SAMPLES)
        if even_start not in lookup or even_start + FRAME_SAMPLES not in lookup:
            continue
        first, second = lookup[even_start], lookup[even_start + FRAME_SAMPLES]
        complete[index] = True
        paired_keep[index] = bool(keep[first] and keep[second])
    return complete, paired_keep


def deployed_word(decisions, channel):
    """Return the channel's deployed word; a channel without one is never replayed."""
    rows = [row for row in decisions["channels"] if row["channel"] == channel]
    if len(rows) != 1:
        raise ValueError(
            f"the decisions must hold exactly one row for channel {channel}"
        )
    if rows[0].get("deployed_word") is None:
        raise ValueError(
            f"channel {channel} has no deployed word ({rows[0].get('outcome')}: "
            f"{rows[0].get('reason')}); its frames are masked, "
            "so there is no detector word to replay"
        )
    return int(rows[0]["deployed_word"])


def numerical_fixture(candidate, freq_id, word):
    """Copy one bound profile into an explicitly unqualified numerical fixture."""
    fixture = copy.deepcopy(candidate)
    rows = [
        row for row in fixture["profiles"] if row.get("receiver_channel_id") == freq_id
    ]
    if len(rows) != 1:
        raise ValueError("the candidate must bind exactly one receiver profile")
    calibration = rows[0]["fine_calibration"]
    selection = calibration.get("provenance", {}).get("threshold_selection", {})
    if selection.get("qualification") != "archive_review_only":
        raise ValueError("the source must remain an archive-review-only candidate")
    if selection.get("selection_granularity") != "detector_frame":
        raise ValueError(
            "the source threshold must refer to individual detector frames"
        )
    if int(calibration["cfar_multiplier_q16"]) != word:
        raise ValueError("candidate and decision word differ")
    calibration["provenance"]["source_threshold_selection"] = copy.deepcopy(selection)
    calibration["provenance"]["threshold_selection"] = {
        "qualification": "offline_numerical_fixture",
        "selection_granularity": "detector_frame",
        "note": "Literal numerical test parameters; no physical calibration or activation authorization.",
    }
    fixture["offline_test_only"] = True
    fixture["offline_test_note"] = (
        "The schema status calibrated means numerical readiness in this temporary test only. "
        "The source remains archive-review-only. This is not a calibration candidate or export."
    )
    return fixture, rows[0]


def base_config(streams, freq_id, frames, fixture_name):
    if streams <= 0 or streams % 2:
        raise ValueError("native input-stream count must be positive and even")
    config = {
        "type": "config",
        "log_level": "INFO",
        "log_config_usage": "fatal_error",
        "buffer_depth": 4,
        "mlock_frames": False,
        "cpu_affinity": [0, 1],
        "num_times": FRAME_SAMPLES,
        "num_local_freq": 1,
        "num_polarizations": 2,
        "num_dishes": streams // 2,
        "frame_arrival_period": SAMPLE_SECONDS * FRAME_SAMPLES,
        "base_dir": "out",
        "pilot_profiles_path": fixture_name,
        "weights_path": "weights.bin",
        "main_pool": {
            "kotekan_metadata_pool": "chordMetadata",
            "num_metadata_objects": 120,
        },
    }
    size = FRAME_SAMPLES * streams
    for name, frame_size in [
        ("voltage", size),
        ("dtv_mask", 1),
        ("dtv_powers", 24),
        ("dtv_fine_support", 8),
    ]:
        config[f"host_{name}_buffer"] = {
            "kotekan_buffer": "standard",
            "num_frames": 4,
            "frame_size": frame_size,
            "metadata_pool": "main_pool",
        }
    config["host_voltage_ringbuffer"] = {
        "kotekan_buffer": "ring",
        "ring_buffer_size": 4 * size,
        "metadata_pool": "main_pool",
    }
    config["read_voltage"] = {
        "kotekan_stage": "rawFileRead",
        "buf": "host_voltage_buffer",
        "base_dir": "input",
        "file_name": "voltage",
        "file_ext": "raw",
        "prefix_hostname": False,
        "end_interrupt": False,
    }
    config["run_send_voltage"] = {
        "gpu_0": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": 0,
            "in_buffers": {"host_voltage_buffer": "host_voltage_buffer"},
            "out_buffers": {"host_voltage_ringbuffer": "host_voltage_ringbuffer"},
            "commands": [
                {
                    "name": "cudaCopyToRingbuffer",
                    "input_size": size,
                    "ring_buffer_size": 4 * size,
                    "in_buf": "host_voltage_buffer",
                    "signal_buf": "host_voltage_ringbuffer",
                    "gpu_mem_output": "voltage_buffer",
                }
            ],
        }
    }
    commands = [
        {
            "name": "cudaPilotProxyDetector",
            "cuda_stream": 3,
            "voltage_name": "voltage",
            "detector_window_samples": 128,
            "dtv_mask_name": "dtv_mask",
            "dtv_powers_name": "dtv_powers",
            "dtv_fine_support_name": "dtv_fine_support",
            "num_frequencies": 1,
            "receiver_channel_id_namespace": "chime_freq_id",
        },
        {"name": "cudaSyncOutput", "cuda_stream": 4},
    ]
    outputs = {}
    for name in ["dtv_mask", "dtv_powers", "dtv_fine_support"]:
        outputs[f"host_{name}"] = f"host_{name}_buffer"
        commands.append(
            {
                "name": "cudaOutputData",
                "cuda_stream": 4,
                "gpu_mem": name + "_buffer",
                "out_buf": "host_" + name,
            }
        )
        config["dump_" + name] = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": f"host_{name}_buffer",
            "file_name": name,
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "allow_ndarray": True,
            "exit_after_n_files": frames,
        }
    config["run_dtv_detector"] = {
        "gpu_0": {
            "kotekan_stage": "cudaProcess",
            "gpu_id": 0,
            "num_cuda_streams": 5,
            "in_buffers": {"host_voltage_ringbuffer": "host_voltage_ringbuffer"},
            "out_buffers": outputs,
            "commands": commands,
        }
    }
    validate_offline_graph(config)
    return config


def validate_offline_graph(config):
    """Accept exactly the file-only test graph, with no mask consumers."""
    expected_stages = {
        "read_voltage": "rawFileRead",
        "dump_dtv_mask": "rawFileWrite",
        "dump_dtv_powers": "rawFileWrite",
        "dump_dtv_fine_support": "rawFileWrite",
        "run_send_voltage/gpu_0": "cudaProcess",
        "run_dtv_detector/gpu_0": "cudaProcess",
    }
    found = {}
    commands = []

    def walk(node, path=""):
        if isinstance(node, dict):
            if "kotekan_stage" in node:
                found[path] = node["kotekan_stage"]
            if "commands" in node:
                commands.extend(command["name"] for command in node["commands"])
            for key, value in node.items():
                walk(value, f"{path}/{key}".strip("/"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}/{index}")

    walk(config)
    if found != expected_stages or commands != [
        "cudaCopyToRingbuffer",
        "cudaPilotProxyDetector",
        "cudaSyncOutput",
        "cudaOutputData",
        "cudaOutputData",
        "cudaOutputData",
    ]:
        raise ValueError(
            "offline replay must use only the fixed file-to-detector-to-file graph"
        )


def run_binary(binary, config, work, name, timeout):
    path = work / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with (work / (name + ".log")).open("w") as log:
        result = subprocess.run(
            [str(binary), "-b", "127.0.0.1:0", "--config", path.name],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError(
            f"{name} returned {result.returncode}: "
            + (work / (name + ".log")).read_text()[-8000:]
        )


def make_metadata(binary, config, streams, freq_id, work, timeout):
    generate = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key
        in {
            "type",
            "log_level",
            "log_config_usage",
            "cpu_affinity",
            "mlock_frames",
            "num_local_freq",
            "num_polarizations",
            "num_dishes",
            "base_dir",
            "main_pool",
            "host_voltage_buffer",
        }
    }
    generate["samples_per_data_set"] = FRAME_SAMPLES
    generate["gen_voltage"] = {
        "kotekan_stage": "testDataGen",
        "out_buf": "host_voltage_buffer",
        "type": "random_signed_offset",
        "seed": 42,
        "first_frame_index": 0,
        "num_frames": 1,
        "wait": False,
        "name": "E",
        "array_shape": [FRAME_SAMPLES, 1, 2, streams // 2],
        "dim_name": ["T", "F", "P", "D"],
        "dim_scaling": [1, 1, 1, 1],
        "manual_freq_ids": [freq_id],
        "meta_time_downsample_factor": 1,
    }
    generate["dump_template"] = {
        "kotekan_stage": "rawFileWrite",
        "in_buf": "host_voltage_buffer",
        "file_name": "template",
        "file_ext": "raw",
        "prefix_hostname": False,
        "num_frames_per_file": 1,
        "allow_ndarray": True,
        "exit_after_n_files": 1,
    }
    run_binary(binary, generate, work, "metadata", timeout)
    paths = list((work / "out").glob("template_*.raw"))
    if len(paths) != 1:
        raise ValueError("metadata fixture did not write exactly one frame")
    check_metadata(paths[0], freq_id, [FRAME_SAMPLES, 1, 2, streams // 2])
    with paths[0].open("rb") as stream:
        size = struct.unpack("<I", stream.read(4))[0]
        metadata = stream.read(size)
    if size != reference.CHORD_METADATA_SIZE:
        raise ValueError("unsupported metadata ABI")
    prefix = reference.CHORD_METADATA_PREFIX.unpack_from(metadata)
    if prefix[:6] != reference.CHORD_METADATA_LIMITS or prefix[-1] != 1:
        raise ValueError("invalid metadata ABI or native-sample period")
    paths[0].unlink()
    return metadata


def cpu_products(block, profile, candidate, weight_bank):
    from pilot_proxy.fine_decision import (
        fine_mask_decision,
        fine_required_multiplier_q16,
    )
    from pilot_proxy.fxfft import fine_power_fx

    streams = block.shape[1]
    real, imag = reference.detector_rows_from_voltage(
        block.reshape(-1),
        0,
        candidate["input_preprocessing"]["time_reverse_detector_windows_before_kernel"],
        num_freq=1,
        num_dishes=streams // 2,
        window_samples=128,
    )
    offset = profile["weight_bank_offset_bytes"]
    weights = weight_bank[offset : offset + profile["weight_bank_nbytes"]]
    projections = reference.expected_row_sums(real, imag, weights)
    fine = fine_power_fx(projections, num_streams=streams)
    # Frame powers are the sums of each term's 256 fine powers, as the stage emits.
    powers = fine.sum(axis=1, dtype=np.uint64)
    calibration = profile["fine_calibration"]
    geometry = {
        "anchor_bin": calibration["anchor_bin"],
        "designated_half_width": calibration["designated_half_width"],
        "bulk_mask": [int(word, 16) for word in calibration["bulk_mask_words_hex"]],
        "cfar_rank": calibration["cfar_rank"],
    }
    decision = fine_mask_decision(
        fine, multiplier_q16=calibration["cfar_multiplier_q16"], **geometry
    )
    required = fine_required_multiplier_q16(fine, **geometry)
    return powers, fine, decision, required


def proxy_excess(powers, profile):
    denominator = int(powers[1]) + int(powers[2])
    if denominator <= 0:
        return None
    return (
        int(powers[0])
        * int(profile["reference_norm_sum_sq"])
        / (denominator * int(profile["target_norm_sq"]))
    ) - 1.0


def validate_weight_norms(profile, weight_bank):
    offset, size = profile["weight_bank_offset_bytes"], profile["weight_bank_nbytes"]
    if offset < 0 or size != 384 or offset + size > len(weight_bank):
        raise ValueError("profile weight slice is not a complete K=128 profile")
    packed = np.frombuffer(weight_bank[offset : offset + size], dtype=np.uint8).reshape(
        3, 128
    )
    real = reference.sign_extend_nibble_twos(packed >> 4).astype(np.int64)
    imag = reference.sign_extend_nibble_twos(packed).astype(np.int64)
    norms = (real * real + imag * imag).sum(axis=1)
    if (
        int(norms[0]) != profile["target_norm_sq"]
        or int(norms[1] + norms[2]) != profile["reference_norm_sum_sq"]
    ):
        raise ValueError("profile proxy norms differ from the exact packed weights")
    if np.any(norms <= 0):
        raise ValueError("all three templates must have positive norms")
    return [int(value) for value in norms]


def summary_for_selection(rows, selected, eligible, shelf):
    values = [
        row["signed_excess"] * shelf
        for row, keep in zip(rows, selected)
        if keep and row["signed_excess"] is not None
    ]
    count = sum(bool(value) for value in selected)
    mean = float(np.mean(values)) if values else None
    return {
        "eligible_frames": int(sum(eligible)),
        "kept_frames": count,
        "kept_fraction": count / sum(eligible) if sum(eligible) else None,
        "signed_proxy_mean": mean,
        "displayed_proxy_mean": max(0.0, mean) if mean is not None else None,
        "events": 1,
        "sampling_guard": None,
        "scope": "Finite single-capture diagnostic; no event-ensemble confidence or policy qualification.",
    }


def replay(args):
    import pilot_proxy.fine_decision
    import pilot_proxy.fxfft

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    source_profile = args.candidate / "pilot_profiles.json"
    source_weights = args.candidate / "weights.bin"
    hashes = {
        "candidate_profiles": sha256(source_profile),
        "weights": sha256(source_weights),
        "decisions": sha256(args.decisions),
        "binary": sha256(args.binary),
        "raw": sha256(args.raw),
        "script": sha256(Path(__file__)),
        "reference": sha256(Path(reference.__file__)),
        "fine_decision": sha256(Path(pilot_proxy.fine_decision.__file__)),
        "fxfft": sha256(Path(pilot_proxy.fxfft.__file__)),
    }
    candidate_hashes = validate_candidate_files(args.candidate)
    candidate = json.loads(source_profile.read_text())
    manifest = json.loads((args.candidate / "weights.manifest.json").read_text())
    if manifest["weights_sha256"] != hashes["weights"]:
        raise ValueError("weight manifest does not identify the source weight bytes")
    weight_bank = source_weights.read_bytes()
    decisions = json.loads(args.decisions.read_text())
    with h5py.File(args.raw, "r") as source:
        raw = source["baseband"]
        if raw.ndim != 2 or raw.dtype != np.dtype("uint8"):
            raise ValueError("native offset-binary uint8 [time,input] is required")
        first = source.attrs.get("time0_fpga_count")
        if not isinstance(first, (int, np.integer)) or int(first) < 0:
            raise ValueError("native integer time0_fpga_count is required")
        first = int(first)
        if not math.isclose(
            float(source.attrs["delta_time"]), SAMPLE_SECONDS, rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError("FPGA tick must be one 2560-ns native sample")
        freq_id = int(source.attrs["freq_id"])
        if not math.isclose(
            float(source.attrs["freq"]) * 1e6,
            800e6 - freq_id * 390625,
            abs_tol=1,
            rel_tol=0,
        ):
            raise ValueError("native receiver identity and center disagree")
        length, streams = raw.shape
        offsets = aligned_offsets(first, length, args.max_frames)
        if not offsets:
            raise ValueError("no globally aligned complete detector frame")
        matches = [
            row
            for row in candidate["profiles"]
            if row.get("receiver_channel_id") == freq_id
        ]
        if len(matches) != 1:
            raise ValueError("exactly one candidate receiver profile is required")
        channel = int(matches[0]["physical_channel"])
        word = deployed_word(decisions, channel)
        fixture, profile = numerical_fixture(candidate, freq_id, word)
        if profile["weight_bank_nbytes"] != 384:
            raise ValueError("replay requires K=128")
        template_norms = validate_weight_norms(profile, weight_bank)
        event = int(source.attrs["event_id"]) if "event_id" in source.attrs else None
    protocol = {
        "schema": SCHEMA,
        "input_hashes": hashes,
        "max_frames": args.max_frames,
        "candidate_files_sha256": candidate_hashes,
        "raw": str(args.raw),
        "frame_origin": 0,
        "pair_origin": 0,
        "frame_samples": FRAME_SAMPLES,
        "sample_seconds": SAMPLE_SECONDS,
        "incomplete_pair_rule": "exclude incomplete pairs from paired diagnostic",
        "channel": channel,
        "freq_id": freq_id,
        "word_q16": str(word),
        "event_id": event,
        "native_first_sample": first,
        "native_samples": length,
        "streams": streams,
        "selected_offsets": offsets,
    }
    protocol["template_norms_sq"] = template_norms
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("output contains a different frozen replay protocol")
    write_json(protocol_path, protocol)
    receipt_path = args.output / "receipt.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get("result") == "PASS":
            if any(previous.get(key) != value for key, value in protocol.items()):
                raise ValueError("saved receipt does not match the frozen protocol")
            for name, expected in previous["output_hashes"].items():
                if sha256(args.output / name) != expected:
                    raise ValueError(f"completed replay output changed: {name}")
            for index in range(previous["frames"]):
                row = json.loads((args.output / f"frame-{index:04d}.json").read_text())
                if (
                    sha256(args.output / f"frame-{index:04d}.npz")
                    != row["products_sha256"]
                ):
                    raise ValueError("completed reference products changed")
            print(f"Already verified: {receipt_path}")
            return 0
    work = Path(tempfile.mkdtemp(prefix="offline-fixture-", dir=args.output))
    (work / "out").mkdir()
    (work / "input").mkdir()
    fixture_name = "offline-numerical-fixture.json"
    write_json(work / fixture_name, fixture)
    shutil.copyfile(source_weights, work / "weights.bin")
    # The stage checks the files it loads against the bundle's sha256sums.txt.
    for name in ("detector_contract.json", "weights.manifest.json"):
        shutil.copyfile(args.candidate / name, work / name)
    sums = {
        name: sha256(work / name)
        for name in ("detector_contract.json", "weights.bin", "weights.manifest.json")
    }
    sums["pilot_profiles.json"] = sha256(work / fixture_name)
    (work / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )
    config = base_config(streams, freq_id, len(offsets), fixture_name)
    metadata = make_metadata(args.binary, config, streams, freq_id, work, args.timeout)
    rows = []
    with h5py.File(args.raw, "r") as source:
        for index, offset in enumerate(offsets):
            sequence = first + offset
            block = np.ascontiguousarray(
                source["baseband"][offset : offset + FRAME_SAMPLES, :]
            )
            frame_hash = hashlib.sha256(block.tobytes()).hexdigest()
            frame_record = args.output / f"frame-{index:04d}.json"
            frame_products = args.output / f"frame-{index:04d}.npz"
            if frame_record.exists() and frame_products.exists():
                row = json.loads(frame_record.read_text())
                if row["input_sha256"] != frame_hash or row["fpga_seq_num"] != sequence:
                    raise ValueError("checkpoint differs from raw input")
            else:
                powers, fine, decision, required = cpu_products(
                    block, profile, candidate, weight_bank
                )
                row = {
                    "frame": index,
                    "fpga_seq_num": sequence,
                    "pair_index": sequence // (2 * FRAME_SAMPLES),
                    "frame_parity": (sequence // FRAME_SAMPLES) % 2,
                    "raw_sample_offset": offset,
                    "input_sha256": frame_hash,
                    "mask": int(decision.mask),
                    "rank_valid": int(decision.valid),
                    "n_bulk": int(decision.n_bulk),
                    "required_q16": (
                        str(required.multiplier_q16)
                        if required.multiplier_q16 is not None
                        else None
                    ),
                    "signed_excess": proxy_excess(powers, profile),
                }
                np.savez_compressed(
                    frame_products, frame_power_u64=powers, fine_power_u64=fine
                )
                row["products_sha256"] = sha256(frame_products)
                write_json(frame_record, row)
            if sha256(frame_products) != row["products_sha256"]:
                raise ValueError("reference product checkpoint changed")
            header = bytearray(metadata)
            struct.pack_into("<i", header, 24, index)
            struct.pack_into("<q", header, 32, sequence)
            struct.pack_into("<q", header, 40, 1)
            path = work / "input" / f"voltage_{index:07d}.raw"
            with path.open("wb") as stream:
                stream.write(struct.pack("<I", len(header)))
                stream.write(header)
                stream.write(block.tobytes())
            rows.append(row)
            print(
                f"Prepared {index + 1}/{len(offsets)}: sequence {sequence}", flush=True
            )
    run_binary(args.binary, config, work, "stage", args.timeout)
    outputs = {
        name: reference.read_raw_frames(str(work / "out" / f"{name}_*.raw"), size)
        for name, size in [("dtv_mask", 1), ("dtv_powers", 24), ("dtv_fine_support", 8)]
    }
    if any(len(frames) != len(rows) for frames in outputs.values()):
        raise ValueError("stage output frame count differs")
    for name, shape in [
        ("dtv_mask", [1]),
        ("dtv_powers", [1, 3]),
        ("dtv_fine_support", [1, 2]),
    ]:
        for path in sorted((work / "out").glob(f"{name}_*.raw")):
            check_metadata(path, freq_id, shape)
    for index, row in enumerate(rows):
        for name, frames in outputs.items():
            if (
                frames[index].fpga_seq_num != row["fpga_seq_num"]
                or frames[index].time_downsampling_fpga != FRAME_SAMPLES
            ):
                raise ValueError(f"{name} timing differs for frame {index}")
        with np.load(args.output / f"frame-{index:04d}.npz") as expected:
            np.testing.assert_array_equal(
                outputs["dtv_powers"][index].payload.view("<u8"),
                expected["frame_power_u64"],
            )
        np.testing.assert_array_equal(outputs["dtv_mask"][index].payload, [row["mask"]])
        np.testing.assert_array_equal(
            outputs["dtv_fine_support"][index].payload.view("<i4"),
            [row["rank_valid"], row["n_bulk"]],
        )
    keep = [row["mask"] == 0 and row["rank_valid"] == 1 for row in rows]
    complete, pair_keep = paired_selection([row["fpga_seq_num"] for row in rows], keep)
    shelf = float(decisions["inputs"]["shelf"])
    for row, kept, paired, eligible in zip(rows, keep, pair_keep, complete):
        row.update(
            frame_keep=bool(kept),
            pair_keep=bool(paired),
            complete_pair=bool(eligible),
            signed_proxy=(
                row["signed_excess"] * shelf
                if row["signed_excess"] is not None
                else None
            ),
        )
    with (args.output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if validate_candidate_files(args.candidate) != candidate_hashes:
        raise ValueError("source candidate changed during replay")
    receipt = {
        **protocol,
        "result": "PASS",
        "frames": len(rows),
        "stage_words_compared": 6 * len(rows),
        "stage_products_compared": [
            "mask",
            "three frame powers",
            "rank_valid",
            "n_bulk",
            "frame identities and periods",
        ],
        "fine_powers": "Independent CPU reference only; the stage does not export fine-power scratch.",
        "candidate_unchanged": True,
        "offline_numerical_fixture": fixture,
        "offline_fixture_sha256": sha256(work / fixture_name),
        "offline_graph": config,
        "discarded_prefix_samples": offsets[0],
        "unused_suffix_samples": length - offsets[-1] - FRAME_SAMPLES,
        "incomplete_pair_frames": int((~complete).sum()),
        "individual": summary_for_selection(
            rows, keep, np.ones(len(rows), bool), shelf
        ),
        "individual_on_complete_pairs": summary_for_selection(
            rows, np.array(keep) & complete, complete, shelf
        ),
        "pair_expanded": summary_for_selection(rows, pair_keep, complete, shelf),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "wall_seconds": time.monotonic() - started,
        "scope": "Offline full detector-stage correctness and finite aligned-pair selection; no runtime pair mask, physical calibration, population policy, or deployment claim.",
    }
    for name in ["metadata.log", "stage.log", "metadata.yaml", "stage.yaml"]:
        shutil.copyfile(work / name, args.output / name)
    receipt["output_hashes"] = {
        name: sha256(args.output / name)
        for name in [
            "frames.csv",
            "stage.log",
            "stage.yaml",
            "metadata.log",
            "metadata.yaml",
        ]
        + [f"frame-{i:04d}.json" for i in range(len(rows))]
        + [f"frame-{i:04d}.npz" for i in range(len(rows))]
    }
    write_json(receipt_path, receipt)
    # The executable fixture and derived voltage files have no purpose outside this test.
    shutil.rmtree(work)
    print(
        f"PASS: {len(rows)} real frames, {int(pair_keep.sum())}/{int(complete.sum())} pair-retained -> {receipt_path}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["binary", "candidate", "decisions", "raw", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    for name in ["binary", "candidate", "decisions", "raw", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
