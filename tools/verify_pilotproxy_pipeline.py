#!/usr/bin/env python3
"""Compare recorded detector outputs with CPU calculations from the input voltages.

The config in config/tests/verify_pilotproxy_pipeline.yaml defines the raw
array layouts. Frame metadata supplies timing and frequency IDs. The runtime
bundle supplies channel bindings, preprocessing, weights and thresholds.
Coarse powers and masks use NumPy integer calculations; fine masks use the
matching pilot-proxy reference package.

Usage:
    python3 tools/verify_pilotproxy_pipeline.py \
        --dump-dir fake_data/pilotproxy_verify --bundle-dir fake_data/pilotproxy_bundle

Returns 0 when all expected captures and detector products agree, 1 otherwise.
"""

import argparse
from dataclasses import dataclass
import glob
import json
import os
import struct
import sys

import numpy as np

# Geometry defaults; must match verify_pilotproxy_pipeline.yaml.
NUM_TIMES = 8192
BLOCK_SAMPLES = 8192
NUM_FREQ = 4
NUM_POL = 2
NUM_DISHES = 8
K = 64
FREQ_IDS = [2408, 1600, 2623, 4000]
NUM_INPUT_FRAMES = 8


@dataclass
class RawFrame:
    payload: np.ndarray
    fpga_seq_num: int
    time_downsampling_fpga: int


def read_raw_frames(pattern, frame_size):
    """Read rawFileWrite dumps: per frame, u32 metadata_size + metadata +
    frame bytes. Retains the timing fields from chordMetadataFormat."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            data = f.read()
        offset = 0
        while offset < len(data):
            if len(data) - offset < 4:
                raise ValueError(
                    f"{path}: truncated metadata length at offset {offset}"
                )
            (metadata_size,) = struct.unpack_from("<I", data, offset)
            offset += 4
            metadata = data[offset : offset + metadata_size]
            if metadata_size < 28 or len(metadata) != metadata_size:
                raise ValueError(f"{path}: missing or truncated timing metadata")
            # Fixed prefix of lib/metadata/chordMetadata.cpp's serialized format.
            if struct.unpack_from("<3i", metadata) != (10, 24, 12288):
                raise ValueError(f"{path}: unsupported CHORD metadata layout")
            sequence, downsampling = struct.unpack_from("<qi", metadata, 16)
            if sequence < 0 or downsampling <= 0:
                raise ValueError(f"{path}: missing or invalid FPGA timing")
            offset += metadata_size
            frame = data[offset : offset + frame_size]
            if len(frame) != frame_size:
                raise ValueError(f"{path}: truncated frame at offset {offset}")
            frames.append(
                RawFrame(np.frombuffer(frame, dtype=np.uint8), sequence, downsampling)
            )
            offset += frame_size
    return frames


def validate_frames(voltage, masks, powers, expected_input_frames):
    """Require a complete capture with matching absolute frame identities."""
    frames_per_block = BLOCK_SAMPLES // NUM_TIMES
    if (
        expected_input_frames <= 0
        or BLOCK_SAMPLES % NUM_TIMES
        or expected_input_frames % frames_per_block
    ):
        raise ValueError("input frame count must contain whole detector blocks")
    num_blocks = expected_input_frames // frames_per_block
    if (len(voltage), len(masks), len(powers)) != (
        expected_input_frames,
        num_blocks,
        num_blocks,
    ):
        raise ValueError(
            f"expected {expected_input_frames} voltage / {num_blocks} mask / "
            f"{num_blocks} power frames; got {len(voltage)} / {len(masks)} / {len(powers)}"
        )
    start = voltage[0].fpga_seq_num
    downsampling = voltage[0].time_downsampling_fpga
    for index, frame in enumerate(voltage):
        if (
            frame.fpga_seq_num != start + index * NUM_TIMES * downsampling
            or frame.time_downsampling_fpga != downsampling
        ):
            raise ValueError(f"voltage frame {index}: discontinuous FPGA timing")
    for name, frames in (("mask", masks), ("power", powers)):
        for block, frame in enumerate(frames):
            expected_sequence = start + block * BLOCK_SAMPLES * downsampling
            if (
                frame.fpga_seq_num != expected_sequence
                or frame.time_downsampling_fpga != BLOCK_SAMPLES * downsampling
            ):
                raise ValueError(
                    f"{name} frame {block}: FPGA timing does not match voltage "
                    f"(expected {expected_sequence}, {BLOCK_SAMPLES * downsampling}; "
                    f"got {frame.fpga_seq_num}, {frame.time_downsampling_fpga})"
                )
    return num_blocks


def sign_extend_nibble_twos(nibble):
    """Two's-complement 4-bit sign extension of an unsigned nibble array."""
    return ((nibble.astype(np.int16) & 0xF) ^ 8) - 8


def detector_rows_from_voltage(
    block_bytes,
    freq_index,
    time_reverse_windows,
    *,
    num_freq=NUM_FREQ,
    num_dishes=NUM_DISHES,
):
    """[T, F, P, D] offset-binary bytes -> (re, im) int16 arrays of shape
    [rows, K] in the kernel's row-major stream-major order."""
    arr = block_bytes.reshape(BLOCK_SAMPLES, num_freq, NUM_POL, num_dishes)
    channel = arr[:, freq_index, :, :]  # [T, P, D]
    # streams [S, T] with s = p * D + d, then rows [S * W, K]
    streams = channel.transpose(1, 2, 0).reshape(NUM_POL * num_dishes, BLOCK_SAMPLES)
    rows = streams.reshape(-1, BLOCK_SAMPLES // K, K).reshape(-1, K)
    if time_reverse_windows:
        rows = rows[:, ::-1]  # adapter flip assumed by the weight templates
    real = ((rows.astype(np.int16) >> 4) & 0xF) - 8  # offset-binary decode
    imag = (rows.astype(np.int16) & 0xF) - 8
    return real, imag


def expected_row_sums(real, imag, weights_packed):
    """Exact complex projections [terms, rows, 2], computed on the CPU."""
    w = np.frombuffer(weights_packed, dtype=np.uint8).reshape(3, K)
    w_real = sign_extend_nibble_twos(w >> 4)
    w_imag = sign_extend_nibble_twos(w)
    projections = []
    for term in range(3):
        # row . conj(w): re = xr*wr + xi*wi ; im = xi*wr - xr*wi
        z_real = (real * w_real[term] + imag * w_imag[term]).sum(axis=1, dtype=np.int64)
        z_imag = (imag * w_real[term] - real * w_imag[term]).sum(axis=1, dtype=np.int64)
        projections.append(np.stack((z_real, z_imag), axis=-1))
    return np.stack(projections)


def expected_powers(real, imag, weights_packed):
    """Exact uint64 power sums for the 3 packed weight terms [3, K]."""
    projections = expected_row_sums(real, imag, weights_packed)
    return (projections * projections).sum(axis=(1, 2), dtype=np.int64).tolist()


def expected_products(
    block_bytes,
    freq_ids,
    bundle,
    weight_bank,
    *,
    num_dishes=NUM_DISHES,
    decision_mode="auto",
):
    """Reference coarse powers and coarse/fine masks for a complete detector block.

    Fine decisions use PilotProxy's fixed-point Python FFT and integer CFAR
    reference. No GPU output is used to construct the expected products.
    """
    profiles = {row.get("chord_channel_id"): row for row in bundle["profiles"]}
    reverse = bundle["input_preprocessing"][
        "time_reverse_detector_windows_before_kernel"
    ]
    powers = np.zeros((len(freq_ids), 3), dtype=np.uint64)
    masks = np.zeros(len(freq_ids), dtype=np.uint8)
    for index, freq_id in enumerate(freq_ids):
        profile = profiles.get(freq_id)
        if profile is None:
            continue
        offset = profile["weight_bank_offset_bytes"]
        weights = weight_bank[offset : offset + profile["weight_bank_nbytes"]]
        real, imag = detector_rows_from_voltage(
            block_bytes, index, reverse, num_freq=len(freq_ids), num_dishes=num_dishes
        )
        projections = expected_row_sums(real, imag, weights)
        powers[index] = (projections * projections).sum(axis=(1, 2), dtype=np.uint64)
        calibration = profile["fine_calibration"]
        if decision_mode == "auto" and calibration["status"] == "calibrated":
            from pilot_proxy.fine_decision import fine_mask_decision
            from pilot_proxy.fxfft import fine_power_fx

            fine = fine_power_fx(projections, num_streams=NUM_POL * num_dishes)
            masks[index] = fine_mask_decision(
                fine,
                anchor_bin=calibration["anchor_bin"],
                designated_half_width=calibration["designated_half_width"],
                bulk_mask=[
                    int(word, 16) for word in calibration["bulk_mask_words_hex"]
                ],
                cfar_rank=calibration["cfar_rank"],
                multiplier_q16=calibration["cfar_multiplier_q16"],
            ).mask
        else:
            target, lower, upper = map(int, powers[index])
            denominator = lower + upper
            masks[index] = int(
                denominator != 0
                and target * profile["positive_excess_half_threshold_den"]
                > profile["positive_excess_half_threshold_num"] * denominator
            )
    return masks, powers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", default="fake_data/pilotproxy_verify")
    parser.add_argument("--bundle-dir", default="fake_data/pilotproxy_bundle")
    parser.add_argument(
        "--num-input-frames",
        type=int,
        default=NUM_INPUT_FRAMES,
        help="expected voltage frame count (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(os.path.join(args.bundle_dir, "pilot_profiles.json")) as f:
        pilot_profiles = json.load(f)
    with open(os.path.join(args.bundle_dir, "weights.bin"), "rb") as f:
        weight_bank = f.read()
    time_reverse_windows = bool(
        pilot_profiles["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
    )
    print(
        "bundle input_preprocessing: time_reverse_detector_windows_before_kernel =",
        time_reverse_windows,
    )
    profiles_by_id = {
        row["chord_channel_id"]: row
        for row in pilot_profiles["profiles"]
        if row.get("chord_channel_id") is not None
    }
    if not profiles_by_id:
        print("FAIL: bundle carries no chord_channel_id entries")
        return 1

    voltage_size = NUM_TIMES * NUM_FREQ * NUM_POL * NUM_DISHES
    try:
        voltage = read_raw_frames(
            os.path.join(args.dump_dir, "voltage_*.raw"), voltage_size
        )
        masks = read_raw_frames(os.path.join(args.dump_dir, "dtv_mask_*.raw"), NUM_FREQ)
        powers = read_raw_frames(
            os.path.join(args.dump_dir, "dtv_powers_*.raw"), NUM_FREQ * 3 * 8
        )
        num_blocks = validate_frames(voltage, masks, powers, args.num_input_frames)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1

    frames_per_block = BLOCK_SAMPLES // NUM_TIMES
    failures = 0
    for block in range(num_blocks):
        block_bytes = np.concatenate(
            [
                frame.payload
                for frame in voltage[
                    block * frames_per_block : (block + 1) * frames_per_block
                ]
            ]
        )
        got_mask = masks[block].payload
        got_powers = (
            np.frombuffer(powers[block].payload.tobytes(), dtype="<u8")
            .reshape(NUM_FREQ, 3)
            .tolist()
        )
        for f, freq_id in enumerate(FREQ_IDS):
            profile = profiles_by_id.get(freq_id)
            if profile is None:
                want_powers, want_mask = [0, 0, 0], 0
            else:
                offset = profile["weight_bank_offset_bytes"]
                nbytes = profile["weight_bank_nbytes"]
                real, imag = detector_rows_from_voltage(
                    block_bytes, f, time_reverse_windows
                )
                want_powers = expected_powers(
                    real, imag, weight_bank[offset : offset + nbytes]
                )
                num = want_powers[0]
                den = want_powers[1] + want_powers[2]
                half_num = profile["positive_excess_half_threshold_num"]
                half_den = profile["positive_excess_half_threshold_den"]
                want_mask = int(den != 0 and num * half_den > half_num * den)
            ok_powers = got_powers[f] == want_powers
            ok_mask = int(got_mask[f]) == want_mask
            status = "ok" if (ok_powers and ok_mask) else "MISMATCH"
            label = (
                f"pilot ch {profile['physical_channel']}" if profile else "non-pilot"
            )
            print(
                f"block {block} f={f} (freq_id {freq_id}, {label}): "
                f"mask got={int(got_mask[f])} want={want_mask}; powers "
                f"got={got_powers[f]} want={want_powers} -> {status}"
            )
            if not (ok_powers and ok_mask):
                failures += 1

    if failures:
        print(f"FAIL: {failures} mismatching channel-block(s)")
        return 1
    print(f"PASS: {num_blocks} blocks x {NUM_FREQ} channels bit-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
