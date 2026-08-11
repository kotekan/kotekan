#!/usr/bin/env python3
"""Offline verifier for config/tests/verify_pilotproxy_pipeline.yaml.

Reads the rawFileWrite dumps produced by the harness (voltage frames plus
the detector's dtv_mask / dtv_powers products), recomputes the exact
expected uint64 coarse power sums and the coarse rational mask from the
dumped voltage and the PilotProxy runtime bundle, and compares bit-for-bit.

The recomputation is self-contained (numpy only) and mirrors the locked
detector contract (external/pilotproxy/f_statistic.h):

- voltage bytes are int4x2_swapped_withoffset: stored = value + 8 per
  nibble, imaginary low nibble, real high nibble (decoding nibble - 8 is
  exactly the stage's XOR 0x88 conversion);
- detector rows: stream s = p * num_dishes + d (P outer, D inner), windows
  of K = 128 consecutive samples per stream; each window is time-reversed
  when the bundle's input_preprocessing requests it (CHORD bundles set
  time_reverse_detector_windows_before_kernel = true: post-spectral-sense
  weight templates assume the adapter flip);
- weights: packed two's-complement int4, real high nibble; the kernel dot
  product multiplies samples by conj(weight);
- P[term] = sum over rows |row . conj(w_term)|^2, exact integers;
- mask = (den != 0) and (P_target * half_den > half_num * den), with
  num = P_target, den = P_ref_lower + P_ref_upper and the per-channel
  rational half-threshold from pilot_profiles.json.

Usage:
    python3 tools/verify_pilotproxy_pipeline.py \
        --dump-dir data/pilotproxy_verify --bundle-dir data/pilotproxy_bundle

Exit code 0 on bit-exact agreement, 1 otherwise.
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

# Geometry defaults; must match verify_pilotproxy_pipeline.yaml.
NUM_TIMES = 8192
BLOCK_SAMPLES = 16384
NUM_FREQ = 4
NUM_POL = 2
NUM_DISHES = 8
K = 128
FREQ_IDS = [2408, 1600, 2623, 4000]


def read_raw_frames(pattern, frame_size):
    """Read rawFileWrite dumps: per frame, u32 metadata_size + metadata +
    frame bytes. Returns the concatenated frame payloads in file order."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            data = f.read()
        offset = 0
        while offset < len(data):
            (metadata_size,) = struct.unpack_from("<I", data, offset)
            offset += 4 + metadata_size
            frame = data[offset : offset + frame_size]
            if len(frame) != frame_size:
                raise ValueError(f"{path}: truncated frame at offset {offset}")
            frames.append(np.frombuffer(frame, dtype=np.uint8))
            offset += frame_size
    return frames


def sign_extend_nibble_twos(nibble):
    """Two's-complement 4-bit sign extension of an unsigned nibble array."""
    return ((nibble.astype(np.int16) & 0xF) ^ 8) - 8


def detector_rows_from_voltage(block_bytes, freq_index, time_reverse_windows):
    """[T, F, P, D] offset-binary bytes -> (re, im) int16 arrays of shape
    [rows, K] in the kernel's row-major stream-major order."""
    arr = block_bytes.reshape(BLOCK_SAMPLES, NUM_FREQ, NUM_POL, NUM_DISHES)
    channel = arr[:, freq_index, :, :]  # [T, P, D]
    # streams [S, T] with s = p * D + d, then rows [S * W, K]
    streams = channel.transpose(1, 2, 0).reshape(NUM_POL * NUM_DISHES, BLOCK_SAMPLES)
    rows = streams.reshape(-1, BLOCK_SAMPLES // K, K).reshape(-1, K)
    if time_reverse_windows:
        rows = rows[:, ::-1]  # adapter flip assumed by the weight templates
    real = ((rows.astype(np.int16) >> 4) & 0xF) - 8  # offset-binary decode
    imag = (rows.astype(np.int16) & 0xF) - 8
    return real, imag


def expected_powers(real, imag, weights_packed):
    """Exact uint64 power sums for the 3 packed weight terms [3, K]."""
    w = np.frombuffer(weights_packed, dtype=np.uint8).reshape(3, K)
    w_real = sign_extend_nibble_twos(w >> 4)
    w_imag = sign_extend_nibble_twos(w)
    powers = []
    for term in range(3):
        # row . conj(w): re = xr*wr + xi*wi ; im = xi*wr - xr*wi
        z_real = (real * w_real[term] + imag * w_imag[term]).sum(axis=1, dtype=np.int64)
        z_imag = (imag * w_real[term] - real * w_imag[term]).sum(axis=1, dtype=np.int64)
        power = (z_real * z_real + z_imag * z_imag).sum(dtype=np.int64)
        powers.append(int(power))
    return powers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", default="data/pilotproxy_verify")
    parser.add_argument("--bundle-dir", default="data/pilotproxy_bundle")
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
    voltage = read_raw_frames(
        os.path.join(args.dump_dir, "voltage_*.raw"), voltage_size
    )
    masks = read_raw_frames(os.path.join(args.dump_dir, "dtv_mask_*.raw"), NUM_FREQ)
    powers = read_raw_frames(
        os.path.join(args.dump_dir, "dtv_powers_*.raw"), NUM_FREQ * 3 * 8
    )

    frames_per_block = BLOCK_SAMPLES // NUM_TIMES
    num_blocks = min(len(voltage) // frames_per_block, len(masks), len(powers))
    if num_blocks == 0:
        print("FAIL: no complete blocks found in", args.dump_dir)
        return 1

    failures = 0
    for block in range(num_blocks):
        block_bytes = np.concatenate(
            voltage[block * frames_per_block : (block + 1) * frames_per_block]
        )
        got_mask = masks[block]
        got_powers = (
            np.frombuffer(powers[block].tobytes(), dtype="<u8")
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
