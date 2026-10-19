"""T3 quantization: the int4 outputs must use the 4-bit range sensibly, and FRB3 must encode
FRB2 within half a quantization step.

Input: complex Gaussian noise with `NOISE_RMS` LSB per component on every input, plus a weak
point-source tone (lib/testing/testSkySignal.hpp), with the production gains.

For the upchannelized voltages (every U), the BB beams and the FRB3 beams the check measures the
rms, the fraction of zeros, the fraction at ±7 (saturated) and the fraction of -8 (no data) over
the used frequency slots, and compares them with a Gaussian noise model. It fails if the rms is
below 0.5 or above 3.5 LSB, more than 5% are saturated, or a used slot holds -8. FRB3 is
dequantized with its offset and scale and compared with FRB2.

See docs/integration-tests/T3-quantization.md.
"""

import math

import numpy as np

from kit import (
    audit,
    histogram,
    large_seq,
    read_output,
    require,
    rounded_gaussian,
    upchan_window,
    used_channels,
    value,
)

UPCHAN_FACTORS = [2, 4, 8, 16, 32, 64]
NUM_TAPS = 4
VARIANTS = ["noise"]
NOISE_RMS = 2.8
TONE_AMPLITUDE = 1
RMS_MIN, RMS_MAX, SATURATED_MAX = 0.5, 3.5, 0.05


def variables(tel, config, variant):
    outputs = [f"upchan_U{U}_voltage" for U in UPCHAN_FACTORS]
    outputs += [
        "bb_beams",
        "frb2_beams",
        "frb3_beams",
        "frb3_beams_offsetscale",
        "frb_pl_mask",
    ]
    return {
        "first_fpga_seq_num": large_seq(config),
        "input_kind": "sky",
        "sources": [
            {
                "amplitude": TONE_AMPLITUDE,
                "direction": [
                    float(np.sin(np.radians(0.5))),
                    float(np.sin(np.radians(0.8))),
                ],
                "tone_cycles_per_frame": round(19.5 / 64 * value(config, "num_times")),
            }
        ],
        "noise_rms": NOISE_RMS,
        "outputs": outputs,
        "write_frames": {name: 1 for name in outputs},
    }


def int4_line(name, hist, sigma, failures):
    """Summarize the histogram of one output against the model `sigma` (before rounding) and
    record a failure if the quantization is catastrophic."""
    values = np.arange(-8, 8)
    total = hist.sum()
    nodata = hist[0] / total
    used = hist[1:].sum()
    rms = math.sqrt((hist[1:] * values[1:] ** 2).sum() / used)
    zeros = hist[8] / used
    saturated = (hist[1] + hist[15]) / used
    model_rms, model_zeros, model_saturated = rounded_gaussian(sigma)
    line = (
        f"{name}: rms {rms:.2f} LSB (model {model_rms:.2f}), zeros {zeros:.1%} "
        f"({model_zeros:.1%}), ±7 {saturated:.2%} ({model_saturated:.2%}), -8 {nodata:.2%}"
    )
    problems = []
    if not RMS_MIN <= rms <= RMS_MAX:
        problems.append(f"rms outside [{RMS_MIN}, {RMS_MAX}]")
    if saturated > SATURATED_MAX:
        problems.append(f"more than {SATURATED_MAX:.0%} saturated")
    if nodata > 0:
        problems.append("-8 in used slots")
    if problems:
        line += "  <- " + ", ".join(problems)
        failures.append(name)
    return line


def check(out_dir, var, config):
    SEQ0 = var["first_fpga_seq_num"]
    T_STAR = (NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2
    channels = np.array(value(config, "frequency_channels"))
    num_dishes = len(config["telescope"]["dish_inputs"])
    results = []
    failures = []

    # The input is rounded and clamped to [-7, 7]; this rms enters every later stage
    sigma_in = rounded_gaussian(var["noise_rms"])[0]

    # Upchannelized voltages [Tbar, Fbar, P, D]: Ē = (G/U) Σ_s W(s) E(s) ..., so noise has the
    # rms σ_in sqrt(Σ_s W(s)²) G / U before rounding
    for U in UPCHAN_FACTORS:
        name = f"upchan_U{U}_voltage"
        ebar = read_output(out_dir, name)
        audit(
            ebar, upchan_factor=U, first_seq=SEQ0 + T_STAR, frequency_channels=channels,
        )
        gain = np.array(config[f"run_set_upchan_U{U}_gain"]["upchan_gain"], dtype=float)
        window = upchan_window(NUM_TAPS, U)
        sigma = sigma_in * math.sqrt((window ** 2).sum() * (gain ** 2).mean()) / U
        hist = histogram(used_channels(ebar, 1), offset=True)
        results.append(int4_line(name, hist, sigma, failures))

    # BB beams [Thi, B, F, P, T]
    bb = read_output(out_dir, "bb_beams")
    audit(bb, first_seq=SEQ0, frequency_channels=channels)
    # J = bb_gain · Σ_d exp(iφ_d) E_d, with |A| = 127.5 a for 2^7 a (calcBBPhase)
    sigma = sigma_in * math.sqrt(num_dishes) * value(config, "bb_gain") * 127.5 / 128
    results.append(
        int4_line("bb_beams", histogram(bb.data, offset=True), sigma, failures)
    )

    # FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256] (float16), FRB3 the same with two samples
    # per byte (int4x2, the earlier sample in the low nibble), offset and scale per chunk of 256
    # samples [Ttilde256, R, Fbar, offset/scale], and the FRB packet loss mask [Ttilde, Fbar]
    frb2 = read_output(out_dir, "frb2_beams")
    frb3 = read_output(out_dir, "frb3_beams")
    offsetscale = read_output(out_dir, "frb3_beams_offsetscale")
    pl_mask = read_output(out_dir, "frb_pl_mask")
    for output in [frb2, frb3, offsetscale, pl_mask]:
        audit(output, first_seq=SEQ0 + T_STAR)
    num_freqs = len(np.atleast_1d(frb2.attr("coarse_freq")))
    num_times = frb2.data.shape[3]
    chunk = value(config, "frb_chunk_size")
    require(
        num_times == chunk,
        f"frb2_beams: {num_times} samples per frame, expected one chunk of {chunk}",
    )
    valid = pl_mask.data[:, :num_freqs].reshape(-1, num_times, num_freqs) != 0
    valid = valid.transpose(0, 2, 1)  # [Ttilde256, Fbar, Ttildelo256]

    # Quantization: q = clamp(round((x - offset) / scale), -7, 7), offset the chunk mean and
    # scale = 2 · stddev_cutoff · stddev / 15, so noise has the rms 15 / (2 stddev_cutoff)
    # before rounding. Invalid samples are -8.
    commands = config["run_frb23"]["gpu_0"]["commands"]
    quantize = next(c for c in commands if c["name"] == "cudaQuantize4")
    stddev_cutoff = quantize["stddev_cutoff"]
    sigma = 15 / (2 * stddev_cutoff)
    hist = np.zeros(16)
    num_wrong_nodata = 0
    worst = 0.0
    num_bad = 0
    num_checked = 0
    num_zero_scale = 0
    for beam in range(frb2.data.shape[1]):
        x = frb2.data[:, beam, :num_freqs].astype(np.float64)
        q = frb3.data[:, beam, :num_freqs]
        q = np.stack([q & 0xF, q >> 4], axis=-1).reshape(x.shape).astype(np.int8)
        q = (q ^ 8) - 8
        offset = offsetscale.data[:, beam, :num_freqs, 0:1].astype(np.float64)
        scale = offsetscale.data[:, beam, :num_freqs, 1:2].astype(np.float64)
        hist += np.bincount(q.ravel() + 8, minlength=16)
        num_wrong_nodata += np.count_nonzero((q == -8) != ~valid)
        num_zero_scale += np.count_nonzero(scale == 0)
        # float16 rounding of 1/scale, -offset/scale and of the fused multiply-add
        allowance = 2.0 ** -9 * (np.abs(x) + np.abs(offset) + scale)
        y = (x - offset) / np.where(scale > 0, scale, 1)
        inner = valid & (np.abs(q) < 7)
        err = np.abs(q - y) * scale - scale / 2 - allowance
        outer = valid & (np.abs(q) == 7)
        short = (6.5 - np.sign(q) * y) * scale - allowance
        bad = (inner & (err > 0)) | (outer & (short > 0))
        num_bad += np.count_nonzero(bad)
        num_checked += np.count_nonzero(valid)
        use = inner & (scale > 0)
        if use.any():
            worst = max(worst, float(np.abs(q - y)[use].max()))
    results.append(int4_line("frb3_beams", hist, sigma, failures))
    line = (
        f"frb3_beams vs frb2_beams: |q - (x - offset) / scale| <= {worst:.3f} below "
        f"saturation, {num_bad} of {num_checked} valid samples beyond half a step, "
        f"{valid.mean():.2%} valid, {num_wrong_nodata} samples with -8 not matching the "
        f"mask, {num_zero_scale} chunks with scale 0"
    )
    problems = []
    if num_bad:
        problems.append("beyond half a step")
    if num_wrong_nodata:
        problems.append("-8 does not match the mask")
    if not valid.all():
        problems.append("invalid samples without packet loss")
    if problems:
        line += "  <- " + ", ".join(problems)
        failures.append("frb3_beams vs frb2_beams")
    results.append(line)

    require(
        not failures,
        f"{len(failures)} outputs out of range:\n    " + "\n    ".join(results),
    )
    return results
