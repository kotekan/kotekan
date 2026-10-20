"""T3 quantization: the int4 outputs must use the 4-bit range sensibly, and FRB3 must encode
FRB2 within half a quantization step.

Input: complex Gaussian noise with `NOISE_RMS` LSB per component on every input, plus a weak
point-source tone (lib/testing/testSkySignal.hpp), with the production gains.

For the upchannelized voltages (every U), the BB beams and the 4-bit FRB3 beams the check
measures the rms, the fraction of zeros, the fraction at ±7 (saturated) and the fraction of -8
(no data) over the used frequency slots, and compares them with a Gaussian noise model. It fails
if the rms is below 0.5 or above 3.5 LSB, more than 5% are saturated, or a used slot holds -8.
FRB3 (4 bits, or 8 bits for CHIME, which has no BB beamformer) is dequantized with its offset and
scale and compared with FRB2.

See docs/integration-tests/T3-quantization.md.
"""

import math

import numpy as np

from kit import (
    NUM_TAPS,
    audit,
    frb3_codes,
    has_bb,
    histogram,
    large_seq,
    read_output,
    require,
    rounded_gaussian,
    upchan_t_star,
    upchan_window,
    used_channels,
    value,
)

VARIANTS = ["noise"]
NOISE_RMS = 2.8
TONE_AMPLITUDE = 1
RMS_MIN, RMS_MAX, SATURATED_MAX = 0.5, 3.5, 0.05


def variables(tel, config, variant):
    outputs = [f"upchan_U{U}_voltage" for U in config["upchan_factors"]]
    outputs += ["bb_beams"] * has_bb(config) + [
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
    T_STAR = upchan_t_star(config)
    channels = np.array(value(config, "frequency_channels"))
    num_dishes = len(config["telescope"]["dish_inputs"])
    results = []
    failures = []

    # The input is rounded and clamped to [-7, 7]; this rms enters every later stage
    sigma_in = rounded_gaussian(var["noise_rms"])[0]

    # Upchannelized voltages [Tbar, Fbar, P, D]: Ē = (G/U) Σ_s W(s) E(s) ..., so noise has the
    # rms σ_in sqrt(Σ_s W(s)²) G / U before rounding
    for U in config["upchan_factors"]:
        name = f"upchan_U{U}_voltage"
        ebar = read_output(out_dir, name)
        audit(
            ebar, upchan_factor=U, first_seq=SEQ0 + T_STAR, frequency_channels=channels,
        )
        gain = np.array(
            value(config, f"run_set_upchan_U{U}_gain")["upchan_gain"], dtype=float
        )
        window = upchan_window(NUM_TAPS, U)
        sigma = sigma_in * math.sqrt((window ** 2).sum() * (gain ** 2).mean()) / U
        hist = histogram(used_channels(ebar, 1), offset=True)
        results.append(int4_line(name, hist, sigma, failures))

    # BB beams [Thi, B, F, P, T]
    if "bb_beams" in var["outputs"]:
        bb = read_output(out_dir, "bb_beams")
        audit(bb, first_seq=SEQ0, frequency_channels=channels)
        # J = bb_gain · Σ_d exp(iφ_d) E_d, with |A| = 127.5 a for 2^7 a (calcBBPhase)
        sigma = sigma_in * math.sqrt(num_dishes) * value(config, "bb_gain") * 127.5 / 128
        results.append(
            int4_line("bb_beams", histogram(bb.data, offset=True), sigma, failures)
        )

    # FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256] (float16), FRB3 codes with offset and scale
    # (kit.frb3_codes), and the FRB packet loss mask [Ttilde, Fbar]
    frb2 = read_output(out_dir, "frb2_beams")
    pl_mask = read_output(out_dir, "frb_pl_mask")
    for name in ["frb2_beams", "frb3_beams", "frb3_beams_offsetscale"]:
        audit(read_output(out_dir, name), first_seq=SEQ0 + T_STAR)
    audit(pl_mask, first_seq=SEQ0 + T_STAR)
    num_freqs = len(np.atleast_1d(frb2.attr("coarse_freq")))
    num_times = frb2.data.shape[3]
    chunk = value(config, "frb_chunk_size")
    require(
        num_times == chunk,
        f"frb2_beams: {num_times} samples per frame, expected one chunk of {chunk}",
    )
    valid = pl_mask.data[:, :num_freqs].reshape(-1, num_times, num_freqs) != 0
    valid = valid.transpose(0, 2, 1)[:, None]  # [Ttilde256, 1, Fbar, Ttildelo256]
    q_all, offset_all, scale_all, nodata, qmin, qmax = frb3_codes(out_dir, num_freqs)

    # Quantization: q = clamp(round((x - offset) / scale), qmin, qmax); invalid samples are
    # `nodata`. For 4 bits the offset is the chunk mean and scale = 2 · stddev_cutoff ·
    # stddev / 15, so noise has the rms 15 / (2 stddev_cutoff) before rounding; 8 bits map the
    # chunk's range to the codes 1 to 254.
    if nodata == -8:
        commands = value(config, "run_frb23")["gpu_0"]["commands"]
        quantize = next(c for c in commands if c["name"] == "cudaQuantize4")
        sigma = 15 / (2 * quantize["stddev_cutoff"])
        hist = np.bincount(q_all.astype(int).ravel() + 8, minlength=16)
        results.append(int4_line("frb3_beams", hist, sigma, failures))
    num_wrong_nodata = num_bad = num_checked = num_zero_scale = 0
    worst = 0.0
    used = set()
    # A few beams at a time, to bound the memory
    for beam in range(0, frb2.data.shape[1], 16):
        beams = slice(beam, beam + 16)
        x = frb2.data[:, beams, :num_freqs].astype(np.float64)
        q, offset, scale = q_all[:, beams], offset_all[:, beams], scale_all[:, beams]
        ok = np.broadcast_to(valid, q.shape)
        num_wrong_nodata += np.count_nonzero((q == nodata) != ~ok)
        num_zero_scale += np.count_nonzero(np.broadcast_to(scale, q.shape) == 0)
        # float16 rounding of 1/scale, -offset/scale and of the fused multiply-add
        allowance = 2.0 ** -9 * (np.abs(x) + np.abs(offset) + scale)
        y = (x - offset) / np.where(scale > 0, scale, 1)
        # 4 bits clamp to ±7; 8 bits map the chunk's range to the codes without clamping
        inner = ok & (q > qmin) & (q < qmax) if nodata == -8 else ok
        err = np.abs(q - y) * scale - scale / 2 - allowance
        outer = ok & ~inner
        short = np.where(q == qmax, qmax - 0.5 - y, y - qmin - 0.5) * scale - allowance
        num_bad += np.count_nonzero((inner & (err > 0)) | (outer & (short > 0)))
        num_checked += np.count_nonzero(ok)
        use = inner & (scale > 0)
        if use.any():
            worst = max(worst, float(np.abs(q - y)[use].max()))
        used |= set(np.unique(q[ok]).tolist())
    line = (
        f"frb3_beams vs frb2_beams: codes {int(min(used))} to {int(max(used))} "
        f"({len(used)} distinct), |q - (x - offset) / scale| <= {worst:.3f} below "
        f"saturation, {num_bad} of {num_checked} valid samples beyond half a step, "
        f"{valid.mean():.2%} valid, {num_wrong_nodata} samples with {nodata} not matching the "
        f"mask, {num_zero_scale} samples with scale 0"
    )
    problems = []
    if num_bad:
        problems.append("beyond half a step")
    if num_wrong_nodata:
        problems.append(f"{nodata} does not match the mask")
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
