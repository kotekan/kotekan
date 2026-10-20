"""T5 gains: `upchan_gain` (per fine channel), `bb_gain` and `frb1_input_scale` (CHIME: the FRB
feed gains, which loadFeedGains takes from `default_gains` without gain files) must scale their
outputs as configured.

Two runs see the same input, complex Gaussian noise with `NOISE_RMS` LSB per component plus a
weak point-source tone (lib/testing/testSkySignal.hpp). The `reference` run has the production
gains. The `scaled` run multiplies the gain of each fine channel by a distinct factor in
[1, 2) (`upchan_scaling`), `bb_gain` by `BB_SCALING` and `frb1_input_scale` by `FRB1_SCALING`.

Checks:
- Upchannelizer, every U and fine channel u: the ratio of the noise σ before rounding, inferred
  from the int4 rms with the rounded-Gaussian model, equals the gain ratio within
  `INT4_TOLERANCE`.
- BB: the same for all beams, with the ratio `BB_SCALING`.
- FRB1, every U and u: the intensity ratio equals FRB1_SCALING² times the ratio of the
  upchannelized power Σ|Ē|² (measured; 1 for U = 1) within `FRB1_TOLERANCE`.

See docs/integration-tests/T5-gains.md.
"""

import math

import numpy as np

from kit import (
    audit,
    find_stages,
    has_bb,
    histogram,
    large_seq,
    read_output,
    require,
    rounded_gaussian,
    upchan_t_star,
    used_channels,
    value,
)

RUNS = ["reference", "scaled"]
NOISE_RMS = 1.2
TONE_AMPLITUDE = 0.1
BB_SCALING = 2
# Below 1: setFRB1Phase refuses weights whose worst-case FRB1 intensity could overflow float16
# (lib/stages/frb1IntensityBound.hpp), and for CHORD that bound allows at most 1.5 times the
# production frb1_input_scale
FRB1_SCALING = 0.5
INT4_TOLERANCE = 0.03
FRB1_TOLERANCE = 0.01


def upchan_scaling(U):
    """The factors applied to the production gains of the fine channels in the scaled run:
    1 + k/U for k = 0, ..., U-1, in steps of U/2 - 1 (mod U), so that they are distinct and
    neighbouring channels differ by at least 14%."""
    stride = max(U // 2 - 1, 1)
    return 1 + np.arange(U) * stride % U / U


def upchan_gain(config, U):
    return np.array(
        value(config, f"run_set_upchan_U{U}_gain")["upchan_gain"], dtype=float
    )


def frb1_scale(config):
    """The scale of the FRB1 weights: `frb1_input_scale` (setFRB1Phase), or for CHIME the
    magnitude of the default feed gain of loadFeedGains."""
    gains = find_stages(config, "loadFeedGains")
    if gains:
        return abs(complex(*gains[0]["default_gains"]))
    return value(config, "frb1_input_scale")


def variables(tel, config, run):
    outputs = [f"upchan_U{U}_voltage" for U in config["upchan_factors"]]
    outputs += ["bb_beams"] * has_bb(config) + ["frb1_beams"]
    var = {
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
    if run == "scaled":
        var["upchan_gains"] = {
            str(U): (upchan_gain(config, U) * upchan_scaling(U)).tolist()
            for U in config["upchan_factors"]
        }
        if has_bb(config):
            var["bb_gain"] = BB_SCALING * value(config, "bb_gain")
        gains = find_stages(config, "loadFeedGains")
        if gains:
            var["frb_default_gains"] = [FRB1_SCALING * g for g in gains[0]["default_gains"]]
        else:
            var["frb1_input_scale"] = FRB1_SCALING * value(config, "frb1_input_scale")
    return var


def int4_stats(data):
    """rms (over values other than -8), sum of squares, fraction at ±7 and number of -8 of the
    components of int4x2_swapped_withoffset bytes."""
    hist = histogram(data, offset=True)
    squares = (hist[1:] * np.arange(-7, 8) ** 2).sum()
    used = hist[1:].sum()
    return math.sqrt(squares / used), squares, (hist[1] + hist[15]) / used, hist[0]


def unrounded_sigma(rms):
    """The σ of a Gaussian that has the given rms after rounding and clamping to [-7, 7]."""
    lo, hi = 0.01, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if rounded_gaussian(mid)[0] < rms:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def frequency_labels(output, axis):
    """`output.data` restricted to the used frequency slots, and their upchannelization factors
    and fine-channel indices."""
    num = len(np.atleast_1d(output.attr("coarse_freq")))
    factor = np.atleast_1d(output.attr("freq_upchan_factor"))[:num]
    index = np.atleast_1d(output.attr("freq_upchan_index"))[:num]
    return used_channels(output, axis), factor, index


def deviation_line(name, deviation, tolerance, failures, details):
    """Summarize the relative deviations of the measured ratios from the expected ones."""
    worst = float(np.abs(deviation).max())
    line = (
        f"{name}: measured / expected ratio - 1 in [{deviation.min():+.4f}, "
        f"{deviation.max():+.4f}]; {details}"
    )
    if not worst <= tolerance:
        line += f"  <- beyond ±{tolerance:.0%}"
        failures.append(name)
    return line


def check(runs):
    ref_dir, ref_var, ref_config = runs["reference"]
    out_dir, var, config = runs["scaled"]
    SEQ0 = var["first_fpga_seq_num"]
    T_STAR = upchan_t_star(config)
    channels = np.array(value(config, "frequency_channels"))
    results = []
    failures = []

    # Upchannelized voltages [Tbar, Fbar, P, D]. Ē_u is linear in G_u before rounding, so the
    # unrounded σ scales with float16(G_u). The power Σ|Ē|² per fine channel is kept for FRB1.
    power_ratio = {1: np.ones(1)}
    for U in config["upchan_factors"]:
        name = f"upchan_U{U}_voltage"
        stats = []
        indices = []
        for directory in [ref_dir, out_dir]:
            ebar = read_output(directory, name)
            audit(
                ebar,
                upchan_factor=U,
                first_seq=SEQ0 + T_STAR,
                frequency_channels=channels,
            )
            data, factor, index = frequency_labels(ebar, 1)
            stats.append(np.array([int4_stats(data[:, index == u]) for u in range(U)]))
            indices.append(index)
        require(
            np.array_equal(indices[0], indices[1]),
            f"{name}: the runs have different fine channels",
        )
        ref, scaled = stats
        require(
            ref[:, 3].sum() == 0 and scaled[:, 3].sum() == 0,
            f"{name}: -8 in used slots",
        )
        gain = upchan_gain(ref_config, U)
        expected = np.float16(gain * upchan_scaling(U)).astype(float) / np.float16(gain)
        sigma = np.array([[unrounded_sigma(rms) for rms in s[:, 0]] for s in stats])
        raw = scaled[:, 0] / ref[:, 0] / expected - 1
        results.append(
            deviation_line(
                name,
                sigma[1] / sigma[0] / expected - 1,
                INT4_TOLERANCE,
                failures,
                f"gain ratios {expected.min():.3f} to {expected.max():.3f}, rms "
                f"{ref[:, 0].min():.2f} to {ref[:, 0].max():.2f} (reference), "
                f"{scaled[:, 0].min():.2f} to {scaled[:, 0].max():.2f} (scaled), at most "
                f"{scaled[:, 2].max():.2%} at ±7; the raw rms ratio deviates by "
                f"{raw.min():+.4f} to {raw.max():+.4f}",
            )
        )
        power_ratio[U] = scaled[:, 1] / ref[:, 1]

    # BB beams [Thi, B, F, P, T]. calcBBPhase realizes bb_gain as 2^7 a / 2^s with a in
    # (1/2, 1], so doubling bb_gain lowers s by one and keeps the phases A.
    if "bb_beams" in var["outputs"]:
        stats = []
        for directory in [ref_dir, out_dir]:
            bb = read_output(directory, "bb_beams")
            audit(bb, first_seq=SEQ0, frequency_channels=channels)
            stats.append(int4_stats(bb.data))
        (ref_rms, _, ref_sat, ref_nodata), (rms, _, sat, nodata) = stats
        require(ref_nodata == 0 and nodata == 0, "bb_beams: -8")
        expected = value(config, "bb_gain") / value(ref_config, "bb_gain")
        require(
            expected == BB_SCALING, f"bb_beams: bb_gain ratio {expected} in the configs"
        )
        results.append(
            deviation_line(
                "bb_beams",
                np.array([unrounded_sigma(rms) / unrounded_sigma(ref_rms) / expected - 1]),
                INT4_TOLERANCE,
                failures,
                f"gain ratio {expected}, rms {ref_rms:.3f} (reference), {rms:.3f} (scaled), "
                f"{ref_sat:.2%} and {sat:.2%} at ±7; the raw rms ratio deviates by "
                f"{rms / ref_rms / expected - 1:+.4f}",
            )
        )

    # FRB1 beams [Ttilde, Fbar, beamQ, beamP]. I = |input_gain · W · FFT(Ē)|², summed over the
    # beam grid, is W² Σ|Ē|² times a constant (Parseval), so the ratio per fine channel is
    # FRB1_SCALING² times the upchannelized power ratio.
    sums = []
    for directory in [ref_dir, out_dir]:
        frb1 = read_output(directory, "frb1_beams")
        audit(frb1, first_seq=SEQ0 + T_STAR, frequency_channels=channels)
        data, factor, index = frequency_labels(frb1, 1)
        sums.append(data.sum(axis=(0, 2, 3), dtype=np.float64))
    expected_scaling = frb1_scale(config) / frb1_scale(ref_config)
    require(
        abs(expected_scaling - FRB1_SCALING) < 1e-12,
        f"frb1_beams: FRB1 weight ratio {expected_scaling} in the configs",
    )
    for U in [U for U in [1] + config["upchan_factors"] if np.any(factor == U)]:
        ratio = np.array(
            [
                sums[1][(factor == U) & (index == u)].sum()
                / sums[0][(factor == U) & (index == u)].sum()
                for u in range(U)
            ]
        )
        expected = expected_scaling ** 2 * power_ratio[U]
        results.append(
            deviation_line(
                f"frb1_beams U={U}",
                ratio / expected - 1,
                FRB1_TOLERANCE,
                failures,
                f"intensity ratios {ratio.min():.4f} to {ratio.max():.4f}",
            )
        )

    require(
        not failures,
        f"{len(failures)} outputs do not scale as configured:\n    "
        + "\n    ".join(results),
    )
    return results
