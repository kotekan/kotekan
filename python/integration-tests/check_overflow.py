"""T4 overflow: full-scale coherent input must saturate the outputs without wraparound, sign
flips, NaN or Inf.

Input, without noise (lib/testing/testSkySignal.hpp):
- `full_scale`: 7 + 7i on every input, constant, i.e. a coherent source at zenith at the centre
  of every coarse channel.
- `full_scale_tone`: an amplitude-7 tone from BB beam `BB_BEAM` (CHIME, without BB: from T2's
  FRB2 source beam), at 1/128 cycles per sample (a fine-channel centre for U=64). The input
  repeats every 128 samples.

First-stage RFI excision is off: a signal of constant power has spectral kurtosis 0 and would be
flagged entirely, and the correlator would sum nothing.

Checks:
- n2k_correlation: V_ij = Σ_t E_i conj(E_j) over the sub-integration, exactly.
- Upchannelizer: wherever the predicted component exceeds ±(6.5 + UPCHAN_MARGIN) the output is
  ±7, wherever it exceeds 0.5 + UPCHAN_MARGIN in magnitude the output has its sign, and there
  is no -8. Away from rounding ties the output is the predicted value, rounded and clamped.
- BB: equal to the integer beamformer applied to the written phases A and shifts s, exactly.
- FRB1: finite and at most the bound for int4 input (lib/stages/frb1IntensityBound.hpp).
- FRB2, FRB3 offset and scale: finite; FRB3: no no-data code (4 bits: -8, 8 bits: 0).

See docs/integration-tests/T4-overflow.md.
"""

import numpy as np

import check_phases
from kit import (
    NUM_TAPS,
    audit,
    decode_int4x2,
    find_stages,
    frb3_codes,
    grid_beam_direction,
    has_bb,
    large_seq,
    read_output,
    require,
    upchan_t_star,
    upchan_window,
    used_channels,
    value,
)

VARIANTS = ["full_scale", "full_scale_tone"]
BB_BEAM = (3, 1)
PERIOD = 128  # input period in samples (both variants)
# The upchannelizer computes in float16; predictions within this margin of a rounding
# threshold are not checked
UPCHAN_MARGIN = 0.02
# The BB kernel's intermediate shift σ by number of dishes: pathfinder, CHORD, CHIME
# (shift-parameter-σ in julia/output/bb_<telescope>.yaml)
BB_SIGMA = {64: 2, 512: 3, 1024: 4}
# The float16 rounding allowance of lib/stages/frb1IntensityBound.hpp
FRB1_TOLERANCE = 0.01


def variables(tel, config, variant):
    if variant == "full_scale":
        source = {
            "amplitude": 7 * np.sqrt(2),
            "phase": np.pi / 4,
            "direction": [0, 0],
            "tone_cycles_per_frame": 0,
        }
    else:
        source = {
            "amplitude": 7,
            "direction": (
                grid_beam_direction(config, "bb", *BB_BEAM)
                if has_bb(config)
                else check_phases.source_direction(config, "frb2")
            ),
            "tone_cycles_per_frame": value(config, "num_times") // PERIOD,
        }
    outputs = [f"upchan_U{U}_voltage" for U in config["upchan_factors"]]
    outputs += ["n2k_correlation"]
    outputs += ["bb_beams", "bb_phase", "bb_shift"] * has_bb(config)
    outputs += [
        "frb1_beams",
        "frb2_beams",
        "frb3_beams",
        "frb3_beams_offsetscale",
    ]
    if variant == "full_scale_tone":
        outputs.append("voltage")
    return {
        "first_fpga_seq_num": large_seq(config),
        "input_kind": "sky",
        "sources": [source],
        "rfi_excision_enabled": False,
        "outputs": outputs,
        "write_frames": {name: 1 for name in outputs},
    }


def input_period(out_dir, var, config):
    """One period of the input, E[t, F, P, D] for FPGA samples SEQ0 + t, t < PERIOD."""
    num_freqs = len(value(config, "frequency_channels"))
    num_dishes = value(config, "num_dishes")
    if "voltage" not in var["outputs"]:
        return np.full((PERIOD, num_freqs, 2, num_dishes), 7 + 7j)
    voltage = read_output(out_dir, "voltage")
    audit(voltage, first_seq=var["first_fpga_seq_num"])
    periods = voltage.data.reshape(-1, PERIOD, *voltage.data.shape[1:])
    require(
        np.all(periods == periods[:1]),
        f"voltage: the input does not repeat every {PERIOD} samples",
    )
    require(not np.any(int4_components(periods[0]) == -8), "voltage: -8")
    return decode_int4x2(periods[0])


def check_correlator(out_dir, e, results, failures):
    """V_ij = Σ_t E_i conj(E_j), stations i = p D + d, stored as 16x16 tiles of the lower
    triangle: [Tc, F, tile, ilo, jlo, re/im] with tile = ihi (ihi + 1) / 2 + jhi, ihi >= jhi
    (external/n2k/include/n2k/Correlator.hpp)."""
    corr = read_output(out_dir, "n2k_correlation")
    audit(corr)
    vis = corr.data.astype(np.int64)  # [Tc, F, tile, 16, 16, 2]
    num_times = corr.time_downsampling
    stations = e.reshape(PERIOD, e.shape[1], -1)  # [t, F, i]
    v = np.einsum("tfi,tfj->fij", stations, stations.conj()) * (num_times // PERIOD)
    num_blocks = stations.shape[2] // 16
    ihi, jhi = np.tril_indices(num_blocks)
    tiles = v.reshape(v.shape[0], num_blocks, 16, num_blocks, 16)[:, ihi, :, jhi]
    tiles = tiles.transpose(1, 0, 2, 3)  # [F, tile, ilo, jlo]
    predicted = np.stack([tiles.real, tiles.imag], axis=-1).round().astype(np.int64)
    wrong = vis != predicted[None]
    line = (
        f"n2k_correlation: {wrong.sum()} of {wrong.size} values differ from Σ E_i conj(E_j) "
        f"over {num_times} samples; |V| <= {np.abs(vis).max()} "
        f"(int32 limit {2 ** 31 - 1})"
    )
    if wrong.any():
        index = np.argwhere(wrong)[0]
        line += (
            f"  <- e.g. {vis[tuple(index)]} instead of {predicted[tuple(index[1:])]} "
            f"at [Tc, F, tile, ilo, jlo, c] = {index.tolist()}"
        )
        failures.append("n2k_correlation")
    results.append(line)


def components(z):
    return np.stack([z.real, z.imag])


def int4_components(data):
    """Real and imaginary parts [2, ...] of int4x2_swapped_withoffset bytes, as int8."""
    data = np.asarray(data, dtype=np.uint8)
    return np.stack([(data >> 4).astype(np.int8) - 8, (data & 0xF).astype(np.int8) - 8])


def check_upchan(out_dir, var, config, e, results, failures):
    """Ē_u(L) = (G_u/U) Σ_s W(s) exp(-2πi (u - (U-1)/2) s/U) E(L - (M-1) U/2 + s)."""
    SEQ0 = var["first_fpga_seq_num"]
    t_star = upchan_t_star(config)
    channels = list(value(config, "frequency_channels"))
    for U in config["upchan_factors"]:
        name = f"upchan_U{U}_voltage"
        out = read_output(out_dir, name)
        audit(out, upchan_factor=U, first_seq=SEQ0 + t_star)
        ebar = int4_components(used_channels(out, 1))  # [C, Tbar, Fbar, P, D]
        coarse = np.atleast_1d(out.attr("coarse_freq"))
        num_coarse = len(coarse) // U
        local = [channels.index(c) for c in coarse.reshape(num_coarse, U)[:, 0]]

        gain = np.array(
            value(config, f"run_set_upchan_U{U}_gain")["upchan_gain"], dtype=float
        )
        s = np.arange(NUM_TAPS * U)
        c = np.arange(U) - (U - 1) / 2
        kernel = (
            np.float16(gain).astype(float)[:, None]
            / U
            * upchan_window(NUM_TAPS, U)[None, :]
            * np.exp(-2j * np.pi * np.outer(c, s) / U)
        )  # [U, MU]
        # The output repeats every PERIOD / U samples
        num_out = PERIOD // U
        start = out.fpga_seq_num - (NUM_TAPS - 1) * U // 2 - SEQ0
        window = (start + U * np.arange(num_out)[:, None] + s[None, :]) % PERIOD
        predicted = np.einsum("us,jsfpd->jfupd", kernel, e[:, local][window])
        predicted = components(predicted)[:, None]  # [C, 1, j, F, U, P, D]
        measured = ebar.reshape(2, -1, num_out, num_coarse, U, *ebar.shape[3:])

        nodata = np.count_nonzero(measured == -8)
        high = predicted > 6.5 + UPCHAN_MARGIN
        low = predicted < -6.5 - UPCHAN_MARGIN
        signed = np.abs(predicted) > 0.5 + UPCHAN_MARGIN
        tie = np.abs(np.abs(predicted) % 1 - 0.5) < UPCHAN_MARGIN
        num_bad_high = np.count_nonzero(high & (measured != 7))
        num_bad_low = np.count_nonzero(low & (measured != -7))
        num_flipped = np.count_nonzero(
            signed & (np.sign(measured) != np.sign(predicted))
        )
        num_unrounded = np.count_nonzero(
            ~tie & (measured != np.clip(np.rint(predicted), -7, 7))
        )
        size = measured.size
        num_high = np.count_nonzero(np.broadcast_to(high | low, measured.shape))
        line = (
            f"{name}: {num_high / size:.1%} predicted beyond ±{6.5 + UPCHAN_MARGIN} "
            f"(max |pred| {np.abs(predicted).max():.1f}), "
            f"{np.count_nonzero(np.abs(measured) == 7) / size:.1%} at ±7; "
            f"{num_bad_high + num_bad_low} not saturated, {num_flipped} sign flips, "
            f"{nodata} -8, {num_unrounded} not clamp(round(pred)) away from ties"
        )
        if num_bad_high or num_bad_low or num_flipped or nodata or num_unrounded:
            line += "  <- wrong saturation or rounding"
            failures.append(name)
        results.append(line)


def check_bb(out_dir, var, e, results, failures):
    """J = clamp(shift(Σ_w shift(Σ_{d in w} A_bd E_d, σ), s - σ), -7, 7) per component, with
    shift(x, n) = (x + 2^(n-1)) >> n and σ = `BB_SIGMA[D]` (julia/kernels/bb.jl). Each warp
    sums a block w of 2^(σ+4) consecutive dishes, shifts its sum by σ and packs it to int16
    (saturating); the blocks are added in int32."""
    SEQ0 = var["first_fpga_seq_num"]
    bb = read_output(out_dir, "bb_beams")  # [Thi, B, F, P, T]
    audit(bb, first_seq=SEQ0)
    phase = read_output(out_dir, "bb_phase")  # [Tbb, F, P, B, D, C]
    shift = read_output(out_dir, "bb_shift").data  # [F, P, B]
    a = phase.data[0, ..., 0].astype(float) + 1j * phase.data[0, ..., 1]
    sigma = BB_SIGMA[e.shape[3]]
    require(
        np.all(shift > sigma),
        f"bb_shift: values {np.unique(shift)} must exceed {sigma}",
    )
    block = 1 << (sigma + 4)
    num_blocks = e.shape[3] // block
    partial = np.einsum(
        "fpbwd,tfpwd->wtfpb",
        a.reshape(*a.shape[:3], num_blocks, block),
        e.reshape(*e.shape[:3], num_blocks, block),
    )
    partial = components(partial).round().astype(np.int64)  # [C, w, t, F, P, B]
    partial = (partial + (1 << (sigma - 1))) >> sigma
    ju = np.clip(partial, -32768, 32767).sum(axis=1)
    s = shift[None, None].astype(np.int64) - sigma
    unclamped = (ju + (1 << (s - 1))) >> s  # [C, t, F, P, B]
    predicted = np.clip(unclamped, -7, 7)
    j = int4_components(bb.data).transpose(0, 1, 5, 3, 4, 2)  # [C, Thi, T, F, P, B]
    measured = j.reshape(2, -1, PERIOD, *j.shape[3:])
    wrong = measured != predicted[:, None]
    line = (
        f"bb_beams: {wrong.sum()} of {wrong.size} components differ from the integer "
        f"prediction; {np.count_nonzero(np.abs(unclamped) > 7) / unclamped.size:.1%} clamped "
        f"(max |unclamped| {np.abs(unclamped).max()}), max |int16 partial sum| "
        f"{np.abs(partial).max()} over {num_blocks} blocks of {block} dishes, "
        f"{np.count_nonzero(np.abs(measured) == 7) / measured.size:.1%} at ±7, "
        f"{np.count_nonzero(measured == -8)} -8"
    )
    if wrong.any():
        flipped = np.count_nonzero(
            wrong & (np.sign(measured) == -np.sign(predicted[:, None]))
        )
        line += f"  <- {flipped} with the wrong sign"
        failures.append("bb_beams")
    results.append(line)


def check_frb(out_dir, var, config, results, failures):
    SEQ0 = var["first_fpga_seq_num"]
    t_star = upchan_t_star(config)

    # FRB1 [Ttilde, Fbar, beamQ, beamP]. For weights of magnitude w on the D dishes of an M x N
    # grid and |E| <= 7 sqrt(2), I <= 98 / (4 M N) · P · (w D)²  (lib/stages/frb1IntensityBound.hpp).
    # w is `frb1_input_scale` (setFRB1Phase) or, for CHIME, the default gain of loadFeedGains,
    # since the test has no gain files.
    frb1 = read_output(out_dir, "frb1_beams")
    audit(frb1, first_seq=SEQ0 + t_star)
    intensity = used_channels(frb1, 1).astype(np.float64)
    M, N = intensity.shape[3] // 2, intensity.shape[2] // 2
    num_dishes = len(config["telescope"]["dish_inputs"])
    gains = find_stages(config, "loadFeedGains")
    if gains:
        weight = abs(complex(*gains[0]["default_gains"]))
    else:
        weight = config.get("frb1_input_scale", 1)
    bound = 98 / (4 * M * N) * 2 * (num_dishes * weight) ** 2
    peak = intensity.max()
    line = f"frb1_beams: max I {peak:.1f}, bound {bound:.1f} (ratio {peak / bound:.4f})"
    if not peak <= bound * (1 + FRB1_TOLERANCE):
        line += "  <- above the bound"
        failures.append("frb1_beams")
    results.append(line)

    # FRB2 [Ttildehi256, R, Fbar, Ttildelo256] float16; FRB3 codes (kit.frb3_codes)
    frb2 = read_output(out_dir, "frb2_beams")
    for name in ["frb2_beams", "frb3_beams", "frb3_beams_offsetscale"]:
        audit(read_output(out_dir, name), first_seq=SEQ0 + t_star)
    num_freqs = len(np.atleast_1d(frb2.attr("coarse_freq")))
    results.append(
        f"frb2_beams: finite, max {frb2.data[:, :, :num_freqs].astype(np.float64).max():.1f}"
    )
    q, offset, scale, nodata_code, qmin, qmax = frb3_codes(out_dir, num_freqs)
    nodata = np.count_nonzero(q == nodata_code)
    extreme = np.count_nonzero((q == qmin) | (q == qmax)) / q.size
    zero_scale = np.count_nonzero(np.broadcast_to(scale, q.shape) == 0) / q.size
    line = (
        f"frb3_beams: {nodata} of {q.size} values {nodata_code} (no data), {extreme:.1%} at "
        f"{qmin} or {qmax}; offset and scale finite, {zero_scale:.1%} of the samples with "
        "scale 0"
    )
    if nodata:
        line += "  <- no data without packet loss"
        failures.append("frb3_beams")
    results.append(line)


def check(out_dir, var, config):
    results = []
    failures = []
    e = input_period(out_dir, var, config)
    check_correlator(out_dir, e, results, failures)
    check_upchan(out_dir, var, config, e, results, failures)
    if "bb_beams" in var["outputs"]:
        check_bb(out_dir, var, e, results, failures)
    check_frb(out_dir, var, config, results, failures)
    require(
        not failures, f"{len(failures)} outputs wrong:\n    " + "\n    ".join(results),
    )
    return results
