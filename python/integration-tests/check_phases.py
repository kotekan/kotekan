"""T2 phases: a point-source tone must show the predicted phases and beam positions.

Input: one tone from direction n, the same on every input apart from the geometric phase
exp(-2πi ν n·r_d / c) (lib/testing/testSkySignal.hpp), in Gaussian noise of 2.8 LSB per
component. The tone is at 19.5/64 cycles per sample in every coarse channel, moved to a
fine-channel centre for the largest U (64; CHIME: U=16, 4.5/16), and away from the mirror channel
for every U. Its amplitude (0.75 LSB) keeps the upchannelizer outputs below saturation with the
production gains, which multiply a tone at a fine-channel centre by 0.81 sqrt(U); the noise
dithers the int4 rounding. The BB source beam (0.75 sqrt(D)) saturates for D > 64, which the BB
peak check tolerates. Phases and amplitudes are estimated coherently, by averaging over time
against the predicted tone.

Variants: the source is on a BB beam (`bb`, not for CHIME, which has no BB beamformer), on an
FRB2 beam (`frb2`), or between beams (`offgrid`). The variant `n2` checks the visibilities with a
stronger tone (`n2_amplitude`, 2 for the pathfinder) that still passes RFI excision; fringe
stopping is off, since Earth rotation would add up to a few degrees over one integration.
"""

import glob
import math
import os

import h5py
import numpy as np

from kit import (
    audit,
    bb_peak_line,
    C0,
    NUM_TAPS,
    direction_vector,
    decode_int4x2,
    frb2_beam_positions,
    frb2_peak_line,
    frb2_tone_power,
    geometric_phases,
    grid_beam_direction,
    has_bb,
    large_seq,
    read_output,
    require,
    rounded_gaussian,
    sky_frequency,
    upchan_t_star,
    upchan_window,
    used_channels,
    value,
)

VARIANTS = ["bb", "frb2", "offgrid", "n2"]
AMPLITUDE = 0.75
NOISE = 2.8
BB_BEAM = (3, 1)
FRB2_BEAM = (12, 5)
# The N2 tone's spectral kurtosis lies this many standard deviations inside the RFI threshold
N2_SK_MARGIN = 1.2


def variants(config):
    return [v for v in VARIANTS if v != "bb" or has_bb(config)]


def frb2_beam(config):
    """The FRB2 source beam R: FRB2_BEAM (x, y) clamped to the beam grid (CHIME: 4 x 256)."""
    bx = min(FRB2_BEAM[0], value(config, "frb2_num_beams_x") - 1)
    by = min(FRB2_BEAM[1], value(config, "frb2_num_beams_y") - 1)
    return bx + value(config, "frb2_num_beams_x") * by


def source_direction(config, variant):
    if variant == "bb":
        return grid_beam_direction(config, "bb", *BB_BEAM)
    if variant == "frb2":
        return [float(x) for x in frb2_beam_positions(config)[frb2_beam(config)]]
    return [float(np.sin(np.radians(0.5))), float(np.sin(np.radians(0.8)))]


def n2_amplitude(config):
    """The amplitude A of the N2 tone. A tone in noise of rms s has the spectral kurtosis
    SK = 1 - (A² / (A² + 2 s²))². RFI excision flags a sample block if its feed-averaged SK
    differs from 1 by more than `rfi_sk_rfimask_sigmas` times its standard deviation, about
    2 / sqrt(M N) for M samples per block and N inputs. A puts SK `N2_SK_MARGIN` standard
    deviations inside that threshold: A = 2 for the pathfinder (N = 128; 71% of the samples
    pass) and 1.47 for CHORD (N = 1024; 99.5% pass)."""
    sigmas = value(config, "rfi_sk_rfimask_sigmas") - N2_SK_MARGIN
    num_inputs = 2 * value(config, "num_dishes")
    deviation = (
        sigmas * 2 / math.sqrt(value(config, "rfi_downsampling_factor") * num_inputs)
    )
    power = 2 * rounded_gaussian(NOISE)[0] ** 2
    x = math.sqrt(deviation)  # A² / (A² + 2 s²)
    return math.sqrt(power * x / (1 - x))


def tone_cycles(config):
    """19.5/64 cycles per sample, moved down to the nearest fine-channel centre for the largest
    upchannelization factor."""
    u_max = max(config["upchan_factors"])
    f_tone = (np.floor(19.5 / 64 * u_max) + 0.5) / u_max
    return round(f_tone * value(config, "num_times"))


def variables(tel, config, variant):
    source = {
        "amplitude": n2_amplitude(config) if variant == "n2" else AMPLITUDE,
        "direction": source_direction(config, variant),
        "tone_cycles_per_frame": tone_cycles(config),
    }
    var = {
        "first_fpga_seq_num": large_seq(config),
        "input_kind": "sky",
        "sources": [source],
        "noise_rms": NOISE,
    }
    if variant == "n2":
        var["do_fringestop"] = False
        var["outputs"] = ["n2_vis"]
        var["write_frames"] = {"n2_vis": 2 * value(config, "num_frequencies")}
        return var
    outputs = ["voltage"] + ["bb_beams", "bb_beam_positions"] * has_bb(config)
    outputs += ["frb1_beams", "frb2_beams"]
    outputs += [f"upchan_U{U}_voltage" for U in config["upchan_factors"]]
    write_frames = {name: 2 for name in outputs}
    write_frames.update(
        {"voltage": 1, "bb_beam_positions": 1, "frb1_beams": 1, "frb2_beams": 1}
    )
    var["outputs"] = outputs
    var["write_frames"] = write_frames
    return var


def phase_stats(ratio):
    """Mean phase (deg, weighted by amplitude), 99th percentile of |phase| (deg), and median
    amplitude of complex ratios measured / predicted."""
    mean = np.degrees(np.angle(ratio.sum()))
    worst = np.percentile(np.abs(np.degrees(np.angle(ratio))), 99)
    return mean, worst, np.median(np.abs(ratio))


def check(out_dir, var, config):
    if "n2_vis" in var["outputs"]:
        return check_n2(out_dir, var, config)
    SEQ0 = var["first_fpga_seq_num"]
    t_star = upchan_t_star(config)
    num_times = value(config, "num_times")
    channels = np.array(value(config, "frequency_channels"))
    direction = var["sources"][0]["direction"]
    k = var["sources"][0]["tone_cycles_per_frame"]
    f_tone = k / num_times  # cycles per sample within a coarse channel
    nu = sky_frequency(config, channels, f_tone)
    phases = geometric_phases(config, direction, nu)  # [F, D]
    exists = np.abs(phases[0]) > 0
    results = []

    # Input: E[t, f, p, d] = A exp(2πi f_tone (t - t0)) phases[f, d] + noise. Average
    # E conj(model) over time, per input, on a subset of the frequencies.
    voltage = read_output(out_dir, "voltage")
    audit(voltage, first_seq=SEQ0, frequency_channels=channels)
    e = decode_int4x2(voltage.data[:2048, ::8]).astype(np.complex64)  # [T, F/8, P, D]
    t = voltage.fpga_seq_num + np.arange(e.shape[0]) - SEQ0
    unit = (
        np.exp(2j * np.pi * f_tone * t)[:, None, None, None]
        * phases[None, ::8, None, :]
    )
    ratio = (e * unit.conj()).mean(axis=0)[:, :, exists] / AMPLITUDE  # [F/8, P, D]
    mean, worst, amp = phase_stats(ratio)
    require(
        abs(mean) <= 1 and worst <= 30 and 0.95 <= amp <= 1.05,
        f"voltage: tone phase {mean:+.2f} deg (99%: {worst:.1f}), amplitude x{amp:.3f}",
    )
    results.append(
        f"voltage: tone phase {mean:+.2f} deg (99%: {worst:.1f}), amplitude x{amp:.3f}"
    )

    # Baseband beams [Thi, B, F, P, T]: the power must peak in the predicted beam
    if "bb_beams" in var["outputs"]:
        bb = read_output(out_dir, "bb_beams")
        audit(bb, first_seq=SEQ0, frequency_channels=channels)
        num_beams = bb.data.shape[1]
        positions = read_output(out_dir, "bb_beam_positions").data[:num_beams]  # [B, 2]
        results.append(bb_peak_line(bb.data, positions, config, phases, nu, AMPLITUDE))

    # Upchannelized voltages [Tbar, Fbar, P, D]. For the tone,
    #     Ē_u(L) = G_u A R_u exp(2πi f_tone (w - t0)) phases,   w = L - (M-1) U / 2,
    #     R_u = (1/U) Σ_s W(s) exp(2πi (f_tone - (u - (U-1)/2) / U) s),
    # where L is the output label and w the start of its input window. The fine channel with the
    # largest coherent amplitude must be the predicted one, and there the time-averaged ratio of
    # measured to predicted must have phase 0 and amplitude 1.
    for U in config["upchan_factors"]:
        name = f"upchan_U{U}_voltage"
        ebar_out = read_output(out_dir, name)
        audit(
            ebar_out,
            upchan_factor=U,
            first_seq=SEQ0 + t_star,
            frequency_channels=channels,
        )
        ebar = decode_int4x2(used_channels(ebar_out, 1)).astype(
            np.complex64
        )  # [Tbar, Fbar, P, D]
        coarse = np.atleast_1d(ebar_out.attr("coarse_freq"))
        fine = np.atleast_1d(ebar_out.attr("freq_upchan_index"))
        num_coarse = len(coarse) // U
        ebar = ebar.reshape(ebar.shape[0], num_coarse, U, *ebar.shape[2:])[..., exists]
        coarse = coarse.reshape(num_coarse, U)[:, 0]
        fine = fine.reshape(num_coarse, U)
        require(
            np.all(fine == np.arange(U)),
            f"{name}: freq_upchan_index is not 0..U-1 per channel",
        )

        labels = ebar_out.fpga_seq_num + U * np.arange(ebar.shape[0])
        start = labels - (NUM_TAPS - 1) * U // 2 - SEQ0
        geometry = phases[np.searchsorted(channels, coarse)][:, None, :][
            ..., exists
        ]  # [coarse, 1, D]
        unit = np.exp(2j * np.pi * f_tone * start)[:, None, None, None] * geometry[None]
        coherent = np.einsum(
            "tcupd,tcpd->cu", ebar, np.broadcast_to(unit, ebar[:, :, 0].shape).conj()
        )
        u_peak = round(f_tone * U + (U - 1) / 2)
        bad = np.abs(coherent).argmax(axis=1) != u_peak
        require(
            not bad.any(),
            f"{name}: tone in fine channel {np.unique(np.abs(coherent).argmax(axis=1)[bad]).tolist()}, expected {u_peak}",
        )

        gain = np.array(
            value(config, f"run_set_upchan_U{U}_gain")["upchan_gain"], dtype=float
        )[u_peak]
        s = np.arange(NUM_TAPS * U)
        response = (
            upchan_window(NUM_TAPS, U)
            * np.exp(2j * np.pi * (f_tone - (u_peak - (U - 1) / 2) / U) * s)
        ).sum() / U
        ratio = (ebar[:, :, u_peak] * unit.conj()).mean(axis=0) / (
            gain * AMPLITUDE * response
        )
        mean, worst, amp = phase_stats(ratio)
        require(
            abs(mean) <= 2 and worst <= 10,
            f"{name}: phase off by {mean:+.1f} deg on average, {worst:.1f} deg (99th percentile)",
        )
        require(0.8 <= amp <= 1.25, f"{name}: amplitude {amp:.2f} times the prediction")
        results.append(
            f"{name}: fine channel {u_peak}, phase {mean:+.1f} deg (99%: {worst:.1f}), amplitude x{amp:.2f}"
        )

    # FRB1 beams [Ttilde, Fbar, beamQ, beamP]: a spatial FFT over the dish grid. Beam p peaks at
    # p = 2 θ_M mod 2M with θ_M = M (n·σ_M) ν / c, σ_M the grid spacing along M (and likewise q
    # along N); this is the beam model the FRB2 weights assume (cudaCalcFRB2WeightsKernel.cu).
    # In each coarse channel, the tone is in the fine channel with the strongest beam.
    frb1 = read_output(out_dir, "frb1_beams")
    audit(frb1, first_seq=SEQ0 + t_star)
    data = used_channels(frb1, 1).astype(np.float64).sum(axis=0)  # [Fbar, Q, P]
    coarse = np.atleast_1d(frb1.attr("coarse_freq"))
    num_Q, num_P = data.shape[1:]
    M, N = num_P // 2, num_Q // 2
    tel = config["telescope"]
    n = direction_vector(direction)
    # M runs along grid x and N along y, unless the beamformer swaps them (CHIME)
    swap = num_P != 2 * value(config, "num_dishes_x")
    axis_M, axis_N = (1, 0) if swap else (0, 1)
    sep_M, sep_N = [
        float(tel[f"dish_separation_{a}_m"]) for a in ("xy"[axis_M], "xy"[axis_N])
    ]
    wrong = []
    checked = 0
    for chan in np.unique(coarse):
        slots = np.nonzero(coarse == chan)[0]
        slot = slots[data[slots].max(axis=(1, 2)).argmax()]
        nu_tone = sky_frequency(config, chan, f_tone)
        two_theta_M = 2 * M * n[axis_M] * sep_M * nu_tone / C0
        two_theta_N = 2 * N * n[axis_N] * sep_N * nu_tone / C0
        q, p = np.unravel_index(data[slot].argmax(), data[slot].shape)
        dp = (p - two_theta_M + M) % (2 * M) - M
        dq = (q - two_theta_N + N) % (2 * N) - N
        checked += 1
        if abs(dp) > 1 or abs(dq) > 1:
            mirror_p = abs((p + two_theta_M + M) % (2 * M) - M) <= 1
            mirror_q = abs((q + two_theta_N + N) % (2 * N) - N) <= 1
            wrong.append(
                (
                    int(chan),
                    int(p),
                    int(q),
                    round(two_theta_M % (2 * M), 1),
                    round(two_theta_N % (2 * N), 1),
                    mirror_p,
                    mirror_q,
                )
            )
    require(
        not wrong,
        f"frb1_beams: peak off the predicted beam in {len(wrong)} of {checked} channels; "
        f"(channel, p, q, 2θ_M, 2θ_N, p mirrored, q mirrored): {wrong[:4]}",
    )
    results.append(
        f"frb1_beams: peak at the predicted (p, q) in all {checked} channels"
    )

    # FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256]: for the `frb2` variant, the source beam
    # must hold the peak tone power
    frb2 = read_output(out_dir, "frb2_beams")
    audit(frb2, first_seq=SEQ0 + t_star)
    if direction == source_direction(config, "frb2"):
        results.append(frb2_peak_line(frb2, frb2_beam(config)))
    else:
        tone = frb2_tone_power(frb2)
        results.append(
            f"frb2_beams: peaks in beams {np.unique(tone.argmax(axis=0)).tolist()} (not checked)"
        )

    return results


def check_n2(out_dir, var, config):
    """Visibilities V_ab (upper triangle, element a = p D + d) of the tone: arg V_ab must equal
    the geometric phase difference between inputs a and b, |V_ab| the tone power."""
    source = var["sources"][0]
    f_tone = source["tone_cycles_per_frame"] / value(config, "num_times")
    amplitude = source["amplitude"]
    files = sorted(glob.glob(os.path.join(out_dir, "n2_vis.*.h5")))
    require(files, "n2_vis: no output files")
    results = []
    deltas = {"E_a conj(E_b)": [], "conj(E_a) E_b": []}
    ratios = []
    valid = []
    for filename in files:
        with h5py.File(filename, "r") as f:
            if f.attrs["n_valid_fpga_ticks"] == 0:
                continue
            vis = f["vis"][()]
            freq_id = int(f.attrs["freq_id"])
            valid.append(
                f.attrs["n_valid_fpga_ticks"] / f.attrs["frame_length_fpga_ticks"]
            )
        vis = vis[:, 0] + 1j * vis[:, 1]
        nu = sky_frequency(config, freq_id, f_tone)
        g = geometric_phases(config, source["direction"], np.array([nu]))[0]
        e = np.concatenate([g, g])  # element a = p D + d
        ia, ib = np.triu_indices(len(e))
        use = (np.abs(e[ia] * e[ib]) > 0) & (ia != ib)
        predicted = amplitude ** 2 * e[ia[use]] * e[ib[use]].conj()
        measured = vis[use]
        deltas["E_a conj(E_b)"].append(np.angle(measured * predicted.conj()))
        deltas["conj(E_a) E_b"].append(np.angle(measured * predicted))
        ratios.append(np.median(np.abs(measured)) / amplitude ** 2)
    require(ratios, "n2_vis: no visibility with valid samples")
    summary = {}
    for name, delta in deltas.items():
        delta = np.degrees(np.concatenate(delta))
        mean = np.degrees(np.angle(np.exp(1j * np.radians(delta)).mean()))
        summary[name] = (mean, np.percentile(np.abs(delta), 99))
    mean, worst = summary["E_a conj(E_b)"]
    other_mean, other_worst = summary["conj(E_a) E_b"]
    ratio = np.median(ratios)
    require(
        abs(mean) <= 2 and worst <= 10,
        f"n2_vis: phase of V_ab vs E_a conj(E_b): mean {mean:.1f} deg, 99% {worst:.1f} deg "
        f"(vs conj(E_a) E_b: mean {other_mean:.1f}, 99% {other_worst:.1f}); "
        f"{np.mean(valid):.1%} of the samples valid",
    )
    require(0.9 <= ratio <= 1.1, f"n2_vis: |V_ab| is {ratio:.2f} times the tone power")
    results.append(
        f"n2_vis: {len(ratios)} visibilities, V_ab = E_a conj(E_b) within {mean:+.2f} deg (99%: {worst:.1f}), |V_ab| x{ratio:.3f}, "
        f"A {amplitude:.3f}, {np.mean(valid):.1%} of the samples valid"
    )
    return results
