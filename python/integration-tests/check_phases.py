"""T2 phases: a point-source tone must show the predicted phases and beam positions.

Input: one noise-free tone of amplitude 5 from direction n, the same on every input apart from
the geometric phase exp(-2πi ν n·r_d / c) (lib/testing/testSkySignal.hpp). The tone is at
19.5/64 cycles per sample in every coarse channel: on a fine-channel centre for U=64, and away
from the mirror channel for every U.

Variants: the source is on a BB beam (`bb`), on an FRB2 beam (`frb2`), or between beams
(`offgrid`). The variant `n2` checks the visibilities: a noise-free tone is flagged as RFI by
the spectral kurtosis, so it uses a weaker tone (amplitude 2) in noise (2.8 LSB per component).
Fringe stopping is off; Earth rotation would add up to a few degrees over one integration.
"""

import glob
import os

import h5py
import numpy as np

from kit import (
    audit,
    C0,
    channel_width,
    direction_vector,
    decode_int4x2,
    geometric_phases,
    grid_beam_direction,
    large_seq,
    read_output,
    require,
    upchan_window,
    used_channels,
    value,
)

UPCHAN_FACTORS = [2, 4, 8, 16, 32, 64]
NUM_TAPS = 4
VARIANTS = ["bb", "frb2", "offgrid", "n2"]
AMPLITUDE = 5
BB_BEAM = (3, 1)
FRB2_BEAM = (12, 5)


def source_direction(config, variant):
    if variant == "bb":
        return grid_beam_direction(config, "bb", *BB_BEAM)
    if variant == "frb2":
        return grid_beam_direction(config, "frb2", *FRB2_BEAM)
    return [float(np.sin(np.radians(0.5))), float(np.sin(np.radians(0.8)))]


def tone_cycles(config):
    return round(19.5 / 64 * value(config, "num_times"))


def variables(tel, config, variant):
    if variant == "n2":
        return {
            "first_fpga_seq_num": large_seq(config),
            "input_kind": "sky",
            "sources": [
                {
                    "amplitude": 2,
                    "direction": source_direction(config, variant),
                    "tone_cycles_per_frame": tone_cycles(config),
                }
            ],
            "noise_rms": 2.8,
            "do_fringestop": False,
            "outputs": ["n2_vis"],
            "write_frames": {"n2_vis": 2 * value(config, "num_frequencies")},
        }
    outputs = ["voltage", "bb_beams", "bb_beam_positions", "frb1_beams", "frb2_beams"]
    outputs += [f"upchan_U{U}_voltage" for U in UPCHAN_FACTORS]
    write_frames = {name: 2 for name in outputs}
    write_frames.update(
        {"voltage": 1, "bb_beam_positions": 1, "frb1_beams": 1, "frb2_beams": 1}
    )
    return {
        "first_fpga_seq_num": large_seq(config),
        "input_kind": "sky",
        "sources": [
            {
                "amplitude": AMPLITUDE,
                "direction": source_direction(config, variant),
                "tone_cycles_per_frame": tone_cycles(config),
            }
        ],
        "outputs": outputs,
        "write_frames": write_frames,
    }


def check(out_dir, var, config):
    if "n2_vis" in var["outputs"]:
        return check_n2(out_dir, var, config)
    SEQ0 = var["first_fpga_seq_num"]
    num_times = value(config, "num_times")
    channels = np.array(value(config, "frequency_channels"))
    direction = var["sources"][0]["direction"]
    k = var["sources"][0]["tone_cycles_per_frame"]
    f_tone = k / num_times  # cycles per sample within a coarse channel
    nu = (channels + f_tone) * channel_width(config)
    phases = geometric_phases(config, direction, nu)  # [F, D]
    results = []

    # Input: E[t, f, p, d] = A exp(2πi f_tone (t - t0)) phases[f, d]
    voltage = read_output(out_dir, "voltage")
    audit(voltage, first_seq=SEQ0, frequency_channels=channels)
    e = decode_int4x2(voltage.data[:256])
    t = voltage.fpga_seq_num + np.arange(e.shape[0]) - SEQ0
    model = (
        AMPLITUDE
        * np.exp(2j * np.pi * f_tone * t)[:, None, None, None]
        * phases[None, :, None, :]
    )
    err = max(np.abs(e.real - model.real).max(), np.abs(e.imag - model.imag).max())
    require(err <= 0.5 + 1e-3, f"voltage: differs from the sky model by {err:.2f} LSB")
    results.append(f"voltage: matches the sky model within {err:.2f} LSB")

    # Baseband beams [Thi, B, F, P, T]: the beam with the largest predicted response must have
    # the largest power, wherever that prediction is unambiguous and the signal is well above the
    # int4 rounding noise (rms >= 2 LSB in the strongest beam)
    bb = read_output(out_dir, "bb_beams")
    audit(bb, first_seq=SEQ0, frequency_channels=channels)
    num_beams = bb.data.shape[1]
    positions = read_output(out_dir, "bb_beam_positions").data[:num_beams]  # [B, 2]
    power = (np.abs(decode_int4x2(bb.data)) ** 2).mean(axis=(0, 3, 4))  # [B, F]
    predicted = np.stack(
        [
            np.abs((phases * geometric_phases(config, pos, nu).conj()).sum(axis=1)) ** 2
            for pos in positions
        ]
    )  # [B, F]
    ranked = np.sort(predicted, axis=0)
    clear = (ranked[-1] >= 1.25 * ranked[-2]) & (power.max(axis=0) >= 2 ** 2)
    require(clear.any(), "bb_beams: no frequency with an unambiguous predicted peak")
    wrong = clear & (power.argmax(axis=0) != predicted.argmax(axis=0))
    require(
        not wrong.any(),
        f"bb_beams: peak in beam {power.argmax(axis=0)[wrong][:8].tolist()} instead of "
        f"{predicted.argmax(axis=0)[wrong][:8].tolist()} for {wrong.sum()} of {clear.sum()} frequencies",
    )
    results.append(
        f"bb_beams: peak in the predicted beam for all {clear.sum()} unambiguous frequencies"
    )

    # Upchannelized voltages [Tbar, Fbar, P, D]: the tone is in the predicted fine channel, with
    # the predicted phase. A sample labelled L comes from the input window starting at
    # L - (M-1) U / 2.
    for U in UPCHAN_FACTORS:
        name = f"upchan_U{U}_voltage"
        ebar_out = read_output(out_dir, name)
        audit(
            ebar_out,
            upchan_factor=U,
            first_seq=SEQ0
            + (NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2,
            frequency_channels=channels,
        )
        ebar = decode_int4x2(used_channels(ebar_out, 1))  # [Tbar, Fbar, P, D]
        coarse = np.atleast_1d(ebar_out.attr("coarse_freq"))
        fine = np.atleast_1d(ebar_out.attr("freq_upchan_index"))
        num_coarse = len(coarse) // U
        ebar = ebar.reshape(ebar.shape[0], num_coarse, U, *ebar.shape[2:])
        coarse = coarse.reshape(num_coarse, U)[:, 0]
        fine = fine.reshape(num_coarse, U)
        require(
            np.all(fine == np.arange(U)),
            f"{name}: freq_upchan_index is not 0..U-1 per channel",
        )
        u_peak = round(f_tone * U + (U - 1) / 2)
        power = (np.abs(ebar) ** 2).sum(axis=(0, 3, 4))  # [coarse, U]
        bad = power.argmax(axis=1) != u_peak
        require(
            not bad.any(),
            f"{name}: tone in fine channel {np.unique(power.argmax(axis=1)[bad]).tolist()}, expected {u_peak}",
        )

        labels = ebar_out.fpga_seq_num + U * np.arange(ebar.shape[0])
        start = labels - (NUM_TAPS - 1) * U // 2 - SEQ0
        s = np.arange(NUM_TAPS * U)
        response = (
            upchan_window(NUM_TAPS, U)
            * np.exp(2j * np.pi * (f_tone - (u_peak - (U - 1) / 2) / U) * s)
        ).sum() / U
        f_index = np.searchsorted(channels, coarse)
        expected = (
            AMPLITUDE
            * response
            * np.exp(2j * np.pi * f_tone * start)[:, None, None, None]
            * phases[f_index][None, :, None, :]
        )  # [Tbar, coarse, P, D]
        measured = ebar[:, :, u_peak]
        expected = np.broadcast_to(expected, measured.shape)
        use = np.abs(expected) >= 1
        delta = np.degrees(np.angle(measured[use] * expected[use].conj()))
        mean = np.degrees(np.angle(np.exp(1j * np.radians(delta)).mean()))
        worst = np.percentile(np.abs(delta), 99)
        ratio = np.median(np.abs(measured[use]) / np.abs(expected[use]))
        require(
            abs(mean) <= 2 and worst <= 15,
            f"{name}: phase off by {mean:.1f} deg on average, {worst:.1f} deg (99th percentile)",
        )
        require(
            0.8 <= ratio <= 1.25, f"{name}: amplitude {ratio:.2f} times the prediction"
        )
        results.append(
            f"{name}: fine channel {u_peak}, phase {mean:+.1f} deg (99%: {worst:.1f}), amplitude x{ratio:.2f}"
        )

    # FRB1 beams [Ttilde, Fbar, beamQ, beamP]: a spatial FFT over the dish grid. Beam p peaks at
    # p = 2 θ_M mod 2M with θ_M = M (n·σ_M) ν / c, σ_M the grid spacing along M (and likewise q
    # along N); this is the beam model the FRB2 weights assume (cudaCalcFRB2WeightsKernel.cu).
    frb1 = read_output(out_dir, "frb1_beams")
    audit(
        frb1,
        first_seq=SEQ0
        + (NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2,
    )
    data = used_channels(frb1, 1).astype(np.float64).sum(axis=0)  # [Fbar, Q, P]
    coarse = np.atleast_1d(frb1.attr("coarse_freq"))
    num_Q, num_P = data.shape[1:]
    M, N = num_P // 2, num_Q // 2
    tel = config["telescope"]
    n = direction_vector(direction)
    wrong = []
    checked = 0
    for chan in np.unique(coarse):
        slots = np.nonzero(coarse == chan)[0]
        slot = slots[data[slots].sum(axis=(1, 2)).argmax()]
        nu_tone = (chan + f_tone) * channel_width(config)
        two_theta_M = 2 * M * n[0] * float(tel["dish_separation_x_m"]) * nu_tone / C0
        two_theta_N = 2 * N * n[1] * float(tel["dish_separation_y_m"]) * nu_tone / C0
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
    # must hold the maximum in every fine channel that carries the tone. Grating lobes (the dish
    # spacing is many wavelengths) can give other beams the same power; such ties are allowed
    # within the float16 resolution.
    frb2 = read_output(out_dir, "frb2_beams")
    audit(
        frb2,
        first_seq=SEQ0
        + (NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2,
    )
    power = used_channels(frb2, 2).astype(np.float64).sum(axis=(0, 3))  # [R, Fbar]
    strong = power.max(axis=0) >= 0.1 * power.max()
    if direction == grid_beam_direction(config, "frb2", *FRB2_BEAM):
        expected_beam = FRB2_BEAM[0] + value(config, "frb2_num_beams_x") * FRB2_BEAM[1]
        share = power[expected_beam, strong] / power[:, strong].max(axis=0)
        bad = share < 0.995
        if bad.any():
            peaks = power[:, strong].argmax(axis=0)[bad]
            require(
                False,
                f"frb2_beams: beam {expected_beam} has only {share[bad].min():.3f} of the peak "
                f"power (peak in beams {np.unique(peaks).tolist()}) in {bad.sum()} of "
                f"{strong.sum()} channels",
            )
        ties = (power[:, strong] >= 0.995 * power[:, strong].max(axis=0)).sum(
            axis=0
        ) > 1
        results.append(
            f"frb2_beams: beam {expected_beam} holds the peak in all {strong.sum()} channels "
            f"with signal ({ties.sum()} ties with grating lobes)"
        )
    else:
        peaks = power.argmax(axis=0)[strong]
        results.append(
            f"frb2_beams: peaks in beams {np.unique(peaks).tolist()} (not checked)"
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
    for filename in files:
        with h5py.File(filename, "r") as f:
            if f.attrs["n_valid_fpga_ticks"] == 0:
                continue
            vis = f["vis"][()]
            freq_id = int(f.attrs["freq_id"])
        vis = vis[:, 0] + 1j * vis[:, 1]
        nu = (freq_id + f_tone) * channel_width(config)
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
        f"(vs conj(E_a) E_b: mean {other_mean:.1f}, 99% {other_worst:.1f})",
    )
    require(0.9 <= ratio <= 1.1, f"n2_vis: |V_ab| is {ratio:.2f} times the tone power")
    results.append(
        f"n2_vis: {len(ratios)} visibilities, V_ab = E_a conj(E_b) within {mean:+.2f} deg (99%: {worst:.1f}), |V_ab| x{ratio:.3f}"
    )
    return results
