"""T6 dish locations: every subsystem must place each dish where `telescope.dish_inputs` says.

Variants:
- `onehot`: the inputs `ONEHOT` carry a tone in turn, each in its own window of T/8 samples in
  the first N2 accumulation bin. N2: only the autocorrelation of each of these elements is
  non-zero, and it equals Σ|E|² / n_valid; the written dish grid indices and feed positions
  match the config. BB: in each window, Σ_t J_b conj(E_d) / Σ_t |E_d|² has the calcBBPhase phase
  2π ν n_b·r_d / c for every beam, frequency and window, and the other polarization is zero.
  Fringe stopping is off: N2Accumulate truncates fringe-stopped values to integers.
- `pairs`: each pair of dishes in `PAIRS` carries its own tone from the zenith, on a U=64
  fine-channel centre. In the fine channel of the tone, FRB1 beam (p, q) must show the fringe
  (1 + cos π (p Δm / M + q Δn / N)) / 2 of the pair's grid separation (Δm, Δn).
- `half`: the northern half of the dishes is Missing (and masked bad); T2's `frb2` variant
  (check_phases.py) must still pass, with predictions from the present dishes only.

`onehot` and `pairs` have no noise and first-stage RFI excision off, since a tone has spectral
kurtosis 0.

See docs/integration-tests/T6-dish-locations.md.
"""

import glob
import os

import h5py
import numpy as np

import check_phases
from kit import (
    audit,
    C0,
    sky_frequency,
    decode_int4x2,
    direction_vector,
    dish_positions,
    frame_seq_nums,
    large_seq,
    read_output,
    require,
    used_channels,
    value,
)

VARIANTS = ["onehot", "pairs", "half"]

# (dish, polarization): grid corners (0, 0), (6, 0), (6, 8) and the single dish at (0, 9), and
# (3, 4), (5, 2), (1, 6), (4, 7) inside
ONEHOT = [(0, 0), (6, 1), (63, 0), (62, 1), (31, 0), (19, 1), (43, 1), (53, 0)]
ONEHOT_AMPLITUDE = 6.1
ONEHOT_CYCLES = 104  # 13 cycles per window
# bb_gain 0.6 keeps |J| <= 4 and avoids a gain near 1/2 or 1, where int4 rounding snaps a
# small rotation of a lattice point back onto the lattice (bias up to 5°)
ONEHOT_BB_GAIN = 0.6
BB_TOLERANCE = 5  # degrees

# Pairs of dishes with grid separations (1, 0), (0, 1), (2, 3), (2, -3), (6, 0) and (0, 9)
PAIRS = [(23, 24), (12, 19), (35, 58), (50, 31), (14, 20), (0, 63)]
PAIR_AMPLITUDE = 5.2
# The upchannelizers that resolve the pairs' tones (1/8 cycles per sample apart)
PAIR_UPCHAN_FACTORS = [16, 32, 64]
PATTERN_TOLERANCE = 0.01

UPCHAN_FACTORS = [2, 4, 8, 16, 32, 64]


def pair_cycles(config, j):
    """Tone of pair j: (j - 2.5)/8 + 1/128 cycles per sample, a U=64 fine-channel centre."""
    return round(((j - 2.5) / 8 + 1 / 128) * value(config, "num_times"))


def n2_bin_start(config, seq0):
    """The first N2 accumulation bin boundary at or after `seq0`."""
    length = config["run_N2_accumulate"]["num_subintegrations_per_bin"] * value(
        config, "n2k_samples_per_data_set"
    )
    return -(-seq0 // length) * length


def onehot_windows(config, seq0):
    start = n2_bin_start(config, seq0)
    length = value(config, "num_times") // len(ONEHOT)
    return [(start + i * length, start + (i + 1) * length) for i in range(len(ONEHOT))]


def variables(tel, config, variant):
    seq0 = large_seq(config)
    var = {
        "first_fpga_seq_num": seq0,
        "input_kind": "sky",
        "rfi_excision_enabled": False,
    }
    if variant == "onehot":
        windows = onehot_windows(config, seq0)
        num_frames = (windows[-1][1] - 1 - seq0) // value(config, "num_times") + 1
        var["sources"] = [
            {
                "amplitude": ONEHOT_AMPLITUDE,
                "direction": [0, 0],
                "tone_cycles_per_frame": ONEHOT_CYCLES,
                "active_inputs": [[dish, pol]],
                "window": list(window),
            }
            for (dish, pol), window in zip(ONEHOT, windows)
        ]
        var["bb_gain"] = ONEHOT_BB_GAIN
        var["do_fringestop"] = False
        var["outputs"] = ["voltage", "bb_beams", "bb_beam_positions", "n2_vis"]
        var["write_frames"] = {
            "voltage": num_frames,
            "bb_beams": num_frames,
            "bb_beam_positions": 1,
            "n2_vis": value(config, "num_frequencies"),
        }
    elif variant == "pairs":
        var["sources"] = [
            {
                "amplitude": PAIR_AMPLITUDE,
                "direction": [0, 0],
                "tone_cycles_per_frame": pair_cycles(config, j),
                "active_inputs": [[d, p] for d in pair for p in range(2)],
            }
            for j, pair in enumerate(PAIRS)
        ]
        # Unit gains keep the upchannelized tones unsaturated
        var["upchan_gains"] = {str(U): [1.0] * U for U in UPCHAN_FACTORS}
        var["outputs"] = ["frb1_beams"]
        var["write_frames"] = {"frb1_beams": 1}
    else:
        var = check_phases.variables(tel, config, "frb2")
        var["half_filled"] = True
        var["bf_mode"] = "auto"
    return var


def config_dish_attrs(config):
    """The dish grid indices [D, 2] (-1 unless an ArrayDish) and feed positions [D, 3] that the
    config implies, and which dishes are listed."""
    tel = config["telescope"]
    num_dishes = value(config, "num_dishes")
    grid = np.full((num_dishes, 2), -1)
    pos = np.zeros((num_dishes, 3))
    listed = np.zeros(num_dishes, dtype=bool)
    for dish in tel["dish_inputs"]:
        d = dish["dish_idx"]
        if dish["type"] == "ArrayDish":
            grid[d] = [dish["grid_x_idx"], dish["grid_y_idx"]]
        disp = dish["feed_pos_disp_m"]
        pos[d] = [
            float(tel["dish_separation_x_m"]) * dish["grid_x_idx"] + disp[0],
            float(tel["dish_separation_y_m"]) * dish["grid_y_idx"] + disp[1],
            disp[2],
        ]
        listed[d] = True
    return grid, pos, listed


def check_dish_attrs(name, attrs, config):
    """The `dish_grid_indices` and `feed_positions_m` attributes written with `name`."""
    grid, pos, listed = config_dish_attrs(config)
    written_grid = np.asarray(attrs["dish_grid_indices"])
    written_pos = np.asarray(attrs["feed_positions_m"])
    require(
        written_grid.shape == grid.shape and written_pos.shape == pos.shape,
        f"{name}: dish attributes have shapes {written_grid.shape}, {written_pos.shape}",
    )
    bad = np.nonzero(np.any(written_grid != grid, axis=1))[0]
    require(
        not bad.size,
        f"{name}: dish_grid_indices of dishes {bad[:8].tolist()} are "
        f"{written_grid[bad[:8]].tolist()}, expected {grid[bad[:8]].tolist()}",
    )
    bad = np.nonzero(np.any(np.abs(written_pos - pos) > 1e-9, axis=1) & listed)[0]
    require(
        not bad.size,
        f"{name}: feed_positions_m of dishes {bad[:8].tolist()} are "
        f"{written_pos[bad[:8]].tolist()}, expected {pos[bad[:8]].tolist()}",
    )
    missing = int((grid[:, 0] < 0).sum())
    return f"{name}: dish_grid_indices and feed_positions_m match the config ({missing} dishes missing)"


def check_onehot(out_dir, var, config):
    SEQ0 = var["first_fpga_seq_num"]
    num_times = value(config, "num_times")
    num_dishes = value(config, "num_dishes")
    channels = np.array(value(config, "frequency_channels"))
    windows = [tuple(source["window"]) for source in var["sources"]]
    results = []

    # Input [T, F, P, D]: each listed input is non-zero only in its window, all others are zero
    voltage = read_output(out_dir, "voltage")
    audit(voltage, first_seq=SEQ0, frequency_channels=channels)
    require(
        windows[-1][1] <= SEQ0 + voltage.data.shape[0],
        "voltage: does not cover the windows",
    )
    nonzero = (voltage.data != 0x88).any(axis=1)  # [T, P, D]
    expected = np.zeros_like(nonzero)
    for (dish, pol), (begin, end) in zip(ONEHOT, windows):
        expected[begin - SEQ0 : end - SEQ0, pol, dish] = True
    require(
        np.array_equal(nonzero, expected),
        "voltage: non-zero outside the one-hot windows and inputs",
    )
    results.append(
        f"voltage: only input (d, p) is non-zero in its window, for {len(ONEHOT)} windows"
    )

    def window_voltage(i):
        (dish, pol), (begin, end) = ONEHOT[i], windows[i]
        return decode_int4x2(voltage.data[begin - SEQ0 : end - SEQ0, :, pol, dish])

    # N2 visibilities, upper triangle of elements a = p D + d: V_ab = Σ E_a conj(E_b) / n_valid.
    # The windows do not overlap, so only the autocorrelations of the one-hot inputs are non-zero.
    files = sorted(glob.glob(os.path.join(out_dir, "n2_vis.*.h5")))
    require(files, "n2_vis: no output files")
    num_elements = 2 * num_dishes
    ia, ib = np.triu_indices(num_elements)
    auto = {a: np.nonzero((ia == a) & (ib == a))[0][0] for a in range(num_elements)}
    power = [(np.abs(window_voltage(i)) ** 2).sum(axis=0) for i in range(len(ONEHOT))]
    worst = 0.0
    num_checked = 0
    for filename in files:
        with h5py.File(filename, "r") as f:
            vis = f["vis"][()]
            n_valid = int(f.attrs["n_valid_fpga_ticks"])
            freq_id = int(f.attrs["freq_id"])
            start = int(f.attrs["fpga_start_tick"])
            length = int(f.attrs["frame_length_fpga_ticks"])
            if filename == files[0]:
                results.append(check_dish_attrs("n2_vis", f.attrs, config))
        require(n_valid > 0, f"{filename}: no valid samples")
        vis = vis[:, 0] + 1j * vis[:, 1]
        f_index = np.searchsorted(channels, freq_id)
        require(
            channels[f_index] == freq_id, f"n2_vis: freq_id {freq_id} not configured"
        )
        predicted = np.zeros(len(vis))
        for i, ((dish, pol), (begin, end)) in enumerate(zip(ONEHOT, windows)):
            if begin >= start and end <= start + length:
                predicted[auto[pol * num_dishes + dish]] = power[i][f_index] / n_valid
            else:
                require(
                    end <= start or begin >= start + length,
                    f"n2_vis: window {i} straddles the bin at {start}",
                )
        wrong = (vis != 0) != (predicted != 0)
        if wrong.any():
            at = np.nonzero(wrong)[0][:4]
            require(
                False,
                f"n2_vis freq_id {freq_id}: products (a, b) {list(zip(ia[at].tolist(), ib[at].tolist()))} "
                f"are {vis[at].tolist()}, expected {predicted[at].tolist()}; expected non-zero at "
                f"{[(int(ia[k]), int(ib[k])) for k in np.nonzero(predicted)[0]]}",
            )
        used = predicted != 0
        worst = max(worst, np.abs(vis[used] / predicted[used] - 1).max())
        num_checked += used.any()
    require(num_checked, "n2_vis: no accumulation holds the windows")
    require(
        worst <= 1e-5, f"n2_vis: autocorrelations differ from Σ|E|²/n by {worst:.2e}"
    )
    results.append(
        f"n2_vis: in {num_checked} of {len(files)} visibilities only the autocorrelations of "
        f"elements {sorted(pol * num_dishes + dish for dish, pol in ONEHOT)} are non-zero, "
        f"equal to Σ|E|²/n within {worst:.1e}; all other {len(ia) - len(ONEHOT)} products are 0"
    )

    # Baseband beams [Thi, B, F, P, T], J_b = bb_gain Σ_d A_bd E_d with
    # A_bd ∝ exp(2πi ν n_b·r_d / c), ν the coarse channel centre (calcBBPhase).
    bb = read_output(out_dir, "bb_beams")
    audit(bb, first_seq=SEQ0, frequency_channels=channels)
    seqs = frame_seq_nums(bb)
    num_beams = bb.data.shape[1]
    beams = read_output(out_dir, "bb_beam_positions").data[:num_beams]  # [B, 2]
    n = np.array([direction_vector(b) for b in beams])  # [B, 3]
    pos, exists = dish_positions(config)
    nu = sky_frequency(config, channels)
    gain = value(config, "bb_gain") * 127.5 / 128
    residuals = []
    amplitudes = []
    for i, ((dish, pol), (begin, end)) in enumerate(zip(ONEHOT, windows)):
        frame = np.searchsorted(seqs, begin, side="right") - 1
        t0 = begin - seqs[frame]
        require(
            frame >= 0 and t0 + end - begin <= num_times,
            f"bb_beams: window {i} is not inside one frame",
        )
        j = decode_int4x2(
            bb.data[frame, :, :, :, t0 : t0 + end - begin]
        )  # [B, F, P, L]
        require(
            not np.any(j[:, :, 1 - pol]),
            f"bb_beams: polarization {1 - pol} responds to input (d, p) = ({dish}, {pol})",
        )
        e = window_voltage(i)  # [L, F]
        ratio = np.einsum("bfl,lf->bf", j[:, :, pol], e.conj()) / (np.abs(e) ** 2).sum(
            axis=0
        )
        phase = 2 * np.pi * np.outer(n @ pos[dish], nu) / C0  # [B, F]
        residuals.append(np.degrees(np.angle(ratio * np.exp(-1j * phase))))
        amplitudes.append(np.abs(ratio) / gain)
    residuals = np.abs(np.array(residuals))  # [window, B, F]
    amplitudes = np.array(amplitudes)
    worst = residuals.max()
    i, b, f = np.unravel_index(residuals.argmax(), residuals.shape)
    line = (
        f"bb_beams: arg(J_b / E_d) - 2π ν n_b·r_d / c within {worst:.2f} deg "
        f"(99%: {np.percentile(residuals, 99):.2f}) over {len(ONEHOT)} inputs, {num_beams} "
        f"beams and {len(channels)} frequencies; |J_b / E_d| / bb_gain "
        f"{amplitudes.min():.3f} to {amplitudes.max():.3f}"
    )
    require(
        worst <= BB_TOLERANCE and 0.9 <= amplitudes.min() and amplitudes.max() <= 1.1,
        f"{line}; worst at input {ONEHOT[i]}, beam {b}, channel {channels[f]}",
    )
    results.append(line)
    return results


def fringe(config, pair, num_P, num_Q):
    """The FRB1 intensity (1 + cos π (p Δm/M + q Δn/N)) / 2 [Q, P] of a pair of dishes with
    equal signals, and the level halfway between its peak and the next lower value."""
    grid, _, _ = config_dish_attrs(config)
    if config.get("frb1_swap_MN", False):
        grid = grid[:, ::-1]
    dm, dn = grid[pair[0]] - grid[pair[1]]
    M, N = num_P // 2, num_Q // 2
    q, p = np.meshgrid(np.arange(num_Q), np.arange(num_P), indexing="ij")
    pattern = (1 + np.cos(np.pi * (p * dm / M + q * dn / N))) / 2
    levels = np.unique(pattern.round(9))
    return pattern, (1 + levels[-2]) / 2, (int(dm), int(dn))


def check_pairs(out_dir, var, config):
    SEQ0 = var["first_fpga_seq_num"]
    t_star = (
        (check_phases.NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2
    )
    num_times = value(config, "num_times")
    channels = np.array(value(config, "frequency_channels"))
    frb1 = read_output(out_dir, "frb1_beams")
    audit(frb1, first_seq=SEQ0 + t_star, frequency_channels=channels)
    data = used_channels(frb1, 1).astype(np.float64).sum(axis=0)  # [Fbar, Q, P]
    num = data.shape[0]
    coarse = np.atleast_1d(frb1.attr("coarse_freq"))[:num]
    factor = np.atleast_1d(frb1.attr("freq_upchan_factor"))[:num]
    index = np.atleast_1d(frb1.attr("freq_upchan_index"))[:num]
    num_Q, num_P = data.shape[1:]
    results = []
    failures = []
    for j, (pair, source) in enumerate(zip(PAIRS, var["sources"])):
        pattern, threshold, sep = fringe(config, pair, num_P, num_Q)
        f_tone = source["tone_cycles_per_frame"] / num_times
        peaks = pattern >= threshold
        worst = 0.0
        wrong_peaks = []
        checked = 0
        for U in PAIR_UPCHAN_FACTORS:
            u = int(np.floor(f_tone * U + (U - 1) / 2 + 0.5))
            for slot in np.nonzero((factor == U) & (index == u))[0]:
                intensity = data[slot] / data[slot].max()
                worst = max(worst, np.abs(intensity - pattern).max())
                if not np.array_equal(intensity >= threshold, peaks):
                    wrong_peaks.append((int(coarse[slot]), U))
                checked += 1
        line = (
            f"frb1_beams pair {pair} (Δm, Δn) = {sep}: {checked} channels (U = "
            f"{PAIR_UPCHAN_FACTORS}), {peaks.sum()} peak bins as predicted"
            f"{'' if not wrong_peaks else ' except in ' + str(wrong_peaks[:4])}, "
            f"max |I / I_max - fringe| {worst:.4f}"
        )
        if not checked or wrong_peaks or worst > PATTERN_TOLERANCE:
            line += "  <- wrong fringe"
            failures.append(pair)
        results.append(line)
    require(
        not failures,
        f"{len(failures)} of {len(PAIRS)} pairs wrong:\n    " + "\n    ".join(results),
    )
    return results


def check_half(out_dir, var, config):
    _, exists = dish_positions(config)
    require(
        exists.sum() == value(config, "num_dishes") // 2,
        f"config: {exists.sum()} dishes present",
    )
    voltage = read_output(out_dir, "voltage")
    require(
        np.all(voltage.data[..., ~exists] == 0x88), "voltage: Missing dishes non-zero"
    )
    positions = read_output(out_dir, "bb_beam_positions")
    results = [
        f"{exists.sum()} of {len(exists)} dishes present; Missing dishes carry no input",
        check_dish_attrs("bb_beam_positions", positions.attrs, config),
    ]
    return results + check_phases.check(out_dir, var, config)


def check(out_dir, var, config):
    if "n2_vis" in var["outputs"]:
        return check_onehot(out_dir, var, config)
    if var.get("half_filled"):
        return check_half(out_dir, var, config)
    return check_pairs(out_dir, var, config)
