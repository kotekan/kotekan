"""T7 beam pointings: the beams setBBBeams and setFRBBeams are configured with must be the beams
the BB and FRB2 beamformers form, for every phase and weights lifetime.

Input: T2's weak tone (check_phases.py) in Gaussian noise, from one direction that is a BB beam
and an FRB2 beam at the same time. The BB phase and FRB1 weights live for 3 frames
(`bf_mask_lifetime_frames`), the FRB2 weights for one FRB2 frame
(`frb2_weights_lifetime_in_samples`), so both are recomputed several times during the run.

Variants:
- `grid`: both setters in `grid_degrees` mode. BB is a single column of beams (num_x = 1, as for
  CHIME) whose x is the midpoint of an asymmetric range and whose y spacing is that of FRB2;
  FRB2 is the production grid. The source is FRB2 beam (12, 5) and the BB beam in the middle of
  the column, which have the same position.
- `manual`: both setters in `manual` mode, with beams at irregular positions and random 64-bit
  ids, in shuffled order. The source is off the production grids.

CHIME's production FRB2 beams are `manual`: there `grid` keeps them, the source is T2's FRB2
source beam, and the BB column (num_x 1) is centred on it with the FRB2 y spacing at the source;
`manual` jitters the FRB2 beams.

Checks: the written positions and ids equal the configured ones, one frame per lifetime with
`fpga_seq_num` seq0 + k L (BB) and seq0 + T* + k L (FRB2); the BB power peaks in the predicted
beam and the FRB2 source beam holds the peak (T2's criteria); every BB beam is non-zero; the tone
in the source beam has the same coherent amplitude and phase (BB) or power (FRB2) in every window,
in particular across lifetime boundaries.

See docs/integration-tests/T7-beam-pointings.md.
"""

import math

import numpy as np

import check_phases
from kit import (
    audit,
    bb_peak_line,
    find_stages,
    frb2_beam_positions,
    has_bb,
    sky_frequency,
    decode_int4x2,
    frame_seq_nums,
    frb2_peak_line,
    geometric_phases,
    grid_point,
    large_seq,
    read_output,
    require,
    upchan_t_star,
    used_channels,
    value,
)

VARIANTS = ["grid", "manual"]
BF_MASK_LIFETIME_FRAMES = 3
# Frames written: 10 BB frames span 4 BB lifetimes; 3 FRB2 frames span 3 FRB2 lifetimes and 6
# FRB1 lifetimes
BB_FRAMES = 10
FRB2_FRAMES = 3
# The FRB2 grid beam that the `grid` source sits on
FRB2_GRID_BEAM = (12, 5)
# Half the x range of the single BB column, in degrees: the midpoint is the FRB2 beam's x
BB_COLUMN_HALF_WIDTH = 0.5
# `manual`: source direction (degrees), and its index in the BB and FRB2 beam lists
MANUAL_SOURCE_DEGREES = (0.5, 0.8)
BB_MANUAL_SOURCE = 11
FRB2_MANUAL_SOURCE = 173
MANUAL_SEED = 7
# Window lengths of the continuity check: BB samples and FRB2 (Ttilde) samples
BB_WINDOW = 512
FRB2_WINDOW = 16
# A window may deviate from the median by this many robust standard deviations
CONTINUITY_SIGMAS = 6


def sin_deg(angle):
    """sin of an angle in degrees, as setBBBeams and setFRBBeams compute it."""
    return math.sin(angle * (math.pi / 180.0))


def manual_layouts(num_bb, frb2_angles):
    """Beam lists for the `manual` variant: BB beams at random positions at least 0.4° apart,
    FRB2 beams at the production positions `frb2_angles` (degrees) jittered by up to 0.1° per
    axis; the source replaces the nearest FRB2 beam. Both lists are shuffled and have random
    64-bit ids."""
    rng = np.random.default_rng(MANUAL_SEED)
    source = np.array(MANUAL_SOURCE_DEGREES)
    bb = []
    while len(bb) < num_bb - 1:
        angle = rng.uniform(-2.5, 2.5, 2)
        if all(np.hypot(*(angle - other)) >= 0.4 for other in bb + [source]):
            bb.append(angle)
    bb.insert(BB_MANUAL_SOURCE, source)

    frb2 = [np.array(angle) + rng.uniform(-0.1, 0.1, 2) for angle in frb2_angles]
    del frb2[int(np.argmin([np.hypot(*(a - source)) for a in frb2]))]
    frb2 = [frb2[i] for i in rng.permutation(len(frb2))]
    frb2.insert(FRB2_MANUAL_SOURCE, source)

    def beams(angles):
        ids = rng.integers(2 ** 64, size=len(angles), dtype=np.uint64)
        require(len(set(ids)) == len(ids), "manual beam ids are not unique")
        return [
            {"id": int(i), "x_dir_grid": sin_deg(a[0]), "y_dir_grid": sin_deg(a[1])}
            for i, a in zip(ids, angles)
        ]

    return beams(bb), beams(frb2)


def frb2_grid(config):
    """(num_x, num_y), (x_min, x_max, y_min, y_max) of the production FRB2 grid."""
    block = "run_set_frb2_beams"
    return (
        (value(config, "num_x", block), value(config, "num_y", block)),
        tuple(value(config, k, block) for k in ["x_min", "x_max", "y_min", "y_max"]),
    )


def frb2_angles(config):
    """The production FRB2 beam positions in degrees, beam by beam."""
    if find_stages(config, "setFRBBeams")[0]["mode"] == "manual":
        return [
            [math.degrees(math.asin(c)) for c in pos]
            for pos in frb2_beam_positions(config)
        ]
    (nx, ny), (x_min, x_max, y_min, y_max) = frb2_grid(config)
    return [
        [grid_point(x_min, x_max, nx, bx), grid_point(y_min, y_max, ny, by)]
        for by in range(ny)
        for bx in range(nx)
    ]


def variables(tel, config, variant):
    manual_frb2 = find_stages(config, "setFRBBeams")[0]["mode"] == "manual"
    if variant == "grid" and manual_frb2:
        # The source is T2's FRB2 beam. The BB column has the FRB2 y spacing at the source and is
        # centred on it.
        positions = frb2_beam_positions(config)
        source = check_phases.frb2_beam(config)
        direction = [float(c) for c in positions[source]]
        layouts = {}
        if has_bb(config):
            x, y = [math.degrees(math.asin(c)) for c in direction]
            nx = value(config, "frb2_num_beams_x")
            spacing = math.degrees(math.asin(positions[source + nx][1])) - y
            num_y = value(config, "bb_num_beams")
            bb_y_min = y - (num_y - 1) // 2 * spacing
            layouts["bb_beam_layout"] = {
                "fixed_mode": "grid_degrees",
                "num_x": 1,
                "num_y": num_y,
                "x_min": x - BB_COLUMN_HALF_WIDTH,
                "x_max": x + BB_COLUMN_HALF_WIDTH,
                "y_min": bb_y_min,
                "y_max": bb_y_min + (num_y - 1) * spacing,
            }
    elif variant == "grid":
        (nx, ny), (x_min, x_max, y_min, y_max) = frb2_grid(config)
        x = grid_point(x_min, x_max, nx, FRB2_GRID_BEAM[0])
        y = grid_point(y_min, y_max, ny, FRB2_GRID_BEAM[1])
        # The column has the FRB2 y spacing and is centred on the source, so that grating lobes
        # are far from it at the low frequencies. The grids have dyadic spacings, so both
        # setters compute y exactly.
        num_y = value(config, "bb_num_beams")
        bb_source = (num_y - 1) // 2
        spacing = (y_max - y_min) / (ny - 1)
        bb_y_min = y - bb_source * spacing
        bb_layout = {
            "fixed_mode": "grid_degrees",
            "num_x": 1,
            "num_y": num_y,
            "x_min": x - BB_COLUMN_HALF_WIDTH,
            "x_max": x + BB_COLUMN_HALF_WIDTH,
            "y_min": bb_y_min,
            "y_max": bb_y_min + (num_y - 1) * spacing,
        }
        require(
            grid_point(bb_layout["x_min"], bb_layout["x_max"], 1, 0) == x
            and grid_point(bb_layout["y_min"], bb_layout["y_max"], num_y, bb_source)
            == y,
            "the BB grid beam is not at the FRB2 grid beam",
        )
        direction = [sin_deg(x), sin_deg(y)]
        layouts = {"bb_beam_layout": bb_layout}
    else:
        num_bb = value(config, "bb_num_beams") if has_bb(config) else 0
        bb, frb2 = manual_layouts(num_bb, frb2_angles(config))
        direction = [
            frb2[FRB2_MANUAL_SOURCE]["x_dir_grid"],
            frb2[FRB2_MANUAL_SOURCE]["y_dir_grid"],
        ]
        layouts = {"frb2_beam_layout": {"mode": "manual", "beams": frb2}}
        if has_bb(config):
            layouts["bb_beam_layout"] = {"fixed_mode": "manual", "fixed_beams": bb}
    outputs = ["bb_beams", "bb_beam_positions", "bb_beam_ids"] * has_bb(config) + [
        "frb2_beams",
        "frb2_beam_positions",
        "frb2_beam_ids",
    ]
    bb_lifetimes = -(-BB_FRAMES // BF_MASK_LIFETIME_FRAMES)
    return {
        "first_fpga_seq_num": large_seq(config),
        "input_kind": "sky",
        "sources": [
            {
                "amplitude": check_phases.AMPLITUDE,
                "direction": direction,
                "tone_cycles_per_frame": check_phases.tone_cycles(config),
            }
        ],
        "noise_rms": check_phases.NOISE,
        "bf_mask_lifetime_frames": BF_MASK_LIFETIME_FRAMES,
        "frb2_weights_lifetime_in_samples": value(config, "frb_num_output_times")
        * value(config, "frb_downsampling_factor"),
        **layouts,
        "outputs": outputs,
        "write_frames": {
            "bb_beams": BB_FRAMES,
            "bb_beam_positions": bb_lifetimes,
            "bb_beam_ids": bb_lifetimes,
            "frb2_beams": FRB2_FRAMES,
            "frb2_beam_positions": FRB2_FRAMES,
            "frb2_beam_ids": FRB2_FRAMES,
        },
    }


def configured_beams(config, block, mode_key, beams_key):
    """Positions [B, 2] (float32, as written) and ids [B] that the setter `block` (a config block
    or its name) is configured with."""
    stage = config[block] if isinstance(block, str) else block
    mode = stage[mode_key]
    if mode == "manual":
        beams = stage[beams_key]
        positions = [[b["x_dir_grid"], b["y_dir_grid"]] for b in beams]
        ids = [b["id"] for b in beams]
    else:
        require(mode == "grid_degrees", f"{block}: mode {mode} is not checked")
        nx, ny = value(config, "num_x", block), value(config, "num_y", block)
        x_min, x_max, y_min, y_max = [
            value(config, k, block) for k in ["x_min", "x_max", "y_min", "y_max"]
        ]  # grid_degrees only occurs at the top level
        positions = [
            [
                sin_deg(grid_point(x_min, x_max, nx, bx)),
                sin_deg(grid_point(y_min, y_max, ny, by)),
            ]
            for by in range(ny)
            for bx in range(nx)
        ]
        ids = list(range(nx * ny))
    return np.array(positions, dtype=np.float32), np.array(ids, dtype=np.uint64)


def check_beam_stream(out_dir, prefix, positions, ids, seq0, lifetime, origin):
    """Every written frame of `<prefix>_beam_positions` and `<prefix>_beam_ids` must hold the
    configured beams, and frame k must have fpga_seq_num seq0 + k lifetime."""
    num = len(ids)
    for name, expected in [
        (f"{prefix}_beam_positions", positions),
        (f"{prefix}_beam_ids", ids),
    ]:
        out = read_output(out_dir, name)
        seqs = frame_seq_nums(out)
        frames = out.data.reshape(len(seqs), *expected.shape)
        require(
            frames.dtype == expected.dtype,
            f"{name}: type {frames.dtype}, expected {expected.dtype}",
        )
        wrong = np.nonzero(
            np.any((frames != expected).reshape(len(seqs), num, -1), axis=(0, 2))
        )[0]
        require(
            not wrong.size,
            f"{name}: beams {wrong[:4].tolist()} are {frames[0, wrong[:4]].tolist()}, "
            f"expected {expected[wrong[:4]].tolist()}",
        )
        expected_seqs = seq0 + lifetime * np.arange(len(seqs))
        require(
            np.array_equal(seqs, expected_seqs),
            f"{name}: fpga_seq_num {seqs.tolist()}, expected {origin} + k {lifetime}: "
            f"{expected_seqs.tolist()}",
        )
        downsampling = {int(a["time_downsampling_fpga"]) for a in out.frame_attrs}
        require(
            downsampling == {lifetime},
            f"{name}: time_downsampling_fpga {sorted(downsampling)}, expected {lifetime}",
        )
    return (
        f"{prefix}_beam_positions, {prefix}_beam_ids: {len(seqs)} frames hold the {num} "
        f"configured beams exactly, fpga_seq_num = {origin} + k {lifetime}"
    )


def continuity(name, windows, boundaries, unit):
    """`windows` [K] are estimates of the same complex (or real) quantity, normalized to 1, in
    consecutive windows. All must agree with the median within CONTINUITY_SIGMAS robust standard
    deviations; `boundaries` are window indices that start a new lifetime."""
    median = np.median(windows.real) + 1j * np.median(windows.imag)
    deviation = windows - median
    sigma = 1.4826 * max(
        np.median(np.abs(deviation.real)), np.median(np.abs(deviation.imag))
    )
    sigma = max(sigma, 1e-6)
    worst = np.abs(deviation).max()
    steps = np.abs(windows[boundaries] - windows[np.array(boundaries) - 1]).astype(
        float
    )
    line = (
        f"{name}: {len(windows)} windows within {worst / sigma:.1f} σ (σ = {100 * sigma:.3f}% "
        f"of the {unit}); steps at {len(boundaries)} lifetime boundaries "
        f"{np.round(100 * steps, 3).tolist()}%"
    )
    require(
        worst <= CONTINUITY_SIGMAS * sigma,
        f"{line}; worst window {np.abs(deviation).argmax()}",
    )
    return line


def check(out_dir, var, config):
    SEQ0 = var["first_fpga_seq_num"]
    t_star = upchan_t_star(config)
    num_times = value(config, "num_times")
    channels = np.array(value(config, "frequency_channels"))
    source = var["sources"][0]
    direction = source["direction"]
    f_tone = source["tone_cycles_per_frame"] / num_times
    amplitude = source["amplitude"]
    nu = sky_frequency(config, channels, f_tone)
    phases = geometric_phases(config, direction, nu)  # [F, D]
    results = []

    # Configured beams; the source must be one of them
    with_bb = "bb_beams" in var["outputs"]
    at_source = np.float32(direction)
    stage = find_stages(config, "setFRBBeams")[0]
    frb2_positions, frb2_ids = configured_beams(
        config,
        "run_set_frb2_beams" if "run_set_frb2_beams" in config else stage,
        "mode",
        "beams",
    )
    frb2_source = np.nonzero(np.all(frb2_positions == at_source, axis=1))[0]
    if with_bb:
        bb_positions, bb_ids = configured_beams(
            config, "run_set_bb_beams", "fixed_mode", "fixed_beams"
        )
        # The closest beam: setBBBeams computes grid positions from degrees, which can round the
        # position of a manual FRB2 beam (CHIME) differently
        distance = np.abs(bb_positions - at_source).sum(axis=1)
        bb_source = np.nonzero(distance == distance.min())[0][:1]
        require(
            distance.min() < 1e-6,
            f"source {direction} is {distance.min():.2g} from the closest BB beam",
        )
    else:
        bb_source = np.zeros(1, dtype=int)
    require(
        len(bb_source) == 1 and len(frb2_source) == 1,
        f"source {direction} is BB beams {bb_source.tolist()} and FRB2 beams {frb2_source.tolist()}",
    )
    bb_source, frb2_source = int(bb_source[0]), int(frb2_source[0])
    results.append(
        f"source at ({math.degrees(math.asin(direction[0])):.5f}°, "
        f"{math.degrees(math.asin(direction[1])):.5f}°): "
        + (f"BB beam {bb_source} (id {bb_ids[bb_source]}), " if with_bb else "")
        + f"FRB2 beam {frb2_source} (id {frb2_ids[frb2_source]})"
    )

    # Written positions and ids, one frame per lifetime
    frb2_lifetime = value(config, "frb2_weights_lifetime_in_samples")
    frb1_lifetime = value(config, "frb1_phase_lifetime_in_samples")
    if with_bb:
        bb_lifetime = value(config, "bb_phase_lifetime_in_samples")
        results.append(
            check_beam_stream(
                out_dir, "bb", bb_positions, bb_ids, SEQ0, bb_lifetime, "seq0"
            )
        )
    results.append(
        check_beam_stream(
            out_dir,
            "frb2",
            frb2_positions,
            frb2_ids,
            SEQ0 + t_star,
            frb2_lifetime,
            "seq0 + T*",
        )
    )

    if with_bb:
        # BB [Thi, B, F, P, T]: the power peaks in the predicted beam (T2), every beam is non-zero
        bb = read_output(out_dir, "bb_beams")
        audit(bb, first_seq=SEQ0, frequency_channels=channels)
        require(
            bb.data.shape[0] == BB_FRAMES,
            f"bb_beams: {bb.data.shape[0]} frames written",
        )
        results.append(
            bb_peak_line(
                bb.data[:2],
                bb_positions.astype(np.float64),
                config,
                phases,
                nu,
                amplitude,
            )
        )
        lut = np.abs(decode_int4x2(np.arange(256))) ** 2
        rms = np.sqrt(
            [lut[bb.data[:, b]].mean() for b in range(bb.data.shape[1])]
        )  # [B]
        require(
            np.all(rms > 0),
            f"bb_beams: beams {np.nonzero(rms == 0)[0].tolist()} are zero",
        )
        results.append(
            f"bb_beams: all {len(rms)} beams non-zero, rms {rms.min():.2f} to {rms.max():.2f} LSB"
        )

        # BB continuity: the coherent tone c_wfp = Σ_t J_src(t) exp(-2πi f_tone (t - seq0)) in windows
        # of BB_WINDOW samples, projected on its mean over windows
        seqs = frame_seq_nums(bb)
        t = (seqs[:, None] + np.arange(num_times)[None, :] - SEQ0).reshape(
            -1
        )  # [Thi T]
        src = decode_int4x2(bb.data[:, bb_source]).astype(
            np.complex64
        )  # [Thi, F, P, T]
        src = np.moveaxis(src, -1, 1).reshape(-1, *src.shape[1:3])  # [Thi T, F, P]
        tone = (
            src * np.exp(-2j * np.pi * f_tone * t).astype(np.complex64)[:, None, None]
        )
        c = tone.reshape(-1, BB_WINDOW, *tone.shape[1:]).sum(axis=1)  # [W, F, P]
        mean = c.mean(axis=0)
        r = (c * mean.conj()).sum(axis=(1, 2)) / (np.abs(mean) ** 2).sum()
        starts = t.reshape(-1, BB_WINDOW)[:, 0]
        boundaries = [
            int(k) for k in np.nonzero((starts % bb_lifetime == 0) & (starts > 0))[0]
        ]
        line = continuity("bb_beams continuity", r, boundaries, "coherent tone")
        phase = np.degrees(np.angle(r)).max() - np.degrees(np.angle(r)).min()
        results.append(f"{line}; phase range {phase:.3f}°")

    # FRB2 [Ttildehi256, R, Fbar, Ttildelo256]: the source beam holds the peak (T2)
    frb2 = read_output(out_dir, "frb2_beams")
    audit(frb2, first_seq=SEQ0 + t_star)
    require(
        frb2.data.shape[0] == FRB2_FRAMES,
        f"frb2_beams: {frb2.data.shape[0]} frames written",
    )
    results.append(frb2_peak_line(frb2, frb2_source))

    # FRB2 continuity: the tone power in the source beam, summed over the fine channel of the tone
    # in every coarse channel, minus the median over beams, in windows of FRB2_WINDOW samples
    data = used_channels(frb2, 2)  # [Thi, R, Fbar, Tlo]
    coarse = np.atleast_1d(frb2.attr("coarse_freq"))[: data.shape[2]]
    power = data[:, frb2_source].sum(axis=(0, 2), dtype=np.float64)  # [Fbar]
    slots = [
        np.nonzero(coarse == chan)[0][power[coarse == chan].argmax()]
        for chan in np.unique(coarse)
    ]
    series = data[:, :, slots, :].sum(axis=2, dtype=np.float64)  # [Thi, R, Tlo]
    series = np.moveaxis(series, 1, 0).reshape(data.shape[1], -1)  # [R, Thi Tlo]
    excess = series[frb2_source] - np.median(series, axis=0)
    w = excess.reshape(-1, FRB2_WINDOW).mean(axis=1)
    w = w / np.median(w)
    step = value(config, "frb_downsampling_factor")
    labels = (
        frame_seq_nums(frb2)[:, None] + step * np.arange(data.shape[3])[None, :]
    ).reshape(-1)
    starts = labels.reshape(-1, FRB2_WINDOW)[:, 0] - SEQ0 - t_star
    boundaries = [
        int(k) for k in np.nonzero((starts % frb1_lifetime == 0) & (starts > 0))[0]
    ]
    line = continuity(
        "frb2_beams continuity", w.astype(complex), boundaries, "tone power"
    )
    results.append(
        f"{line} (FRB1 lifetime {frb1_lifetime}, FRB2 lifetime {frb2_lifetime} samples)"
    )
    return results
