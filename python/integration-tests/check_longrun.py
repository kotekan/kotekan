"""T10 long run: with everything that changes over time changing at once, the pipeline must run
for many ring buffer wraps and lifetime boundaries without stalling, dropping a frame, switching
late, or growing in memory.

Input: T2's weak tone (check_phases.py) on FRB2 beam (12, 5) in Gaussian noise of 2.8 LSB per
component, with the production gains, from `large_seq`. Random packet loss at `LOSS_RATE`, with
lost voltages 0+0i (T8); random bad inputs changing every `CHANGE_LIFETIMES` lifetimes (T9), poisoned
with random bytes where their packets are not lost; BB phase and FRB1 weights lifetimes of
T9's mask lifetime (3 frames; CHIME: 6) and an FRB2 weights lifetime of one FRB2 frame (T7).

Variants: `long` runs `RUN_FRAMES["long"]` frames, about 10 minutes of kotekan time on cx67.
`short` (not run by default) runs the same setup for a few seconds of simulated time.

Outputs, all for the same simulated span: both bad feed mask streams, BB and FRB2 beam positions
and ids, the FRB PL mask, the N2 lost-sample counts per frame (n2k_plcounts), the N2 integrations
of the first `N2_FREQUENCIES` frequencies, and about `FRB2_WRITTEN` FRB2 frames spread over the
run (hdf5FileWrite frame decimation).

Checks:
- Every writer received its number of frames, with contiguous fpga_seq_num; the N2 integrations
  are contiguous from the first bin boundary; the decimated writers wrote the configured frames.
- Mask streams equal the schedule at every frame (T9); beam positions and ids equal the
  configuration, one frame per lifetime at seq0 + k L (BB) and seq0 + T* + k L (FRB2) (T7).
- n2k_plcounts and every FRB PL mask frame equal the propagated loss exactly (T8); the measured
  loss rate agrees with `LOSS_RATE` within `LOSS_SIGMAS` binomial standard deviations.
- N2: n_pl, n_valid and n_rfi of every written integration are exact, the flags are the AND of
  the masks of its frames (T9), and no visibility or weight is NaN or Inf.
- FRB2: no NaN or Inf, and the source beam holds the peak in every written frame (T2).
- Memory: kotekan's resident memory does not grow by more than `MEMORY_TOLERANCE_MIB` after
  `MEMORY_WARMUP` seconds (see `check_memory`).

CHIME has one bad feed mask stream and no BB (see check_badfeeds.py), and poisons with
`random_valid` bytes; its run has as many frames, which also holds whole N2 integrations of 240
frames.

See docs/integration-tests/T10-long-run.md.
"""

import json
import math
import os
import re

import h5py
import numpy as np

import check_phases
from check_badfeeds import (
    Schedule,
    check_masks,
    lifetime_frames,
    n2_masked,
    result_line,
)
from check_beams import check_beam_stream, configured_beams
from check_dishes import n2_bin_start
from check_packetloss import Loss, frb_frame_length, frb_valid, t_star
from kit import (
    CheckFailed,
    audit,
    find_stages,
    frame_files,
    frb2_peak_line,
    has_bb,
    has_command,
    large_seq,
    read_frame,
    read_output,
    value,
)

VARIANTS = ["long"]
# Simulated voltage frames, a multiple of 72 (lifetimes, schedule periods, FRB and N2 frames;
# CHIME: 720)
RUN_FRAMES = {"long": 8640, "short": 216}
CHANGE_LIFETIMES = 3
BF_COUNT = [3, 5]
BF_SEED = 10
LOSS_RATE = 1e-4
LOSS_SEED = 10
LOSS_SIGMAS = 3
N2_FREQUENCIES = 16
FRB2_WRITTEN = 4
# Resident memory: sampling interval and warm-up (s), and the allowed growth after warm-up
MEMORY_INTERVAL = 30
MEMORY_WARMUP = 60
MEMORY_TOLERANCE_MIB = 32
# Frames per chunk when predicting the loss
CHUNK_FRAMES = 48


def n2_bin_length(config):
    return find_stages(config, "N2Accumulate")[0][
        "num_subintegrations_per_bin"
    ] * value(config, "n2k_samples_per_data_set")


def frame_counts(config, seq0, num_frames):
    """The frames each writer receives while the input covers `num_frames` frames from `seq0`.
    Outputs whose frames start at seq0 + T* or at an N2 bin boundary get the frames that end
    within the input."""
    T = value(config, "num_times")
    frb = (num_frames * T - t_star(config)) // frb_frame_length(config)
    bin0 = n2_bin_start(config, seq0)
    integrations = (seq0 + num_frames * T - bin0) // n2_bin_length(config)
    counts = {"n2_bf_mask": num_frames} if n2_masked(config) else {}
    counts["n2k_plcounts"] = num_frames
    counts["bf_mask"] = num_frames // lifetime_frames(config)
    if has_bb(config):
        counts["bb_beam_positions"] = num_frames // lifetime_frames(config)
        counts["bb_beam_ids"] = num_frames // lifetime_frames(config)
    counts |= {
        "frb_pl_mask": frb,
        "frb2_beam_positions": frb,
        "frb2_beam_ids": frb,
        "frb2_beams": frb,
        "n2_vis": integrations * value(config, "num_frequencies"),
    }
    return counts, integrations


def decimation(config, counts):
    """hdf5FileWrite [write_x_frames, per_y_frames]: the first N2_FREQUENCIES frequencies of
    every N2 integration, and FRB2 frames spread over the run."""
    return {
        "n2_vis": [N2_FREQUENCIES, value(config, "num_frequencies")],
        "frb2_beams": [1, max(counts["frb2_beams"] // FRB2_WRITTEN, 1)],
    }


def decimated_frames(count, x, y):
    """The frames of `count` that hdf5FileWrite writes with write_x_frames x and per_y_frames y:
    as documented, the first x of every y."""
    return [k for k in range(count) if k % y < x]


def variables(tel, config, variant):
    seq0 = large_seq(config)
    counts, _ = frame_counts(config, seq0, RUN_FRAMES[variant])
    return {
        "first_fpga_seq_num": seq0,
        "input_kind": "sky",
        "sources": [
            {
                "amplitude": check_phases.AMPLITUDE,
                "direction": check_phases.source_direction(config, "frb2"),
                "tone_cycles_per_frame": check_phases.tone_cycles(config),
            }
        ],
        "noise_rms": check_phases.NOISE,
        "loss_rate": LOSS_RATE,
        "loss_seed": LOSS_SEED,
        "lost_voltage": "zero",
        "bf_mask_lifetime_frames": lifetime_frames(config),
        "bf_change_frames": CHANGE_LIFETIMES * lifetime_frames(config),
        "bf_random_count": BF_COUNT,
        "bf_seed": BF_SEED,
        # Transpose2048_chime traps on -8 (julia/kernels/xpose2048.jl)
        "bad_voltage": (
            "random_valid"
            if has_command(config, "cudaTranspose2048_chime")
            else "random"
        ),
        "frb2_weights_lifetime_in_samples": frb_frame_length(config),
        "outputs": list(counts),
        "write_frames": counts,
        "write_decimation": decimation(config, counts),
    }


def section(results, failures, name, function, *args):
    """Run one check that raises CheckFailed, recording its result lines or its failure."""
    try:
        lines = function(*args)
        results.extend([lines] if isinstance(lines, str) else lines)
    except CheckFailed as error:
        results.append(f"{error}  <- wrong")
        failures.append(name)


def check_counts(out_dir, config, counts, results, failures):
    """Every per-frame writer received all its frames (file numbers) with contiguous
    fpga_seq_num; single-file writers hold all frames."""
    lines = []
    for name, count in counts.items():
        if name in ["n2_vis", "frb2_beams"]:
            continue
        if name in ["frb_pl_mask", "n2k_plcounts"]:
            out = read_output(out_dir, name)
            frame_len = (
                value(config, "frb_num_output_times") if name == "frb_pl_mask" else 1
            )
            num = out.data.shape[0] // frame_len
        else:
            files = frame_files(out_dir, name)
            num = len(files)
            if list(files) != list(range(count)):
                num = f"{num} (frames {list(files)[:3]}...{list(files)[-3:]})"
        lines.append(f"{name} {num}")
        if num != count:
            failures.append(f"{name} frames")
            lines[-1] += f" <- expected {count}"
    results.append("frames: " + ", ".join(lines))


def predicted_lost(var, config, seq0, num_frames):
    """The lost samples [frame, F] of every correlation frame."""
    T = value(config, "num_times")
    num_freqs = value(config, "num_frequencies")
    lost = []
    for k0 in range(0, num_frames, CHUNK_FRAMES):
        n = min(CHUNK_FRAMES, num_frames - k0)
        loss = Loss(var, seq0 + k0 * T, n * T, num_freqs)
        seq = seq0 + T * np.arange(k0, k0 + n)
        lost.append(loss.num_lost(seq, seq + T))
    return np.concatenate(lost)


def check_loss(out_dir, var, config, lost, results, failures):
    """n2k_plcounts equals the prediction; the measured loss rate agrees with LOSS_RATE."""
    T = value(config, "num_times")
    out = read_output(out_dir, "n2k_plcounts")
    audit(out, first_seq=var["first_fpga_seq_num"])
    measured = out.data.reshape(out.data.shape[0], -1)
    n = min(len(measured), len(lost))
    wrong = measured[:n] != lost[:n]
    results.append(
        result_line(
            "n2k_plcounts",
            np.count_nonzero(wrong),
            f"{lost[:n].sum()} lost samples predicted in {n} frames, {measured[:n].sum()} "
            f"counted, {np.count_nonzero(wrong)} entries differ",
            failures,
        )
    )
    # A packet holds 16 samples of 4 frequencies; it counts 16 lost samples in each of them
    packets = n * (T // 16) * (lost.shape[1] // 4)
    total = int(measured[:n].sum())
    lost_packets = total / 64
    expected = LOSS_RATE * packets
    sigma = math.sqrt(packets * LOSS_RATE * (1 - LOSS_RATE))
    deviation = (lost_packets - expected) / sigma
    results.append(
        result_line(
            "loss rate",
            total % 64 != 0 or abs(deviation) > LOSS_SIGMAS,
            f"{lost_packets:.0f} of {packets} packets lost ({lost_packets / packets:.4%}), "
            f"expected {expected:.0f} ± {sigma:.0f} ({LOSS_RATE:.4%}): {deviation:+.2f} σ",
            failures,
        )
    )


def check_frb_mask(out_dir, var, config, results, failures):
    """Every FRB PL mask frame equals the propagated loss (T8)."""
    seq0 = var["first_fpga_seq_num"]
    num_freqs = value(config, "num_frequencies")
    span = frb_frame_length(config)
    out = read_output(out_dir, "frb_pl_mask")
    audit(out, first_seq=seq0 + t_star(config))
    nt = value(config, "frb_num_output_times")
    num_frames = out.data.shape[0] // nt
    invalid = 0
    wrong = 0
    first_wrong = None
    for k0 in range(0, num_frames, CHUNK_FRAMES // 6):
        k1 = min(k0 + CHUNK_FRAMES // 6, num_frames)
        # The FRB windows of these frames lie within [seq0 + k0 span, seq0 + k1 span + 512)
        loss = Loss(var, seq0 + k0 * span, (k1 - k0) * span + 512, num_freqs)
        labels = out.fpga_seq_num + out.time_downsampling * np.arange(k0 * nt, k1 * nt)
        expected = frb_valid(loss, config, labels)
        measured = out.data[k0 * nt : k1 * nt, : expected.shape[1]] != 0
        differ = measured != expected
        invalid += np.count_nonzero(~expected)
        wrong += np.count_nonzero(differ)
        if differ.any() and first_wrong is None:
            t, f = np.argwhere(differ)[0]
            first_wrong = [k0 * nt + int(t), int(f)]
    results.append(
        result_line(
            "frb_pl_mask",
            wrong,
            f"{num_frames} frames, {invalid} invalid samples predicted, {wrong} differ",
            failures,
            first_wrong,
        )
    )


def check_n2(out_dir, var, config, sched, lost, integrations, results, failures):
    """Contiguous integrations from the first bin boundary, the documented frequencies, exact
    counts and flags, finite visibilities and weights."""
    T = value(config, "num_times")
    seq0 = sched.seq0
    channels = list(value(config, "frequency_channels"))
    num_freqs = len(channels)
    length = n2_bin_length(config)
    bin0 = n2_bin_start(config, seq0)
    files = frame_files(out_dir, "n2_vis")
    x, y = var["write_decimation"]["n2_vis"]
    expected_frames = decimated_frames(integrations * num_freqs, x, y)
    results.append(
        result_line(
            "n2_vis frames",
            list(files) != expected_frames,
            f"{len(files)} written, frequencies {sorted({k % y for k in files})} of every "
            f"integration; expected {len(expected_frames)}, the first {x} of every {y}",
            failures,
        )
    )
    starts = set()
    wrong_place = 0
    wrong_counts = 0
    wrong_flags = 0
    num_flagged = []
    not_finite = 0
    for k, filename in files.items():
        with h5py.File(filename, "r") as f:
            attrs = dict(f.attrs)
            not_finite += np.count_nonzero(~np.isfinite(f["vis"][()]))
            not_finite += np.count_nonzero(~np.isfinite(f["weight"][()]))
            flags = f["flags"][()] != 0
        start = int(attrs["fpga_start_tick"])
        f_local = channels.index(int(attrs["freq_id"]))
        starts.add(start)
        wrong_place += (
            start != bin0 + (k // num_freqs) * length
            or f_local != k % num_freqs
            or int(attrs["frame_length_fpga_ticks"]) != length
        )
        frames = (start - seq0) // T + np.arange(length // T)
        n_pl = int(lost[frames, f_local].sum())
        wrong_counts += (
            int(attrs["n_pl_fpga_ticks"]) != n_pl
            or int(attrs["n_valid_fpga_ticks"]) != length - n_pl
            or int(attrs["n_rfi_fpga_ticks"]) != 0
        )
        good = np.all(sched.good(seq0 + T * frames), axis=0)
        if not n2_masked(config):
            good[:] = True
        wrong_flags += np.any(flags != good)
        num_flagged.append(np.count_nonzero(~good))
    starts = sorted(starts)
    contiguous = starts == [bin0 + i * length for i in range(len(starts))]
    results.append(
        result_line(
            "n2_vis integrations",
            (not contiguous) + wrong_place,
            f"{len(starts)} contiguous integrations of {length} samples from the first bin "
            f"at seq0 + {bin0 - seq0}: {'yes' if contiguous else 'NO'}; {wrong_place} files "
            "with the wrong start, frequency or length",
            failures,
        )
    )
    results.append(
        result_line(
            "n2_vis counts",
            wrong_counts,
            f"n_pl, n_valid = length - n_pl and n_rfi = 0 wrong in {wrong_counts} of "
            f"{len(files)} files",
            failures,
        )
    )
    results.append(
        result_line(
            "n2_vis flags",
            wrong_flags,
            f"flags = AND of the frame masks ({min(num_flagged, default=0)} to "
            f"{max(num_flagged, default=0)} elements flagged per integration) wrong in "
            f"{wrong_flags} files",
            failures,
        )
    )
    results.append(
        result_line(
            "n2_vis values",
            not_finite,
            f"{not_finite} visibilities or weights NaN or Inf",
            failures,
        )
    )


def check_frb2(out_dir, var, config, counts, results, failures):
    """The documented frames are written; each is finite and peaks in the source beam."""
    seq0 = var["first_fpga_seq_num"]
    x, y = var["write_decimation"]["frb2_beams"]
    expected_frames = decimated_frames(counts["frb2_beams"], x, y)
    files = frame_files(out_dir, "frb2_beams")
    results.append(
        result_line(
            "frb2_beams frames",
            list(files) != expected_frames,
            f"frames {list(files)} written; expected the first {x} of every {y}: "
            f"{expected_frames}",
            failures,
        )
    )
    for k, filename in files.items():
        section(
            results, failures, f"frb2_beams {k}", frb2_frame, filename, k, config, seq0
        )


def frb2_frame(filename, k, config, seq0):
    """FRB2 frame k: its fpga_seq_num, no NaN or Inf, the peak in the source beam (T2)."""
    frame = read_frame(filename, "frb2_beams")
    audit(frame, first_seq=seq0 + t_star(config) + k * frb_frame_length(config))
    return f"frame {k}: " + frb2_peak_line(frame, check_phases.frb2_beam(config))


def check_memory(out_dir, results, failures):
    """Resident memory after warm-up: the median of the last third of the samples exceeds the
    median of the first third by at most MEMORY_TOLERANCE_MIB. A leak raises every later
    sample; a transient allocation, such as hdf5FileWrite compressing one large frame, raises
    single samples and does not move the medians."""
    with open(os.path.join(out_dir, "memory.json")) as f:
        samples = json.load(f)
    t = np.array([s[0] for s in samples])
    rss = np.array([s[1] for s in samples]) / 1024  # MiB
    gpu = [s[2] for s in samples if s[2] is not None]
    after = rss[t >= MEMORY_WARMUP]
    if len(after) < 6:
        results.append(
            f"memory: {len(after)} samples after the warm-up of {MEMORY_WARMUP} s, "
            "too few to check"
        )
        if not samples:
            failures.append("memory")
        return
    third = len(after) // 3
    first = np.median(after[:third])
    growth = np.median(after[-third:]) - first
    every = max(len(t) // 12, 1)
    series = ", ".join(f"{t[i]:.0f} s {rss[i]:.0f}" for i in range(0, len(t), every))
    results.append(
        result_line(
            "memory",
            growth > MEMORY_TOLERANCE_MIB,
            f"VmRSS after {MEMORY_WARMUP} s: {first:.0f} MiB, growth {growth:+.1f} MiB "
            f"(median of the last vs the first third of {len(after)} samples), tolerance "
            f"{MEMORY_TOLERANCE_MIB} MiB; largest sample {after.max() - first:+.0f} MiB; "
            f"GPU {min(gpu, default=0)} to {max(gpu, default=0)} MiB; samples (MiB): {series}",
            failures,
        )
    )


def frame_rate(out_dir, stream):
    """Frames per wall-clock second over the run, from the drain of the one-frame output
    `stream` in the kotekan log, in tenths of the run."""
    pattern = re.compile(rf"drain_{stream} frame (\d+) \(duration ([0-9.e+-]+) sec\)")
    points = []
    with open(out_dir + ".log") as log:
        for line in log:
            match = pattern.search(line)
            if match:
                points.append((int(match.group(1)), float(match.group(2))))
    if len(points) < 20:
        return "frame rate: no log"
    marks = [points[i * (len(points) - 1) // 10] for i in range(11)]
    rates = [(b[0] - a[0]) / max(b[1] - a[1], 1e-9) for a, b in zip(marks, marks[1:])]
    return (
        f"frame rate: {points[-1][0] + 1} frames in {points[-1][1]:.0f} s, per tenth of the "
        f"run {' '.join(f'{r:.1f}' for r in rates)} frames/s"
    )


def check(out_dir, var, config):
    seq0 = var["first_fpga_seq_num"]
    num_frames = var["write_frames"]["n2k_plcounts"]
    counts, integrations = frame_counts(config, seq0, num_frames)
    sched = Schedule(var, config)
    results = []
    failures = []
    results.append(
        frame_rate(out_dir, "n2_bf_mask" if n2_masked(config) else "n2k_plcounts")
    )
    check_counts(out_dir, config, counts, results, failures)
    check_masks(out_dir, config, sched, results, failures)
    stage = find_stages(config, "setFRBBeams")[0]
    frb2_positions, frb2_ids = configured_beams(
        config,
        "run_set_frb2_beams" if "run_set_frb2_beams" in config else stage,
        "mode",
        "beams",
    )
    streams = []
    if has_bb(config):
        bb_positions, bb_ids = configured_beams(
            config, "run_set_bb_beams", "fixed_mode", "fixed_beams"
        )
        streams.append(
            (
                "bb",
                bb_positions,
                bb_ids,
                seq0,
                value(config, "bb_phase_lifetime_in_samples"),
                "seq0",
            )
        )
    for prefix, positions, ids, first, lifetime, origin in streams + [
        (
            "frb2",
            frb2_positions,
            frb2_ids,
            seq0 + t_star(config),
            value(config, "frb2_weights_lifetime_in_samples"),
            "seq0 + T*",
        ),
    ]:
        section(
            results,
            failures,
            f"{prefix} beams",
            check_beam_stream,
            out_dir,
            prefix,
            positions,
            ids,
            first,
            lifetime,
            origin,
        )
    lost = predicted_lost(var, config, seq0, num_frames)
    check_loss(out_dir, var, config, lost, results, failures)
    check_frb_mask(out_dir, var, config, results, failures)
    check_n2(out_dir, var, config, sched, lost, integrations, results, failures)
    check_frb2(out_dir, var, config, counts, results, failures)
    check_memory(out_dir, results, failures)
    if failures:
        raise CheckFailed(
            f"{len(failures)} checks failed ({', '.join(failures)}):\n    "
            + "\n    ".join(results)
        )
    return results
