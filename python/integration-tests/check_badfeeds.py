"""T9 bad feed mask: bad feeds must be excluded from every subsystem exactly while they are
marked bad, so that their voltages do not matter.

Input: T2's weak tone (check_phases.py) in Gaussian noise of 2.8 LSB per component, with the
production gains. The bad feed schedule is `testBadFeeds` (lib/testing/testSkySignal.hpp): 3 to 5
random inputs per period of one mask lifetime, starting at the first sample. The bad feed mask
lifetime of RFI, BB and FRB1 is `LIFETIME_FRAMES` frames (see `lifetime_frames`); N2Accumulate gets a one-frame mask
stream with the same content. The run starts `N2_LEAD_FRAMES` frames before an N2 bin boundary,
so that the N2 bins begin and end at mask changes.

Runs, in pairs with the same masks:
- `full`, `full_reference`: all dishes present.
- `half`, `half_reference`: the northern half of the dishes is Missing (`half_filled`), and
  `bf_mode: auto` marks them bad as well.
In `full` and `half` the voltages of the bad inputs and of the Missing dishes are random bytes
(`bad_voltage: random`; CHIME: `random_valid`, without -8, on which its transpose traps); in the
reference runs they carry the normal signal.

Checks:
- Both mask streams equal the schedule, frame by frame, with fpga_seq_num stepping by their
  lifetimes, and agree at every sample.
- BB phases A are zero exactly for the bad inputs of each lifetime frame, and the FRB1 weights W
  exactly for the grid cells of bad inputs (as setFRB1Phase places the dishes). BB beams equal
  the reference bit for bit. FRB1 and FRB2 samples equal the reference bit for bit unless their
  input window holds a poisoned sample of an input that their weight frame does not mask; FRB3
  values of such clean samples are within one quantization step of the reference.
- RFI: S012tilde is the sum of the single-feed S012 over the good inputs of each 256-sample
  block, so S0tilde is 256 times the number of good inputs; S012tilde, SKtilde and the RFI mask
  equal the reference, and the single-feed S012 of good inputs too.
- N2: the per-element flags of each integration are the AND of the masks of the correlation
  frames it accumulates; visibilities and weights of products of good elements, and the sample
  counts, equal the reference.

CHIME has no BB beamformer and no FRB U = 1; its FRB1 weights are the feed gains of
processFRBFeedGains.
CHIMETelescope has no dish types for `bf_mode: auto`, so the half-filled CHIME run lists the
Missing inputs with `bf_mode: manual`.

See docs/integration-tests/T9-bad-feed-mask.md.
"""

import glob
import os

import h5py
import numpy as np

import check_phases
from check_dishes import n2_bin_start
from check_packetloss import frb3_values, frb_factors, t_star
from kit import (
    NUM_TAPS,
    audit,
    dish_positions,
    find_stages,
    frame_seq_nums,
    frb1_swap_MN,
    has_bb,
    has_command,
    large_seq,
    read_output,
    require,
    value,
)

RUNS = ["full_reference", "full", "half_reference", "half"]
LIFETIME_FRAMES = 3
SEED = 9
COUNT = [3, 5]
# The first N2 bin starts this many frames after the first sample
N2_LEAD_FRAMES = 3
N2_INTEGRATIONS = 2
# Frames written: voltages; BB, RFI; FRB1; FRB2 and FRB3 (masks: for both N2 integrations)
VOLTAGE_FRAMES = 1
DATA_FRAMES = 7
FRB1_FRAMES = 1
FRB2_FRAMES = 2


def lifetime_frames(config):
    """The bad feed mask lifetime in frames: LIFETIME_FRAMES, or at least twice the
    upchannelizer output offset T*. The FRB1 weights made from mask frame k apply from T* after
    its start (processFRBFeedGains, setFRB1Phase), so for T* of a whole lifetime (CHIME: 3
    frames) no FRB sample would have the weights of its own mask frame."""
    T = value(config, "num_times")
    return max(LIFETIME_FRAMES, -(-2 * t_star(config) // T))


def render_half(tel):
    """The config with `half_filled`, for its Missing dishes."""
    from run import render_config

    return render_config(tel, {"half_filled": True})


def n2_bin_frames(config):
    return (
        find_stages(config, "N2Accumulate")[0]["num_subintegrations_per_bin"]
        * value(config, "n2k_samples_per_data_set")
        // value(config, "num_times")
    )


def mask_frames(config):
    return N2_LEAD_FRAMES + n2_bin_frames(config) * N2_INTEGRATIONS


def variables(tel, config, run):
    var = check_phases.variables(tel, config, "offgrid")
    del var["outputs"], var["write_frames"]
    T = value(config, "num_times")
    var["first_fpga_seq_num"] = (
        n2_bin_start(config, large_seq(config)) - N2_LEAD_FRAMES * T
    )
    var["bf_mask_lifetime_frames"] = lifetime_frames(config)
    var["bf_change_frames"] = lifetime_frames(config)
    var["bf_random_count"] = COUNT
    var["bf_seed"] = SEED
    if run.startswith("half"):
        var["half_filled"] = True
        if config["telescope"]["name"] == "CHORDTelescope":
            var["bf_mode"] = "auto"
        else:
            # inventBFMask's `auto` needs CHORDTelescope's dish types
            _, exists = dish_positions(render_half(tel))
            var["bf_mode"] = "manual"
            var["bf_manual_bad_feeds"] = np.nonzero(~np.tile(exists, 2))[0].tolist()
        # As config/chord/pathfinder.j2: the default 0.8 counts masked inputs as missing data and
        # flags everything when more than 20% of the inputs are masked
        var["rfi_feed_averaged_min_good_frac"] = 0.1
    if not run.endswith("reference"):
        # Transpose2048_chime traps on -8 (julia/kernels/xpose2048.jl)
        chime_transpose = has_command(config, "cudaTranspose2048_chime")
        var["bad_voltage"] = "random_valid" if chime_transpose else "random"
    lifetimes = mask_frames(config) // lifetime_frames(config)
    write_frames = {
        "bf_mask": lifetimes,
        "n2_bf_mask": mask_frames(config),
        "voltage": VOLTAGE_FRAMES,
    }
    if has_bb(config):
        write_frames["bb_phase"] = DATA_FRAMES // lifetime_frames(config) + 1
        write_frames["bb_beams"] = DATA_FRAMES
    write_frames |= {
        "rfi_S012": DATA_FRAMES,
        "rfi_S012tilde": DATA_FRAMES,
        "rfi_SKtilde": DATA_FRAMES,
        "rfi_RFImask": DATA_FRAMES,
        "frb1_beams": FRB1_FRAMES,
        "frb2_beams": FRB2_FRAMES,
        "frb3_beams": FRB2_FRAMES,
        "frb3_beams_offsetscale": FRB2_FRAMES,
        "n2_vis": N2_INTEGRATIONS * value(config, "num_frequencies"),
    }
    for U in frb_factors(config):
        write_frames[f"frb1_U{U}_phase"] = lifetimes
    var["outputs"] = list(write_frames)
    var["write_frames"] = write_frames
    return var


def splitmix64(x):
    """The splitmix64 finalizer on a Python int (lib/testing/testSkySignal.hpp)."""
    m = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    return x ^ (x >> 31)


class Schedule:
    """The bad inputs of a run: `testBadFeeds` (lib/testing/testSkySignal.hpp) plus, with
    `bf_mode: auto`, the Missing dishes. Inputs are p D + d, the order of the bad feed mask."""

    def __init__(self, var, config):
        self.seq0 = var["first_fpga_seq_num"]
        self.change = var["bf_change_frames"] * value(config, "num_times")
        self.num_dishes = value(config, "num_dishes")
        self.num_inputs = value(config, "num_polarizations") * self.num_dishes
        self.count = var["bf_random_count"]
        self.seed_hash = splitmix64(var["bf_seed"])
        self.base = np.ones(self.num_inputs, dtype=bool)
        if var.get("bf_mode") == "auto":
            _, exists = dish_positions(config)
            self.base = np.tile(exists, self.num_inputs // self.num_dishes)
        self.base[var.get("bf_manual_bad_feeds", [])] = False

    def random_bad(self, k):
        """The random bad inputs of period k."""

        def h(i):
            return splitmix64(((k << 16) | i) ^ self.seed_hash)

        lo, hi = self.count
        n = lo + h(0) % (hi - lo + 1)
        bad = []
        i = 1
        while len(bad) < n:
            x = h(i) % self.num_inputs
            if x not in bad:
                bad.append(x)
            i += 1
        return bad

    def period(self, seq):
        return (np.asarray(seq) - self.seq0) // self.change

    def good_in_period(self, k):
        good = self.base.copy()
        good[self.random_bad(int(k))] = False
        return good

    def good(self, seq):
        """Which inputs are good at FPGA samples `seq`: [..., I]."""
        k = self.period(seq)
        periods = np.unique(k)
        table = np.array([self.good_in_period(p) for p in periods])
        return table[np.searchsorted(periods, k)]


def first_index(flags):
    """The index of the first true entry of `flags`."""
    return [int(i) for i in np.unravel_index(np.argmax(flags), np.shape(flags))]


def result_line(name, num_wrong, text, failures, where=None):
    line = f"{name}: {text}"
    if num_wrong:
        line += "  <- wrong" if where is None else f"  <- first at {where}"
        failures.append(name)
    return line


def check_masks(out_dir, config, sched, results, failures):
    """Both mask streams equal the schedule and agree at every sample."""
    T = value(config, "num_times")
    streams = {}
    L = value(config, "bf_mask_lifetime_in_samples")
    for name, lifetime in [("bf_mask", L), ("n2_bf_mask", T)]:
        out = read_output(out_dir, name)
        audit(out, first_seq=sched.seq0)
        require(
            out.time_downsampling == lifetime,
            f"{name}: lifetime {out.time_downsampling}, expected {lifetime}",
        )
        seq = frame_seq_nums(out)
        mask = out.data.reshape(len(seq), -1) != 0
        expected = sched.good(seq)
        wrong = mask != expected
        changes = np.count_nonzero(np.any(expected[1:] != expected[:-1], axis=1))
        results.append(
            result_line(
                name,
                wrong.any(),
                f"{len(seq)} frames of {lifetime} samples from seq0, {changes} changes, "
                f"{(~expected).sum(axis=1).min()} to {(~expected).sum(axis=1).max()} bad "
                f"inputs per frame, {np.count_nonzero(wrong)} entries differ from the schedule",
                failures,
                first_index(wrong) if wrong.any() else None,
            )
        )
        streams[name] = (seq, mask)
    # The two streams at the first sample of every one-frame mask frame
    seq, n2_mask = streams["n2_bf_mask"]
    bf_seq, bf_mask = streams["bf_mask"]
    index = (seq - bf_seq[0]) // L
    use = index < len(bf_seq)
    differ = np.any(n2_mask[use] != bf_mask[index[use]], axis=1)
    results.append(
        result_line(
            "mask streams",
            differ.any(),
            f"bf_mask and n2_bf_mask agree in {np.count_nonzero(~differ)} of "
            f"{np.count_nonzero(use)} frames",
            failures,
        )
    )


def check_voltage(out_dir, ref_dir, sched, results, failures):
    """The poisoned run differs from the reference exactly in the bad inputs."""
    out = read_output(out_dir, "voltage")
    ref = read_output(ref_dir, "voltage")
    audit(out, first_seq=sched.seq0)
    seq = out.fpga_seq_num + np.arange(out.data.shape[0])
    bad = ~sched.good(seq)  # [T, I]
    differ = (out.data != ref.data).reshape(*out.data.shape[:2], -1)  # [T, F, I]
    bad = np.broadcast_to(bad[:, None, :], differ.shape)
    wrong = np.count_nonzero(differ & ~bad)
    results.append(
        result_line(
            "voltage",
            wrong,
            f"{wrong} good values differ from the reference; "
            f"{np.count_nonzero(differ & bad) / max(np.count_nonzero(bad), 1):.4f} of the bad "
            "values are poisoned",
            failures,
        )
    )


def check_bb_phase(out_dir, sched, results, failures):
    """A [Tbb, F, P, B, D, C] is zero exactly for the bad inputs of each lifetime frame."""
    out = read_output(out_dir, "bb_phase")
    audit(out, first_seq=sched.seq0)
    seq = frame_seq_nums(out)
    nonzero = np.any(out.data != 0, axis=5)  # [Tbb, F, P, B, D]
    good = sched.good(seq).reshape(len(seq), 1, -1, 1, sched.num_dishes)
    wrong = nonzero != np.broadcast_to(good, nonzero.shape)
    results.append(
        result_line(
            "bb_phase",
            wrong.any(),
            f"{len(seq)} frames, A zero for {np.count_nonzero(~good)} bad (frame, input) pairs, "
            f"{np.count_nonzero(wrong)} entries wrong",
            failures,
            first_index(wrong) if wrong.any() else None,
        )
    )


def frb1_cells(config):
    """The dish at every FRB1 grid cell [N, M], placed as setFRB1Phase places them
    (kotekan::frb_dish_index_at_location): ArrayDishes at their grid indices, the other dishes
    in the empty cells in order, padding (>= num_dishes) after them."""
    M = value(config, "num_dishes_x")
    N = value(config, "num_dishes_y")
    D = value(config, "num_dishes")
    cell = np.full((N, M), -1)
    for dish in config["telescope"]["dish_inputs"]:
        if dish["type"] == "ArrayDish":
            cell[dish["grid_y_idx"], dish["grid_x_idx"]] = dish["dish_idx"]
    placed = set(cell[cell >= 0].tolist())
    surplus = (d for d in range(M * N) if d not in placed)
    for m in range(M):
        for n in range(N):
            if cell[n, m] < 0:
                cell[n, m] = next(surplus)
    require(
        sorted(cell.ravel().tolist())[:D] == list(range(D)),
        "frb1 cells: dishes missing",
    )
    # [N, M] with M along grid y (CHIME)
    return cell.T if frb1_swap_MN(config) else cell


def check_frb1_weights(out_dir, config, sched, results, failures):
    """W [TW, Fbar, P, N, M, C] is zero exactly at the grid cells of bad inputs, and
    frb1_input_scale (CHIME: the default feed gain of loadFeedGains) elsewhere, in every used
    fine channel."""
    cell = frb1_cells(config)
    on_grid = cell < sched.num_dishes
    gains = find_stages(config, "loadFeedGains")
    weight = (
        gains[0]["default_gains"] if gains else [value(config, "frb1_input_scale"), 0]
    )
    L = value(config, "bf_mask_lifetime_in_samples")
    num_wrong = 0
    num_zero = 0
    first_wrong = None
    for U in frb_factors(config):
        name = f"frb1_U{U}_phase"
        out = read_output(out_dir, name)
        num = U * (
            value(config, f"upchan_U{U}_max_channel")
            - value(config, f"upchan_U{U}_min_channel")
        )
        # The unused fine channels are NaN
        out.data = out.data[:, :num]
        audit(out, first_seq=sched.seq0 + t_star(config))
        seq = frame_seq_nums(out) - t_star(config)
        w = out.data.astype(np.float64)  # [TW, Fbar, P, N, M, C]
        good = sched.good(seq).reshape(len(seq), -1, sched.num_dishes)  # [TW, P, D]
        expected_good = np.where(on_grid, good[:, :, np.where(on_grid, cell, 0)], True)
        expected = np.stack([np.where(expected_good, w, 0.0) for w in weight], axis=-1)[
            :, None
        ]
        wrong = w != np.broadcast_to(expected, w.shape)
        require(out.time_downsampling == L, f"{name}: lifetime {out.time_downsampling}")
        num_wrong += np.count_nonzero(wrong)
        num_zero += np.count_nonzero(w == 0)
        if wrong.any() and first_wrong is None:
            first_wrong = f"U={U} {first_index(wrong)}"
    results.append(
        result_line(
            "frb1 weights",
            num_wrong,
            f"all U, {num_zero} zero weights as predicted, {num_wrong} wrong",
            failures,
            first_wrong,
        )
    )


def equal_line(name, measured, reference, clean, failures):
    """Clean samples must equal the reference bit for bit. Report how many other samples
    differ, to show that the poison reaches them."""
    clean = np.broadcast_to(clean, measured.shape)
    differ = measured != reference
    wrong = np.count_nonzero(differ & clean)
    line = (
        f"{name}: {wrong} of {np.count_nonzero(clean)} clean values differ from the reference; "
        f"{np.count_nonzero(differ & ~clean)} of {np.count_nonzero(~clean)} exposed values differ"
    )
    if wrong:
        line += f"  <- first at index {first_index(differ & clean)}"
        failures.append(name)
    return line


def frb_clean(sched, config, labels):
    """Whether FRB samples with labels `labels` [J] are independent of the poison: [J, Fbar].
    An FRB sample of upchannelization factor U depends on the input samples
    [L - (M-1) U/2, L + 192 + (M-1) U/2) (U > 1) or [L, L + 192) (U = 1), and is weighted with
    weight frame floor((L - seq0 - T*) / lifetime). It is exposed if a poisoned input in its
    window is good in that weight frame. Windows span at most two mask periods."""
    tds = value(config, "frb_downsampling_factor")
    L = value(config, "bf_mask_lifetime_in_samples")
    labels = np.asarray(labels)
    weight_frame = (labels - sched.seq0 - t_star(config)) // L
    weight_seq = sched.seq0 + weight_frame * L
    num_out = max(
        value(config, f"upchan_U{U}_max_output_channel") for U in frb_factors(config)
    )
    clean = np.zeros((len(labels), num_out), dtype=bool)
    good_w = sched.good(weight_seq)  # [J, I]
    for U in frb_factors(config):
        fmin = value(config, f"upchan_U{U}_min_channel")
        fmax = value(config, f"upchan_U{U}_max_channel")
        lo = value(config, f"upchan_U{U}_min_output_channel")
        hi = value(config, f"upchan_U{U}_max_output_channel")
        reach = (NUM_TAPS - 1) * U // 2 if U > 1 else 0
        exposed = np.zeros(len(labels), dtype=bool)
        for edge in [labels - reach, labels + tds + reach - 1]:
            poisoned = ~sched.good(np.maximum(edge, sched.seq0))
            exposed |= np.any(poisoned & good_w, axis=1)
        clean[:, lo:hi] = np.repeat(~exposed[:, None], U * (fmax - fmin), axis=1)
    return clean


def check_frb(out_dir, ref_dir, config, sched, results, failures):
    ts = t_star(config)
    tds = value(config, "frb_downsampling_factor")

    # FRB1 beams [Ttilde, Fbar, beamQ, beamP]
    out = read_output(out_dir, "frb1_beams")
    ref = read_output(ref_dir, "frb1_beams")
    audit(out, first_seq=sched.seq0 + ts)
    labels = out.fpga_seq_num + out.time_downsampling * np.arange(out.data.shape[0])
    clean = frb_clean(sched, config, labels)
    num = clean.shape[1]
    results.append(
        equal_line(
            "frb1_beams",
            out.data[:, :num].view(np.uint16),
            ref.data[:, :num].view(np.uint16),
            clean[:, :, None, None],
            failures,
        )
    )

    # FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256]
    out = read_output(out_dir, "frb2_beams")
    ref = read_output(ref_dir, "frb2_beams")
    audit(out, first_seq=sched.seq0 + ts)
    nt = out.data.shape[3]
    labels = out.fpga_seq_num + tds * np.arange(out.data.shape[0] * nt)
    clean = frb_clean(sched, config, labels)  # [J, Fbar]
    clean = clean.reshape(out.data.shape[0], nt, num).transpose(0, 2, 1)[:, None]
    results.append(
        equal_line(
            "frb2_beams",
            out.data[:, :, :num].view(np.uint16),
            ref.data[:, :, :num].view(np.uint16),
            clean,
            failures,
        )
    )

    # FRB3: one offset and scale per 256 samples, so an exposed sample changes its chunk
    q, xhat, scale, offset, (nodata, qmin, qmax) = frb3_values(out_dir, num)
    q_ref, xhat_ref, scale_ref, offset_ref, _ = frb3_values(ref_dir, num)
    x = out.data[:, :, :num].astype(np.float64)
    clean = np.broadcast_to(clean, q.shape)

    def chunks(a):
        # 4 bits: 256 samples per chunk, 8 bits: 16 (and 16 frequencies, over which the
        # exposure does not change)
        return a.reshape(*a.shape[:3], -1, 256 if nodata == -8 else 16)

    whole = np.all(chunks(clean), axis=4)
    same_chunk = np.all(chunks(q == q_ref), axis=4)
    same_chunk &= np.all(
        chunks(np.broadcast_to((scale == scale_ref) & (offset == offset_ref), q.shape)),
        axis=4,
    )
    wrong_whole = np.count_nonzero(whole & ~same_chunk)
    use = clean & (q > qmin) & (q < qmax) & (q_ref > qmin) & (q_ref < qmax)
    if nodata != -8:
        use = clean
    allowance = 2.0 ** -9 * (2 * np.abs(x) + np.abs(offset) + np.abs(offset_ref))
    allowance = allowance + 2.0 ** -9 * (scale + scale_ref)
    excess = np.abs(xhat - xhat_ref) - (scale + scale_ref) / 2 - allowance
    num_beyond = np.count_nonzero(use & (excess > 0))
    step = np.maximum(scale, scale_ref)
    ratio = (np.abs(xhat - xhat_ref) / np.where(step > 0, step, 1))[use]
    worst = float(ratio.max()) if ratio.size else 0.0
    results.append(
        result_line(
            "frb3_beams",
            wrong_whole + num_beyond,
            f"{np.count_nonzero(whole)} of {whole.size} chunks clean and "
            f"{wrong_whole} of them differ; {num_beyond} of {np.count_nonzero(use)} clean "
            f"unsaturated values beyond one step of the reference (max {worst:.3f} steps)",
            failures,
        )
    )


def check_bb(out_dir, ref_dir, sched, results, failures):
    out = read_output(out_dir, "bb_beams")
    ref = read_output(ref_dir, "bb_beams")
    audit(out, first_seq=sched.seq0)
    results.append(equal_line("bb_beams", out.data, ref.data, True, failures))


def check_rfi(out_dir, ref_dir, sched, results, failures):
    """S012tilde = Σ_good S012 per 256-sample block; good single-feed S012, S012tilde, SKtilde
    and the RFI mask equal the reference."""
    s012 = read_output(out_dir, "rfi_S012")  # [Trfi, F, 3, P, D]
    s012_ref = read_output(ref_dir, "rfi_S012")
    tilde = read_output(out_dir, "rfi_S012tilde")  # [Trfi, F, 3]
    audit(s012, first_seq=sched.seq0)
    audit(tilde, first_seq=sched.seq0)
    ds = s012.time_downsampling
    seq = s012.fpga_seq_num + ds * np.arange(s012.data.shape[0])
    good = sched.good(seq)  # [Trfi, I]
    data = s012.data.reshape(*s012.data.shape[:3], -1)  # [Trfi, F, 3, I]
    expected = np.where(good[:, None, None, :], data, 0).sum(axis=3, dtype=np.uint64)
    n = min(len(expected), tilde.data.shape[0])
    wrong = tilde.data[:n] != expected[:n]
    s0 = tilde.data[:n, :, 0]
    n_good = good[:n].sum(axis=1)
    wrong_s0 = s0 != (ds * n_good)[:, None]
    results.append(
        result_line(
            "rfi_S012tilde",
            np.count_nonzero(wrong) + np.count_nonzero(wrong_s0),
            f"{n} blocks of {ds} samples: Σ_good S012 differs in {np.count_nonzero(wrong)} "
            f"entries; S0tilde = {ds} × good inputs ({n_good.min()} to {n_good.max()}) except "
            f"{np.count_nonzero(wrong_s0)} entries",
            failures,
            first_index(wrong | wrong_s0[:, :, None])
            if wrong.any() or wrong_s0.any()
            else None,
        )
    )
    results.append(
        equal_line(
            "rfi_S012",
            data,
            s012_ref.data.reshape(data.shape),
            good[:, None, None, :],
            failures,
        )
    )
    for name in ["rfi_S012tilde", "rfi_SKtilde", "rfi_RFImask"]:
        out = read_output(out_dir, name)
        ref = read_output(ref_dir, name)
        audit(out, first_seq=sched.seq0)
        a = out.data.view(np.uint8)
        b = ref.data.view(np.uint8)
        wrong = np.count_nonzero(a != b)
        text = f"{wrong} of {a.size} bytes differ"
        if name == "rfi_RFImask":
            valid = np.unpackbits(a).mean()
            text += f"; {valid:.4f} of the samples unflagged"
            wrong += valid == 0
        results.append(result_line(f"{name} vs reference", wrong, text, failures))


def read_n2(out_dir):
    """Integrations by fpga_start_tick: lists of (freq_id, attrs, vis, weight, flags)."""
    integrations = {}
    for filename in sorted(glob.glob(os.path.join(out_dir, "n2_vis.*.h5"))):
        with h5py.File(filename, "r") as f:
            attrs = dict(f.attrs)
            entry = (
                int(attrs["freq_id"]),
                attrs,
                f["vis"][()],
                f["weight"][()],
                f["flags"][()],
            )
        integrations.setdefault(int(attrs["fpga_start_tick"]), []).append(entry)
    require(integrations, "n2_vis: no output files")
    return {k: sorted(v, key=lambda e: e[0]) for k, v in sorted(integrations.items())}


def n2_flags(start, entries, config, sched, results, failures):
    """The flags of the integration at `start` must be the AND of the masks of its correlation
    frames. Returns the expected good elements [I]."""
    T = value(config, "num_times")
    num_freqs = value(config, "num_frequencies")
    require(len(entries) == num_freqs, f"n2_vis at {start}: {len(entries)} frequencies")
    length = int(entries[0][1]["frame_length_fpga_ticks"])
    seq = start + T * np.arange(length // T)
    expected = np.all(sched.good(seq), axis=0)  # [I]
    prior = sched.good(start - T)
    flags = np.array([e[4] for e in entries]) != 0  # [F, I]
    wrong = flags != expected[None, :]
    extra = np.nonzero(np.any(~flags & expected, axis=0))[0]
    missed = np.nonzero(np.any(flags & ~expected, axis=0))[0]
    line = (
        f"n2_vis [{start}, {start + length}): {np.count_nonzero(~expected)} elements bad in "
        f"some frame; flags wrong for {np.count_nonzero(np.any(wrong, axis=0))} elements"
    )
    if extra.size:
        line += (
            f" (flagged although good: {extra.tolist()}; bad in the frame before the bin: "
            f"{[int(e) for e in extra if not prior[e]]})"
        )
    if missed.size:
        line += f" (not flagged although bad: {missed.tolist()})"
    results.append(result_line(f"n2 flags {start}", wrong.any(), line, failures))
    return expected


def check_n2(out_dir, ref_dir, config, sched, results, failures):
    """Flags are the AND of the masks of the integrated correlation frames; products of good
    elements and the counts equal the reference."""
    T = value(config, "num_times")
    num_freqs = value(config, "num_frequencies")
    out = read_n2(out_dir)
    ref = read_n2(ref_dir)
    require(list(out) == list(ref), "n2_vis: the runs integrate different samples")
    first = sched.seq0 + N2_LEAD_FRAMES * T
    require(
        list(out)[0] == first,
        f"n2_vis: first integration at {list(out)[0]}, expected {first}",
    )
    for start, entries in out.items():
        expected = n2_flags(start, entries, config, sched, results, failures)
        ia, ib = np.triu_indices(sched.num_inputs)
        good_product = expected[ia] & expected[ib]
        num_vis = 0
        num_weight = 0
        num_counts = 0
        num_bad_differ = 0
        for e, r in zip(entries, ref[start]):
            require(e[0] == r[0], f"n2_vis at {start}: frequencies differ")
            differ = np.any(e[2].view(np.uint32) != r[2].view(np.uint32), axis=-1)
            num_vis += np.count_nonzero(differ & good_product)
            num_bad_differ += np.count_nonzero(differ & ~good_product)
            num_weight += np.count_nonzero(
                (e[3].view(np.uint32) != r[3].view(np.uint32)) & good_product
            )
            for key in ["n_valid_fpga_ticks", "n_rfi_fpga_ticks", "n_pl_fpga_ticks"]:
                num_counts += int(e[1][key]) != int(r[1][key])
        n_valid = min(int(e[1]["n_valid_fpga_ticks"]) for e in entries)
        results.append(
            result_line(
                f"n2_vis {start}",
                num_vis + num_weight + num_counts + (n_valid == 0),
                f"n_valid ≥ {n_valid}; "
                f"{np.count_nonzero(good_product)} good products per frequency: {num_vis} "
                f"visibilities, {num_weight} weights and {num_counts} counts differ from the "
                f"reference; {num_bad_differ} of {num_freqs * np.count_nonzero(~good_product)} "
                "products of bad elements differ",
                failures,
            )
        )


def check(runs):
    results = []
    failures = []
    for run, (out_dir, var, run_config) in runs.items():
        sched = Schedule(var, run_config)
        results.append(f"[{run}]")
        check_masks(out_dir, run_config, sched, results, failures)
        if has_bb(run_config):
            check_bb_phase(out_dir, sched, results, failures)
        check_frb1_weights(out_dir, run_config, sched, results, failures)
        if run.endswith("reference"):
            continue
        ref_dir = runs[run + "_reference"][0]
        check_voltage(out_dir, ref_dir, sched, results, failures)
        if has_bb(run_config):
            check_bb(out_dir, ref_dir, sched, results, failures)
        check_frb(out_dir, ref_dir, run_config, sched, results, failures)
        check_rfi(out_dir, ref_dir, sched, results, failures)
        check_n2(out_dir, ref_dir, run_config, sched, results, failures)
    require(
        not failures,
        f"{len(failures)} checks failed ({', '.join(failures)}):\n    "
        + "\n    ".join(results),
    )
    return results
