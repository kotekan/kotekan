"""T8 packet loss: every packet-loss (PL) mask must be the propagation of the injected loss, and
lost samples must not change valid data.

Input: T2's weak tone (check_phases.py) in Gaussian noise of 2.8 LSB per component, with the
production gains. The noise repeats every frame, so one frame of voltages is the whole input.

Runs, all with the same input:
- `reference`: no packet loss.
- `sparse`: random loss at `SPARSE_RATE` (a few packets per frame) plus the loss boxes of
  `loss_boxes`; lost voltages are zero, as the receivers write them. The correlator and the RFI
  statistics rely on this; they do not read the PL mask for the voltages.
- `dense`: random loss at `DENSE_RATE` plus the same boxes; lost voltages are -8-8i (no data).

CHIME receives lost samples per group of 4 frequencies (lost_samples_buffer_<n>), from which
zeroSamples zeroes the lost voltages, also in `dense`, and lostSamplesToPLMask makes the PL mask;
the FRB chain has no U = 1. The HFB upchannelizer (U = 128) is checked as the others, with the
output offset of the HFB chain.

The loss model is `testPacketLoss` (lib/testing/testSkySignal.hpp): random loss drops packets of
16 FPGA samples of 4 frequencies for all inputs; boxes have the PL mask resolution of 2 samples.

Checks:
- For every run, the input PL mask, the expanded mask, the upchannelized mask for every U, the
  BB mask and the FRB mask equal the propagated loss exactly. An upchannelized sample L is valid iff the input
  samples [L - (M-1) U/2, L - (M-1) U/2 + M U) are; an FRB sample L iff [L, L + 192) (U = 1) or
  [L - (M-1) U/2, L + 192 + (M-1) U/2) (U > 1) are; a BB sample iff its input sample is.
- Valid upchannelized, BB, FRB1 and FRB2 samples equal the reference run bit for bit. Invalid
  FRB3 samples are -8; valid ones are within one quantization step of the reference.
- N2 (`reference` and `sparse`): the per-frame counts and lost-sample counts, and the valid and
  lost samples of the integration, equal the prediction; Σ V n_valid equals the reference sum
  minus Σ E_a conj(E_b) over the lost samples, within float32 rounding.

See docs/integration-tests/T8-packet-loss.md.
"""

import glob
import math
import os

import h5py
import numpy as np

import check_phases
from kit import (
    NUM_TAPS,
    audit,
    decode_int4x2,
    find_stages,
    frame_seq_nums,
    frb3_codes,
    has_bb,
    has_value,
    hfb_t_star,
    hfb_upchan_factors,
    large_seq,
    read_output,
    require,
    upchan_t_star,
    value,
)

RUNS = ["reference", "sparse", "dense"]
SPARSE_RATE = 1e-4
DENSE_RATE = 1e-2
LOSS_SEED = 8
# Frames written: masks and N2 counts, upchannelized voltages and BB, FRB PL mask, FRB beams
MASK_FRAMES = 8
DATA_FRAMES = 4
FRB_MASK_FRAMES = 2
FRB_FRAMES = 1
# Frames covered by the loss prediction beyond the end of the first N2 integration
PREDICTED_MARGIN_FRAMES = 6
# float32 rounding of the visibilities, relative to |Σ V n| of both runs
N2_TOLERANCE = 1e-6


def t_star(config):
    return upchan_t_star(config)


def frb_factors(config):
    """The upchannelization factors of the FRB chain, with U = 1 where it has one."""
    return [
        U
        for U in [1] + config["upchan_factors"]
        if has_value(config, f"upchan_U{U}_min_channel")
    ]


def predicted_frames(config):
    """The frames from seq0 to the end of the first N2 integration, and a margin."""
    seq0 = large_seq(config)
    T = value(config, "num_times")
    length = find_stages(config, "N2Accumulate")[0][
        "num_subintegrations_per_bin"
    ] * value(config, "n2k_samples_per_data_set")
    end = -(-seq0 // length) * length + length
    return (end - seq0) // T + PREDICTED_MARGIN_FRAMES


def frb_frame_length(config):
    return value(config, "frb_num_output_times") * value(
        config, "frb_downsampling_factor"
    )


def loss_boxes(config):
    """Lost FPGA samples [begin, end): inside the first T* samples, at the start and the end of
    a frame, across a frame boundary, across the boundary of the first two FRB frames, and
    single sample pairs at every even offset within a packet, so that window edges are resolved
    to one sample for every U."""
    seq0 = large_seq(config)
    T = value(config, "num_times")
    frb = t_star(config) + frb_frame_length(config)
    boxes = [
        (t_star(config) - 16, t_star(config)),
        (T, T + 32),
        (2 * T - 48, 2 * T),
        (3 * T - 32, 3 * T + 48),
        (frb - 16, frb + 16),
    ]
    boxes += [(T + T // 2 + 514 * k, T + T // 2 + 514 * k + 2) for k in range(1, 8)]
    return [[seq0 + begin, seq0 + end] for begin, end in boxes]


def variables(tel, config, run):
    var = check_phases.variables(tel, config, "offgrid")
    del var["outputs"], var["write_frames"]
    var["noise_period_frames"] = 1
    var["do_fringestop"] = False
    masks = ["pl_mask", "pl_expanded_mask"] + ["bb_pl_mask"] * has_bb(config)
    masks += [f"{chain}_expanded_pl_mask" for chain, _, _, _ in upchan_chains(config)]
    data = [f"{chain}_voltage" for chain, _, _, _ in upchan_chains(config)]
    data += ["bb_beams"] * has_bb(config)
    frb = ["frb1_beams", "frb2_beams", "frb3_beams", "frb3_beams_offsetscale"]
    write_frames = {name: MASK_FRAMES for name in masks}
    write_frames.update({name: DATA_FRAMES for name in data})
    write_frames.update({name: FRB_FRAMES for name in frb})
    write_frames["frb_pl_mask"] = FRB_MASK_FRAMES
    if run != "dense":
        write_frames["n2k_counts"] = MASK_FRAMES
        write_frames["n2k_plcounts"] = MASK_FRAMES
        write_frames["n2_vis"] = value(config, "num_frequencies")
    if run == "reference":
        write_frames["voltage"] = 1
    else:
        var["loss_rate"] = SPARSE_RATE if run == "sparse" else DENSE_RATE
        var["loss_seed"] = LOSS_SEED
        var["loss_boxes"] = loss_boxes(config)
        var["lost_voltage"] = "zero" if run == "sparse" else "nodata"
    var["outputs"] = list(write_frames)
    var["write_frames"] = write_frames
    return var


def splitmix64(x):
    """The splitmix64 finalizer on uint64 arrays (lib/testing/testSkySignal.hpp)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


class Loss:
    """The injected loss of a run (testPacketLoss in lib/testing/testSkySignal.hpp), per pair of
    FPGA samples (the PL mask resolution) in [seq0, seq0 + num_samples) and per group of 4
    frequencies."""

    def __init__(self, var, seq0, num_samples, num_freqs):
        self.seq0 = seq0
        pair = seq0 // 2 + np.arange(num_samples // 2)
        p = (pair // 8).astype(np.uint64)[:, None]  # packet: 16 samples
        q = np.arange(num_freqs // 4, dtype=np.uint64)[None, :]
        seed = splitmix64(np.array([var.get("loss_seed", 1)], dtype=np.uint64))
        h = splitmix64(((p << np.uint64(16)) | q) ^ seed)
        lost = h < np.uint64(int(math.ldexp(var.get("loss_rate", 0), 64)))
        for begin, end in var.get("loss_boxes", []):
            lost[max((begin - seq0) // 2, 0) : max((end - seq0) // 2, 0)] = True
        self.pairs = lost  # [pair, F4]
        self.cum = np.concatenate(
            [np.zeros((1, lost.shape[1]), dtype=np.int64), np.cumsum(lost, axis=0)]
        )
        self.freq4 = np.arange(num_freqs) // 4

    def lost(self, seq):
        """Whether FPGA samples `seq` are lost: [..., F]."""
        return self.pairs[(np.asarray(seq) - self.seq0) // 2][..., self.freq4]

    def num_lost(self, begin, end):
        """The number of lost samples in [begin, end) (even): [..., F]."""
        i = (np.asarray(begin) - self.seq0) // 2
        j = (np.asarray(end) - self.seq0) // 2
        return 2 * (self.cum[j] - self.cum[i])[..., self.freq4]

    def any_lost(self, begin, end):
        """Whether any sample in [begin, end) is lost: [..., F]."""
        i = (np.asarray(begin) - self.seq0) // 2
        j = (np.asarray(end) - 1 - self.seq0) // 2 + 1
        return (self.cum[j] - self.cum[i] > 0)[..., self.freq4]


def bits(data):
    """The bits of uint1x8 bytes along the last axis, least significant first."""
    return np.unpackbits(np.asarray(data, dtype=np.uint8), axis=-1, bitorder="little")


def mask_line(name, measured, expected, failures):
    """Compare a mask (1 = valid) with the expected validity, broadcast over the inputs."""
    measured = np.asarray(measured, dtype=bool)
    expected = np.broadcast_to(expected, measured.shape)
    wrong = measured != expected
    line = (
        f"{name}: {np.count_nonzero(~expected)} of {expected.size} invalid as predicted, "
        f"{np.count_nonzero(~measured)} measured, {np.count_nonzero(wrong)} differ"
    )
    if wrong.any():
        line += f"  <- first at index {np.argwhere(wrong)[0].tolist()}"
        failures.append(name)
    return line


def upchan_chains(config):
    """(output prefix, U, prefix of the channel range keys, output offset T*) of every
    upchannelizer: the FRB chain and the HFB chain (CHIME)."""
    chains = [
        (f"upchan_U{U}", U, f"upchan_U{U}", t_star(config))
        for U in config["upchan_factors"]
    ]
    chains += [
        (f"upchan_hfb_U{U}", U, f"upchan_hfb_upchan_U{U}", hfb_t_star(config, U))
        for U in hfb_upchan_factors(config)
    ]
    return chains


def upchan_channels(config, U, keys=None):
    """The local coarse channels [Fmin, Fmax) of upchannelization factor U (channel range keys
    `<keys>_min_channel`, default `upchan_U<U>`)."""
    keys = keys or f"upchan_U{U}"
    return (
        value(config, f"{keys}_min_channel"),
        value(config, f"{keys}_max_channel"),
    )


def upchan_valid(loss, config, U, labels, keys=None):
    """Validity [Tbar, channel] of the upchannelized samples with labels `labels`: the window
    [L - (M-1) U/2, L - (M-1) U/2 + M U) must not contain lost samples."""
    fmin, fmax = upchan_channels(config, U, keys)
    begin = np.asarray(labels) - (NUM_TAPS - 1) * U // 2
    return ~loss.any_lost(begin, begin + NUM_TAPS * U)[..., fmin:fmax]


def frb_valid(loss, config, labels):
    """Validity [Ttilde, Fbar] of the FRB samples with labels `labels`. An FRB1 sample sums
    Td = 192/U upchannelized samples, so it depends on the input samples
    [L - (M-1) U/2, L + 192 + (M-1) U/2) for U > 1, and on [L, L + 192) for U = 1."""
    tds = value(config, "frb_downsampling_factor")
    labels = np.asarray(labels)
    num_out = max(
        value(config, f"upchan_U{U}_max_output_channel") for U in frb_factors(config)
    )
    valid = np.zeros((labels.shape[0], num_out), dtype=bool)
    for U in frb_factors(config):
        fmin, fmax = upchan_channels(config, U)
        lo = value(config, f"upchan_U{U}_min_output_channel")
        hi = value(config, f"upchan_U{U}_max_output_channel")
        require(hi - lo == U * (fmax - fmin), f"U={U}: inconsistent output channels")
        reach = (NUM_TAPS - 1) * U // 2 if U > 1 else 0
        lost = loss.any_lost(labels - reach, labels + tds + reach)[..., fmin:fmax]
        valid[:, lo:hi] = np.repeat(~lost, U, axis=1)
    return valid


def check_masks(out_dir, config, loss, results, failures):
    seq0 = loss.seq0
    ts = t_star(config)

    # Input PL mask [T2hi64, F4, P, D8, T2lo64]: bit j of a row covers samples 2j and 2j + 1
    out = read_output(out_dir, "pl_mask")
    audit(out, first_seq=seq0)
    seq = out.fpga_seq_num + 128 * np.arange(out.data.shape[0])[:, None]
    seq = seq + 2 * np.arange(64)[None, :]  # [T2hi64, 64]
    expected = ~loss.lost(seq)[..., ::4]  # [T2hi64, 64, F4]
    expected = expected.transpose(0, 2, 1)[:, :, None, None, :]
    results.append(mask_line("pl_mask", bits(out.data), expected, failures))

    # Expanded PL mask [Thi64, F, P, D8, Tlo64], one bit per sample
    out = read_output(out_dir, "pl_expanded_mask")
    audit(out, first_seq=seq0)
    seq = out.fpga_seq_num + 64 * np.arange(out.data.shape[0])[:, None]
    seq = seq + np.arange(64)[None, :]
    expected = ~loss.lost(seq).transpose(0, 2, 1)[:, :, None, None, :]
    results.append(mask_line("pl_expanded_mask", bits(out.data), expected, failures))

    # BB PL mask [Thi64, F, P, Tlo64]: the AND over dishes of the expanded mask
    if has_bb(config):
        out = read_output(out_dir, "bb_pl_mask")
        audit(out, first_seq=seq0)
        results.append(
            mask_line("bb_pl_mask", bits(out.data), expected[:, :, :, 0], failures)
        )

    # Upchannelized PL masks [Thi64, F, P, D8, Tlo64]: bit tbar has the label L = seq + U tbar
    for chain, U, keys, chain_t_star in upchan_chains(config):
        name = f"{chain}_expanded_pl_mask"
        out = read_output(out_dir, name)
        audit(out, first_seq=seq0 + chain_t_star)
        labels = out.fpga_seq_num + U * (
            64 * np.arange(out.data.shape[0])[:, None] + np.arange(64)[None, :]
        )
        valid = upchan_valid(loss, config, U, labels, keys)  # [Thi64, 64, channel]
        measured = bits(out.data[:, : valid.shape[2]])
        expected = valid.transpose(0, 2, 1)[:, :, None, None, :]
        results.append(mask_line(name, measured, expected, failures))

    # FRB PL mask [Ttilde, Fbar]
    out = read_output(out_dir, "frb_pl_mask")
    audit(out, first_seq=seq0 + ts)
    labels = out.fpga_seq_num + out.time_downsampling * np.arange(out.data.shape[0])
    expected = frb_valid(loss, config, labels)
    measured = out.data[:, : expected.shape[1]]
    results.append(mask_line("frb_pl_mask", measured, expected, failures))


def equal_line(name, measured, reference, valid, failures):
    """Valid samples must equal the reference bit for bit. Report how many invalid samples
    differ, to show that the lost voltages reach them."""
    valid = np.broadcast_to(valid, measured.shape)
    differ = measured != reference
    wrong = np.count_nonzero(differ & valid)
    line = (
        f"{name}: {wrong} of {np.count_nonzero(valid)} valid values differ from the "
        f"reference; {np.count_nonzero(differ & ~valid)} of {np.count_nonzero(~valid)} "
        "invalid values differ"
    )
    if wrong:
        line += f"  <- first at index {np.argwhere(differ & valid)[0].tolist()}"
        failures.append(name)
    return line


def check_data(out_dir, ref_dir, config, loss, results, failures):
    seq0 = loss.seq0
    ts = t_star(config)

    # Upchannelized voltages [Tbar, Fbar, P, D], Fbar = U channel + u
    for chain, U, keys, chain_t_star in upchan_chains(config):
        name = f"{chain}_voltage"
        out = read_output(out_dir, name)
        ref = read_output(ref_dir, name)
        audit(out, upchan_factor=U, first_seq=seq0 + chain_t_star)
        labels = out.fpga_seq_num + U * np.arange(out.data.shape[0])
        valid = np.repeat(upchan_valid(loss, config, U, labels, keys), U, axis=1)
        num = valid.shape[1]
        results.append(
            equal_line(
                name,
                out.data[:, :num],
                ref.data[:, :num],
                valid[:, :, None, None],
                failures,
            )
        )

    # BB beams [Thi, B, F, P, T], one Thi per frame
    if has_bb(config):
        out = read_output(out_dir, "bb_beams")
        ref = read_output(ref_dir, "bb_beams")
        audit(out, first_seq=seq0)
        seq = frame_seq_nums(out)[:, None] + np.arange(out.data.shape[-1])[None, :]
        valid = ~loss.lost(seq).transpose(0, 2, 1)  # [Thi, F, T]
        results.append(
            equal_line(
                "bb_beams", out.data, ref.data, valid[:, None, :, None], failures
            )
        )

    # FRB1 beams [Ttilde, Fbar, beamQ, beamP], FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256]
    out = read_output(out_dir, "frb1_beams")
    ref = read_output(ref_dir, "frb1_beams")
    audit(out, first_seq=seq0 + ts)
    labels = out.fpga_seq_num + out.time_downsampling * np.arange(out.data.shape[0])
    valid = frb_valid(loss, config, labels)
    num = valid.shape[1]
    results.append(
        equal_line(
            "frb1_beams",
            out.data[:, :num].view(np.uint16),
            ref.data[:, :num].view(np.uint16),
            valid[:, :, None, None],
            failures,
        )
    )

    out = read_output(out_dir, "frb2_beams")
    ref = read_output(ref_dir, "frb2_beams")
    audit(out, first_seq=seq0 + ts)
    tds = value(config, "frb_downsampling_factor")
    labels = out.fpga_seq_num + tds * np.arange(out.data.shape[3])
    valid = frb_valid(loss, config, labels).T  # [Fbar, Ttildelo256]
    num = valid.shape[0]
    results.append(
        equal_line(
            "frb2_beams",
            out.data[:, :, :num].view(np.uint16),
            ref.data[:, :, :num].view(np.uint16),
            valid[None, None],
            failures,
        )
    )
    check_frb3(out_dir, ref_dir, out.data[:, :, :num], valid, results, failures)


def frb3_values(out_dir, num):
    """FRB3 codes q [Ttildehi256, R, Fbar, Ttildelo256], the dequantized values offset + scale q,
    their scales and offsets, and the codes for no data and the lowest and highest values (see
    kit.frb3_codes)."""
    q, offset, scale, nodata, qmin, qmax = frb3_codes(out_dir, num)
    return q, offset + scale * q, scale, offset, (nodata, qmin, qmax)


def check_frb3(out_dir, ref_dir, frb2, valid, results, failures):
    """Invalid samples have the no-data code. Valid samples below saturation (4 bits) in both runs
    encode the same FRB2 value, so their dequantized values differ by at most half a step of each
    run plus the float16 allowance of T3."""
    num = frb2.shape[2]
    q, xhat, scale, offset, (nodata, qmin, qmax) = frb3_values(out_dir, num)
    q_ref, xhat_ref, scale_ref, offset_ref, _ = frb3_values(ref_dir, num)
    x = frb2.astype(np.float64)
    valid = np.broadcast_to(valid[None, None], q.shape)
    wrong_nodata = np.count_nonzero((q == nodata) != ~valid)
    if nodata == -8:
        use = valid & (np.abs(q) < 7) & (np.abs(q_ref) < 7)
    else:
        use = valid
    allowance = 2.0 ** -9 * (2 * np.abs(x) + np.abs(offset) + np.abs(offset_ref))
    allowance = allowance + 2.0 ** -9 * (scale + scale_ref)
    excess = np.abs(xhat - xhat_ref) - (scale + scale_ref) / 2 - allowance
    num_beyond = np.count_nonzero(use & (excess > 0))
    step = np.maximum(scale, scale_ref)
    worst = float((np.abs(xhat - xhat_ref) / np.where(step > 0, step, 1))[use].max())
    line = (
        f"frb3_beams: {np.count_nonzero(q == nodata)} values {nodata}, {wrong_nodata} not matching the "
        f"mask; {num_beyond} of {np.count_nonzero(use)} valid unsaturated values beyond one "
        f"step of the reference (max {worst:.3f} steps)"
    )
    if wrong_nodata or num_beyond:
        line += "  <- wrong"
        failures.append("frb3_beams")
    results.append(line)


def check_n2_counts(out_dir, config, loss, results, failures):
    """Per correlation frame and frequency: n2k_plcounts is the number of lost samples, and
    every n2k_counts entry the number of valid samples (no RFI flags)."""
    T = value(config, "num_times")
    plcounts = read_output(out_dir, "n2k_plcounts")
    counts = read_output(out_dir, "n2k_counts")
    audit(plcounts, first_seq=loss.seq0)
    audit(counts, first_seq=loss.seq0)
    seq = plcounts.fpga_seq_num + T * np.arange(plcounts.data.shape[0])
    lost = loss.num_lost(seq, seq + T)  # [frame, F]
    wrong = plcounts.data != lost
    line = (
        f"n2k_plcounts: {lost.sum()} lost samples predicted in {len(seq)} frames, "
        f"{plcounts.data.sum()} counted, {np.count_nonzero(wrong)} entries differ"
    )
    if wrong.any():
        failures.append("n2k_plcounts")
    results.append(line)
    seq = counts.fpga_seq_num + T * np.arange(counts.data.shape[0])
    valid = T - loss.num_lost(seq, seq + T)
    wrong = counts.data != valid[:, :, None, None, None]
    line = f"n2k_counts: {np.count_nonzero(wrong)} of {wrong.size} entries differ from T - lost"
    if wrong.any():
        line += f"  <- first at index {np.argwhere(wrong)[0].tolist()}"
        failures.append("n2k_counts")
    results.append(line)


def read_n2(out_dir, config):
    """Visibilities [F, products] and attributes [F] of the first integration, by local
    frequency."""
    channels = list(value(config, "frequency_channels"))
    vis = [None] * len(channels)
    attrs = [None] * len(channels)
    for filename in sorted(glob.glob(os.path.join(out_dir, "n2_vis.*.h5"))):
        with h5py.File(filename, "r") as f:
            f_local = channels.index(int(f.attrs["freq_id"]))
            require(vis[f_local] is None, f"n2_vis: two integrations of {f_local}")
            v = f["vis"][()]
            vis[f_local] = v[:, 0].astype(np.float64) + 1j * v[:, 1]
            attrs[f_local] = dict(f.attrs)
    require(all(v is not None for v in vis), "n2_vis: frequencies missing")
    return np.array(vis), attrs


def check_n2(out_dir, ref_dir, config, loss, results, failures):
    """The lost samples of the integration [start, start + length) must be counted as lost, the
    rest as valid, and Σ V n_valid must equal the reference sum minus the lost samples' sum."""
    vis, attrs = read_n2(out_dir, config)
    vis_ref, attrs_ref = read_n2(ref_dir, config)
    start = np.array([int(a["fpga_start_tick"]) for a in attrs])
    length = np.array([int(a["frame_length_fpga_ticks"]) for a in attrs])
    require(
        np.all(start == start[0]) and np.all(length == length[0]),
        "n2_vis: frequencies integrate different samples",
    )
    require(
        [int(a["fpga_start_tick"]) for a in attrs_ref] == start.tolist(),
        "n2_vis: the runs integrate different samples",
    )
    start, length = int(start[0]), int(length[0])
    lost = loss.num_lost(start, start + length)  # [F]
    n_valid = np.array([int(a["n_valid_fpga_ticks"]) for a in attrs])
    n_pl = np.array([int(a["n_pl_fpga_ticks"]) for a in attrs])
    n_rfi = np.array([int(a["n_rfi_fpga_ticks"]) for a in attrs])
    n_ref = np.array([int(a["n_valid_fpga_ticks"]) for a in attrs_ref])
    n_rfi_ref = np.array([int(a["n_rfi_fpga_ticks"]) for a in attrs_ref])
    line = (
        f"n2_vis counts: integration [{start}, {start + length}), {lost.sum()} lost samples "
        f"predicted, n_pl {n_pl.sum()}, n_valid {n_valid.min()} to {n_valid.max()} "
        f"(reference {n_ref.min()} to {n_ref.max()}), n_rfi {n_rfi.sum()} "
        f"(reference {n_rfi_ref.sum()})"
    )
    if (
        np.any(n_pl != lost)
        or np.any(n_valid != length - lost)
        or np.any(n_ref != length)
        or n_rfi.any()
        or n_rfi_ref.any()
    ):
        line += "  <- wrong"
        failures.append("n2_vis counts")
    results.append(line)

    # Σ_lost E_a conj(E_b) from the reference voltages, which repeat every frame
    T = value(config, "num_times")
    voltage = read_output(ref_dir, "voltage")
    audit(voltage, first_seq=loss.seq0)
    require(voltage.data.shape[0] == T, "voltage: expected one frame")
    seq = start + np.arange(length)
    lost_samples = loss.lost(seq)  # [t, F]
    ia, ib = np.triu_indices(2 * value(config, "num_dishes"))  # element a = p D + d
    worst = 0.0
    worst_abs = 0.0
    largest_lost = 0.0
    num_beyond = 0
    for f in range(vis.shape[0]):
        t = (seq[lost_samples[:, f]] - loss.seq0) % T
        e = decode_int4x2(voltage.data[t, f]).reshape(len(t), -1)
        e = e.astype(np.complex128)
        s_lost = (e.T @ e.conj())[ia, ib]
        s_ref = vis_ref[f] * n_ref[f]
        s = vis[f] * n_valid[f]
        delta = np.abs(s - (s_ref - s_lost))
        scale = np.abs(s_ref) + np.abs(s)
        num_beyond += np.count_nonzero(delta > N2_TOLERANCE * scale)
        worst = max(worst, float((delta / np.where(scale > 0, scale, 1)).max()))
        worst_abs = max(worst_abs, float(delta.max()))
        largest_lost = max(largest_lost, float(np.abs(s_lost).max()))
    line = (
        f"n2_vis: Σ V n_valid vs reference minus lost samples: max |Δ| {worst_abs:.2f}, "
        f"relative {worst:.2e}; {num_beyond} products beyond {N2_TOLERANCE:g}; "
        f"max |Σ_lost E_a conj(E_b)| {largest_lost:.0f}"
    )
    if num_beyond:
        line += "  <- wrong"
        failures.append("n2_vis")
    results.append(line)


def check(runs):
    ref_dir, ref_var, config = runs["reference"]
    seq0 = ref_var["first_fpga_seq_num"]
    num_samples = predicted_frames(config) * value(config, "num_times")
    num_freqs = value(config, "num_frequencies")
    results = []
    failures = []
    for run, (out_dir, var, _) in runs.items():
        loss = Loss(var, seq0, num_samples, num_freqs)
        results.append(f"[{run}] {loss.pairs.mean():.4%} of the samples lost")
        check_masks(out_dir, config, loss, results, failures)
        if run != "reference":
            check_data(out_dir, ref_dir, config, loss, results, failures)
        if run != "dense":
            check_n2_counts(out_dir, config, loss, results, failures)
        if run == "sparse":
            check_n2(out_dir, ref_dir, config, loss, results, failures)
    require(
        not failures,
        f"{len(failures)} checks failed ({', '.join(failures)}):\n    "
        + "\n    ".join(results),
    )
    return results
