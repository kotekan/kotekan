"""T9r bad feed mask via REST: a POST to /updatable_config/bad_inputs must change the bad feed
masks once, at a mask frame boundary, in both mask streams at the same sample, and every
subsystem must follow the masks.

Input: as T9 (check_badfeeds.py), but the masks come from two bufferBadInputs stages, one per
lifetime (that of T9 for RFI, BB and FRB1, one frame for N2Accumulate), as
config/chord/pathfinder.j2 produces them. All inputs are good until `BAD_INPUTS` are posted, as
soon as kotekan's REST server answers, with the FPGA time of sample `START_FRAMES` frames and one
sample after the first as start time. bufferBadInputs applies an update by FPGA time at the next
lifetime boundary, so both streams must change at the same frame.

Checks:
- Each stream starts all good and changes exactly once, to `BAD_INPUTS`, at the first boundary at
  or after the start time.
- The two streams agree at every sample.
- BB phases, FRB1 weights, S0tilde and the N2 flags follow the written masks (as in T9).

CHIME has one stream (its N2Accumulate reads no mask, see check_badfeeds.py) and no BB; it posts
the bad inputs in cylinder order, as config/chime_upgrade_frb.j2, which for these inputs is the
mask order.

See docs/integration-tests/T9-bad-feed-mask.md.
"""

import numpy as np

import check_badfeeds
import check_phases
from check_badfeeds import (
    check_bb_phase,
    check_frb1_weights,
    lifetime_frames,
    n2_flags,
    n2_masked,
    read_n2,
    result_line,
)
from check_dishes import n2_bin_start
from check_packetloss import frb_factors
from kit import audit, frame_seq_nums, has_bb, large_seq, read_output, require, value

VARIANTS = ["rest"]
POST_DELAY = 0
START_FRAMES = 180
BAD_INPUTS = [5, 17, 70, 100]
MASK_FRAMES = 240
N2_INTEGRATIONS = 9


def variables(tel, config, variant):
    var = check_phases.variables(tel, config, "offgrid")
    T = value(config, "num_times")
    var["first_fpga_seq_num"] = check_badfeeds_first_seq(config)
    var["bf_source"] = "rest"
    var["bf_mask_lifetime_frames"] = lifetime_frames(config)
    lifetimes = MASK_FRAMES // lifetime_frames(config)
    write_frames = {"bf_mask": lifetimes}
    if n2_masked(config):
        write_frames["n2_bf_mask"] = MASK_FRAMES
    if has_bb(config):
        write_frames["bb_phase"] = lifetimes
    write_frames["rfi_S012tilde"] = MASK_FRAMES
    if n2_masked(config):
        write_frames["n2_vis"] = N2_INTEGRATIONS * value(config, "num_frequencies")
    for U in frb_factors(config):
        write_frames[f"frb1_U{U}_phase"] = lifetimes
    var["outputs"] = list(write_frames)
    var["write_frames"] = write_frames
    return var


def start_seq(config):
    return (
        check_badfeeds_first_seq(config) + START_FRAMES * value(config, "num_times") + 1
    )


def check_badfeeds_first_seq(config):
    return n2_bin_start(
        config, large_seq(config)
    ) - check_badfeeds.N2_LEAD_FRAMES * value(config, "num_times")


def posts(tel, config, variant):
    # The UNIX time of FPGA sample start_seq, as the telescope computes it.
    dt_ns = (
        1e9 * value(config, "num_samples_per_frame") / value(config, "adc_frequency")
    )
    start_time = (config["gps_time"]["frame0_nano"] + start_seq(config) * dt_ns) / 1e9
    payload = {"bad_inputs": BAD_INPUTS, "start_time": start_time, "update_id": "T9r"}
    return [(POST_DELAY, "/updatable_config/bad_inputs", payload)]


class WrittenMask:
    """A written mask stream, used in place of check_badfeeds.Schedule."""

    def __init__(self, out, config):
        self.seq = frame_seq_nums(out)
        self.seq0 = int(self.seq[0])
        self.lifetime = out.time_downsampling
        self.mask = out.data.reshape(len(self.seq), -1) != 0
        self.num_dishes = value(config, "num_dishes")
        self.num_inputs = self.mask.shape[1]

    def good(self, seq):
        index = (np.asarray(seq) - self.seq0) // self.lifetime
        require(
            np.all((index >= 0) & (index < len(self.seq))),
            "a written output extends beyond the written mask frames",
        )
        return self.mask[index]


def check_stream(out_dir, name, lifetime, config, results, failures):
    """The stream starts all good and changes once, to BAD_INPUTS. Returns the stream and the
    sequence number of the change."""
    out = read_output(out_dir, name)
    audit(out)
    stream = WrittenMask(out, config)
    require(stream.lifetime == lifetime, f"{name}: lifetime {stream.lifetime}")
    after = np.ones(stream.num_inputs, dtype=bool)
    after[BAD_INPUTS] = False
    changes = np.nonzero(np.any(stream.mask[1:] != stream.mask[:-1], axis=1))[0] + 1
    T = value(config, "num_times")
    boundary = value(config, "bf_mask_lifetime_in_samples")
    expected = -(-(start_seq(config) - stream.seq0) // boundary) * boundary // lifetime
    wrong = (
        list(changes) != [expected]
        or not stream.mask[0].all()
        or np.any(stream.mask[changes[0]] != after)
    )
    change_seq = int(stream.seq[changes[0]]) if len(changes) else None
    results.append(
        result_line(
            name,
            wrong,
            f"{len(stream.seq)} frames of {lifetime} samples, changes at frames "
            f"{[(int(stream.seq[c]) - stream.seq0) // T for c in changes]} after seq0, expected "
            f"{expected * lifetime // T} "
            f"(of {(int(stream.seq[-1]) + lifetime - stream.seq0) // T} written)",
            failures,
        )
    )
    return stream, change_seq


def check_s0tilde(out_dir, stream, results, failures):
    tilde = read_output(out_dir, "rfi_S012tilde")
    audit(tilde, first_seq=stream.seq0)
    ds = tilde.time_downsampling
    seq = tilde.fpga_seq_num + ds * np.arange(tilde.data.shape[0])
    n_good = stream.good(seq).sum(axis=1)
    wrong = tilde.data[:, :, 0] != (ds * n_good)[:, None]
    results.append(
        result_line(
            "rfi_S012tilde",
            np.count_nonzero(wrong),
            f"{len(seq)} blocks: S0tilde = {ds} × good inputs ({n_good.min()} to "
            f"{n_good.max()}) except {np.count_nonzero(wrong)} entries",
            failures,
        )
    )


def check(out_dir, var, config):
    T = value(config, "num_times")
    results = []
    failures = []
    bf, bf_change = check_stream(
        out_dir,
        "bf_mask",
        value(config, "bf_mask_lifetime_in_samples"),
        config,
        results,
        failures,
    )
    require(bf.seq0 == var["first_fpga_seq_num"], f"bf_mask starts at {bf.seq0}")
    if n2_masked(config):
        n2, n2_change = check_stream(
            out_dir, "n2_bf_mask", T, config, results, failures
        )
        require(n2.seq0 == var["first_fpga_seq_num"], f"n2_bf_mask starts at {n2.seq0}")
        use = n2.seq < bf.seq[-1] + bf.lifetime
        differ = np.any(n2.mask[use] != bf.good(n2.seq[use]), axis=1)
        line = (
            f"bf_mask and n2_bf_mask differ in {np.count_nonzero(differ)} of "
            f"{np.count_nonzero(use)} frames"
        )
        if bf_change is not None and n2_change is not None:
            line += (
                f" (n2_bf_mask changes {(n2_change - bf_change) // T:+d} frames later)"
            )
        results.append(result_line("mask streams", differ.any(), line, failures))

    if has_bb(config):
        check_bb_phase(out_dir, bf, results, failures)
    check_frb1_weights(out_dir, config, bf, results, failures)
    check_s0tilde(out_dir, bf, results, failures)
    if n2_masked(config):
        for start, entries in read_n2(out_dir).items():
            n2_flags(start, entries, config, n2, results, failures)
    require(
        not failures,
        f"{len(failures)} checks failed ({', '.join(failures)}):\n    "
        + "\n    ".join(results),
    )
    return results
