"""T1 timing: a time-limited signal must appear at the predicted FPGA sequence numbers in every
output.

Input: a constant voltage (real part 4, every dish, frequency and polarization) in the FPGA
samples [box_begin, box_end), zero elsewhere. The box starts mid-frame.
"""

import numpy as np

from kit import (
    audit,
    decode_int4x2,
    frame_seq_nums,
    power_centroid,
    read_output,
    require,
    used_channels,
)

UPCHAN_FACTORS = [2, 4, 8, 16, 32, 64]
NUM_TAPS = 4  # kotekan::upchan_default_num_taps


def variables(tel, config):
    num_times = config["num_times"]
    frb_frame = config["frb_num_output_times"] * config["frb_downsampling_factor"]
    num_frames = -(-frb_frame // num_times)
    box_begin = 3 * num_times + 1000
    outputs = ["voltage", "bb_beams", "frb1_beams", "frb2_beams"]
    outputs += [f"upchan_U{U}_voltage" for U in UPCHAN_FACTORS]
    write_frames = {name: num_frames for name in outputs}
    write_frames["frb1_beams"] = 1
    write_frames["frb2_beams"] = 1
    return {
        "input_kind": "box",
        "box_begin": box_begin,
        "box_end": box_begin + 3000,
        "outputs": outputs,
        "write_frames": write_frames,
    }


def check(out_dir, var, config):
    box_begin, box_end = var["box_begin"], var["box_end"]
    centre = (box_begin + box_end - 1) / 2
    channels = config["frequency_channels"]
    t_star = (NUM_TAPS - 1) * config["max_upchannelization_factor"] // 2
    tds = config["frb_downsampling_factor"]
    results = []

    # The input itself, to validate the generator and the labelling
    voltage = read_output(out_dir, "voltage")
    audit(voltage, first_seq=0, frequency_channels=channels)
    labels = voltage.fpga_seq_num + np.arange(voltage.data.shape[0])
    nonzero = labels[(voltage.data != 0x88).any(axis=(1, 2, 3))]
    require(
        nonzero.min() == box_begin
        and nonzero.max() == box_end - 1
        and len(nonzero) == box_end - box_begin,
        f"voltage: signal in [{nonzero.min()}, {nonzero.max()}], expected [{box_begin}, {box_end - 1}]",
    )
    results.append(f"voltage: signal in [{box_begin}, {box_end})")

    # Baseband beams: [Thi, B, F, P, T], per-frame files, one Thi per frame
    bb = read_output(out_dir, "bb_beams")
    audit(bb, first_seq=0, frequency_channels=channels)
    seqs = frame_seq_nums(bb)
    num_times = bb.data.shape[-1]
    labels = (seqs[:, None] + np.arange(num_times)[None, :]).reshape(-1)
    power = (np.abs(decode_int4x2(bb.data)) ** 2).sum(axis=(1, 2, 3)).reshape(-1)
    nonzero = labels[power > 0]
    require(
        nonzero.min() == box_begin and nonzero.max() == box_end - 1,
        f"bb_beams: signal in [{nonzero.min()}, {nonzero.max()}], expected [{box_begin}, {box_end - 1}]",
    )
    results.append(f"bb_beams: signal in [{nonzero.min()}, {nonzero.max()}]")

    # Upchannelized voltages: [Tbar, Fbar, P, D]; a sample's label is the beginning of the
    # centred upchannelized sample, which covers U FPGA samples
    for U in UPCHAN_FACTORS:
        name = f"upchan_U{U}_voltage"
        ebar = read_output(out_dir, name)
        audit(ebar, upchan_factor=U, first_seq=t_star, frequency_channels=channels)
        labels = ebar.fpga_seq_num + U * np.arange(ebar.data.shape[0]) + (U - 1) / 2
        power = (np.abs(decode_int4x2(used_channels(ebar, 1))) ** 2).sum(axis=(1, 2, 3))
        mean, support = power_centroid(labels, power)
        require(
            abs(mean - centre) <= U / 4,
            f"{name}: power centroid at {mean:.1f}, expected {centre:.1f} +- {U / 4}",
        )
        reach = NUM_TAPS * U
        require(
            support.min() >= box_begin - reach and support.max() <= box_end + reach,
            f"{name}: signal in [{support.min()}, {support.max()}] reaches beyond the PFB window",
        )
        results.append(f"{name}: centroid {mean:.1f} (expected {centre:.1f})")

    # FRB beams: FRB1 [Ttilde, Fbar, beamQ, beamP], FRB2 [Ttildehi256, R, Fbar, Ttildelo256]
    frb1 = read_output(out_dir, "frb1_beams")
    audit(frb1, first_seq=t_star)
    labels = frb1.fpga_seq_num + tds * np.arange(frb1.data.shape[0]) + (tds - 1) / 2
    power = used_channels(frb1, 1).astype(np.float64).sum(axis=(1, 2, 3))
    mean, _ = power_centroid(labels, power)
    require(
        abs(mean - centre) <= tds / 4,
        f"frb1_beams: power centroid at {mean:.1f}, expected {centre:.1f} +- {tds / 4}",
    )
    results.append(f"frb1_beams: centroid {mean:.1f}")

    frb2 = read_output(out_dir, "frb2_beams")
    audit(frb2, first_seq=t_star)
    seqs = frame_seq_nums(frb2)
    num_lo = frb2.data.shape[-1]
    labels = (seqs[:, None] + tds * np.arange(num_lo)[None, :]).reshape(-1) + (
        tds - 1
    ) / 2
    power = used_channels(frb2, 2).astype(np.float64).sum(axis=(1, 2)).reshape(-1)
    mean, _ = power_centroid(labels, power)
    require(
        abs(mean - centre) <= tds / 4,
        f"frb2_beams: power centroid at {mean:.1f}, expected {centre:.1f} +- {tds / 4}",
    )
    results.append(f"frb2_beams: centroid {mean:.1f}")

    return results
