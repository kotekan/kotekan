"""Shared helpers for the X-engine integration tests.

Reads the HDF5 files written by ``hdf5FileWrite`` (see
docs/sphinx/user/file_formats/hdf5_frames.rst), decodes kotekan's element types, and labels
samples with their FPGA sequence numbers.
"""

import glob
import math
import os

import h5py
import hdf5plugin  # noqa: F401 (registers the bitshuffle filter used by hdf5FileWrite)
import numpy as np


class CheckFailed(Exception):
    pass


def require(condition, message):
    """Fail the check with `message` unless `condition` holds."""
    if not condition:
        raise CheckFailed(message)


class Output:
    """One written buffer: the frames' data, dimension names and per-frame attributes.

    `data` holds all frames concatenated along axis 0. For a single file the attributes are
    stored once; for per-frame files `frame_attrs` lists every frame's attributes.
    """

    def __init__(self, name, data, attrs, frame_attrs):
        self.name = name
        self.data = data
        self.attrs = attrs
        self.frame_attrs = frame_attrs
        self.dim_names = [str(n) for n in attrs["dim_names"]]
        self.type = str(attrs["type"])

    def attr(self, key):
        return self.attrs[key]

    @property
    def fpga_seq_num(self):
        return int(self.attrs["fpga_seq_num"])

    @property
    def time_downsampling(self):
        return int(self.attrs["time_downsampling_fpga"])


def _str(value):
    return value.decode() if isinstance(value, bytes) else value


def _read_attrs(dset):
    attrs = {}
    for key, value in dset.attrs.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
            value = [_str(v) for v in value]
        attrs[key] = _str(value)
    return attrs


def read_output(out_dir, name):
    """Read output `name` from `out_dir`, as a single file or as per-frame files."""
    single = os.path.join(out_dir, name + ".h5")
    if os.path.exists(single):
        files = [single]
    else:
        files = sorted(glob.glob(os.path.join(out_dir, name + ".[0-9]*.h5")))
    require(files, f"{name}: no output files in {out_dir}")
    datas = []
    frame_attrs = []
    for filename in files:
        with h5py.File(filename, "r") as f:
            dset = f[name]
            datas.append(dset[()])
            frame_attrs.append(_read_attrs(dset))
    data = np.concatenate(datas, axis=0)
    return Output(name, data, frame_attrs[0], frame_attrs)


def frame_files(out_dir, name):
    """The per-frame files of output `name` by frame number, the number of the frame among all
    frames the writer received (hdf5FileWrite names them `<name>.<frame>.h5`)."""
    files = glob.glob(os.path.join(out_dir, name + ".[0-9]*.h5"))
    return dict(sorted((int(f.rsplit(".", 2)[1]), f) for f in files))


def read_frame(filename, name):
    """One per-frame file of output `name`, as an `Output` of one frame."""
    with h5py.File(filename, "r") as f:
        dset = f[name]
        attrs = _read_attrs(dset)
        return Output(name, dset[()], attrs, [attrs])


def used_channels(output, axis):
    """`output.data` restricted to the frequency slots listed in `coarse_freq`. Buffers are
    sized for the largest channel count; the remaining slots hold no data."""
    num = len(np.atleast_1d(output.attr("coarse_freq")))
    return np.take(output.data, np.arange(num), axis=axis)


def decode_int4x2(data):
    """Decode int4x2_swapped_withoffset bytes: real in the high nibble, imaginary in the low,
    both offset by 8 (so 0x00 is -8-8i, the no-data value)."""
    data = np.asarray(data, dtype=np.uint8)
    re = (data >> 4).astype(np.int8) - 8
    im = (data & 0xF).astype(np.int8) - 8
    return re + 1j * im


def time_axis(output, axis, step):
    """FPGA sequence numbers of the samples along time `axis` of a single-file output with
    a split time axis: axis 0 advances by `time_downsampling_fpga`, `axis` by `step`."""
    shape = output.data.shape
    seq0 = output.fpga_seq_num
    slow = seq0 + np.arange(shape[0]) * output.time_downsampling
    if axis == 0:
        return slow
    fast = np.arange(shape[axis]) * step
    return (slow[:, None] + fast[None, :]).reshape(-1)


def frame_seq_nums(output):
    """The first FPGA sequence number of every frame (per-frame files only)."""
    return np.array([int(a["fpga_seq_num"]) for a in output.frame_attrs])


def audit(output, upchan_factor=None, first_seq=None, frequency_channels=None):
    """Metadata checks that apply to every output."""
    name = output.name
    if output.type.startswith("float"):
        require(np.all(np.isfinite(output.data)), f"{name}: contains NaN or Inf")
    if len(output.frame_attrs) > 1:
        seqs = frame_seq_nums(output)
        step = (
            output.data.shape[0] // len(output.frame_attrs) * output.time_downsampling
        )
        require(
            np.all(np.diff(seqs) == step),
            f"{name}: frames are not contiguous: fpga_seq_num {seqs.tolist()}",
        )
    if first_seq is not None:
        require(
            output.fpga_seq_num == first_seq,
            f"{name}: first fpga_seq_num is {output.fpga_seq_num}, expected {first_seq}",
        )
    if "coarse_freq" in output.attrs and frequency_channels is not None:
        coarse = np.atleast_1d(output.attr("coarse_freq"))
        valid = coarse[coarse >= 0]
        require(
            np.all(np.isin(valid, frequency_channels)),
            f"{name}: coarse_freq {sorted(set(valid.tolist()) - set(frequency_channels))} "
            "are not configured channels",
        )
    if upchan_factor is not None and "freq_upchan_factor" in output.attrs:
        factor = np.atleast_1d(output.attr("freq_upchan_factor"))
        index = np.atleast_1d(output.attr("freq_upchan_index"))
        used = factor > 0
        require(
            np.all(factor[used] == upchan_factor),
            f"{name}: freq_upchan_factor {sorted(set(factor[used].tolist()))} != {upchan_factor}",
        )
        require(
            np.all((0 <= index[used]) & (index[used] < upchan_factor)),
            f"{name}: freq_upchan_index out of range [0, {upchan_factor})",
        )


def power_centroid(labels, power):
    """The power-weighted mean of the sample labels, and the labels with non-zero power."""
    total = power.sum()
    require(total > 0, "no signal power")
    return float((labels * power).sum() / total), labels[power > 0]


C0 = 299792458.0  # speed of light [m/s]


NUM_TAPS = 4  # kotekan::upchan_default_num_taps


def output_alignment(config):
    """The upchannelizer output alignment `upchan_output_alignment` (default 1, CHIME: 3 frames),
    in FPGA samples."""
    return (
        value(config, "upchan_output_alignment")
        if "upchan_output_alignment" in config
        else 1
    )


def upchan_t_star(config):
    """The upchannelizer output offset T* = (taps - 1) U_max / 2, rounded up to a multiple of
    the output alignment (`upchan_output_offset`)."""
    align = output_alignment(config)
    t_star = (NUM_TAPS - 1) * value(config, "max_upchannelization_factor") // 2
    return -(-t_star // align) * align


def large_seq(config):
    """A large start FPGA sequence number. N2Accumulate needs frames to start at multiples of
    the frame length, and the voltage stream must start at a multiple of the upchannelizer
    output alignment; 7 frames past 2**33, rounded up to both, aligns it with no lifetime or
    integration."""
    step = math.lcm(value(config, "num_times"), output_alignment(config))
    return -(-(2 ** 33 + 7 * value(config, "num_times")) // step) * step


def value(config, key, block=None):
    """A top-level config value, or one in the top-level `block`; kotekan arithmetic expressions
    such as `5.0 / 4` that refer to other top-level values are evaluated. A key that is not at
    the top level is looked up in the top-level blocks (CHIME's `FRB`), and expressions there
    refer to that block first."""
    scope = config if block is None else config[block]
    if key not in scope and block is None:
        blocks = [b for b, v in config.items() if isinstance(v, dict) and key in v]
        require(blocks, f"config: no value {key}")
        return value(config, key, blocks[0])
    v = scope[key]
    if not isinstance(v, str):
        return v

    class Names(dict):
        def __missing__(self, name):
            if block is not None and name in config[block]:
                return value(config, name, block)
            return value(config, name)

    return eval(v, {"__builtins__": {}}, Names())


def has_bb(config):
    """Whether the config has the baseband beamformer (CHIME has none)."""
    return "host_bb_beams_buffer" in config


def frb1_swap_MN(config):
    """Whether FRB1 beam axis P (M) runs along grid y instead of x (CHIME)."""
    return value(config, "frb1_num_beams_P") != 2 * value(config, "num_dishes_x")


def has_value(config, key):
    """Whether `key` is a top-level value or one in a top-level block (see `value`)."""
    return key in config or any(
        isinstance(v, dict) and key in v for v in config.values()
    )


def has_command(config, name):
    """Whether any cudaProcess stage of the config runs the GPU command `name`."""
    if isinstance(config, dict):
        return config.get("name") == name or any(
            has_command(v, name) for v in config.values()
        )
    if isinstance(config, list):
        return any(has_command(v, name) for v in config)
    return False


def find_stages(config, kotekan_stage):
    """The config blocks of every stage of type `kotekan_stage`, at any depth."""
    found = []
    for v in config.values():
        if isinstance(v, dict):
            if v.get("kotekan_stage") == kotekan_stage:
                found.append(v)
            found += find_stages(v, kotekan_stage)
    return found


def dish_positions(config):
    """Dish positions [D, 3] in the grid frame, and which dishes exist, from
    `telescope.dish_inputs`. Dishes listed with `type: Missing` do not exist."""
    tel = config["telescope"]
    num_dishes = value(config, "num_dishes")
    dx = float(tel["dish_separation_x_m"])
    dy = float(tel["dish_separation_y_m"])
    pos = np.zeros((num_dishes, 3))
    exists = np.zeros(num_dishes, dtype=bool)
    for dish in tel["dish_inputs"]:
        if dish["type"] == "Missing":
            continue
        d = dish["dish_idx"]
        disp = dish["feed_pos_disp_m"]
        pos[d] = [
            dx * dish["grid_x_idx"] + disp[0],
            dy * dish["grid_y_idx"] + disp[1],
            disp[2],
        ]
        exists[d] = True
    return pos, exists


def channel_width(config):
    return value(config, "adc_frequency") / value(config, "num_samples_per_frame")


def sky_frequency(config, channels, f_tone=0):
    """The sky frequency (Hz) of `f_tone` cycles per sample in coarse `channels`. As in
    `Telescope`, Nyquist zone z (`telescope.nyquist_zone`, default 1) starts at floor(z/2) times
    the ADC frequency and counts up in odd and down in even zones (CHIME)."""
    zone = config["telescope"].get("nyquist_zone", 1)
    sign = 1 if zone % 2 else -1
    return zone // 2 * value(config, "adc_frequency") + sign * (
        np.asarray(channels) + f_tone
    ) * channel_width(config)


def direction_vector(direction):
    nx, ny = direction
    return np.array([nx, ny, np.sqrt(1 - nx ** 2 - ny ** 2)])


def geometric_phases(config, direction, frequencies):
    """exp(-2πi ν n·r_d / c) for frequencies [F] (Hz) and all dishes: [F, D]. Missing dishes
    get 0. This is the convention of lib/testing/testSkySignal.hpp."""
    pos, exists = dish_positions(config)
    delay = pos @ direction_vector(direction) / C0
    return np.exp(-2j * np.pi * np.outer(frequencies, delay)) * exists


def grid_beam_direction(config, prefix, bx, by):
    """The position of beam (bx, by) of a `grid_degrees` beam grid (setBBBeams, setFRBBeams)
    with `<prefix>_num_beams_x`, `<prefix>_beam_separation_x` etc."""
    angles = []
    for b, axis in [(bx, "x"), (by, "y")]:
        num = value(config, f"{prefix}_num_beams_{axis}")
        sep = value(config, f"{prefix}_beam_separation_{axis}")
        angles.append(np.radians(-sep * (num - 1) / 2 + sep * b))
    return [float(np.sin(a)) for a in angles]


def frb2_beam_positions(config):
    """The FRB2 beam positions [R, 2] that setFRBBeams produces for its `grid_degrees` or
    `manual` mode."""
    stage = find_stages(config, "setFRBBeams")[0]
    if stage["mode"] == "manual":
        return np.array([[b["x_dir_grid"], b["y_dir_grid"]] for b in stage["beams"]])
    num_x, num_y = value(config, "frb2_num_beams_x"), value(config, "frb2_num_beams_y")
    return np.array(
        [
            grid_beam_direction(config, "frb2", bx, by)
            for by in range(num_y)
            for bx in range(num_x)
        ]
    )


def grid_point(lo, hi, n, i):
    """Point `i` of `n` points spaced uniformly over [lo, hi], as `Beams::grid_point` computes it
    (lib/utils/beamUtil.cpp); a single point sits at the midpoint."""
    if n == 1:
        return (lo + hi) / 2
    return (lo * (n - i - 1) + hi * i) / (n - 1)


def bb_peak_line(bb_data, positions, config, phases, nu, amplitude):
    """Baseband beams `bb_data` [Thi, B, F, P, T] (int4 bytes) of a tone of `amplitude` with geometric phases `phases`
    [F, D] at frequencies `nu` [F], beams at `positions` [B, 2]. J = bb_gain Σ_d exp(iφ_bd) E_d,
    so the predicted tone power in beam b is (A bb_gain |Σ_d exp(2πi ν (n_b - n)·r_d / c)|)^2 on
    top of a noise floor that is the same for all beams. Where the prediction is unambiguous,
    the measured power must peak in the predicted beam."""
    power = (np.abs(decode_int4x2(bb_data)) ** 2).mean(axis=(0, 3, 4))  # [B, F]
    gain = value(config, "bb_gain") * 127.5 / 128
    predicted = np.stack(
        [
            (
                amplitude
                * gain
                * np.abs(
                    (phases * geometric_phases(config, pos, nu).conj()).sum(axis=1)
                )
            )
            ** 2
            for pos in positions
        ]
    )  # [B, F]
    ranked = np.sort(predicted, axis=0)
    clear = (ranked[-1] >= 1.25 * ranked[-2]) & (ranked[-1] - ranked[-2] >= 1)
    require(clear.any(), "bb_beams: no frequency with an unambiguous predicted peak")
    wrong = clear & (power.argmax(axis=0) != predicted.argmax(axis=0))
    require(
        not wrong.any(),
        f"bb_beams: peak in beam {power.argmax(axis=0)[wrong][:8].tolist()} instead of "
        f"{predicted.argmax(axis=0)[wrong][:8].tolist()} for {wrong.sum()} of {clear.sum()} frequencies",
    )
    beams = np.unique(predicted.argmax(axis=0)[clear]).tolist()
    return (
        f"bb_beams: peak in the predicted beam for all {clear.sum()} unambiguous frequencies "
        f"(beams {beams})"
    )


def frb2_tone_power(frb2):
    """The tone power [R, coarse] of FRB2 beams [Ttildehi256, R, Fbar, Ttildelo256]: in each
    coarse channel, the fine channel with the largest power, minus the noise floor (the median
    over beams)."""
    power = used_channels(frb2, 2).sum(axis=(0, 3), dtype=np.float64)  # [R, Fbar]
    coarse = np.atleast_1d(frb2.attr("coarse_freq"))
    slots = np.array(
        [
            np.nonzero(coarse == chan)[0][power[:, coarse == chan].max(axis=0).argmax()]
            for chan in np.unique(coarse)
        ]
    )
    return power[:, slots] - np.median(power[:, slots], axis=0)


def frb2_peak_line(frb2, expected_beam):
    """The source beam must hold the maximum tone power in each coarse channel. Grating lobes
    (the dish spacing is many wavelengths) can give other beams the same tone power; such ties
    are allowed within 2%."""
    tone = frb2_tone_power(frb2)
    share = tone[expected_beam] / tone.max(axis=0)
    bad = share < 0.98
    if bad.any():
        peaks = tone.argmax(axis=0)[bad]
        require(
            False,
            f"frb2_beams: beam {expected_beam} has only {share[bad].min():.3f} of the peak "
            f"tone power (peak in beams {np.unique(peaks).tolist()}) in {bad.sum()} of "
            f"{tone.shape[1]} channels",
        )
    ties = (tone >= 0.98 * tone.max(axis=0)).sum(axis=0) > 1
    return (
        f"frb2_beams: beam {expected_beam} holds the peak in all {tone.shape[1]} channels "
        f"({ties.sum()} ties with grating lobes)"
    )


def frb3_codes(out_dir, num):
    """FRB3 codes q, offsets and scales, each [Ttildehi256, R, Fbar, Ttildelo256] for the first
    `num` frequencies, and the codes for no data and the lowest and highest values. A value is
    encoded as offset + scale q.
    - 4 bits (cudaQuantize4): int4x2, two's complement, the earlier sample in the low nibble;
      no data -8, values -7 to 7; offset and scale per 256 samples [Ttilde256, R, Fbar, 2].
    - 8 bits (cudaQuantize8, CHIME): [Ttilde256, R4, Fbar64, Ttilde16_lo16, Rlo4, Fbar16_lo4,
      Fbarlo16, Ttildelo16]; no data 0, values 1 to 254; offset and scale per 16 frequencies and
      16 samples [Ttilde256, R4, Fbar64, Ttilde16_lo16, Rlo4, Fbar16_lo4, 2]."""
    frb3 = read_output(out_dir, "frb3_beams").data
    offsetscale = read_output(out_dir, "frb3_beams_offsetscale").data
    if frb3.ndim == 4:
        frb3 = frb3[:, :, :num]
        offsetscale = offsetscale[:, :, :num].astype(np.float64)
        q = np.stack([frb3 & 0xF, frb3 >> 4], axis=-1).reshape(*frb3.shape[:3], -1)
        q = ((q.astype(np.int8) ^ 8) - 8).astype(np.float64)
        return q, offsetscale[..., 0:1], offsetscale[..., 1:2], -8, -7, 7
    n = frb3.shape[0]
    q = frb3.transpose(0, 1, 4, 2, 5, 6, 3, 7).reshape(n, -1, 256, 256)[:, :, :num]
    offsetscale = offsetscale.transpose(0, 1, 4, 2, 5, 3, 6).reshape(n, -1, 16, 16, 2)
    offsetscale = np.repeat(np.repeat(offsetscale, 16, axis=2), 16, axis=3)[:, :, :num]
    offsetscale = offsetscale.astype(np.float64)
    return q.astype(np.float64), offsetscale[..., 0], offsetscale[..., 1], 0, 1, 254


def upchan_window(num_taps, U):
    """The sinc-Hanning PFB window of the upchannelizer, without the 1/U factor
    (julia/kernels/upchan.jl, `Wkernel`)."""
    s = np.arange(num_taps * U)
    x = (2 * s - (num_taps * U - 1)) / (2 * num_taps * U)
    return np.cos(np.pi * x) ** 2 * np.sinc(num_taps * x)


def histogram(data, offset):
    """Counts of the int4 values -8..7 (index value + 8) over both nibbles of the bytes `data`,
    offset-encoded (int4x2_swapped_withoffset) or two's complement (int4x2)."""
    counts = np.bincount(np.asarray(data, dtype=np.uint8).ravel(), minlength=256)
    nibbles = [np.arange(256) >> 4, np.arange(256) & 0xF]
    hist = np.zeros(16)
    for nibble in nibbles:
        code = nibble if offset else nibble ^ 8
        hist += np.bincount(code, weights=counts, minlength=16)
    return hist


def rounded_gaussian(sigma):
    """rms, fraction of zeros and fraction at ±7 of a Gaussian with standard deviation `sigma`,
    rounded to the nearest integer and clamped to [-7, 7]."""
    k = np.arange(-7, 8)
    edges = np.concatenate([[-np.inf], k[:-1] + 0.5, [np.inf]])
    cdf = np.array([0.5 * (1 + math.erf(x / (sigma * math.sqrt(2)))) for x in edges])
    p = np.diff(cdf)
    return math.sqrt((p * k ** 2).sum()), p[7], p[0] + p[-1]
