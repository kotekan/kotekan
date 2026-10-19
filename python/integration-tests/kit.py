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


def large_seq(config):
    """A large start FPGA sequence number. N2Accumulate needs frames to start at multiples of
    the frame length; 7 frames past 2**33 aligns it with no lifetime or integration."""
    return 2 ** 33 + 7 * value(config, "num_times")


def value(config, key):
    """A top-level config value; kotekan arithmetic expressions such as `5.0 / 4` that refer
    to other top-level values are evaluated."""
    v = config[key]
    if not isinstance(v, str):
        return v

    class Names(dict):
        def __missing__(self, name):
            return value(config, name)

    return eval(v, {"__builtins__": {}}, Names())


def dish_positions(config):
    """Dish positions [D, 3] in the grid frame, and which dishes exist, from
    `telescope.dish_inputs`."""
    tel = config["telescope"]
    num_dishes = value(config, "num_dishes")
    dx = float(tel["dish_separation_x_m"])
    dy = float(tel["dish_separation_y_m"])
    pos = np.zeros((num_dishes, 3))
    exists = np.zeros(num_dishes, dtype=bool)
    for dish in tel["dish_inputs"]:
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
