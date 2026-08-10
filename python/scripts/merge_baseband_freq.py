#!/usr/bin/env python3
"""Merge kotekan baseband HDF5 dumps into a single full-band file.

Each node in a CHORD/kotekan pipeline writes ``hdf5FileWrite`` dumps that only
contain the frequencies that node processed (see
``docs/sphinx/user/file_formats/hdf5_frames.rst``).  The datasets are
``[T, F, P, D]`` with the time axis growing as frames are appended, and the
``coarse_freq`` attribute naming the channels held in the file.

This tool takes the dumps from several nodes, checks that they describe the same
observation and the same time samples, and writes one file whose frequency axis
is the union of all of them.  Files belonging to one node are also concatenated
along time, so a whole acquisition can be merged in one pass.

Memory is bounded by processing the output in time blocks: only one block of the
merged array (plus one source slab) is ever resident, and at most
``--max-open-files`` HDF5 files are open at a time.

Examples
--------

Merge every dump under a set of per-node directories::

    merge_baseband_freq.py -o /data/merged.h5 /scratch/node*/baseband

Check the inputs without writing anything::

    merge_baseband_freq.py -o /data/merged.h5 --dry-run /scratch/node*/baseband

Build a virtual dataset instead of copying the ~TB of data::

    merge_baseband_freq.py -o /data/merged_vds.h5 --vds /scratch/node*/baseband
"""

import argparse
import glob
import math
import os
import sys
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

try:
    # Registers the bitshuffle/zstd filter used by hdf5FileWrite when
    # `use_compression` is on.  Import before h5py by convention.
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - only matters for compressed inputs
    hdf5plugin = None

import h5py


# Attributes that must be identical in every source file: a mismatch means the
# files do not describe the same data and merging them would be meaningless.
CRITICAL_ATTRS = (
    "name",
    "type",
    "dim_names",
    "dim_scalings",
    "num_polarizations",
    "num_dishes",
    "seq_length_nsec",
    "time_downsampling_fpga",
    "chord_metadata_version",
)

# Attributes that should also be identical, but where a mismatch is merely
# suspicious.  Reported as a warning, or as an error under --strict.
ADVISORY_ATTRS = (
    "telescope_name",
    "gps_time_enabled",
    "itrs_lat_deg",
    "itrs_lon_deg",
    "grid_orientation",
    "grid_size_x",
    "grid_size_y",
    "feed_separation_x_m",
    "feed_separation_y_m",
    "dish_grid_indices",
    "feed_positions_m",
    "rfi_frame_excision_enabled",
    "rfi_frame_excision_thresholds",
)

# Attributes recomputed for the merged file rather than copied from a source.
DERIVED_ATTRS = (
    "coarse_freq",
    "freq_upchan_factor",
    "freq_upchan_index",
    "fpga_seq_num",
    "fpga_seq_time_nsec",
)

# int4x2 samples are offset-encoded, so a zero sample is 0x8 in each nibble.
OFFSET_INT4_ZERO = 0x88


class MergeError(RuntimeError):
    """Fatal inconsistency between the files being merged."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return "{:.2f} {}".format(n, unit)
        n /= 1024.0


def parse_size(text):
    """Parse a byte count such as ``512M``, ``4GiB`` or ``1073741824``."""

    s = str(text).strip().upper().replace("IB", "").replace("B", "")
    factor = 1
    if s and s[-1] in "KMGT":
        factor = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[s[-1]]
        s = s[:-1]
    return int(float(s) * factor)


def parse_int(text):
    """Parse an integer that may be written in hex (``0x88``) or decimal."""

    s = str(text).strip()
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def attrs_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.array_equal(a, b))
    return bool(a == b)


def as_selection(indices):
    """Turn an index array into a slice when it is evenly strided.

    Frequency subsets are usually a regular stride through the merged axis (node
    ``k`` of ``n`` holds every ``n``-th channel), and a slice lets both numpy and
    HDF5 handle the copy as a single strided selection instead of a gather.
    """

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return slice(0, 0)
    if indices.size == 1:
        return slice(int(indices[0]), int(indices[0]) + 1)
    step = int(indices[1] - indices[0])
    if step > 0 and np.array_equal(indices, np.arange(indices.size) * step + indices[0]):
        return slice(int(indices[0]), int(indices[-1]) + 1, step)
    return indices


# ---------------------------------------------------------------------------
# source description
# ---------------------------------------------------------------------------


@dataclass
class SourceFile:
    """Everything we need to know about one dump file, read from its header."""

    path: str
    host: str
    dataset: str
    shape: tuple
    dtype: np.dtype
    chunks: tuple
    coarse_freq: np.ndarray
    upchan_index: np.ndarray
    upchan_factor: np.ndarray
    seq_start: int
    seq_step: int
    seq_length_nsec: int
    time_ns: int
    attrs: dict = field(repr=False, default_factory=dict)

    @property
    def n_time(self):
        return int(self.shape[0])

    @property
    def n_freq(self):
        return int(self.shape[1])

    @property
    def trailing(self):
        return tuple(int(d) for d in self.shape[2:])

    @property
    def seq_end(self):
        return self.seq_start + self.n_time * self.seq_step


def auto_dataset_name(h5file, path):
    names = [k for k, v in h5file.items() if isinstance(v, h5py.Dataset)]
    if len(names) != 1:
        raise MergeError(
            "{}: expected exactly one root dataset, found {}; pass --dataset".format(
                path, names or "none"
            )
        )
    return names[0]


def host_from_path(path, dataset):
    """Recover the writing host from the ``<host>_<file_name>.<n>.h5`` name."""

    base = os.path.basename(path)
    marker = "_{}.".format(dataset)
    idx = base.find(marker)
    return base[:idx] if idx > 0 else ""


def scan_file(path, dataset=None):
    """Read one file's header.  No sample data is touched."""

    try:
        h5file = h5py.File(path, "r")
    except OSError as exc:
        raise MergeError(
            "cannot open {}: {}\n"
            "(a file left open by a crashed writer may need `h5clear -s {}`)".format(
                path, exc, path
            )
        )

    with h5file:
        name = dataset or auto_dataset_name(h5file, path)
        if name not in h5file:
            raise MergeError("{}: no dataset named '{}'".format(path, name))
        dset = h5file[name]
        attrs = dict(dset.attrs)

        if dset.ndim < 3:
            raise MergeError(
                "{}: dataset '{}' has rank {}, expected at least 3 ([T, F, ...])".format(
                    path, name, dset.ndim
                )
            )
        if "coarse_freq" not in attrs:
            raise MergeError(
                "{}: no 'coarse_freq' attribute, cannot place its channels".format(path)
            )
        if "fpga_seq_num" not in attrs:
            raise MergeError(
                "{}: no 'fpga_seq_num' attribute, cannot place it in time".format(path)
            )

        coarse_freq = np.asarray(attrs["coarse_freq"], dtype=np.int64).ravel()
        if coarse_freq.size != dset.shape[1]:
            raise MergeError(
                "{}: coarse_freq has {} entries but the frequency axis is {}".format(
                    path, coarse_freq.size, dset.shape[1]
                )
            )

        n_freq = coarse_freq.size
        upchan_index = np.asarray(
            attrs.get("freq_upchan_index", np.zeros(n_freq)), dtype=np.int64
        ).ravel()
        upchan_factor = np.asarray(
            attrs.get("freq_upchan_factor", np.ones(n_freq)), dtype=np.int64
        ).ravel()

        # One row of the dataset spans this many FPGA ticks.  Prefer the explicit
        # attribute; `dim_scalings[0]` carries the same factor for buffers that
        # were downsampled without recording it.
        seq_step = int(attrs.get("time_downsampling_fpga", 0))
        if seq_step <= 0:
            scalings = np.asarray(attrs.get("dim_scalings", [1]), dtype=np.int64).ravel()
            seq_step = int(scalings[0]) if scalings.size else 1
        if seq_step <= 0:
            seq_step = 1

        return SourceFile(
            path=os.path.abspath(path),
            host=host_from_path(path, name),
            dataset=name,
            shape=tuple(int(d) for d in dset.shape),
            dtype=dset.dtype,
            chunks=tuple(int(c) for c in dset.chunks) if dset.chunks else None,
            coarse_freq=coarse_freq,
            upchan_index=upchan_index,
            upchan_factor=upchan_factor,
            seq_start=int(attrs["fpga_seq_num"]),
            seq_step=seq_step,
            seq_length_nsec=int(attrs.get("seq_length_nsec", 0)),
            time_ns=int(attrs["fpga_seq_time_nsec"]) if "fpga_seq_time_nsec" in attrs else None,
            attrs=attrs,
        )


def expand_inputs(paths, pattern):
    """Expand directories and globs into a sorted list of files."""

    found = []
    for entry in paths:
        if os.path.isdir(entry):
            found.extend(glob.glob(os.path.join(entry, "**", pattern), recursive=True))
        else:
            matches = glob.glob(entry)
            found.extend(matches if matches else [entry])
    # Deduplicate while keeping a stable, reproducible order.
    return sorted(set(os.path.abspath(p) for p in found))


# ---------------------------------------------------------------------------
# frequency groups
# ---------------------------------------------------------------------------


@dataclass
class FreqGroup:
    """All files that carry one particular set of frequency channels."""

    coarse_freq: np.ndarray
    upchan_index: np.ndarray
    upchan_factor: np.ndarray
    files: list
    dst_sel: object = None  # slice or index array into the merged frequency axis
    gaps: list = field(default_factory=list)

    @property
    def n_freq(self):
        return int(self.coarse_freq.size)

    @property
    def seq_start(self):
        return self.files[0].seq_start

    @property
    def seq_end(self):
        return self.files[-1].seq_end

    @property
    def hosts(self):
        return sorted({f.host for f in self.files if f.host})

    @property
    def label(self):
        hosts = self.hosts
        who = ",".join(hosts) if hosts else os.path.basename(self.files[0].path)
        return "{} [{}..{}]".format(who, self.coarse_freq[0], self.coarse_freq[-1])


def group_sources(sources):
    """Bucket files by their frequency set; each bucket is one writer/node."""

    buckets = OrderedDict()
    for src in sources:
        key = (src.coarse_freq.tobytes(), src.upchan_index.tobytes())
        buckets.setdefault(key, []).append(src)

    groups = []
    for files in buckets.values():
        files.sort(key=lambda f: f.seq_start)
        first = files[0]
        groups.append(
            FreqGroup(
                coarse_freq=first.coarse_freq,
                upchan_index=first.upchan_index,
                upchan_factor=first.upchan_factor,
                files=files,
            )
        )
    groups.sort(key=lambda g: (int(g.coarse_freq.min()), int(g.upchan_index.min())))
    return groups


def check_group_time_continuity(group, log):
    """Verify one node's files tile time without overlapping, record any gaps."""

    for prev, cur in zip(group.files, group.files[1:]):
        if cur.seq_start < prev.seq_end:
            raise MergeError(
                "{}: overlapping time coverage between\n  {} (ticks {}..{})\n  {} "
                "(ticks {}..{})\nTwo writers appear to hold the same frequencies for "
                "the same time.".format(
                    group.label,
                    prev.path,
                    prev.seq_start,
                    prev.seq_end,
                    cur.path,
                    cur.seq_start,
                    cur.seq_end,
                )
            )
        if cur.seq_start > prev.seq_end:
            group.gaps.append((prev.seq_end, cur.seq_start))
            log(
                "warning: {}: {} ticks missing between {} and {}".format(
                    group.label,
                    cur.seq_start - prev.seq_end,
                    os.path.basename(prev.path),
                    os.path.basename(cur.path),
                )
            )


def check_consistency(sources, strict, log):
    """Cross-check the attributes that must agree for a merge to make sense."""

    ref = sources[0]
    problems = []

    for src in sources[1:]:
        if src.dtype != ref.dtype:
            problems.append(
                "{}: dtype {} != {} ({})".format(src.path, src.dtype, ref.dtype, ref.path)
            )
        if src.trailing != ref.trailing:
            problems.append(
                "{}: trailing dimensions {} != {} ({})".format(
                    src.path, src.trailing, ref.trailing, ref.path
                )
            )
        if src.seq_step != ref.seq_step:
            problems.append(
                "{}: time downsampling {} != {} ({})".format(
                    src.path, src.seq_step, ref.seq_step, ref.path
                )
            )
        for key in CRITICAL_ATTRS:
            if key in ref.attrs and key in src.attrs:
                if not attrs_equal(ref.attrs[key], src.attrs[key]):
                    problems.append(
                        "{}: attribute '{}' differs from {}".format(src.path, key, ref.path)
                    )
            elif (key in ref.attrs) != (key in src.attrs):
                problems.append(
                    "{}: attribute '{}' present in only one of the files".format(src.path, key)
                )

    if problems:
        raise MergeError("inconsistent source files:\n  " + "\n  ".join(problems))

    for src in sources[1:]:
        for key in ADVISORY_ATTRS:
            in_ref, in_src = key in ref.attrs, key in src.attrs
            if in_ref and in_src and attrs_equal(ref.attrs[key], src.attrs[key]):
                continue
            if not in_ref and not in_src:
                continue
            msg = "attribute '{}' differs between {} and {}".format(
                key, os.path.basename(ref.path), os.path.basename(src.path)
            )
            if strict:
                raise MergeError(msg)
            log("warning: " + msg)


def build_freq_axis(groups, order, allow_duplicates, log):
    """Assign each group's channels a position on the merged frequency axis."""

    coarse = np.concatenate([g.coarse_freq for g in groups])
    upchan_idx = np.concatenate([g.upchan_index for g in groups])
    upchan_fac = np.concatenate([g.upchan_factor for g in groups])
    n_freq = coarse.size

    # A channel is identified by (coarse channel, upchannelisation index) so that
    # upchannelised data with repeated coarse ids still sorts correctly.
    pairs = np.stack([coarse, upchan_idx], axis=1)
    _, first_idx, counts = np.unique(pairs, axis=0, return_index=True, return_counts=True)
    if np.any(counts > 1):
        dup = pairs[np.sort(first_idx[counts > 1])][:5]
        msg = "{} channels are claimed by more than one node, e.g. {}".format(
            int(np.sum(counts[counts > 1])), dup.tolist()
        )
        if not allow_duplicates:
            raise MergeError(msg + " (use --allow-duplicate-freq to keep them all)")
        log("warning: " + msg)

    if order == "sorted":
        sorted_pos = np.lexsort((upchan_idx, coarse))
        dst = np.empty(n_freq, dtype=np.int64)
        dst[sorted_pos] = np.arange(n_freq, dtype=np.int64)
    else:  # "concat": keep whole nodes contiguous, in group order
        dst = np.arange(n_freq, dtype=np.int64)

    offset = 0
    for group in groups:
        group.dst_sel = as_selection(dst[offset : offset + group.n_freq])
        offset += group.n_freq

    out_coarse = np.empty(n_freq, dtype=coarse.dtype)
    out_upchan_idx = np.empty(n_freq, dtype=upchan_idx.dtype)
    out_upchan_fac = np.empty(n_freq, dtype=upchan_fac.dtype)
    out_coarse[dst] = coarse
    out_upchan_idx[dst] = upchan_idx
    out_upchan_fac[dst] = upchan_fac

    # Report holes in the band so a missing node is obvious.
    if np.all(out_upchan_fac == 1):
        span = np.unique(out_coarse)
        expected = np.arange(span[0], span[-1] + 1)
        missing = np.setdiff1d(expected, span)
        if missing.size:
            log(
                "warning: {} coarse channels between {} and {} are not covered by any "
                "node (e.g. {})".format(
                    missing.size, span[0], span[-1], missing[:8].tolist()
                )
            )

    return out_coarse, out_upchan_idx, out_upchan_fac


# ---------------------------------------------------------------------------
# time grid
# ---------------------------------------------------------------------------


def check_time_alignment(groups, seq_step, tolerance_ns, log):
    """Confirm the nodes really are sampling the same ticks at the same times."""

    origin = groups[0].seq_start
    for group in groups:
        for src in group.files:
            if (src.seq_start - origin) % seq_step:
                raise MergeError(
                    "{}: starts at tick {}, which is not on the {}-tick sample grid "
                    "defined by {}".format(
                        src.path, src.seq_start, seq_step, groups[0].files[0].path
                    )
                )

    # fpga_seq_time_nsec must track fpga_seq_num, or the FPGA timestamps and the
    # sequence numbers disagree and "aligned in time" cannot be trusted.
    ref = None
    for group in groups:
        for src in group.files:
            if src.time_ns is None or not src.seq_length_nsec:
                continue
            if ref is None:
                ref = src
                continue
            expected = ref.time_ns + (src.seq_start - ref.seq_start) * ref.seq_length_nsec
            delta = src.time_ns - expected
            if abs(delta) > tolerance_ns:
                raise MergeError(
                    "{}: timestamp {} ns is {} ns away from the {} ns implied by tick {} "
                    "and {} -- the nodes are not aligned in time".format(
                        src.path, src.time_ns, delta, expected, src.seq_start, ref.path
                    )
                )

    starts = {g.seq_start for g in groups}
    ends = {g.seq_end for g in groups}
    if len(starts) > 1 or len(ends) > 1:
        log(
            "warning: nodes cover different time ranges (starts {}, ends {})".format(
                sorted(starts), sorted(ends)
            )
        )


def resolve_time_range(groups, coverage, start_seq, end_seq, seq_step):
    if coverage == "intersection":
        seq0 = max(g.seq_start for g in groups)
        seq1 = min(g.seq_end for g in groups)
    else:
        seq0 = min(g.seq_start for g in groups)
        seq1 = max(g.seq_end for g in groups)

    if start_seq is not None:
        seq0 = max(seq0, start_seq)
    if end_seq is not None:
        seq1 = min(seq1, end_seq)

    # Snap onto the sample grid defined by the first node.
    origin = groups[0].seq_start
    seq0 += (-(seq0 - origin)) % seq_step
    seq1 -= (seq1 - origin) % seq_step

    if seq1 <= seq0:
        raise MergeError(
            "no time samples selected ({} coverage gives ticks {}..{}); the nodes may "
            "not overlap in time".format(coverage, seq0, seq1)
        )
    return seq0, seq1


def plan_reads(group, seq0, seq1, seq_step):
    """List the (file, source row, count, destination row) copies for a range."""

    starts = [f.seq_start for f in group.files]
    first = max(bisect_right(starts, seq0) - 1, 0)
    plan = []
    for src in group.files[first:]:
        if src.seq_start >= seq1:
            break
        lo, hi = max(src.seq_start, seq0), min(src.seq_end, seq1)
        if hi <= lo:
            continue
        plan.append(
            (
                src.path,
                (lo - src.seq_start) // seq_step,
                (hi - lo) // seq_step,
                (lo - seq0) // seq_step,
            )
        )
    return plan


# ---------------------------------------------------------------------------
# bounded pool of open HDF5 files
# ---------------------------------------------------------------------------


class FilePool:
    """Keep at most ``max_open`` HDF5 files open, evicting least-recently-used."""

    def __init__(self, max_open, rdcc_nbytes):
        self.max_open = max(1, int(max_open))
        self.rdcc_nbytes = int(rdcc_nbytes)
        self._open = OrderedDict()
        self.opens = 0

    def get(self, path):
        handle = self._open.get(path)
        if handle is not None:
            self._open.move_to_end(path)
            return handle
        while len(self._open) >= self.max_open:
            _, evicted = self._open.popitem(last=False)
            evicted.close()
        # A chunk cache sized to a couple of chunks keeps compressed sources from
        # being decompressed twice when a read straddles a chunk boundary.
        handle = h5py.File(path, "r", rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=521)
        self._open[path] = handle
        self.opens += 1
        return handle

    def close(self):
        for handle in self._open.values():
            handle.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# parallel reader workers (optional)
# ---------------------------------------------------------------------------

_WORKER = {}


def _worker_init(shm_name, shape, dtype_str, dataset, max_open, rdcc_nbytes):
    from multiprocessing import shared_memory

    # The workers share the parent's resource tracker, so the segment stays
    # registered exactly once and the parent's unlink() cleans it up.
    shm = shared_memory.SharedMemory(name=shm_name)
    _WORKER["shm"] = shm
    _WORKER["buf"] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    _WORKER["dataset"] = dataset
    _WORKER["pool"] = FilePool(max_open, rdcc_nbytes)


def _worker_read(task):
    path, src_row, count, dst_row, sel = task
    dset = _WORKER["pool"].get(path)[_WORKER["dataset"]]
    buf = _WORKER["buf"]
    scratch = np.empty((count,) + dset.shape[1:], dtype=buf.dtype)
    dset.read_direct(scratch, np.s_[src_row : src_row + count])
    buf[dst_row : dst_row + count, sel] = scratch
    return count


# ---------------------------------------------------------------------------
# output creation
# ---------------------------------------------------------------------------


def choose_chunks(shape, itemsize, chunk_time, chunk_freq, target_bytes):
    """Pick a chunk shape, defaulting to whole frequency planes like kotekan."""

    n_freq = shape[1] if chunk_freq is None else min(chunk_freq, shape[1])
    trailing = math.prod(shape[2:]) if len(shape) > 2 else 1
    row_bytes = n_freq * trailing * itemsize
    if chunk_time is None:
        chunk_time = max(1, target_bytes // max(row_bytes, 1))
    chunk_time = int(min(max(1, chunk_time), shape[0]))
    return (chunk_time, n_freq) + tuple(shape[2:])


def default_fill_value(dtype, type_attr):
    if dtype == np.uint8 and "withoffset" in str(type_attr):
        return OFFSET_INT4_ZERO
    return 0


def compression_options(name, level):
    """h5py keyword arguments for the requested output compression."""

    if name == "none":
        return {}
    if name == "gzip":
        return dict(compression="gzip", compression_opts=min(level, 9))
    if hdf5plugin is None:
        raise MergeError("--compression {} needs the hdf5plugin package".format(name))
    if name == "bitshuffle":
        # The format hdf5FileWrite itself writes, so merged files stay readable
        # by the same tooling.  Needs a bitshuffle plugin that can encode.
        return dict(hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=level))
    if name == "zstd":
        return dict(hdf5plugin.Zstd(clevel=level))
    raise MergeError("unknown compression '{}'".format(name))


def create_output_dataset(fout, name, shape, dtype, chunks, fill, compression, label):
    try:
        return fout.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, fillvalue=fill, **compression
        )
    except (ValueError, OSError) as exc:
        if not compression:
            raise
        raise MergeError(
            "could not create the output dataset with --compression {}: {}\n"
            "The HDF5 filter is probably unavailable for writing in this "
            "environment; try --compression none or --compression zstd.".format(label, exc)
        )


def write_attributes(dset, ref, coarse, upchan_idx, upchan_fac, seq0, groups, args):
    """Copy the observation metadata forward and record what was merged."""

    for key, value in ref.attrs.items():
        if key not in DERIVED_ATTRS:
            dset.attrs[key] = value

    dset.attrs["coarse_freq"] = coarse.astype(np.asarray(ref.attrs["coarse_freq"]).dtype)
    if "freq_upchan_factor" in ref.attrs:
        dset.attrs["freq_upchan_factor"] = upchan_fac.astype(
            np.asarray(ref.attrs["freq_upchan_factor"]).dtype
        )
    if "freq_upchan_index" in ref.attrs:
        dset.attrs["freq_upchan_index"] = upchan_idx.astype(
            np.asarray(ref.attrs["freq_upchan_index"]).dtype
        )

    dset.attrs["fpga_seq_num"] = np.int64(seq0)
    if ref.time_ns is not None and ref.seq_length_nsec:
        dset.attrs["fpga_seq_time_nsec"] = np.int64(
            ref.time_ns + (seq0 - ref.seq_start) * ref.seq_length_nsec
        )

    dset.attrs["merge_tool"] = "merge_baseband_freq.py"
    dset.attrs["merge_num_sources"] = np.int64(sum(len(g.files) for g in groups))
    dset.attrs["merge_num_nodes"] = np.int64(len(groups))
    dset.attrs["merge_freq_order"] = args.freq_order
    dset.attrs["merge_coverage"] = args.coverage
    hosts = sorted({h for g in groups for h in g.hosts})
    if hosts:
        dset.attrs["merge_source_hosts"] = np.array(hosts, dtype=object)
    if args.record_source_files:
        dset.attrs["merge_source_files"] = np.array(
            [f.path for g in groups for f in g.files], dtype=object
        )


# ---------------------------------------------------------------------------
# merge implementations
# ---------------------------------------------------------------------------


def merge_copy(groups, ref, out_shape, seq0, seq_step, coarse, upchan_idx, upchan_fac, args, log):
    """Stream the sources into a new dataset, one time block at a time."""

    itemsize = ref.dtype.itemsize
    trailing = math.prod(ref.trailing) if ref.trailing else 1
    n_freq_out = out_shape[1]
    max_group_freq = max(g.n_freq for g in groups)

    chunks = None
    if args.layout == "chunked":
        chunks = choose_chunks(
            out_shape, itemsize, args.chunk_time, args.chunk_freq, args.chunk_bytes
        )

    # Size the time block from the memory budget.  Resident at any moment: one
    # block of the merged array, plus the slab each reader is filling.
    bytes_per_row = n_freq_out * trailing * itemsize
    scratch_per_row = max_group_freq * trailing * itemsize
    if args.read_jobs:
        # Readers work concurrently, so give the assembly buffer half the budget
        # and share the rest between the workers' slabs.
        block_rows = (args.memory_limit // 2) // max(bytes_per_row, 1)
        task_rows = (args.memory_limit - args.memory_limit // 2) // max(
            args.read_jobs * scratch_per_row, 1
        )
    else:
        block_rows = args.memory_limit // max(bytes_per_row + scratch_per_row, 1)
        task_rows = block_rows

    if chunks is not None and block_rows > chunks[0]:
        # Chunk-aligned writes avoid a read-modify-write of every output chunk.
        block_rows -= block_rows % chunks[0]
    block_rows = int(max(1, min(block_rows, out_shape[0])))
    task_rows = int(max(1, min(task_rows, block_rows)))

    resident = block_rows * bytes_per_row + max(args.read_jobs, 1) * task_rows * scratch_per_row
    if resident > args.memory_limit:
        log(
            "warning: one time sample needs {}, more than the {} limit".format(
                human_bytes(resident), human_bytes(args.memory_limit)
            )
        )
    log(
        "output {} {} ({}), chunks {}, time block {} rows, ~{} resident".format(
            out_shape,
            ref.dtype,
            human_bytes(math.prod(out_shape) * itemsize),
            chunks,
            block_rows,
            human_bytes(resident),
        )
    )

    fill = args.fill_value
    if fill is None:
        fill = default_fill_value(ref.dtype, ref.attrs.get("type", ""))
    # Any row not covered by every node has to be initialised before the copies.
    needs_fill = bool(
        any(g.gaps for g in groups)
        or any(g.seq_start > seq0 or g.seq_end < seq0 + out_shape[0] * seq_step for g in groups)
    )

    compression = compression_options(args.compression, args.compression_level)

    shm = None
    executor = None
    total_bytes = 0
    started = time.monotonic()

    try:
        if args.read_jobs:
            from multiprocessing import get_context, shared_memory

            shm = shared_memory.SharedMemory(
                create=True, size=block_rows * bytes_per_row
            )
            buf = np.ndarray((block_rows, n_freq_out) + ref.trailing, ref.dtype, buffer=shm.buf)
        else:
            buf = np.empty((block_rows, n_freq_out) + ref.trailing, dtype=ref.dtype)
            scratch = np.empty((task_rows, max_group_freq) + ref.trailing, dtype=ref.dtype)

        with h5py.File(args.output, "w", rdcc_nbytes=args.output_cache) as fout:
            dset = create_output_dataset(
                fout,
                args.dataset_name,
                out_shape,
                ref.dtype,
                chunks,
                fill,
                compression,
                args.compression,
            )
            write_attributes(dset, ref, coarse, upchan_idx, upchan_fac, seq0, groups, args)

            if args.read_jobs:
                executor = get_context("spawn").Pool(
                    processes=args.read_jobs,
                    initializer=_worker_init,
                    initargs=(
                        shm.name,
                        buf.shape,
                        ref.dtype.str,
                        ref.dataset,
                        max(1, args.max_open_files // args.read_jobs),
                        args.source_cache,
                    ),
                )
                pool = None
            else:
                executor = None
                pool = FilePool(args.max_open_files, args.source_cache)

            for row0 in range(0, out_shape[0], block_rows):
                rows = min(block_rows, out_shape[0] - row0)
                view = buf[:rows]
                if needs_fill:
                    view.fill(fill)

                block_seq0 = seq0 + row0 * seq_step
                block_seq1 = block_seq0 + rows * seq_step

                # Split each contiguous span into task_rows pieces: it bounds the
                # slab a reader holds, and gives the worker pool enough units of
                # work to stay busy.
                tasks = []
                for group in groups:
                    for path, src_row, count, dst_row in plan_reads(
                        group, block_seq0, block_seq1, seq_step
                    ):
                        for off in range(0, count, task_rows):
                            take = min(task_rows, count - off)
                            tasks.append(
                                (path, src_row + off, take, dst_row + off, group.dst_sel)
                            )

                if executor is not None:
                    for _ in executor.imap_unordered(_worker_read, tasks):
                        pass
                else:
                    for path, src_row, count, dst_row, sel in tasks:
                        source = pool.get(path)[ref.dataset]
                        slab = scratch[:count]
                        source.read_direct(slab, np.s_[src_row : src_row + count])
                        view[dst_row : dst_row + count, sel] = slab

                dset[row0 : row0 + rows] = view

                total_bytes += rows * bytes_per_row
                elapsed = time.monotonic() - started
                done = (row0 + rows) / out_shape[0]
                log(
                    "  {:5.1f}%  {} written  {:.0f} MiB/s  eta {:.0f}s".format(
                        100.0 * done,
                        human_bytes(total_bytes),
                        total_bytes / (1 << 20) / max(elapsed, 1e-6),
                        elapsed * (1.0 - done) / max(done, 1e-9),
                    )
                )

            if pool is not None:
                log("closed after {} file opens".format(pool.opens))
                pool.close()
            if executor is not None:
                executor.close()
                executor.join()
                executor = None
    finally:
        if executor is not None:
            executor.terminate()
            executor.join()
        if shm is not None:
            shm.close()
            shm.unlink()

    elapsed = time.monotonic() - started
    log(
        "wrote {} in {:.1f}s ({:.0f} MiB/s)".format(
            human_bytes(total_bytes), elapsed, total_bytes / (1 << 20) / max(elapsed, 1e-6)
        )
    )


def merge_vds(groups, ref, out_shape, seq0, seq_step, coarse, upchan_idx, upchan_fac, args, log):
    """Map the sources into one virtual dataset without copying any data."""

    fill = args.fill_value
    if fill is None:
        fill = default_fill_value(ref.dtype, ref.attrs.get("type", ""))

    layout = h5py.VirtualLayout(shape=out_shape, dtype=ref.dtype)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    n_map = 0

    for group in groups:
        by_path = {f.path: f for f in group.files}
        for path, src_row, count, dst_row in plan_reads(
            group, seq0, seq0 + out_shape[0] * seq_step, seq_step
        ):
            src = by_path[path]
            stored = os.path.relpath(path, out_dir) if args.vds_relative else path
            vsource = h5py.VirtualSource(stored, ref.dataset, shape=src.shape, dtype=src.dtype)
            if isinstance(group.dst_sel, slice):
                layout[dst_row : dst_row + count, group.dst_sel] = vsource[
                    src_row : src_row + count
                ]
                n_map += 1
            else:
                # Irregular frequency placement needs one mapping per channel.
                for j, k in enumerate(group.dst_sel):
                    layout[dst_row : dst_row + count, int(k)] = vsource[
                        src_row : src_row + count, j
                    ]
                    n_map += 1

    with h5py.File(args.output, "w") as fout:
        dset = fout.create_virtual_dataset(args.dataset_name, layout, fillvalue=fill)
        write_attributes(dset, ref, coarse, upchan_idx, upchan_fac, seq0, groups, args)
        dset.attrs["merge_mode"] = "virtual"

    log(
        "wrote virtual dataset with {} mappings referencing {} paths".format(
            n_map, "relative" if args.vds_relative else "absolute"
        )
    )


def verify(groups, ref, out_shape, seq0, seq_step, args, log):
    """Spot-check the merged file against the sources at random time samples."""

    count = args.verify or 16
    rng = np.random.default_rng(args.verify_seed)
    rows = np.unique(rng.integers(0, out_shape[0], size=min(count, out_shape[0])))
    checked = 0

    with h5py.File(args.output, "r") as fout:
        if args.dataset_name not in fout:
            raise MergeError(
                "{} has no dataset '{}'".format(args.output, args.dataset_name)
            )
        merged = fout[args.dataset_name]
        if tuple(merged.shape) != tuple(out_shape):
            raise MergeError(
                "{}: shape {} does not match the {} implied by the sources".format(
                    args.output, merged.shape, out_shape
                )
            )
        if int(merged.attrs.get("fpga_seq_num", seq0)) != seq0:
            raise MergeError(
                "{}: starts at tick {}, expected {}".format(
                    args.output, int(merged.attrs["fpga_seq_num"]), seq0
                )
            )
        with FilePool(args.max_open_files, args.source_cache) as pool:
            for row in rows:
                row = int(row)
                seq = seq0 + row * seq_step
                out_row = merged[row]
                for group in groups:
                    plan = plan_reads(group, seq, seq + seq_step, seq_step)
                    if not plan:
                        continue
                    path, src_row, _, _ = plan[0]
                    src_row_data = pool.get(path)[ref.dataset][src_row]
                    if not np.array_equal(out_row[group.dst_sel], src_row_data):
                        raise MergeError(
                            "verification failed at time row {} for {} ({})".format(
                                row, group.label, path
                            )
                        )
                    checked += 1

    log("verified {} node/time samples across {} time rows".format(checked, rows.size))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Inputs may be files, glob patterns or directories (searched recursively).",
    )
    parser.add_argument("inputs", nargs="+", help="source files, globs or directories")
    parser.add_argument("-o", "--output", required=True, help="merged output file")
    parser.add_argument(
        "--pattern", default="*.h5", help="glob used inside input directories (default: *.h5)"
    )
    parser.add_argument(
        "--dataset", default=None, help="source dataset name (default: the only root dataset)"
    )
    parser.add_argument(
        "--dataset-name", default=None, help="dataset name in the output (default: same as input)"
    )

    parser.add_argument(
        "--freq-order",
        choices=("sorted", "concat"),
        default="sorted",
        help="order the merged frequency axis by channel id, or keep each node's "
        "channels contiguous (default: sorted)",
    )
    parser.add_argument(
        "--coverage",
        choices=("intersection", "union"),
        default="intersection",
        help="keep only times covered by every node, or the full span with gaps "
        "filled (default: intersection)",
    )
    parser.add_argument("--start-seq", type=int, default=None, help="first FPGA tick to merge")
    parser.add_argument(
        "--end-seq", type=int, default=None, help="stop before this FPGA tick"
    )

    parser.add_argument(
        "--memory-limit",
        type=parse_size,
        default=2 << 30,
        help="working-set budget for the merged block and source slab (default: 2G)",
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=32,
        help="maximum HDF5 files held open at once (default: 32)",
    )
    parser.add_argument(
        "--source-cache",
        type=parse_size,
        default=16 << 20,
        help="HDF5 chunk cache per open source file (default: 16M)",
    )
    parser.add_argument(
        "--output-cache",
        type=parse_size,
        default=64 << 20,
        help="HDF5 chunk cache for the output file (default: 64M)",
    )
    parser.add_argument(
        "--read-jobs",
        type=int,
        default=0,
        help="read with N worker processes; helps when the sources live on a "
        "parallel or multi-device filesystem (default: 0, single process)",
    )

    parser.add_argument(
        "--layout",
        choices=("chunked", "contiguous"),
        default="chunked",
        help="output dataset layout (default: chunked)",
    )
    parser.add_argument("--chunk-time", type=int, default=None, help="output chunk time extent")
    parser.add_argument(
        "--chunk-freq", type=int, default=None, help="output chunk frequency extent"
    )
    parser.add_argument(
        "--chunk-bytes",
        type=parse_size,
        default=8 << 20,
        help="target size of an output chunk (default: 8M)",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "bitshuffle", "zstd", "gzip"),
        default="none",
        help="compress the output; 'bitshuffle' matches what hdf5FileWrite writes "
        "but needs an encoding-capable plugin (default: none, much faster)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=9, help="compression level (default: 9)"
    )
    parser.add_argument(
        "--fill-value",
        type=parse_int,
        default=None,
        help="value for uncovered samples (default: 0x88 for offset-encoded int4, else 0)",
    )

    parser.add_argument(
        "--vds",
        action="store_true",
        help="write a virtual dataset instead of copying the data; instant, but the "
        "source files must stay in place",
    )
    parser.add_argument(
        "--vds-relative",
        action="store_true",
        help="store source paths relative to the output file (--vds only)",
    )

    parser.add_argument(
        "--time-tolerance-ns",
        type=int,
        default=0,
        help="allowed disagreement between a file's timestamp and its FPGA tick "
        "(default: 0, exact)",
    )
    parser.add_argument(
        "--allow-duplicate-freq",
        action="store_true",
        help="continue when two nodes claim the same channel",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat metadata warnings as errors"
    )
    parser.add_argument(
        "--record-source-files",
        action="store_true",
        help="record every source path in a merge_source_files attribute",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="after merging, compare N random time samples against the sources",
    )
    parser.add_argument("--verify-seed", type=int, default=0, help="seed for --verify sampling")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check an existing output file against the sources without merging",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="run all checks but write nothing"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report warnings")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    def log(message):
        if not args.quiet or message.startswith("warning"):
            print(message, file=sys.stderr, flush=True)

    if args.compression != "none" and args.layout == "contiguous":
        raise MergeError("--compression requires --layout chunked")
    if args.vds and args.compression != "none":
        raise MergeError("--vds cannot compress; it does not copy any data")

    paths = expand_inputs(args.inputs, args.pattern)
    if not paths:
        raise MergeError("no input files matched {}".format(args.inputs))

    output_abs = os.path.abspath(args.output)
    if output_abs in paths:
        raise MergeError("the output file is also an input: {}".format(args.output))

    log("scanning {} files".format(len(paths)))
    sources = [scan_file(p, args.dataset) for p in paths]
    if len({s.dataset for s in sources}) > 1:
        raise MergeError("source files use different dataset names; pass --dataset")

    check_consistency(sources, args.strict, log)

    groups = group_sources(sources)
    for group in groups:
        check_group_time_continuity(group, log)

    # SubsetBaseband does not record which element blocks it kept, so matching
    # trailing dimensions is the only cross-node check available for them.
    if len(groups) > 1 and sources[0].trailing:
        log(
            "note: the dish/polarisation subset is not recorded in the files; the merge "
            "assumes every node wrote the same {} element layout".format(
                "x".join(str(d) for d in sources[0].trailing)
            )
        )

    ref = groups[0].files[0]
    seq_step = ref.seq_step
    check_time_alignment(groups, seq_step, args.time_tolerance_ns, log)

    coarse, upchan_idx, upchan_fac = build_freq_axis(
        groups, args.freq_order, args.allow_duplicate_freq, log
    )
    seq0, seq1 = resolve_time_range(
        groups, args.coverage, args.start_seq, args.end_seq, seq_step
    )
    out_shape = ((seq1 - seq0) // seq_step, int(coarse.size)) + ref.trailing

    log("")
    log("{} node(s), {} file(s):".format(len(groups), len(sources)))
    for group in groups:
        log(
            "  {:<44} {:>5} chan  {:>4} file(s)  ticks {}..{}".format(
                group.label, group.n_freq, len(group.files), group.seq_start, group.seq_end
            )
        )
    log(
        "merged: {} samples x {} channels, ticks {}..{}".format(
            out_shape[0], out_shape[1], seq0, seq1
        )
    )
    log("")

    if args.dataset_name is None:
        args.dataset_name = ref.dataset

    if args.dry_run:
        log("dry run: nothing written")
        return 0

    if args.verify_only:
        if not os.path.exists(output_abs):
            raise MergeError("--verify-only: {} does not exist".format(args.output))
        verify(groups, ref, out_shape, seq0, seq_step, args, log)
        return 0

    out_dir = os.path.dirname(output_abs)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.vds:
        merge_vds(
            groups, ref, out_shape, seq0, seq_step, coarse, upchan_idx, upchan_fac, args, log
        )
    else:
        merge_copy(
            groups, ref, out_shape, seq0, seq_step, coarse, upchan_idx, upchan_fac, args, log
        )

    if args.verify:
        verify(groups, ref, out_shape, seq0, seq_step, args, log)

    log("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MergeError as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
