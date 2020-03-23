"""Read a visBuffer dump into python.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future.utils import native_str

import ctypes
import os
import io

import numpy as np


class time_spec(ctypes.Structure):
    """Struct repr of a timespec type."""

    _fields_ = [("tv", ctypes.c_int64), ("tv_nsec", ctypes.c_uint64)]

    @classmethod
    def from_float(cls, v):
        """Create a time_spec from a float.

        Parameters
        ----------------
        v : float
            The interval in seconds.

        Returns
        -------
        ts : time_spec
        """
        ts = cls()
        ts.tv = np.floor(v).astype(np.int64)
        ts.tv_nsec = ((v % 1.0) * 1e9).astype(np.int64)
        return ts


class timeval(ctypes.Structure):
    """Struct repr of a timeval type."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class VisMetadata(ctypes.Structure):
    """Wrap a visMetadata struct.
    """

    _fields_ = [
        ("fpga_seq", ctypes.c_uint64),
        ("ctime", time_spec),
        ("fpga_length", ctypes.c_uint64),
        ("fpga_total", ctypes.c_uint64),
        ("rfi_total", ctypes.c_uint64),
        ("freq_id", ctypes.c_uint32),
        ("dataset_id", ctypes.c_uint64 * 2),
        ("num_elements", ctypes.c_uint32),
        ("num_prod", ctypes.c_uint32),
        ("num_ev", ctypes.c_uint32),
    ]


class psrCoord(ctypes.Structure):
    """ Struct repr of psrCoord field in ChimeMetadata."""

    _fields_ = [
        ("ra", ctypes.ARRAY(ctypes.c_float, 10)),
        ("dec", ctypes.ARRAY(ctypes.c_float, 10)),
        ("scaling", ctypes.ARRAY(ctypes.c_uint32, 10)),
    ]


class ChimeMetadata(ctypes.Structure):
    """Wrap a ChimeMetadata struct."""

    _fields_ = [
        ("fpga_seq_num", ctypes.c_uint64),
        ("first_packet_recv_time", timeval),
        ("gps_time", time_spec),
        ("lost_timesamples", ctypes.c_int32),
        ("stream_ID", ctypes.c_uint16),
        ("psrCoord", psrCoord),
        ("rfi_zeroed", ctypes.c_uint32),
    ]


class VisBuffer(object):
    """Python representation of a visBuffer dump.

    Access the data through the `vis`, `weight`, `eval`, `evec` and `erms`
    attributes which are all numpy arrays.

    Parameters
    ----------
    buffer : bytearray
        Memory to provide a view of.
    skip : int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    """

    def __init__(self, buffer, skip=4):

        self._buffer = buffer[skip:]

        meta_size = ctypes.sizeof(VisMetadata)

        if len(self._buffer) < meta_size:
            raise ValueError("Buffer too small to contain metadata.")

        self.metadata = VisMetadata.from_buffer(self._buffer[:meta_size])

        self._set_data_arrays()

    def _set_data_arrays(self):

        _data = self._buffer[ctypes.sizeof(VisMetadata) :]

        layout = self.__class__._calculate_layout(
            self.metadata.num_elements, self.metadata.num_prod, self.metadata.num_ev
        )

        for member in layout["members"]:

            arr = np.frombuffer(
                _data[member["start"] : member["end"]], dtype=member["dtype"]
            )
            setattr(self, member["name"], arr)

    @classmethod
    def _calculate_layout(cls, num_elements, num_prod, num_ev):
        """Calculate the buffer layout.

        Parameters
        ----------
        num_elements, num_prod, num_ev : int
            Length of each dimension.

        Returns
        -------
        layout : dict
            Structure of buffer.
        """

        structure = [
            ("vis", np.complex64, num_prod),
            ("weight", np.float32, num_prod),
            ("flags", np.float32, num_elements),
            ("eval", np.float32, num_ev),
            ("evec", np.complex64, num_ev * num_elements),
            ("erms", np.float32, 1),
            ("gain", np.complex64, num_elements),
        ]

        end = 0

        members = []
        maxsize = 0

        for name, dtype, num in structure:

            member = {}

            size = np.dtype(dtype).itemsize

            # Update the maximum size
            maxsize = size if maxsize < size else maxsize

            member["start"] = _offset(end, size)
            end = member["start"] + num * size
            member["end"] = end
            member["size"] = num * size
            member["num"] = num
            member["dtype"] = dtype
            member["name"] = name

            members.append(member)

        struct_end = _offset(members[-1]["end"], maxsize)
        layout = {"size": struct_end, "members": members}
        return layout

    @classmethod
    def from_file(cls, filename):
        """Load a visBuffer from a kotekan dump file.
        """
        import os

        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf)

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as visBuffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of VisBuffers
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of VisBuffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = ctypes.c_int(ctypes.sizeof(VisMetadata))

        for ii, buf in enumerate(buffers):

            with open(pat % ii, "wb+") as fh:
                fh.write(msize_c)
                fh.write(bytearray(buf._buffer))

    @classmethod
    def new_from_params(cls, num_elements, num_prod, num_ev, insert_size=True):
        """Create a new VisBuffer owning its own memory.

        Parameters
        ----------
        num_elements, num_prod, num_ev
            Structural parameters.

        Returns
        -------
        buffer : VisBuffer
        """

        layout = cls._calculate_layout(num_elements, num_prod, num_ev)
        meta_size = ctypes.sizeof(VisMetadata)

        buf = np.zeros(meta_size + layout["size"], dtype=np.uint8)

        # Set the structure in the metadata
        metadata = VisMetadata.from_buffer(buf[:meta_size])
        metadata.num_elements = num_elements
        metadata.num_prod = num_prod
        metadata.num_ev = num_ev

        return cls(buf, skip=0)


def _offset(offset, size):
    """Calculate the start of a member of `size` after `offset` within a
    struct."""
    return ((size - (offset % size)) % size) + offset


class VisRaw(object):
    """Reader for correlator files in the raw format.

    Parses the structure of the binary files and loads them
    into an memmap-ed numpy array.

    Parameters
    ----------
    TODO
    num_time
    num_freq
    metadata
    time
    index_map
    data

    Attributes
    ----------
    data : np.ndarray
        Contains the datasets. Accessed as a numpy record array.
    metadata : dict
        Holds associated metadata, including the index_map.
    valid_frames : np.ndarray
        Indicates whether each frame is populated with valid (1) or not (0)
        data.
    time : np.ndarray
        Is the array of times, in the usual correlator file format.
    """

    def __init__(
        self,
        num_time,
        num_freq,
        num_prod,
        metadata,
        time,
        index_map,
        data,
        valid_frames,
    ):
        self.num_time = num_time
        self.num_freq = num_freq
        self.num_prod = num_prod
        self.metadata = metadata
        self.time = time
        self.index_map = index_map
        self.data = data
        self.valid_frames = valid_frames

    @classmethod
    def from_nparray(
        cls, array, size_frame, num_time, num_freq, num_elements, num_stack, num_ev,
    ):
        layout = VisBuffer._calculate_layout(num_elements, num_stack, num_ev)

        # TODO: remove this when we have fixed the alignment issue in kotekan (see self.from_file)
        dtype_layout = {"names": [], "formats": [], "offsets": []}
        for member in layout["members"]:
            dtype_layout["names"].append(member["name"])
            dtype_layout["offsets"].append(member["start"])
            if member["num"] == 1:
                dtype_layout["formats"].append((member["dtype"]))
            else:
                dtype_layout["formats"].append((member["dtype"], member["num"]))
        dtype_layout["itemsize"] = layout["size"]
        data_struct = np.dtype(dtype_layout)

        # the valid vield (1 byte) is 4-byte-aligned
        align_valid = 4

        frame_struct = np.dtype(
            {
                "names": ["valid", "metadata", "data"],
                "formats": [np.uint8, VisMetadata, data_struct],
                "offsets": [0, align_valid, align_valid + ctypes.sizeof(VisMetadata)],
                "itemsize": size_frame,
            }
        )

        # TODO: Python 3 - use native_str for compatibility
        raw = array.view(dtype=frame_struct)
        data = raw["data"]
        metadata = raw["metadata"]
        valid_frames = raw["valid"]

        ctime = metadata["ctime"]
        fpga_seq = metadata["fpga_seq"]
        num_prod = metadata["num_prod"]
        time = []
        for t in range(num_time):
            for f in range(num_freq):
                time.append((fpga_seq[t, f], ctime[t, f]))
        time = np.array(
            time,
            dtype=[
                (native_str("fpga_count"), np.uint64),
                (native_str("ctime"), time_spec),
            ],
        )

        return cls(
            num_time,
            num_freq,
            num_prod,
            metadata,
            time,
            index_map=None,
            data=data,
            valid_frames=valid_frames,
        )

    def from_file(cls, filename, mode="r", mmap=False):
        """Read correlator files in the raw format.

        Parses the structure of the binary files and loads them
        into an memmap-ed numpy array.

        Parameters
        ----------
        filename : str
            Name of file to open.
        mode : str, optional
            Mode to open file in. Defaults to read only.
        mmap : bool, optional
            Currently ignored. Use an mmap to open the file to avoid loading it all into memory.

        Returns
        -------
        VisRaw
            A VisRaw object giving access to the given file.
        """
        import msgpack

        # Get filenames
        filename = VisRaw._parse_filename(filename)
        meta_path = filename + ".meta"
        data_path = filename + ".data"

        # Read file metadata
        with io.open(meta_path, "rb") as fh:
            metadata = msgpack.load(fh, raw=False)

        index_map = metadata["index_map"]

        # TODO: (Python 3) Used native_str for compatibility here
        time = np.array(
            [(t["fpga_count"], t["ctime"]) for t in index_map["time"]],
            dtype=[
                (native_str("fpga_count"), np.uint64),
                (native_str("ctime"), np.float64),
            ],
        )

        num_freq = metadata["structure"]["nfreq"]
        num_time = metadata["structure"]["ntime"]
        num_prod = len(index_map["prod"])
        num_stack = len(index_map["stack"]) if "stack" in index_map else num_prod
        num_elements = len(index_map["input"])
        num_ev = len(index_map["ev"])

        # TODO: this doesn't work at the moment because kotekan and numpy
        # disagree on how the struct should be aligned. It turns out (as of
        # v1.16) that numpy is correct, so we should switch back, but in the
        # near term we need to force numpy to use the same alignment.

        # data_struct = [
        #     ("vis", np.complex64, self.num_stack),
        #     ("weight", np.float32, self.num_stack),
        #     ("flags", np.float32, self.num_elements),
        #     ("eval", np.float32,  self.num_ev),
        #     ("evec", np.complex64, self.num_ev * self.num_elements),
        #     ("erms", np.float32,  1),
        #     ("gain", np.complex64, self.num_elements),
        # ]
        # data_struct = np.dtype([(native_str(d[0]),) + d[1:] for d in data_struct], align=True)

        layout = VisBuffer._calculate_layout(num_elements, num_stack, num_ev)

        # TODO: remove this when we have fixed the alignment issue in kotekan
        dtype_layout = {"names": [], "formats": [], "offsets": []}
        for member in layout["members"]:
            dtype_layout["names"].append(member["name"])
            dtype_layout["offsets"].append(member["start"])
            dtype_layout["formats"].append((member["dtype"], member["num"]))
        dtype_layout["itemsize"] = layout["size"]
        data_struct = np.dtype(dtype_layout)

        frame_struct = np.dtype(
            {
                "names": ["valid", "metadata", "data"],
                "formats": [np.uint8, VisMetadata, data_struct],
                "itemsize": metadata["structure"]["frame_size"],
            }
        )

        # TODO: Python 3 - use native_str for compatibility
        # Load data into on-disk numpy array
        raw = np.memmap(
            native_str(data_path),
            dtype=frame_struct,
            mode=mode,
            shape=(num_time, num_freq),
        )
        data = raw["data"]
        metadata = raw["metadata"]
        valid_frames = raw["valid"]

        return cls(num_time, num_freq, num_prod, metadata, time, index_map, data)

    @staticmethod
    def _parse_filename(fname):
        return os.path.splitext(fname)[0]

    @classmethod
    def create(cls, name, time, freq, input_, prod, nev, stack=None):
        """Create a VisRaw file that can be written into.

        Parameters
        ----------
        name : str
            Base name of files to write.
        time : list
            Must be a list of dicts with `fpga_count` and `ctime` keys.
        freq, input_, prod : list
            Definitions of other axes. Must be lists, but exact subtypes are not checked.
        nev : int
            Number of eigenvalues/vectors saved.
        stack : list, optional
            Optional definition of a stack axis.
        """
        import msgpack

        # Validate and create the index maps
        if (
            not isinstance(time, list)
            or "fpga_count" not in time[0]
            or "ctime" not in time[0]
        ):
            raise ValueError("Incorrect format for time axis")

        if not isinstance(freq, list):
            raise ValueError("Incorrect format for freq axis")

        if not isinstance(input_, list):
            raise ValueError("Incorrect format for input axis")

        if not isinstance(prod, list):
            raise ValueError("Incorrect format for prod axis")

        index_map = {
            "time": time,
            "freq": freq,
            "input": input_,
            "prod": prod,
            "ev": list(range(nev)),
        }

        if stack is not None:
            if not isinstance(stack, list):
                raise ValueError("Incorrect format for stack axis")
            index_map["stack"] = stack

        # Calculate the structural metadata
        ninput = len(input_)
        nstack = len(stack) if stack is not None else len(prod)
        nfreq = len(freq)
        ntime = len(time)

        msize = ctypes.sizeof(VisMetadata)
        dsize = VisBuffer._calculate_layout(ninput, nstack, nev)["size"]

        structure = {
            "nfreq": nfreq,
            "ntime": ntime,
            "metadata_size": msize,
            "data_size": dsize,
            "frame_size": _offset(
                1 + msize + dsize, 4 * 1024
            ),  # Align to 4kb boundaries
        }

        attributes = {
            "git_version_tag": "hello",
            "weight_type": "inverse_variance",
            "instrument_name": "chime",
        }

        metadata = {
            "index_map": index_map,
            "structure": structure,
            "attributes": attributes,
        }

        # Write metadata to file
        with open(name + ".meta", "wb") as fh:
            msgpack.dump(metadata, fh)

        # Open the rawfile
        rawfile = cls.from_file(name, mode="w+")

        # Set the metadata on the frames that we already have
        rawfile.valid_frames[:] = 1
        rawfile.metadata["num_elements"][:] = ninput
        rawfile.metadata["num_prod"][:] = nstack
        rawfile.metadata["num_ev"][:] = nev
        rawfile.metadata["freq_id"][:] = np.arange(nfreq)[np.newaxis, :]

        fpga = np.array([t["fpga_count"] for t in time])
        ctime = np.array(
            [(int(np.floor(t["ctime"])), int((t["ctime"] % 1.0) * 1e9)) for t in time],
            dtype=[("tv", np.int64), ("tv_nsec", np.uint64)],
        )
        rawfile.metadata["fpga_seq"][:] = fpga[:, np.newaxis]
        rawfile.metadata["ctime"][:] = ctime[:, np.newaxis]

        return rawfile

    def flush(self):
        """Ensure the data is flushed to disk."""
        self.raw.flush()


def simple_visraw_data(filename, ntime, nfreq, ninput):
    """Create a simple VisRaw test file that kotekan can read.

    Parameters
    ----------
    filename : str
        Base name of files that will be written.
    ninput, nfreq, ntime : int
        Number of inputs, frequencies and times to use. These axes are given
        dummy values.

    Returns
    -------
    raw : VisRaw
        A readonly view of the VisRaw file.
    """

    nprod = ninput * (ninput + 1) // 2
    nev = 4

    time = [{"ctime": (10.0 * i), "fpga_count": i} for i in range(ntime)]

    freq = [{"centre": (800 - i * 10.0), "width": 10.0} for i in range(nfreq)]

    input_ = [(i, "test%04i" % i) for i in range(ninput)]

    prod = [(i, j) for i in range(ninput) for j in range(i, ninput)]

    raw = VisRaw.create(filename, time, freq, input_, prod, nev)

    # Set vis data
    raw.data["vis"].real = np.arange(nprod)[np.newaxis, np.newaxis, :]
    raw.data["vis"].imag = np.arange(ntime)[:, np.newaxis, np.newaxis]

    # Set weight data
    raw.data["weight"] = np.arange(nfreq)[np.newaxis, :, np.newaxis]

    # Set eigendata
    raw.data["eval"] = np.arange(nev)[np.newaxis, np.newaxis, :]
    raw.data["evec"] = np.arange(ninput * nev)[np.newaxis, np.newaxis, :]

    # Return read only view
    del raw
    return VisRaw(filename, mode="r")


def freq_id_to_stream_id(f_id):
    """ Convert a frequency ID to a stream ID. """
    pre_encode = (0, (f_id % 16), (f_id // 16), (f_id // 256))
    stream_id = (
        (pre_encode[0] & 0xF)
        + ((pre_encode[1] & 0xF) << 4)
        + ((pre_encode[2] & 0xF) << 8)
        + ((pre_encode[3] & 0xF) << 12)
    )
    return stream_id


class GpuBuffer(object):
    """Python representation of a GPU buffer dump.

    Parameters
    ----------
    buffer : np.ndarray(dtype=np.uint32)
        Visibility buffer as integers in blocked format.
    metadata: ChimeMetadata
        Associated metadata.
    """

    def __init__(self, buffer, metadata):

        self.data = buffer
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        """Load a GpuBuffer from a kotekan dump file.
        """

        with io.FileIO(filename, "rb") as fh:
            # first 4 bytes are metadata size
            fh.seek(4)
            cm = ChimeMetadata()
            fh.readinto(cm)
            buf = np.frombuffer(fh.read(), dtype=np.uint32)

        return cls(buf, cm)

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as GpuBuffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of GpuBuffers
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of GpuBuffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = np.uint32(ctypes.sizeof(ChimeMetadata))

        for ii, buf in enumerate(buffers):

            with open(pat % ii, "wb+") as fh:
                # first write metadata size
                fh.write(msize_c)
                # then metadata itself
                fh.write(buf.metadata)
                # finally visibility data
                fh.write(buf.data.astype(dtype=np.uint32).tobytes())
