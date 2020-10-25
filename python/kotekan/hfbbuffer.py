"""Read a HFBBuffer dump into python.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import ctypes
import os
import io

import numpy as np
from kotekan import timespec


class HFBMetadata(ctypes.Structure):
    """Wrap a HFBMetadata struct."""

    _fields_ = [
        ("fpga_seq", ctypes.c_uint64),
        ("ctime", timespec.time_spec),
        ("freq_id", ctypes.c_uint32),
        ("fpga_total", ctypes.c_uint64),
        ("fpga_length", ctypes.c_uint64),
        ("num_beams", ctypes.c_uint32),
        ("num_subfreq", ctypes.c_uint32),
        ("dataset_id", ctypes.c_uint64 * 2),
    ]


class HFBBuffer(object):
    """Python representation of a HFBBuffer dump.

    Access the data through the `hfb` and `weight`
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

        meta_size = ctypes.sizeof(HFBMetadata)

        if len(self._buffer) < meta_size:
            raise ValueError("Buffer too small to contain metadata.")

        self.metadata = HFBMetadata.from_buffer(self._buffer[:meta_size])

        self._set_data_arrays()

    def _set_data_arrays(self):

        _data = self._buffer[ctypes.sizeof(HFBMetadata) :]

        layout = self.__class__.calculate_layout(
            self.metadata.num_elements, self.metadata.num_prod, self.metadata.num_ev
        )

        for member in layout["members"]:

            arr = np.frombuffer(
                _data[member["start"] : member["end"]], dtype=member["dtype"]
            )
            setattr(self, member["name"], arr)

    @classmethod
    def calculate_layout(cls, num_beams, num_subfreq):
        """Calculate the buffer layout.

        Parameters
        ----------
        num_beams, num_subfreq : int
            Length of each dimension.

        Returns
        -------
        layout : dict
            Structure of buffer.
        """

        structure = [
            ("hfb", np.float32, num_beams * num_subfreq),
            ("weight", np.float32, num_beams * num_subfreq),
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

            # make sure this dimension doesn't get squashed out if it's 1 (for everything but erms)
            if name == "erms":
                member["num"] = num
            else:
                member["num"] = (num,)

            member["dtype"] = dtype
            member["name"] = name

            members.append(member)

        struct_end = _offset(members[-1]["end"], maxsize)
        layout = {"size": struct_end, "members": members}
        return layout

    @classmethod
    def from_file(cls, filename):
        """Load a HFBBuffer from a kotekan dump file."""
        import os

        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf)

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as HFBBuffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of HFBBuffers
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of HFBBuffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = ctypes.c_int(ctypes.sizeof(HFBMetadata))

        for ii, buf in enumerate(buffers):

            with open(pat % ii, "wb+") as fh:
                fh.write(msize_c)
                fh.write(bytearray(buf._buffer))

    @classmethod
    def new_from_params(cls, num_beams, num_subfreq, insert_size=True):
        """Create a new HFBBuffer owning its own memory.

        Parameters
        ----------
        num_beams, num_subfreq
            Structural parameters.

        Returns
        -------
        buffer : HFBBuffer
        """

        layout = cls.calculate_layout(num_beams, num_subfreq)
        meta_size = ctypes.sizeof(HFBMetadata)

        buf = np.zeros(meta_size + layout["size"], dtype=np.uint8)

        # Set the structure in the metadata
        metadata = HFBMetadata.from_buffer(buf[:meta_size])
        metadata.num_beams = num_beams
        metadata.num_subfreq = num_subfreq

        return cls(buf, skip=0)


def _offset(offset, size):
    """Calculate the start of a member of `size` after `offset` within a
    struct."""
    return ((size - (offset % size)) % size) + offset


class HFBRaw(object):
    """Reader for absorber files in the raw format.

    Parses the structure of the binary files and loads them
    into an memmap-ed numpy array.

    Parameters
    ----------
    num_time : int
        Number of time samples.
    num_freq : int
        Number of frequencies.
    num_beams : int
        Number of beams.
    num_subfreq : int
        Number of sub-frequencies.
    metadata : HFBMetadata
        Metadata
    time : np.ndarray
        Is the array of times, in the usual correlator file format.
    index_map
        Index maps.
    data : np.ndarray
        Data.
    valid_frames : np.ndarray
        Validity flags for each frame in data.
    file_metadata
        From .meta raw file (optional).
    comet_manager : comet.Manager
        (optional) A comet manager instance. If this is provided, dataset states will be
        requested from the comet broker to provide gain/flaginput update IDs.

    Attributes
    ----------
    data : np.ndarray
        Contains the datasets. Accessed as a numpy record array.
    metadata : HFBMetadata
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
        num_beams,
        num_subfreq,
        metadata,
        time,
        index_map,
        data,
        valid_frames,
        file_metadata=None,
        comet_manager=None,
    ):
        self.num_time = num_time
        self.num_freq = num_freq
        self.num_beams = num_beams
        self.num_subfreq = num_subfreq
        self.metadata = metadata
        self.time = time
        self.index_map = index_map
        self.data = data
        self.valid_frames = valid_frames
        self.file_metadata = file_metadata

    @classmethod
    def frame_struct(cls, size_frame, num_beams, num_subfreq, align_valid):
        """
        Construct frame struct.

        Parameters
        ----------
        size_frame : int
            Total size of a frame in bytes.
        num_beams : int
            Number of beams in a frame.
        num_subfreq : int
            Number of sub-frequencies in a frame.
        align_valid : int
            If `True`, the valid field of the frame will be padded with 3 bytes to be aligned to 4
            bytes.

        Returns
        -------
        numpy.dtype
            Frame structure.
        """
        layout = HFBBuffer.calculate_layout(num_beams, num_subfreq)

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

        if align_valid:
            # the valid vield (1 byte) is 4-byte-aligned
            align_valid = 4
        else:
            align_valid = 1

        frame_struct = np.dtype(
            {
                "names": ["valid", "metadata", "data"],
                "formats": [np.uint8, HFBMetadata, data_struct],
                "offsets": [0, align_valid, align_valid + ctypes.sizeof(HFBMetadata)],
                "itemsize": size_frame,
            }
        )
        return frame_struct

    @classmethod
    def from_buffer(cls, buffer, size_frame, num_time, num_freq, comet_manager=None):
        """
        Create a HFBRaw object from a buffer.

        Parameters
        ----------
        buffer : buffer_like
            Input data.
        size_frame : int
            Size of a frame in bytes.
        num_time : int
            Number of time samples in the buffer.
        num_freq : int
            Number of frequencies in the buffer.
        comet_manager : comet.Manager
            (optional) A comet manager instance. If this is provided, dataset states will be
            requested from the comet broker to provide index maps and gain/flaginput update IDs.

        Returns
        -------
        HFBRaw
            HFBRaw viewing the data in the buffer.

        Raises
        ------
        ValueError
            If there was a problem parsing the buffer into the HFBRaw structure.
        """
        # Create a simple struct to access the metadata (num_elements, num_prod, num_ev)
        align_valid = 4  # valid field is 4 byte aligned
        frame_struct_simple = np.dtype(
            {
                "names": ["valid", "metadata", "data"],
                "formats": [
                    np.uint8,
                    HFBMetadata,
                    (np.void, size_frame - align_valid - ctypes.sizeof(HFBMetadata)),
                ],
                "offsets": [0, align_valid, align_valid + ctypes.sizeof(HFBMetadata)],
                "itemsize": size_frame,
            }
        )
        raw = buffer.view(dtype=frame_struct_simple)
        num_beams = np.unique(
            raw["metadata"][raw["valid"].astype(np.bool)]["num_beams"]
        )
        if len(num_beams) > 1:
            raise ValueError(
                "Found more than 1 value for `num_beams` in numpy ndarray: {}.".format(
                    num_beams
                )
            )
        num_beams = num_beams[0]
        num_subfreq = np.unique(
            raw["metadata"][raw["valid"].astype(np.bool)]["num_subfreq"]
        )
        if len(num_subfreq) > 1:
            raise ValueError(
                "Found more than 1 value for `num_subfreq` in numpy ndarray: {}.".format(
                    num_subfreq
                )
            )
        num_subfreq = num_subfreq[0]

        # Now that we have some metadata, we can really parse the data...
        frame_struct = cls.frame_struct(
            size_frame, num_beams, num_subfreq, align_valid=True
        )

        raw = buffer.view(dtype=frame_struct)
        data = raw["data"]
        metadata = raw["metadata"]
        valid_frames = raw["valid"]

        ctime = metadata["ctime"]
        fpga_seq = metadata["fpga_seq"]

        num_beams = metadata["num_beams"]
        num_subfreq = metadata["num_subfreq"]

        time = np.ndarray(
            shape=(num_time, num_freq),
            dtype=[("fpga_count", np.uint64), ("ctime", np.float64)],
        )

        # flatten time index map (we only need one value per time slot, but we have one per
        # frame/frequency)
        for t in range(num_time):
            ts = []
            fpga = []
            for f in range(num_freq):
                if valid_frames[t, f].astype(np.bool):
                    ts.append(
                        timespec.time_spec.from_buffer_copy(ctime[t, f]).to_float()
                    )
                    fpga.append(fpga_seq[t, f])
            ts = np.unique(ts)
            fpga = np.unique(fpga)
            if len(ts) != 1 or len(fpga) != 1:
                if len(ts) == 0 and len(fpga) == 0:
                    # the whole time slot is invalid
                    time[t] = (0, 0)
                else:
                    raise ValueError(
                        "Found {} fpga sequences and {} time specs for time index {} "
                        "(Expected one each).".format(len(fpga), len(ts), t)
                    )
            else:
                time[t] = (fpga[0], ts[0])

        # generate index maps
        index_map = {"time": time}
        if comet_manager is not None:
            # add beam, subfreq and freq
            state_axis_map = [
                ("beams", "beam"),
                ("sub-frequencies", "subfreq"),
                ("frequencies", "freq"),
            ]
            ds = np.array(metadata["dataset_id"][valid_frames.astype(np.bool)]).view(
                "u8,u8"
            )
            unique_ds = np.unique(ds)
            state = {names[0]: set() for names in state_axis_map}
            for ds in unique_ds:
                ds_id = "{:016x}{:016x}".format(ds[1], ds[0])
                for names in state_axis_map:
                    state[names[0]].add(comet_manager.get_state(names[0], ds_id))
            for names in state_axis_map:
                if len(state[names[0]]) == 0:
                    state[names[0]] = None
                elif len(state[names[0]]) == 1:
                    state[names[0]] = state[names[0]].pop()
                else:
                    raise ValueError(
                        "Found more than one {} state when looking up dataset IDs "
                        "found in metadata.".format(names[0])
                    )
                if state[names[0]] is not None:
                    index_map[names[1]] = state[names[0]].data["data"]

            # Convert index_map into numpy arrays
            if "beam" in index_map:
                index_map["beam"] = np.array(index_map["beam"])
            if "subfreq" in index_map:
                index_map["subfreq"] = np.array(index_map["subfreq"])
            if "freq" in index_map:
                index_map["freq"] = np.array(
                    [(ff[1]["centre"], ff[1]["width"]) for ff in index_map["freq"]],
                    dtype=[("centre", np.float32), ("width", np.float32)],
                )

        return cls(
            num_time,
            num_freq,
            num_beams,
            num_subfreq,
            metadata,
            time,
            index_map=index_map,
            data=data,
            valid_frames=valid_frames,
            comet_manager=comet_manager,
        )

    @classmethod
    def from_file(cls, filename, mode="r", mmap=False):
        """Read absorber files in the raw format.

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
        HFBRaw
            A HFBRaw object giving access to the given file.
        """
        import msgpack

        # Get filenames
        filename = HFBRaw._parse_filename(filename)
        meta_path = filename + ".meta"
        data_path = filename + ".data"

        # Read file metadata
        with io.open(meta_path, "rb") as fh:
            metadata = msgpack.load(fh, raw=False)

        index_map = metadata["index_map"]

        time = np.array(
            [(t["fpga_count"], t["ctime"]) for t in index_map["time"]],
            dtype=[("fpga_count", np.uint64), ("ctime", np.float64)],
        )

        num_freq = metadata["structure"]["nfreq"]
        num_time = metadata["structure"]["ntime"]
        num_beams = len(index_map["beam"])
        num_subfreqs = len(index_map["subfreq"])

        frame_struct = cls.frame_struct(
            metadata["structure"]["frame_size"],
            num_beams,
            num_subfreqs,
            align_valid=False,
        )

        file_metadata = metadata

        # Load data into on-disk numpy array
        raw = np.memmap(
            data_path, dtype=frame_struct, mode=mode, shape=(num_time, num_freq)
        )
        data = raw["data"]
        metadata = raw["metadata"]
        valid_frames = raw["valid"]

        return cls(
            num_time,
            num_freq,
            num_beams,
            num_subfreqs,
            metadata,
            time,
            index_map,
            data,
            valid_frames,
            file_metadata,
        )

    @staticmethod
    def _parse_filename(fname):
        return os.path.splitext(fname)[0]

    @classmethod
    def create(cls, name, time, freq, beam, subfreq, stack=None):
        """Create a HFBRaw file that can be written into.

        Parameters
        ----------
        name : str
            Base name of files to write.
        time : list
            Must be a list of dicts with `fpga_count` and `ctime` keys.
        freq, beam, subfreq : list
            Definitions of other axes. Must be lists, but exact subtypes are not checked.
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

        if not isinstance(beam, list):
            raise ValueError("Incorrect format for beam axis")

        if not isinstance(subfreq, list):
            raise ValueError("Incorrect format for subfreq axis")

        index_map = {"time": time, "freq": freq, "beam": beam, "subfreq": subfreq}

        # Calculate the structural metadata
        nbeam = len(beam)
        nsubfreq = len(subfreq)
        nfreq = len(freq)
        ntime = len(time)

        msize = ctypes.sizeof(HFBMetadata)
        dsize = HFBBuffer.calculate_layout(nbeam, nsubfreq)["size"]

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
        rawfile.metadata["num_beams"][:] = nbeam
        rawfile.metadata["num_subfreqs"][:] = nsubfreq
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


def simple_hfbraw_data(filename, ntime, nfreq, nbeam, nsubfreq):
    """Create a simple HFBRaw test file that kotekan can read.

    Parameters
    ----------
    filename : str
        Base name of files that will be written.
    nbeam, nsubfreq, nfreq, ntime : int
        Number of beams, frequencies and times to use. These axes are given
        dummy values.

    Returns
    -------
    raw : HFBRaw
        A readonly view of the HFBRaw file.
    """

    time = [{"ctime": (10.0 * i), "fpga_count": i} for i in range(ntime)]

    freq = [{"centre": (800 - i * 10.0), "width": 10.0} for i in range(nfreq)]

    beam = [b for i in range(nbeam)]

    subfreq = [sf for i in range(nsubfreq)]

    raw = HFBRaw.create(filename, time, freq, beam, subfreq)

    # Set vis data
    raw.data["hfb"] = np.arange(nbeam * nsubfreq)[np.newaxis, np.newaxis, :]

    # Set weight data
    raw.data["weight"] = np.arange(nbeam * nsubfreq)[np.newaxis, :, np.newaxis]

    # Return read only view
    del raw
    return HFBRaw(filename, mode="r")


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
