"""Read an N2Buffer dump into python.
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

"""
class CTimeSpec(ctypes.Structure):

    _fields_ = [
        ("tv_sec", ctypes.c_time_t),
        ("tv_nsec", ctypes.c_long),
    ]
"""


class EOP(ctypes.Structure):

    _fields_ = [
        ("t_inst", ctypes.c_int64),
        ("t_ut1", ctypes.c_int64),
        ("delta_UT1_inst", ctypes.c_double),
        ("ERA_deg", ctypes.c_double),
        ("xp_as", ctypes.c_double),
        ("yp_as", ctypes.c_double),
    ]


class N2Metadata(ctypes.Structure):
    """Wrap an N2Metadata struct."""

    _fields_ = [
        ("num_elements", ctypes.c_uint32),
        ("num_prod", ctypes.c_uint32),
        ("num_ev", ctypes.c_uint32),
        ("nfreq", ctypes.c_uint32),
        ("freq_id", ctypes.c_uint32),
        ("freq_Hz", ctypes.c_double),
        ("eop", EOP),
        ("fpga_start_tick", ctypes.c_uint64),
        ("frame_start_time_ns", ctypes.c_uint64),
        ("frame_length_fpga_ticks", ctypes.c_uint64),
        ("n_valid_fpga_ticks", ctypes.c_uint64),
        ("n_rfi_fpga_ticks", ctypes.c_uint64),
    ]


class N2Buffer(object):
    """Python representation of an N2Buffer dump.

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

        meta_size = ctypes.sizeof(N2Metadata)

        # print("Loading N2Buffer")
        # print("File size: {:d}".format(len(buffer)))
        # print("buffer size: {:d}".format(len(self._buffer)))
        # print("meta size: {:d}".format(meta_size))
        # print("data size: {:d}".format(len(self._buffer) - meta_size))

        if len(self._buffer) < meta_size:
            raise ValueError("Buffer too small to contain metadata.")

        self.metadata = N2Metadata.from_buffer(self._buffer[:meta_size])

        self._set_data_arrays()

    def _set_data_arrays(self):

        _data = self._buffer[ctypes.sizeof(N2Metadata) :]

        layout = self.__class__.calculate_layout(
            self.metadata.num_elements, self.metadata.num_prod, self.metadata.num_ev
        )

        for member in layout["members"]:

            arr = np.frombuffer(
                _data[member["start"] : member["end"]], dtype=member["dtype"]
            )
            setattr(self, member["name"], arr)

    @classmethod
    def calculate_layout(cls, num_elements, num_prod, num_ev):
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
            ("emethod", np.int32, 1),
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
        """Load an N2Buffer from a kotekan dump file."""
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf)

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as N2Buffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of N2Buffers
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of N2Buffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = ctypes.c_int(ctypes.sizeof(N2Metadata))

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
        buffer : N2Buffer
        """

        layout = cls.calculate_layout(num_elements, num_prod, num_ev)
        meta_size = ctypes.sizeof(N2Metadata)

        buf = np.zeros(meta_size + layout["size"], dtype=np.uint8)

        # Set the structure in the metadata
        metadata = N2Metadata.from_buffer(buf[:meta_size])
        metadata.num_elements = num_elements
        metadata.num_prod = num_prod
        metadata.num_ev = num_ev

        return cls(buf, skip=0)


def _offset(offset, size):
    """Calculate the start of a member of `size` after `offset` within a
    struct."""
    return ((size - (offset % size)) % size) + offset
