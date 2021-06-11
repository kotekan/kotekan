"""Read a BasebandBuffer dump into python.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import ctypes
import glob
import numpy as np
import os
import io


class BasebandMetadata(ctypes.Structure):
    """Wrap a BasebandMetadata struct."""

    _fields_ = [
        # event and frequency id
        ("event_id", ctypes.c_uint64),
        ("freq_id", ctypes.c_uint64),
        #
        # event start and end at this frequency
        ("event_start_seq", ctypes.c_uint64),
        ("event_end_seq", ctypes.c_uint64),
        #
        # timestamp of the first captured sample
        ("time0_fpga", ctypes.c_uint64),
        ("time0_ctime", ctypes.c_double),
        ("time0_ctime_offset", ctypes.c_double),
        #
        # TOA of the packet with the first sample
        ("first_packet_recv_time", ctypes.c_double),
        #
        # FPGA seq of the first sample in this frame
        ("frame_fpga_seq", ctypes.c_uint64),
        #
        # Number of valid samples in this frame
        ("valid_to", ctypes.c_uint64),
        #
        # Time of FPGA frame=0
        ("fpga0_ns", ctypes.c_uint64),
        #
        # Number of inputs per sample
        ("num_elements", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class BasebandBuffer(object):
    """Python representation of a BasebandBuffer dump.

    Parameters
    ----------
    buffer : bytearray
        Memory to provide a view of.
    skip : int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    """

    meta_size = ctypes.sizeof(BasebandMetadata)

    def __init__(self, buffer, skip=4):

        self._buffer = buffer[skip:]

        if len(self._buffer) < self.meta_size:
            raise ValueError("Buffer too small to contain metadata.")

        self.metadata = BasebandMetadata.from_buffer(self._buffer[: self.meta_size])

    @classmethod
    def from_file(cls, filename):
        """Load a BasebandBuffer from a kotekan dump file."""
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf)

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as BasebandBuffers.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of BasebandBuffers
        """
        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]

    @classmethod
    def to_files(cls, buffers, basename):
        """Write a list of buffers to disk.

        Parameters
        ----------
        buffers : list of BasebandBuffers
            Buffers to write.
        basename : str
            Basename for filenames.
        """
        pat = basename + "_%07d.dump"

        msize_c = ctypes.c_int(ctypes.sizeof(BasebandMetadata))

        for ii, buf in enumerate(buffers):

            with open(pat % ii, "wb+") as fh:
                fh.write(msize_c)
                fh.write(bytearray(buf._buffer))

    @classmethod
    def new_from_params(
        cls, event_id, freq_id, num_elements, frame_size, frame_data=None
    ):
        """Create a new BasebandBuffer owning its own memory.

        Parameters
        ----------
        event_id, freq_id
            Baseband event properties
        num_elements
            Number of telescope inputs
        frame_size:
            length of frame in bytes
        frame_data: optional List(bytes)
            value of samples

        Returns
        -------
        buffer : BasebandBuffer
        """

        # layout = cls.calculate_layout(num_elements, num_prod, num_ev)
        meta_size = ctypes.sizeof(BasebandMetadata)

        buf = np.zeros(meta_size + frame_size, dtype=np.uint8)
        if frame_data:
            buf[meta_size:] = frame_data

        # Set the structure in the metadata
        metadata = BasebandMetadata.from_buffer(buf[:meta_size])
        metadata.event_id = event_id
        metadata.freq_id = freq_id
        metadata.num_elements = num_elements

        return cls(buf, skip=0)
