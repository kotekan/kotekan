"""Read a frbBuffer dump into python.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import ctypes
import os
import io


class FrbPacketHeader(ctypes.Structure):
    """Struct representation of the static section of a FRB network packet, as described in ``FRBHeader``"""

    _fields_ = [
        ("version", ctypes.c_uint32),
        ("nbytes", ctypes.c_int16),
        ("fpga_counts_per_sample", ctypes.c_uint16),
        ("fpga_seq_num", ctypes.c_uint64),
        ("nbeams", ctypes.c_uint16),
        ("nfreq", ctypes.c_uint16),
        ("nupfreq", ctypes.c_uint16),
        ("ntsamp", ctypes.c_uint16),
    ]


class FrbPacket(ctypes.Structure):
    """Struct representation of a FRB network packet, as described in ``CHIMEFRB/ch_frb_io/L0_L1_packet.hpp``"""

    _pack_ = 1

    @classmethod
    def from_file(cls, filename, max_packets=None):
        """Load a list of frbPackets from a kotekan dump file.

        FrbPostProcess stores packets for all 256 streams into a single frame,
        one after the other, so they ought to appear like that in the dumped
        buffer.

        """
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        header = FrbPacketHeader.from_buffer(buf[4:])
        struct_name = ("FrbPacket_" + filename).encode("ascii")
        struct = type(struct_name, (FrbPacket,), {})
        struct._fields_ = [
            ("header", FrbPacketHeader),
            ("beam_ids", ctypes.c_uint16 * header.nbeams),
            ("freq_ids", ctypes.c_uint16 * header.nfreq),
            ("scales", ctypes.c_float * (header.nbeams * header.nfreq)),
            ("offsets", ctypes.c_float * (header.nbeams * header.nfreq)),
            ("data", ctypes.c_ubyte * header.nbytes),
        ]

        npkts = (len(buf) - 4) // ctypes.sizeof(struct)
        if max_packets:
            npkts = min(npkts, max_packets)
        return (struct * npkts).from_buffer(buf[4:])

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as frbPackets.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of frbPackets
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]
