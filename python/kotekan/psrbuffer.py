"""Read a pulsarPostProcess dump into python.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from ctypes import Structure, c_uint, c_uint8, c_uint32, sizeof
import os
import io


class PsrPacketHeader(Structure):
    _fields_ = [
        # word 1
        ("seconds", c_uint, 30),
        ("legacy", c_uint, 1),
        ("invalid", c_uint, 1),
        # word 2
        ("data_frame", c_uint, 24),
        ("ref_epoch", c_uint, 6),
        ("unused", c_uint, 2),
        # word 3
        ("frame_len", c_uint, 24),
        ("log_num_chan", c_uint, 5),
        ("vdif_version", c_uint, 3),
        # word 4
        ("station_id", c_uint, 16),
        ("thread_id", c_uint, 10),
        ("bit_depth", c_uint, 5),
        ("data_type", c_uint, 1),
        # word 5
        ("beam_id", c_uint, 24),
        ("unused", c_uint, 8),
        # word 6
        ("scaling_factor", c_uint32),  # psr_scaling?
        # word 7
        ("eud3", c_uint32),  # unused
        # word 8
        ("dec", c_uint, 16),
        ("ra", c_uint, 16),
    ]


class PsrPacket(Structure):
    """Struct representation of a FRB network packet, as described in ``CHIMEFRB/ch_frb_io/L0_L1_packet.hpp``
    """

    _pack_ = 1

    @classmethod
    def from_file(cls, filename, max_packets=None):
        """Load a list of psrPackets from a kotekan dump file.

        PulsarPostProcess stores packets for all 10 streams into a single
        frame, one after the other, so they ought to appear like that in the
        dumped buffer.

        """
        import os

        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        header = PsrPacketHeader.from_buffer(buf[4:])
        if header.frame_len == 629:
            data_len = 5000
        # elif header.frame_len == 3125:
        #     data_len = 6250
        else:
            raise ValueError("Don't know how to handle this format")

        struct_name = "PsrPacket_" + filename
        struct = type(struct_name, (PsrPacket,), {})
        struct._fields_ = [("header", PsrPacketHeader), ("data", c_uint8 * data_len)]

        npkts = (len(buf) - 4) // sizeof(struct)
        if max_packets:
            npkts = min(npkts, max_packets)
        return (struct * npkts).from_buffer(buf[4:])

    @classmethod
    def load_files(cls, pattern):
        """Read a set of dump files as PsrPackets.

        Parameters
        ----------
        pattern : str
            A globable pattern to read.

        Returns
        -------
        buffers : list of PsrPackets
        """
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]
