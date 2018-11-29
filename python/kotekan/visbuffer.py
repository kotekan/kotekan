"""Read a visBuffer dump into python.
"""
# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import ctypes
import os
import io

import numpy as np


class time_spec(ctypes.Structure):
    """Struct repr of a timespec type."""
    _fields_ = [
        ('tv', ctypes.c_int64),
        ('tv_nsec', ctypes.c_uint64)
    ]


class VisMetadata(ctypes.Structure):
    """Wrap a visMetadata struct.
    """

    _fields_ = [
        ("fpga_seq", ctypes.c_uint64),
        ("ctime", time_spec),
        ("fpga_length", ctypes.c_uint64),
        ("fpga_total", ctypes.c_uint64),
        ("freq_id", ctypes.c_uint32),
        ("dataset_id", ctypes.c_uint64),
        ("num_elements", ctypes.c_uint32),
        ("num_prod", ctypes.c_uint32),
        ("num_ev", ctypes.c_uint32)
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

        _data = self._buffer[ctypes.sizeof(VisMetadata):]

        layout = self.__class__._calculate_layout(self.metadata.num_elements,
                                                  self.metadata.num_prod,
                                                  self.metadata.num_ev)

        for member in layout['members']:

            arr = np.frombuffer(_data[member['start']:member['end']],
                                dtype=member['dtype'])
            setattr(self, member['name'], arr)

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
            ('vis', np.complex64, num_prod),
            ('weight', np.float32, num_prod),
            ('flags', np.float32, num_elements),
            ("eval", np.float32,  num_ev),
            ("evec", np.complex64, num_ev * num_elements),
            ("erms", np.float32,  1),
            ("gain", np.complex64, num_elements)
        ]

        end = 0

        members = []
        maxsize = 0

        for name, dtype, num in structure:

            member = {}

            size = np.dtype(dtype).itemsize

            # Update the maximum size
            maxsize = size if maxsize < size else maxsize

            member['start'] = _offset(end, size)
            end = member['start'] + num * size
            member['end'] = end
            member['size'] = num * size
            member['num'] = num
            member['dtype'] = dtype
            member['name'] = name

            members.append(member)

        struct_end = _offset(members[-1]['end'], maxsize)
        layout = {'size': struct_end, 'members': members}
        return layout

    @classmethod
    def from_file(cls, filename):
        """Load a visBuffer from a kotekan dump file.
        """
        import os

        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, 'rb') as fh:
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

            with open(pat % ii, 'wb+') as fh:
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

        buf = np.zeros(meta_size + layout['size'], dtype=np.uint8)

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
    """Read a raw visibilty file.

    Parameters
    ----------
    filename : string
        Path to either `.meta` or `.data` file, or common root.
    """

    def __init__(self, filename):

        import msgpack

        base = os.path.splitext(filename)[0]

        meta_path = base + '.meta'
        data_path = base + '.data'

        with open(meta_path, 'rb') as fh:
            self.metadata = msgpack.load(fh)

        self.data = []

        frame_size = self.metadata['structure']['frame_size']
        data_size = self.metadata['structure']['data_size']
        metadata_size = self.metadata['structure']['metadata_size']

        nfreq = self.metadata['structure']['nfreq']
        ntime = self.metadata['structure']['ntime']

        with io.FileIO(data_path, 'rb') as fh:

            for ti in range(ntime):

                fs = []

                for fi in range(nfreq):

                    buf = bytearray(data_size + metadata_size + 1)
                    fh.seek((ti * nfreq + fi) * frame_size)
                    fh.readinto(buf)

                    if buf[0] == 0:
                        fs.append(None)
                    else:
                        fs.append(VisBuffer(buf, skip=1))

                self.data.append(fs)