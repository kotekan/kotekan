"""Read a visBuffer dump into python.
"""

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
        ("dataset_id", ctypes.c_uint32),
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

        num_prod = self.metadata.num_prod
        num_elements = self.metadata.num_elements
        num_eigen = self.metadata.num_ev

        structure = [
            ('vis', np.complex64, num_prod),
            ('weight', np.float32, num_prod),
            ("eval", np.float32,  num_eigen),
            ("evec", np.complex64, num_eigen * num_elements),
            ("erms", np.float32,  1)
        ]

        end = 0

        for name, dtype, num in structure:
            size = np.dtype(dtype).itemsize

            start = _offset(end, size)
            end = start + num * size

            arr = np.frombuffer(_data[start:end], dtype=dtype)
            setattr(self, name, arr)

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
        import glob

        return [cls.from_file(fname) for fname in sorted(glob.glob(pattern))]


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
        
        with open(meta_path, 'r') as fh:
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