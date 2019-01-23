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
        ("freq_id", ctypes.c_uint32),
        ("dataset_id", ctypes.c_uint64),
        ("num_elements", ctypes.c_uint32),
        ("num_prod", ctypes.c_uint32),
        ("num_ev", ctypes.c_uint32)
    ]


class psrCoord(ctypes.Structure):
    """ Struct repr of psrCoord field in ChimeMetadata."""

    _fields_ = [("ra", ctypes.ARRAY(ctypes.c_float, 10)),
               ("dec", ctypes.ARRAY(ctypes.c_float, 10)),
               ("scaling", ctypes.ARRAY(ctypes.c_uint32, 10))]


class ChimeMetadata(ctypes.Structure):
    """Wrap a ChimeMetadata struct."""

    _fields_ = [
        ("fpga_seq_num", ctypes.c_uint64),
        ("first_packet_recv_time", timeval),
        ("gps_time", time_spec),
        ("lost_timesamples", ctypes.c_int32),
        ("stream_ID", ctypes.c_uint16),
        ("psrCoord", psrCoord),
        ("rfi_zeroed", ctypes.c_uint32)
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
    """Reader for correlator files in the raw format.

    Parses the structure of the binary files and loads them
    into an memmap-ed numpy array.

    Parameters
    ----------
    filename : str
        Name of file to open.
    mmap : bool, optional
        Use an mmap to open the file to avoid loading it all into memory.

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

    def __init__(self, filename, mmap=False):

        import msgpack

        # Get filenames
        self.filename = self._parse_filename(filename)
        self.meta_path = self.filename + ".meta"
        self.data_path = self.filename + ".data"

        # Read file metadata
        with open(self.meta_path, 'rb') as fh:
            metadata = msgpack.load(fh)

        self.index_map = metadata['index_map']

        self.time = np.array(
            [(t['fpga_count'], t['ctime']) for t in self.index_map['time']],
            dtype=[('fpga_count', np.uint64), ('ctime', np.float64)]
        )

        self.num_freq = metadata['structure']['nfreq']
        self.num_time = metadata['structure']['ntime']
        self.num_prod = len(self.index_map['prod'])
        self.num_elements = len(self.index_map['input'])
        self.num_ev = len(self.index_map['ev'])

        # Packing of the data on disk. First byte indicates if data is present.
        data_struct = np.dtype([
            ('vis', np.complex64, self.num_prod),
            ('weight', np.float32, self.num_prod),
            ('flags', np.float32, self.num_elements),
            ("eval", np.float32,  self.num_ev),
            ("evec", np.complex64, self.num_ev * self.num_elements),
            ("erms", np.float32,  1),
            ("gain", np.complex64, self.num_elements),
        ], align=True)
        frame_struct = np.dtype({
            'names': ['valid', 'metadata', 'data'],
            'formats': [np.uint8, VisMetadata, data_struct],
            'itemsize': metadata['structure']['frame_size']
        })

        # Load data into on-disk numpy array
        self.raw = np.memmap(self.data_path, dtype=frame_struct, mode='r',
                             shape=(self.num_time, self.num_freq))
        self.data = self.raw['data']
        self.metadata = self.raw['metadata']
        self.valid_frames = self.raw['valid']
        self.file_metadata = metadata

    @staticmethod
    def _parse_filename(fname):
        return os.path.splitext(fname)[0]


def freq_id_to_stream_id(f_id):
    """ Convert a frequency ID to a stream ID. """
    pre_encode = (0, (f_id % 16), (f_id / 16), (f_id / 256))
    stream_id = ((pre_encode[0] & 0xF) +
                 ((pre_encode[1] & 0xF) << 4) +
                 ((pre_encode[2] & 0xF) << 8) +
                 ((pre_encode[3] & 0xF) << 12))
    return stream_id


class GpuBuffer(object):
    """Python representation of a GPU buffer dump.

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

    def __init__(self, buffer, metadata):

        self.data = buffer
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        """Load a GpuBuffer from a kotekan dump file.
        """

        with io.FileIO(filename, 'rb') as fh:
            # first 4 bytes are metadata size
            msize = np.frombuffer(fh.read(4), np.uint32)[0]
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

            with open(pat % ii, 'wb+') as fh:
                # first write metadata size
                fh.write(msize_c)
                # then metadata itself
                fh.write(buf.metadata)
                # finally visibility data
                fh.write(buf.data.astype(dtype=np.uint32).tobytes())
