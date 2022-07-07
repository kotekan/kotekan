""" Tracking beam buffer dump into python
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import ctypes
import os
import io
import gc

import numpy as np
from astropy.utils import lazyproperty
from kotekan import timespec


class BeamMetadata(ctypes.Structure):
    """Wrap a BeamMetadata struct.
    """
    _fields_ = [
        ("fpga_seq_start", ctypes.c_uint64),
        ("ctime", timespec.time_spec),
        ("stream_id", ctypes.c_uint64),
        ("dataset_id", ctypes.c_uint64 * 2),
        ("beam_number", ctypes.c_uint32),
        ("ra", ctypes.c_float),
        ("dec", ctypes.c_float),
        ("scaling", ctypes.c_uint32)
    ]


class FreqIDBeamMetadata(ctypes.Structure):
     _fields_ = [
        ("fpga_seq_start", ctypes.c_uint64),
        ("ctime", timespec.time_spec),
        ("stream_id", ctypes.c_uint64),
        ("dataset_id", ctypes.c_uint64 * 2),
        ("beam_number", ctypes.c_uint32),
        ("ra", ctypes.c_float),
        ("dec", ctypes.c_float),
        ("scaling", ctypes.c_uint32),
        ("frequency_bin", ctypes.c_uint32)
    ]


class MergedBeamMetadata(ctypes.Structure):
    _fields_ = [
        ("sub_frame_per_frame", ctypes.c_uint32),
        ("sub_frame_metadata_size", ctypes.c_uint32),
        ("sub_frame_data_size", ctypes.c_uint32),
        ("freq_start", ctypes.c_uint32),
        ("nchan", ctypes.c_uint32),
        ("nframe", ctypes.c_uint32),
        ("n_sample_per_frame", ctypes.c_uint32),
        ("n_pol", ctypes.c_uint32),
        ("fpga_seq_start", ctypes.c_uint64),
        ("ctime", timespec.time_spec)
    ]

# Build uint_8 to complex64 lookup table.
#decoder_levels = np.array([np.nan] +list(range(-7, 8)))
decoder_levels = np.array(list(range(-8, 8)))
lut = (decoder_levels * 1j + decoder_levels[:, np.newaxis]).ravel().astype('c8')

def decode_4bit(words):
    """Decode 4-bit data.
    For a given int8 byte containing bits 76543210,

    """
    return lut[words]
    

class BeamFrameBase(object):
    """Class of Python interface for tracking beam frame.

    The tracking beam frame comes form CHIME correlator.
    It is organized as metadata-data block. The data
    contains 4-4bits complex data.

    Parameters
    ----------
    frame: bytearray
        Memory of tracking beam raw data.
    metadata_cls: `BeamMetadata`or `FreqIDBeamMetadata` class
        The metadata of the beam frame.
    skip : int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    """
    def __init__(self, frame, metadata_cls=None, skip=0):
        self._buffer = frame[skip:]
        if metadata_cls:
            meta_size = ctypes.sizeof(metadata_cls)
            if len(self._buffer) < meta_size:
                raise ValueError("Frame is too small to contain metadata.")
            self.metadata = metadata_cls.from_buffer(self._buffer[:meta_size])
        else:
            self.metadata = None
         
    def _read_frame(self):
        raise NotImplementedError

    def _set_data_arrays(self):
        """Read data from buffer to numpy array.
        """
        raise NotImplementedError

    @classmethod
    def calculate_layout(cls):
        """Calculate the frame layout

        Parameters
        ----------
        frame_size: int, optional
            The frame data size
        num_pol:
            Number of polarization.
        
        Returns
        -------
        layout : dict
            Structure of buffer.
        """
        raise NotImplementedError
        
    @classmethod
    def from_file(cls, filename):
        """Load a BeamBuffer from a kotekan dump file.
        """
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf)


class SingleBeamFrame(BeamFrameBase):
    """ Class of Python interface for single tracking beam frame.
    
    Parameters
    ----------
    frame: bytearray
        Memory of merged tracking beam raw data.
    metadata_cls: `BeamMetadata` class or `FreqIDBeamMetadata`
        Metadata class for a single frame
    samples_per_frame: int
        Time samples for this frame
    num_pol: int
        Number of polarizations in the frame.
    skip: int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    """
    def __init__(self, frame, metadata_cls, samples_per_frame, num_pol, skip=0):
        super().__init__(frame, metadata_cls, skip)
        self.samples_per_frame = samples_per_frame
        self.num_pol = num_pol
        self.metadata_size = ctypes.sizeof(metadata_cls)
        self.data_size = len(self._buffer[ctypes.sizeof(metadata_cls) :])
        print("Single Frame: ", self.data_size, samples_per_frame * num_pol)
        if self.data_size != samples_per_frame * num_pol:
            raise ValueError("Data size in the file does not match the"
                             " `samples_per_frame` and `num_pol` input.")
        self.data = None
        self.sub_frame_per_frame = 1
        self.offset = 0
        self.sample_offset = 0

    def get_single_frame(index):
        if index > 0:
            raise ValueError("Single frame buffer can only get to the frame index of 0.")
        return self

    def read_frame(self, decoder=decode_4bit):
        raw = np.frombuffer(self._buffer[self.metadata_size:], 
                            dtype=np.uint8) 
        decoded = decode_4bit(raw) 
        self.data = decoded.reshape(self.samples_per_frame, self.num_pol)
        self.offset += 1

    def read(self, read_samples):
        if self.sample_offset + read_samples > self.samples_per_frame:
            raise ValueError("Not enough samples in the frame to read.")
        
        if self.data is None:
            self.read_frame()
        offset0 = self.sample_offset
        self.sample_offset += read_samples
        #print("Single frame offset0", offset0, read_samples, self.metadata.frequency_bin, 
        #       self.metadata.ctime.to_float())
        return self.data[offset0 : offset0 + read_samples]

    @classmethod
    def from_file(cls, filename, metadata_cls, samples_per_frame, num_pol, skip=0):
        """Load a BeamBuffer from a kotekan dump file.
        """
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)

        return cls(buf, metadata_cls, samples_per_frame, num_pol, skip)


class MergedBeamFrame(BeamFrameBase):
    """Class of Python interface for merged tracking beam frame.

    To reduce the file I/O for the tracking beam frames, 
    MergedBeamFrame puts the single frames together. Structure:
    merged metadata block + sub-frames.

    Parameters
    ----------
    frame: bytearray
        Memory of merged tracking beam raw data.
    metadata_cls: `MergedBeamMetadata` class, optinal 
        If provide, it will read frame information from the metadata. If not,
        the subframe information `sub_frame_per_frame`, `sub_frame_metadata_size`, 
        `sub_frame_data_size` should be give.
    skip : int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    """
    def __init__(self, frame, single_frame_metadata_cls, samples_per_single_frame,
                 num_pol, single_frame_skip=0, merged_metadata_cls=None, 
                 merged_frame_skip=0, sub_frame_per_frame=None):
        super().__init__(frame, merged_metadata_cls, merged_frame_skip)
        # TODO need to change the design here.
        if self.metadata is None:
            self.sub_frame_per_frame = sub_frame_per_frame
            self.sub_frame_metadata_size = ctypes.sizeof(single_frame_metadata_cls)
            self.sub_frame_data_size = samples_per_single_frame * num_pol
            self.metadata_size = 0
        else:# read information from metadata
            self.sub_frame_per_frame = self.metadata.sub_frame_per_frame
            self.sub_frame_metadata_size = self.metadata.sub_frame_metadata_size
            self.sub_frame_data_size = self.metadata.sub_frame_data_size
            self.metadata_size = ctypes.sizeof(merged_metadata_cls)
            print("signle frame metadata size:", self.sub_frame_metadata_size, ctypes.sizeof(single_frame_metadata_cls))
            if int(self.sub_frame_metadata_size) != int(ctypes.sizeof(single_frame_metadata_cls)):
                raise ValueError("Input single frame metadata size does not match the buffer metadata.")
            print("signle data size: ", self.sub_frame_data_size * num_pol, samples_per_single_frame * num_pol)
            if self.sub_frame_data_size != samples_per_single_frame * num_pol:
                raise ValueError("Input single frame data size does not match the buffer metadata.")
        self.single_frame_metadata_cls = single_frame_metadata_cls
        self.samples_per_single_frame = samples_per_single_frame
        self.num_pol = num_pol
        self.single_frame_skip = single_frame_skip
        # Check the sub-frame information
        if (self.sub_frame_per_frame is None or self.sub_frame_per_frame <=0):
            raise ValueError("Number of sub-frame per frame does not provided"
                             " correctly.")
        if (self.sub_frame_metadata_size is None or self.sub_frame_metadata_size <=0):
            raise ValueError("Sub-frame metadata size does not provided"
                             " correctly.")
        if (self.sub_frame_data_size is None or self.sub_frame_data_size <=0):
            raise ValueError("Sub-frame data size does not provided"
                             " correctly.")
        self.data_size = len(self._buffer[self.metadata_size:])
        print(self.metadata_size, self.data_size, self.sub_frame_per_frame * (self.sub_frame_metadata_size
            + self.sub_frame_data_size))
        print(self.sub_frame_per_frame , self.sub_frame_metadata_size, self.sub_frame_data_size)
        if self.data_size != self.sub_frame_per_frame * (self.sub_frame_metadata_size 
            + self.sub_frame_data_size):
            raise ValueError("The input `sub_frame_per_frame`,"
                             " `sub_frame_metadata_size`, `sub_frame_data_size`"
                             " does not match the buffer data size.")
        self.offset = 0
    
    def get_single_frame(self, index, read_data=False):
        """Read single tracking beam frames.

        Parameter
        ---------
        index: int
            The index of the sub-frame to read.
        read_data: bool, optional
            If reading the data to single beam.
        Return
        ------
        SingleBeamFrame object with data read-in.
        """
        if index >= self.sub_frame_per_frame:
            raise ValueError("Index out off the subframes in the merged frame.")
        sub_frame_size = self.sub_frame_data_size + self.sub_frame_metadata_size
        offset = self.metadata_size + index * sub_frame_size
        sframe_buff = self._buffer[offset:offset + sub_frame_size]
        sframe = SingleBeamFrame(sframe_buff, self.single_frame_metadata_cls, 
            self.samples_per_single_frame, self.num_pol, self.single_frame_skip)
        if read_data:
            sframe.read_data()
        return sframe

    def read(self, index):
        self.offset = index + 1
        return self.get_single_frame(index, read_data=True).data

    @classmethod
    def from_file(cls, filename, single_frame_metadata_cls, samples_per_single_frame,
                  num_pol, single_frame_skip=0, merged_metadata_cls=None,
                  merged_frame_skip=0, sub_frame_per_frame=None):
        """Load a visBuffer from a kotekan dump file.
        """
        filesize = os.path.getsize(filename)

        buf = bytearray(filesize)

        with io.FileIO(filename, "rb") as fh:
            fh.readinto(buf)
        
        return cls(buf, single_frame_metadata_cls, samples_per_single_frame,
                   num_pol, single_frame_skip, merged_metadata_cls,
                   merged_frame_skip, sub_frame_per_frame)

class SortedTrackingBeamReader(object):
    """ CHIME Sorted Tracking Beam raw data stream reader from files. 
    """
    def __init__(self, files, file_type, num_chan, sample_time,
                 single_frame_metadata_cls, samples_per_single_frame,
                 num_pol, samples_per_buffer=None, sub_frame_per_frame=1,
                 single_frame_skip=0, merged_metadata_cls=None,
                 merged_frame_skip=0):
        self.files = files
        self.files.sort()
        self.file_type = file_type
        self.num_chan = num_chan
        self.single_frame_metadata_cls = single_frame_metadata_cls
        self.samples_per_single_frame = samples_per_single_frame
        self.num_pol = num_pol
        self.single_frame_skip = single_frame_skip
        self.merged_metadata_cls = merged_metadata_cls
        self.merged_frame_skip = merged_frame_skip
        self.sub_frame_per_frame = sub_frame_per_frame
        self.frame_offset = 0
        self._frame_buffer = np.zeros((self.samples_per_single_frame, self.num_chan,
            self.num_pol), dtype='c8')

    def _load_file(self, filename):
        frames = []
        if self.file_type == 'merged':
            merged_frame = MergedBeamFrame.from_file(filename, self.single_frame_metadata_cls,
                self.samples_per_single_frame, self.num_pol, self.single_frame_skip,
                self.merged_metadata_cls, self.merged_frame_skip, self.sub_frame_per_frame)
            for ii in range(merged_frame.sub_frame_per_frame):
                frames.append(merged_frame.get_single_frame(ii))
        elif self.file_type == 'single':
            frame = SingleBeamFrame.from_file(filename, self.single_frame_metadata_cls,
                self.samples_per_single_frame, self.num_pol, self.single_frame_skip)
            frames.append(frame)
        return frames
    
    def seek_frame(self, offset):
        self.frame_offset = offset
        return self.frame_offset

    def _read_one_frame(self):
        frames = self._load_file(self.files[self.frame_offset])
        start_fpga = 0
        start_ctime = 0
        for ii, fm in enumerate(frames):
            fm.read_frame()
            if ii == 0:
                start_fpga = fm.metadata.fpga_seq_start
                start_ctime = fm.metadata.ctime
            else:
                #print("ctime test", fm.metadata.ctime.tv, start_ctime.tv, ii)
                assert fm.metadata.fpga_seq_start == start_fpga
                assert fm.metadata.ctime.tv == start_ctime.tv
                assert fm.metadata.ctime.tv_nsec == start_ctime.tv_nsec
            #print(fm.metadata.fpga_seq_start, fm.metadata.frequency_bin, self.frame_offset, self.files[self.frame_offset])
            self._frame_buffer[:, ii, :] = fm.data
        self.frame_offset += 1
        return start_fpga, start_ctime

class TrackingBeamReader(object):
    """CHIME Tracking Beam raw data stream reader from files. 
   
    Parameters
    ----------
    files: list
        List of files to read.
    file_type: str
        Type of tracking beam file.
    num_chan: int
        number of channals
    single_frame_metadata_cls: `BeamMetadata` class or `FreqIDBeamMetadata`
        Metadata class for a single frame
    samples_per_single_frame: int 
        Number of samples for one single tracking beam frame.
    num_pol: int
        Number of polarizations.
    single_frame_skip: int, optional 
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    merged_metadata_cls: `MergedBeamMetadata` class, optinal
        The metaclass for the merged frame files.
    merged_frame_skip: int, optional
        Number of bytes to skip from the beginning of the buffer. Useful for
        raw dumps when the metadata size is given in the first four bytes.
    sub_frame_per_frame: int, optional
        Number of sub-frame in one merged frame.

    Note
    ----
    The sub_frame_per_frame should be the same across the whole data stream.
    If the data files are single frame files, please use sub_frame_per_frame=1.
    This function does not check the droped frames.
    """
    def __init__(self, files, file_type, num_chan, sample_time, 
                 single_frame_metadata_cls, samples_per_single_frame,
                 num_pol, samples_per_buffer=None, sub_frame_per_frame=1,
                 single_frame_skip=0, merged_metadata_cls=None, 
                 merged_frame_skip=0):
        self.files = files
        self.file_type = file_type
        self.num_chan = num_chan
        self.single_frame_metadata_cls = single_frame_metadata_cls
        self.samples_per_single_frame = samples_per_single_frame
        self.num_pol = num_pol
        self.single_frame_skip = single_frame_skip
        self.merged_metadata_cls = merged_metadata_cls
        self.merged_frame_skip = merged_frame_skip
        self.sub_frame_per_frame = sub_frame_per_frame
        self.sample_time = sample_time
        self.frame_time_res = self.samples_per_single_frame * self.sample_time
        self.freq_map = {}
        self.frame_time_axis = self.get_time_axis(self.frame_time_res)
        # May need to change here to save memory.
        if samples_per_buffer is None:
            self.samples_per_buffer = samples_per_single_frame
        else:
            self.samples_per_buffer = samples_per_buffer
        self._data_buffer = np.zeros((self.samples_per_buffer, self.num_chan, 
            self.num_pol), dtype='c8')
        # the fill up status of frequency
        self._chan_offset = np.zeros(num_chan, dtype=int)
        # offset
        self.offset = 0
        self._buffer_offset = 0
        self._file_offset = 0
        self._sample_offset_in_frame = 0
        self._frame_queue = {} 
        for ii in range(self.num_chan):
            self._frame_queue[ii] = [None] * len(self.frame_time_axis)
        self._frame_map = np.zeros((self.num_chan, len(self.frame_time_axis), 3), dtype=int)
        self._frame_map[..., 0] = -1 # avoid the index zero
        self._frame_map[..., 1] = -1 # avoid the index zero
        self.num_frame = len(self.frame_time_axis)
        self.good_chan = np.zeros((self.num_chan, len(self.frame_time_axis)), dtype=int)

    @lazyproperty
    def number_files(self):
        return len(self.files)
    
    @property
    def is_buffer_full(self):
        return np.all(self._chan_offset == self.samples_per_buffer)

    @property
    def _unfilled_chans(self):
        return np.where(self._chan_offset != self.samples_per_buffer)[0]
    
    @property
    def queue_len(self):
        return np.sum(self._frame_map[...,2], axis=1)
    
    def nframe_at_time(self, time_index):
        return np.sum(self._frame_map[:, time_index, 2])

    @property
    def is_queue_empty(self):
        return all(v == 0  for v in self.queue_len)

    @property
    def _queue_start_time(self):
        return np.array([x[0].metadata.ctime.to_float() if len(x) !=0 else 0 for x in self._frame_queue.values()])
    
    @lazyproperty
    def shape(self):
        num_frames = self.number_files * self.sub_frame_per_frame
        num_samples = num_frames * self.samples_per_single_frame / self.num_chan
        return (num_samples, self.num_chan, self.num_pol)
    
    def has_empty_queue(self, time_index):
        return np.sum(self._frame_map[:, time_index, 2]) != self.num_chan

    def file_start_time(self, filename):
        frames = self._load_file(filename)
        return np.array([fm.metadata.ctime.to_float() for fm in frames]).min()
             
    def file_end_time(self, filename):
        frames = self._load_file(filename)
        return np.array([fm.metadata.ctime.to_float() for fm in frames]).max()

    def get_time_axis(self, time_resolution):
        start_time = self.file_start_time(self.files[0])
        end_time = self.file_end_time(self.files[-1])
        return np.arange(start_time, end_time + time_resolution, time_resolution)

    def _load_file(self, filename):
        frames = []
        if self.file_type == 'merged':
            merged_frame = MergedBeamFrame.from_file(filename, self.single_frame_metadata_cls, 
                self.samples_per_single_frame, self.num_pol, self.single_frame_skip, 
                self.merged_metadata_cls, self.merged_frame_skip, self.sub_frame_per_frame)
            for ii in range(merged_frame.sub_frame_per_frame):
                frames.append(merged_frame.get_single_frame(ii))
        elif self.file_type == 'single':
            frame = SingleBeamFrame.from_file(filename, self.single_frame_metadata_cls, 
                self.samples_per_single_frame, self.num_pol, self.single_frame_skip)
            frames.append(frame)
        return frames

    def _get_channel_index(self, freq_ID):
        idx = self.freq_map.get(freq_ID, None)
        if idx is None:
            assigned_chans = list(self.freq_map.values())
            for idx in range(self.num_chan):
                if idx not in assigned_chans:
                    self.freq_map[freq_ID] = idx
                    return idx
            raise RuntimeError("All channels in the buffer are full.")
        else:
            return idx
    
    def _get_channel_index2(self, freq_ID):
        if freq_ID > self.num_chan:
            raise ValueError("Frequency ID is out of the range, change your num_chan when initizing.")
        else:
            return freq_ID

    def _detect_bad_chan(self, start_frame, wait_frame=20):
        # This assumes the frame from each frequency comes in somewhat uniformly.
        # This will not work if parallarized. 
        rest_frame = self.num_frame - start_frame
        if rest_frame < wait_frame:
            raise ValueError("Can not detecte bad channel give the rest data.")
        if start_frame < 0:
            start_frame = 0
        queue_len = np.sum(self._frame_map[:, start_frame::, 2], axis=1)
        good_chans = np.where(queue_len > 0)[0]
        if (np.all(queue_len[good_chans] > wait_frame) and good_chans!= []):
            good_chan_overall = np.where(queue_len != 0)[0]
            for ii in range(wait_frame):
                good_chan_time = np.where(self._frame_map[:, start_frame + ii, 2] != 0)[0]
                self.good_chan[good_chan_time, start_frame + ii] = 1 
            return np.where(queue_len == 0)[0], np.where(queue_len != 0)[0]
        else:
            # Can not decide. 
            return [], []
    
    def _file_2_queue(self, filename, file_index, tol=2e-6):
        frames = self._load_file(filename)
        for ii, fm in enumerate(frames):
            freq_id = fm.metadata.frequency_bin
            chan_idx = self._get_channel_index2(freq_id)
            # Put the frame to the right time
            f_time = fm.metadata.ctime.to_float()
            if f_time > self.frame_time_axis.max() or f_time < self.frame_time_axis.min():
                ValueError("Frame is out of time_axis, please reset the time_axis")
            # Find the time
            closest = np.abs(self.frame_time_axis - f_time).argmin()
            diff = np.abs(self.frame_time_axis[closest] - f_time)
            if diff > tol:
                ValueError("Can not align the frame time {} in the time axis."
                           " The closest one is {} but the difference is {} > {}".
                           format(f_time, self.frame_time_axis[closest], diff, tol))
            self._frame_queue[chan_idx][closest] = fm
            self._frame_map[chan_idx, closest, 0] = ii
            self._frame_map[chan_idx, closest, 1] = file_index
            self._frame_map[chan_idx, closest, 2] = 1

    def _search_time_in_file(self, time_index):
        """ Search the time index in the files. 
        """
        # Check if there is a file in the current record
        registered = np.where(self._frame_map[:, time_index, 1] != -1)[0]
        if registered != []:
            return registered
        # Get the last registerd file
        register_status = np.sum(self._frame_map[...,1], axis=0)
        registered_index = np.where(register_status != -self.num_chan)[0]
        upper_end = np.searchsorted(registered_index, time_index)
        print(upper_end)
        if upper_end == 0:
            search_start = 0
            if registered_index != []:
                first_register = self._frame_map[:, registered_index[0], 1].min()
                search_end = first_register
            else:
                search_end = len(self.files)
        elif upper_end == len(registered_index):
            last_register = self._frame_map[:, registered_index[-1], 1].min()
            search_start = last_register
            search_end = len(self.files)
        else:
            search_start = self._frame_map[:, registered_index[upper_end - 1], 1].min()
            search_end = self._frame_map[:, registered_index[upper_end], 1].min()
        return search_start, search_end

    def clean_queue(self, time_index):
        """Clean queue at one time.
        """
        for v in self._frame_queue.values():
            v[time_index] = None
        self._frame_map[:, time_index, 2] = 0
        n = gc.collect()

    def _reset_buffer(self):
        """Bring the chan offset to zero, allows refill the buffer.
        """
        self._chan_offset.fill(0)
        self._data_buffer.fill(0)

    def _fill_buffer(self, frame_offset, wait_frame=30):
        """ Fill up the buffer. each run loads the sample of one single frame
        """
        # Move to the offset
        print('buffer offset', frame_offset)
        self.frame_offset = frame_offset
        load_size = int(wait_frame * self.num_chan / self.sub_frame_per_frame)
        # This is going to be very slow.
        while not self.is_buffer_full:
            # Load files to the queue is empty.
            load_file = True
            file_index = self._frame_map[:, frame_offset, 1]
            print(file_index)
            if np.sum(file_index) == -self.num_chan: #no file registered
                s_start, s_end = self._search_time_in_file(frame_offset)
                self._file_offset = s_start
            else:
                good_chan_num = np.sum(self.good_chan[:, frame_offset])
                if np.sum(self._frame_map[:, frame_offset, 2]) == good_chan_num: #Frame is loaded
                    load_file = False
                else:
                    s_start = file_index.min() - int(load_size / 2)
                    if s_start < 0:
                        s_start = 0
                    s_end = file_index.max() + int(load_size / 2)
                    if s_end > self.num_frame:
                        s_end = self.num_frame
                    self._file_offset = s_start

            print(frame_offset, self._file_offset, load_file)
            while (self.has_empty_queue(frame_offset) and load_file):
                # Load files
                if self._file_offset >= self.number_files:
                    # if there is frame in the queue this will skip the
                    # existing frames.
                    return -1
                # detect bad channels which does not have any frames come in, 
                # if all good channle has frame stop queue filling
                print("file offset", self.files[self._file_offset])
                self._file_2_queue(self.files[self._file_offset], self._file_offset)
                self._file_offset += 1
                # Check if the request frame get loaded yet
                load_status = np.sum(self._frame_map[:,frame_offset, 2])
                print("load status", load_status)
                if load_status > 0: # have data in the queue for the frame
                    # Check good frame.
                    if self.num_frame - s_start > wait_frame:
                        bad_f, good_f = self._detect_bad_chan(frame_offset - int(wait_frame/2), wait_frame)
                        if bad_f != []:
                            break

            # offload unused frames 
            hold_start = frame_offset - int(wait_frame)
            hold_end = frame_offset + int(wait_frame)
            if hold_start < 0:
                hold_start = 0
            if hold_end > self.num_frame:
                hold_end = self.num_frame
            unuse = np.array(list(range(hold_start)) + list(range(hold_end, self.num_frame)))
            unuse_load = np.where(np.sum(self._frame_map[:, unuse, 2]) != 0)[0]
            remove = unuse[unuse_load]
            print("remove chan", remove)
            
            for ii in remove:
                self.clean_queue(ii)

            # read date to buffer for each channel
            num_bad_frames = 0
            for chan_index in range(self.num_chan):
                in_frame = self._frame_queue[chan_index][frame_offset]
                good_frame = True
                if in_frame is None:
                    good_frame = False
                    num_bad_frames += 1
                else:    
                    if in_frame.data is None:
                        in_frame.read_frame()
                    print("dump data", frame_offset, chan_index, in_frame.metadata.ctime.to_float(), in_frame.metadata.fpga_seq_start)
                chan_offset = self._chan_offset[chan_index]
                num_request_sample = self.samples_per_buffer - chan_offset
                if good_frame:
                    num_exist_sample = in_frame.samples_per_frame - in_frame.sample_offset
                    if num_request_sample >= num_exist_sample:
                        read_number = num_exist_sample
                    else:
                        read_number = num_request_sample
                    #print("chan_offset", chan_offset, read_number)
                    self._data_buffer[chan_offset: chan_offset + read_number, 
                        chan_index, :] = in_frame.read(read_number)
                    self._chan_offset[chan_index] = read_number + chan_offset
                else:
                    self._chan_offset[chan_index] = self.samples_per_buffer
                #print("sample_offse", in_frame.sample_offset)
        self._buffer_offset += self.samples_per_buffer 
        print("number bad frame", num_bad_frames)

    def read(self, offset, nbuffer=-1):
        """ Read data from buffer. Since this program does not have anytime check. It
        can not do the seek in the data set, and can only read integer of buffer size.
        """
        if nbuffer == -1: # Read all the data in a file.
            nbuffer = int(self.shape[0] / self.samples_per_buffer) + 1
        data = np.zeros((nbuffer * self.samples_per_buffer, self.shape[1], self.shape[2]), dtype='c8')
        read_offset = 0
        for nb in range(nbuffer):
            print("READ offset", read_offset)
            self._reset_buffer()
            print("bigest", self._data_buffer.max())
            self._fill_buffer(offset)
            data[read_offset:read_offset + self.samples_per_buffer, ...] = self._data_buffer
            read_offset = read_offset + self.samples_per_buffer
            offset += 1
        return data

        
            
    
        
        




