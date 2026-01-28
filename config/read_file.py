import os
import io
import struct
import numpy as np
import ctypes
import ctypes.util
POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL(ctypes.util.find_library("c"))

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

def raw_baseband_frames(file_name: str, buf: bytearray):
    """Iterates over frames in a raw baseband file."""
    with io.FileIO(file_name, "rb") as raw_file:
        while raw_file.readinto(buf):
            yield buf
        size = os.path.getsize(file_name)
        bufcache_dontneed(raw_file.fileno(), 0, size)
        
def bufcache_dontneed(fd, offs, length):
    """Clean up buffer cache."""
    return libc.posix_fadvise(
        fd, ctypes.c_uint64(offs), ctypes.c_uint64(length), POSIX_FADV_DONTNEED
    )
#frame_size = Nbytes
#dtype = np.complex64
#frame_shape = (nfreq, ninput)
#docker run -d -it -v /home/shiona/:/home -v --name shiona chimefrb/baseband-analysis:latest bash
def read_raw_file(fname):
    num_elements=1
    frames = np.array([])
    metadata_size = 4 #ctypes.sizeof(BasebandMetadata)
    print(metadata_size)
    buf = bytearray(frame_size + metadata_size)
    for b in raw_baseband_frames(fname, buf):
        '''frame_metadata = BasebandMetadata.from_buffer(b)
        event_id, freq_id = (frame_metadata.event_id, frame_metadata.freq_id)
        print(f"event_id: {event_id}")
        print(f"freq_id: {freq_id}")
        print(f"time0_ctime: {frame_metadata.time0_ctime}")
        print(f"time0_ctime_offset: {frame_metadata.time0_ctime_offset}")'''

        bbdata=np.array(b[metadata_size:])
        
        frames=np.append(frames,bbdata) 

    '''frames = []
    with open(fname, "rb") as f:
        while True:
            # read metadata size
            hdr = f.read(4)
            if not hdr:
                break
            (meta_size,) = struct.unpack("I", hdr)
            #buf = bytearray(frame_size + metadata_size)
            # skip metadata
            
            d = f.read(meta_size)
            #arr = np.frombuffer(d, dtype=dtype)#.reshape(frame_shape)
            #print(arr)
            # read frame payload
            data = f.read(frame_size)
            if len(data) < frame_size:
                break

            arr = np.frombuffer(data, dtype=dtype)
            print(arr)
            print(arr.shape)
            arr=arr.reshape(frame_shape)
            frames.append(arr.copy()) '''
    return np.array(frames)
import yaml

with open("/home/shiona/kotekan/config/verify_outrigger_beamformer.yaml") as f:
    cfg = yaml.safe_load(f)
frame_size = cfg['samples_per_data_set']*cfg['num_data_sets']* cfg['num_local_freq'] * cfg['num_pointings'] * 2 * cfg['sizeof_data']
dtype = np.complex64
ntime=cfg['samples_per_data_set']
npointing=cfg['num_pointings']
nfreq=cfg['num_local_freq']
frame_shape = (npointing,nfreq,ntime)
fname='/home/shiona/kotekan_tests/hx4_adc_data_0000000.raw'
frames=read_raw_file(fname)
print(frames)