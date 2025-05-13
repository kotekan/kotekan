#!/usr/bin/env python3

import h5py
import socket
import ctypes

hostname = socket.gethostname()
fh = h5py.File(hostname + "_pulsar.h5", "r")

# output order in network stream is:
# out_index =
#   // beam
#   psr * _udp_pulsar_packet_size * _num_packet_per_stream
#   // packet
#   + frame * _udp_pulsar_packet_size
#   // freq
#   + (thread_id % num_freqs_per_frame) * _timesamples_per_pulsar_packet * _num_pol
#   // time
#   + in_frame_location * _num_pol
#   // pol
#   + p
#   // offset for packet header
#   + udp_pulsar_header_size;
# where

_udp_pulsar_packet_size = 5032
_num_packet_per_stream = 80
_timesamples_per_pulsar_packet = 625
_num_pulsars = 10
_num_pol = 2
num_freqs_per_frame = 4

udp_pulsar_header_size = 32 # VDIF header
def parse_vdif_header(header):
    # VDIF header is described in https://vlbi.org/vlbi-standards/vdif/
    # and the flavour used by CHIME in lib/utils/vdif_functions.h
    class VDIFHeader(ctypes.Structure):
        _fields_ = [
            ("word0", ctypes.c_uint32), ("word1", ctypes.c_uint32),
            ("word2", ctypes.c_uint32), ("word3", ctypes.c_uint32),
            ("word4", ctypes.c_uint32), ("word5", ctypes.c_uint32),
            ("word6", ctypes.c_uint32), ("word7", ctypes.c_uint32),
        ]
    raw_header = VDIFHeader.from_buffer(header)
    parsed_header = dict()
    parsed_header["seconds"] = (raw_header.word0 & 0b00111111111111111111111111111111) >> 0
    parsed_header["legacy"] = (raw_header.word0 & 0b01000000000000000000000000000000) >> 30
    parsed_header["invalid"] = (raw_header.word0 & 0b10000000000000000000000000000000) >> 31
    parsed_header["data_frame"] = (raw_header.word1 & 0b00000000111111111111111111111111) >> 0
    parsed_header["ref_epoch"] = (raw_header.word1 & 0b00111111000000000000000000000000) >> 24
    parsed_header["unused"] = (raw_header.word1 & 0b11000000000000000000000000000000) >> 30
    parsed_header["frame_len"] = (raw_header.word2 & 0b00000000111111111111111111111111) >> 0
    parsed_header["log_num_chan"] = (raw_header.word2 & 0b00011111000000000000000000000000) >> 24
    parsed_header["vdif_version"] = (raw_header.word2 & 0b11100000000000000000000000000000) >> 29
    parsed_header["station_id"] = (raw_header.word3 & 0b00000000000000001111111111111111) >> 0
    parsed_header["thread_id"] = (raw_header.word3 & 0b00000011111111110000000000000000) >> 16
    parsed_header["bits_depth"] = (raw_header.word3 & 0b01111100000000000000000000000000) >> 26
    parsed_header["data_type"] = (raw_header.word3 & 0b10000000000000000000000000000000) >> 31
    parsed_header["eud1"] = (raw_header.word4 & 0b00000000111111111111111111111111) >> 0
    parsed_header["edv"] = (raw_header.word4 & 0b11111111000000000000000000000000) >> 24
    parsed_header["eud2"] = (raw_header.word5 & 0b11111111111111111111111111111111) >> 0
    parsed_header["eud3"] = (raw_header.word6 & 0b11111111111111111111111111111111) >> 0
    parsed_header["eud4"] = (raw_header.word7 & 0b11111111111111111111111111111111) >> 0
    return parsed_header

# and the dummy producer sets data to be:
# 
# int index = i + pol * samples_per_dataset + freq * num_pols * samples_per_dataset +
#              beam * num_freqs * num_pols * samples_per_dataset;
#  frame[index + 0] = beam << 4 | freq;
#  frame[index + 1] = pol | (count & 0xff);
#  frame[index + 2] = (i >> 8) & 0xff;
#  frame[index + 3] = (i >> 0) & 0xff;
# where

samples_per_dataset = 16384
num_pols = 2
num_freqs = 16
num_beams = 16

for packet in range(0,5):
    dset = fh["/%07d/pulsar_output_buffer" % packet]
    for psr in range(0,_num_pulsars):
        for frame in range(0,_num_packet_per_stream):
            frame_index = psr * _udp_pulsar_packet_size * _num_packet_per_stream \
                          + frame * _udp_pulsar_packet_size
            header = parse_vdif_header(dset[frame_index:frame_index+udp_pulsar_header_size])
            # check that header is what we expect it to be:
            # could check "seconds"
            assert(header["legacy"] == 0)
            assert(header["invalid"] == 0)
            # 16 freqs before we get to the next FFT package (and thus next time), 4 freqs per package
            assert(header["data_frame"] == packet // 4 * _num_packet_per_stream + frame)
            assert(header["epoch"] == 36)
            assert(header["unused"] == 0)
            assert(header["frame_len"] == 629)
            assert(header["log_num_chan"] == 3)
            assert(header["vdif_version"] == 1)
            #assert(header["station_id"] == ???)


            for thread_id in range(0, num_freqs_per_frame): # hard-coded, 4 freqs per package
                for p in range(0, _num_pol):
                    for in_frame_location in range(0, _timesamples_per_pulsar_packet):
                        out_index = \
                        psr * _udp_pulsar_packet_size * _num_packet_per_stream \
                        + frame * _udp_pulsar_packet_size \
                        + (thread_id % num_freqs_per_frame) * _timesamples_per_pulsar_packet * _num_pol \
                        + in_frame_location * _num_pol \
                        + p \
                        + udp_pulsar_header_size
    
                        i = (packet // 4 * _num_packet_per_stream * _timesamples_per_pulsar_packet + \
                             frame * _timesamples_per_pulsar_packet + in_frame_location) % samples_per_dataset
                        sub_i = i & 0x3
                        i = i & ~0x3
    
                        freq = thread_id + packet % (num_freqs // num_freqs_per_frame) * num_freqs_per_frame
    
                        count = psr * num_freqs * num_pols * samples_per_dataset // 4 + \
                                thread_id * num_pols * samples_per_dataset // 4  + \
                                p * samples_per_dataset // 4 + \
                                i // 4
                        count *= 2

                        #print(out_index, i, sub_i, psr, frame, thread_id, p, in_frame_location)
    
                        if(sub_i == 0):
                            dat = (psr << 4) |  freq
                            if(dset[out_index] != dat):
                                print(out_index, i, sub_i, packet, psr, frame, thread_id, p, in_frame_location)
                                print("beam:", dset[out_index] >> 4, psr)
                                print("freq:", dset[out_index] & 0xf, freq)
                        if(sub_i == 1):
                            dat = (count & 0xff) | p;
                            if(dset[out_index] != dat): # ignore count for now
                                print(out_index, i, sub_i, packet, psr, frame, thread_id, p, in_frame_location)
                                print("pol:", dset[out_index] & 0x1, p)
                                print("count:", dset[out_index] & 0xfe, count & 0xff)
                        if(sub_i == 2):
                            dat = (i >> 8) & 0xff
                            if(dset[out_index] != dat):
                                print(out_index, i, sub_i, packet, psr, frame, thread_id, p, in_frame_location)
                                print("ihigh:", dset[out_index], (i>>8) & 0xff)
                        if(sub_i == 3):
                            dat = (i >> 0) & 0xff
                            if(dset[out_index] != dat):
                                print(out_index, i, sub_i, packet, psr, frame, thread_id, p, in_frame_location)
                                print("ilow:", dset[out_index], (i>>0) & 0xff)
