#ifndef POWER_STREAM_UTIL
#define POWER_STREAM_UTIL

struct  __attribute__((__packed__)) IntensityHeader {
    int packet_length;      // - packet length
    int header_length;      // - header length
    int samples_per_packet; // - number of samples in packet
    int sample_type;        // - data type of samples in packet
    double raw_cadence;     // - raw sample cadence
    int num_freqs;          // - number of frequencies
    int num_elems;          // - number of elements
    int samples_summed;     // - samples summed for each datum
    uint handshake_idx;     // - frame idx at handshake
    double handshake_utc;   // - UTC time at handshake
};

struct  __attribute__((__packed__)) IntensityPacketHeader {
    int frame_idx;          //- frame idx
    int elem_idx;           //- elem idx
    int samples_summed;     //- number of samples integrated
};

#endif