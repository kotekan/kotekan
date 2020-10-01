#ifndef PULSAR_FUNCTIONS_HPP
#define PULSAR_FUNCTIONS_HPP

#include <inttypes.h>

/**
 * @struct PSRHeader
 * @brief Header for CHIME/Pulsar UDP packet.
 *
 * Current version of the header used for transmission of beamformed
 * data to the CHIME/Pulsar backend, same as VDIF definition. Indices
 * of frequency channels, each 10-b numbers, are encoded in two places:
 * PSRHeader.thread_id and PSRHeader.eud3, with the latter defined as
 * PSRHeader.eud3 = freq_index[3]<<20 + freq_index[2]<<10 + freq_index[1]
 * Also, position info is encoded into PSRHeader.eud4 as:
 * PSRHeader.eud4 = (RA << 16) + (DEC)
 *
 * @author Cherry Ng
 **/

struct PSRHeader {
    uint32_t seconds : 30;
    uint32_t legacy : 1;
    uint32_t invalid : 1;
    uint32_t data_frame : 24;
    uint32_t ref_epoch : 6;
    uint32_t unused : 2;
    uint32_t frame_len : 24;
    uint32_t log_num_chan : 5;
    uint32_t vdif_version : 3;
    uint32_t station_id : 16;
    uint32_t thread_id : 10;
    uint32_t bits_depth : 5;
    uint32_t data_type : 1;
    uint32_t eud1 : 24;
    uint32_t edv : 8;
    uint32_t eud2 : 32;
    uint32_t eud3 : 32;
    uint32_t eud4 : 32;
};

#endif /* PULSAR_FUNCTIONS_HPP */
