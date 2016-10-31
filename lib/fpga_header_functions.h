#ifndef FPGA_HEADER_FUNCTIONS_H
#define FPGA_HEADER_FUNCTIONS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t link_id;
    uint8_t slot_id;
    uint8_t create_id;
    uint8_t unused;
} stream_id_t;

inline uint32_t bin_number(const stream_id_t* stream_id, const int index)
{
    return stream_id->slot_id + stream_id->link_id * 16 + index * 128;
}

inline uint32_t bin_number_16_elem(const stream_id_t * stream_id, const int index)
{
    return stream_id->link_id + index * 8;
}

float freq_from_bin(const int bin);

stream_id_t extract_stream_id(const uint16_t encoded_stream_id);
uint16_t encode_stream_id(const stream_id_t s_stream_id);

#ifdef __cplusplus
}
#endif

#endif