#ifndef FPGA_HEADER_FUNCTIONS_H
#define FPGA_HEADER_FUNCTIONS_H

#include <stdint.h>

typedef struct {
    uint8_t link_id;
    uint8_t slot_id;
    uint8_t create_id;
    uint8_t unused;
} stream_id_t;

uint32_t bin_number(const stream_id_t * stream_id, const int index);

float freq_from_bin(const int bin);

stream_id_t extract_stream_id(const uint16_t encoded_stream_id);

#endif