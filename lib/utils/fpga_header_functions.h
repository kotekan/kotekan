#ifndef FPGA_HEADER_FUNCTIONS_H
#define FPGA_HEADER_FUNCTIONS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t link_id;
    uint8_t slot_id;
    uint8_t crate_id;
    uint8_t unused;
} stream_id_t;

uint32_t bin_number(const stream_id_t* stream_id, const int index);
uint32_t bin_number_16_elem(const stream_id_t* stream_id, const int index);
uint32_t bin_number_chime(const stream_id_t* stream_id);
uint32_t bin_number_multifreq(const stream_id_t* stream_id, const int num_local_freq, int freqidx);
float freq_from_bin(const int bin);

stream_id_t extract_stream_id(const uint16_t encoded_stream_id);
uint16_t encode_stream_id(const stream_id_t s_stream_id);

#ifdef __cplusplus
}
#endif

#endif
