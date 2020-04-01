#include "fpga_header_functions.h"

#include <stdint.h>

stream_id_t extract_stream_id(const uint16_t encoded_stream_id) {
    stream_id_t stream_id;

    stream_id.link_id = encoded_stream_id & 0x000F;
    stream_id.slot_id = (encoded_stream_id & 0x00F0) >> 4;
    stream_id.crate_id = (encoded_stream_id & 0x0F00) >> 8;
    stream_id.unused = (encoded_stream_id & 0xF000) >> 12;

    return stream_id;
}

uint16_t encode_stream_id(const stream_id_t s_stream_id) {
    uint16_t stream_id;

    stream_id = (s_stream_id.link_id & 0xF) + ((s_stream_id.slot_id & 0xF) << 4)
                + ((s_stream_id.crate_id & 0xF) << 8) + ((s_stream_id.unused & 0xF) << 12);

    return stream_id;
}

float freq_from_bin(const int bin) {
    return 800.0 - (float)bin * 400.0 / 1024.0;
}


uint32_t bin_number(const stream_id_t* stream_id, const int index) {
    return stream_id->slot_id + stream_id->link_id * 16 + index * 128;
}

uint32_t bin_number_16_elem(const stream_id_t* stream_id, const int index) {
    return stream_id->link_id + index * 8;
}

uint32_t bin_number_multifreq(const stream_id_t* stream_id, const int num_local_freq, int freqidx) {
    uint32_t freq_id;
    switch (num_local_freq) {
        case 8:
            freq_id = bin_number(stream_id, freqidx);
            break;
        case 128:
            freq_id = bin_number_16_elem(stream_id, freqidx);
            break;
        default:
            freq_id = bin_number_chime(stream_id);
            break;
    }
    return freq_id;
}

// This should use a table, but for now it seems to work in the base CHIME setup.
// TODO replace with a table when the full table mapping data is in kotekan
uint32_t bin_number_chime(const stream_id_t* stream_id) {
    return stream_id->crate_id * 16 + stream_id->slot_id + stream_id->link_id * 32
           + stream_id->unused * 256;
}
