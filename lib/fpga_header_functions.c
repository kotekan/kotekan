#include "fpga_header_functions.h"

stream_id_t extract_stream_id(const uint16_t encoded_stream_id)
{
    stream_id_t stream_id;

    stream_id.link_id = encoded_stream_id & 0x000F;
    stream_id.slot_id = (encoded_stream_id & 0x00F0) >> 4;
    stream_id.create_id = (encoded_stream_id & 0x0F00) >> 8;
    stream_id.unused = (encoded_stream_id & 0xF000) >> 12;

    return stream_id;
}

uint32_t bin_number(const stream_id_t* stream_id, const int index)
{
    return stream_id->slot_id + stream_id->link_id * 16 + index * 128;
}

float freq_from_bin(const int bin)
{
    return 800.0 - (float)bin * 400.0/1024.0;
}