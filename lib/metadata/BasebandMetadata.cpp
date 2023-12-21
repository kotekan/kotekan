#include "BasebandMetadata.hpp"

REGISTER_TYPE_WITH_FACTORY(metadataObject, BasebandMetadata);

struct BasebandMetadataFormat {
    uint64_t event_id;
    uint64_t freq_id;
    uint64_t event_start_fpga;
    uint64_t event_end_fpga;
    uint64_t time0_fpga;
    double time0_ctime;
    double time0_ctime_offset;
    double first_packet_recv_time;
    int64_t frame_fpga_seq;
    int64_t valid_to;
    uint64_t fpga0_ns;
    int32_t num_elements;
    int32_t reserved;
};

size_t BasebandMetadata::get_serialized_size() {
    return sizeof(BasebandMetadataFormat);
}

size_t BasebandMetadata::set_from_bytes(const char* bytes, size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    const BasebandMetadataFormat* fmt = reinterpret_cast<const BasebandMetadataFormat*>(bytes);
    event_id = fmt->event_id;
    freq_id = fmt->freq_id;
    event_start_fpga = fmt->event_start_fpga;
    event_end_fpga = fmt->event_end_fpga;
    time0_fpga = fmt->time0_fpga;
    time0_ctime = fmt->time0_ctime;
    time0_ctime_offset = fmt->time0_ctime_offset;
    first_packet_recv_time = fmt->first_packet_recv_time;
    frame_fpga_seq = fmt->frame_fpga_seq;
    valid_to = fmt->valid_to;
    fpga0_ns = fmt->fpga0_ns;
    num_elements = fmt->num_elements;
    return sz;
}

size_t BasebandMetadata::serialize(char* bytes) {
    size_t sz = get_serialized_size();
    BasebandMetadataFormat* fmt = reinterpret_cast<BasebandMetadataFormat*>(bytes);
    fmt->event_id = event_id;
    fmt->freq_id = freq_id;
    fmt->event_start_fpga = event_start_fpga;
    fmt->event_end_fpga = event_end_fpga;
    fmt->time0_fpga = time0_fpga;
    fmt->time0_ctime = time0_ctime;
    fmt->time0_ctime_offset = time0_ctime_offset;
    fmt->first_packet_recv_time = first_packet_recv_time;
    fmt->frame_fpga_seq = frame_fpga_seq;
    fmt->valid_to = valid_to;
    fmt->fpga0_ns = fpga0_ns;
    fmt->num_elements = num_elements;
    return sz;
}

