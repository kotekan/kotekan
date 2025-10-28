#include "BasebandMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <assert.h> // for assert
#include <memory>   // for allocator

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

void BasebandMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto bb_other = std::dynamic_pointer_cast<const BasebandMetadata>(other);
    assert(bb_other);
    *this = *bb_other;
}

size_t BasebandMetadata::get_serialized_size() {
    return sizeof(BasebandMetadataFormat);
}

size_t BasebandMetadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
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

nlohmann::json BasebandMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const BasebandMetadata& m) {
    assert(j.empty());

    j.emplace("event_id", m.event_id);
    j.emplace("freq_id", m.freq_id);
    j.emplace("event_start_fpga", m.event_start_fpga);
    j.emplace("event_end_fpga", m.event_end_fpga);
    j.emplace("time0_fpga", m.time0_fpga);
    j.emplace("time0_ctime", m.time0_ctime);
    j.emplace("time0_ctime_offset", m.time0_ctime_offset);
    j.emplace("first_packet_recv_time", m.first_packet_recv_time);
    j.emplace("frame_fpga_seq", m.frame_fpga_seq);
    j.emplace("valid_to", m.valid_to);
    j.emplace("fpga0_ns", m.fpga0_ns);
    j.emplace("num_elements", m.num_elements);
}

void from_json(const nlohmann::json& j, BasebandMetadata& m) {
    m.event_id = j.at("event_id");
    m.freq_id = j.at("freq_id");
    m.event_start_fpga = j.at("event_start_fpga");
    m.event_end_fpga = j.at("event_end_fpga");
    m.time0_fpga = j.at("time0_fpga");
    m.time0_ctime = j.at("time0_ctime");
    m.time0_ctime_offset = j.at("time0_ctime_offset");
    m.first_packet_recv_time = j.at("first_packet_recv_time");
    m.frame_fpga_seq = j.at("frame_fpga_seq");
    m.valid_to = j.at("valid_to");
    m.fpga0_ns = j.at("fpga0_ns");
    m.num_elements = j.at("num_elements");
}
