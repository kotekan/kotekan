#include "GnssTelemSend.hpp"

#include "StageFactory.hpp"
#include "gnssTelem.hpp"

#include <cstring>

REGISTER_KOTEKAN_STAGE(GnssTelemSend);

GnssTelemSend::GnssTelemSend(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& buffer_container) :
    bufferSend(config, unique_name, buffer_container), _buf(get_buffer("buf")) {
    if (_buf->frame_size < sizeof(gnss::TelemHeader))
        FATAL_ERROR("{:s}: buf frames of {:d} B cannot hold a {:d} B telemetry header", unique_name,
                    _buf->frame_size, sizeof(gnss::TelemHeader));
}

size_t GnssTelemSend::frame_send_bytes(int frame_id) {
    gnss::TelemHeader h;
    std::memcpy(&h, _buf->frames[frame_id], sizeof(h));
    // GnssTelemPack checks its rows' frames fit at construction, so a longer one is not its
    // frame; it is sent whole, and the gather counts it bad.
    const size_t n = gnss::telem_frame_bytes(h);
    return n <= _buf->frame_size ? n : _buf->frame_size;
}
