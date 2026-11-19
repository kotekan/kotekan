#ifndef GNSS_TELEM_SEND_HPP
#define GNSS_TELEM_SEND_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "bufferSend.hpp"

#include <cstddef>
#include <string>

/**
 * @class GnssTelemSend
 * @brief A bufferSend of telemetry frames that sends each frame at its own length.
 *
 * The rows of a GnssTelemPack host may share one out_buf, sized for the widest row, so that a
 * GPU half sends one stream. Each frame is sent as gnss::telem_frame_bytes of its header, the
 * bytes a row's own buffer would send; the gather's bufferRecv takes the shorter frames with
 * allow_short_frames.
 *
 * @par buffers
 * @buffer buf telemetry frames (gnssTelem.hpp), each at least as long as its header says.
 *
 * Every other key, and the metrics, are bufferSend's.
 */
class GnssTelemSend : public bufferSend {
public:
    GnssTelemSend(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

protected:
    size_t frame_send_bytes(int frame_id) override;

private:
    Buffer* _buf;
};

#endif // GNSS_TELEM_SEND_HPP
