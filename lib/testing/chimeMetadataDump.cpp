#include "chimeMetadataDump.hpp"

#include "Config.hpp"
#include "ICETelescope.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.hpp"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" //
#include "chimeMetadata.hpp"   // for get_first_packet_recv_time, get_fpga_seq_num, get_gps...
#include "kotekanLogging.hpp"  // for INFO

#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdint.h>   // for uint64_t, uint8_t
#include <sys/time.h> // for timeval
#include <time.h>     // for gmtime, strftime, timespec, time_t


REGISTER_KOTEKAN_STAGE(chimeMetadataDump);

chimeMetadataDump::chimeMetadataDump(kotekan::Config& config, const std::string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&chimeMetadataDump::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

chimeMetadataDump::~chimeMetadataDump() {}

void chimeMetadataDump::main_thread() {

    int frame_id = 0;
    uint8_t* frame = nullptr;

    auto& tel = Telescope::instance();

    while (!stop_thread) {

        frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        uint64_t fpga_seq = get_fpga_seq_num(in_buf, frame_id);
        stream_t encoded_stream_id = get_stream_id(in_buf, frame_id);
        ice_stream_id_t stream_id = ice_extract_stream_id(encoded_stream_id);
        timeval time_v = get_first_packet_recv_time(in_buf, frame_id);
        uint64_t lost_samples = get_lost_timesamples(in_buf, frame_id);
        struct timespec time_s = get_gps_time(in_buf, frame_id);

        char time_buf[64];
        time_t temp_time = time_v.tv_sec;
        struct tm* l_time = gmtime(&temp_time);
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

        char gps_time_buf[64];
        time_t temp_gps_time = time_s.tv_sec;
        struct tm* l_gps_time = gmtime(&temp_gps_time);
        strftime(gps_time_buf, sizeof(gps_time_buf), "%Y-%m-%d %H:%M:%S", l_gps_time);

        INFO("Metadata for {:s}[{:d}]: FPGA Seq: {:d}, stream ID = (crate ID: {:d}, "
             "slot ID: {:d}, link ID: {:d}, freq ID: {:d}), lost samples: {:d} freq_bin: {:d}, "
             "freq: {:f} MHz , time stamp: {:d}.{:06d} ({:s}.{:06d}), "
             "GPS time: {:d}.{:06d} ({:s}.{:09d})",
             in_buf->buffer_name, frame_id, fpga_seq, stream_id.crate_id, stream_id.slot_id,
             stream_id.link_id, stream_id.unused, lost_samples, tel.to_freq_id(encoded_stream_id),
             tel.to_freq(encoded_stream_id), time_v.tv_sec, time_v.tv_usec, time_buf,
             time_v.tv_usec, time_s.tv_sec, time_s.tv_nsec, gps_time_buf, time_s.tv_nsec);

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}
