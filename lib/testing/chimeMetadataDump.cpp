#include "chimeMetadataDump.hpp"

#include "Config.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.hpp"          // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" //
#include "chordMetadata.hpp"
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
    in_buf->register_consumer(unique_name);
}

chimeMetadataDump::~chimeMetadataDump() {}

void chimeMetadataDump::main_thread() {

    int frame_id = 0;
    uint8_t* frame = nullptr;

    auto& tel = Telescope::instance();

    while (!stop_thread) {

        frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        uint64_t fpga_seq = get_chord_metadata(in_buf, frame_id)->get_fpga_seq_num();
        timeval time_v = get_chord_metadata(in_buf, frame_id)->get_first_packet_recv_time();
        uint64_t lost_samples = get_chord_metadata(in_buf, frame_id)->get_lost_timesamples();
        struct timespec time_s = get_chord_metadata(in_buf, frame_id)->get_gps_time();
        freq_id_t freq_id = get_chord_metadata(in_buf, frame_id)->get_coarse_freq()[0];

        char time_buf[64];
        time_t temp_time = time_v.tv_sec;
        struct tm* l_time = gmtime(&temp_time);
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

        char gps_time_buf[64];
        time_t temp_gps_time = time_s.tv_sec;
        struct tm* l_gps_time = gmtime(&temp_gps_time);
        strftime(gps_time_buf, sizeof(gps_time_buf), "%Y-%m-%d %H:%M:%S", l_gps_time);

        INFO("Metadata for {:s}[{:d}]: FPGA Seq: {:d}, stream ID = ("
             "freq ID: {:d}), lost samples: {:d} freq_bin: {:d}, "
             "freq: {:f} MHz , time stamp: {:d}.{:06d} ({:s}.{:06d}), "
             "GPS time: {:d}.{:06d} ({:s}.{:09d})",
             in_buf->buffer_name, frame_id, fpga_seq, lost_samples, freq_id,
             tel.to_freq(freq_id), time_v.tv_sec, time_v.tv_usec, time_buf,
             time_v.tv_usec, time_s.tv_sec, time_s.tv_nsec, gps_time_buf, time_s.tv_nsec);

        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}
