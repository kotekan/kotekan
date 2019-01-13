#include "chimeMetadataDump.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "util.h"

#include <time.h>

REGISTER_KOTEKAN_PROCESS(chimeMetadataDump);

chimeMetadataDump::chimeMetadataDump(kotekan::Config& config, const string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&chimeMetadataDump::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

chimeMetadataDump::~chimeMetadataDump() {}

void chimeMetadataDump::main_thread() {

    int frame_id = 0;
    uint8_t* frame = NULL;

    while (!stop_thread) {

        frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (frame == NULL)
            break;

        uint64_t fpga_seq = get_fpga_seq_num(in_buf, frame_id);
        stream_id_t stream_id = get_stream_id_t(in_buf, frame_id);
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

        INFO("Metadata for %s[%d]: FPGA Seq: %" PRIu64
             ", stream ID = {create ID: %d, slot ID: %d, link ID: %d, freq ID: %d}, lost samples: "
             "%" PRIu64 " freq_bin: %d, freq: %f MHz , time stamp: %ld.%06ld (%s.%06ld), GPS time: "
             "%ld.%06ld (%s.%09ld)",
             in_buf->buffer_name, frame_id, fpga_seq, stream_id.crate_id, stream_id.slot_id,
             stream_id.link_id, stream_id.unused, lost_samples, bin_number_chime(&stream_id),
             freq_from_bin(bin_number_chime(&stream_id)), time_v.tv_sec, time_v.tv_usec, time_buf,
             time_v.tv_usec, time_s.tv_sec, time_s.tv_nsec, gps_time_buf, time_s.tv_nsec);

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}
