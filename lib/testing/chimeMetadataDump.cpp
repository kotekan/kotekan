#include "chimeMetadataDump.hpp"
#include "util.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "errors.h"
#include <time.h>

chimeMetadataDump::chimeMetadataDump(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&chimeMetadataDump::main_thread, this)) {

    buf = get_buffer("buf");
    register_consumer(buf, unique_name.c_str());
}

chimeMetadataDump::~chimeMetadataDump() {
}

void chimeMetadataDump::apply_config(uint64_t fpga_seq) {
}

void chimeMetadataDump::main_thread() {

    int frame_id = 0;
    uint8_t * frame = NULL;

    while (!stop_thread) {

        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
        stream_id_t stream_id = get_stream_id_t(buf, frame_id);
        timeval time_v = get_first_packet_recv_time(buf, frame_id);
        uint64_t lost_samples = get_lost_timesamples(buf, frame_id);

        char time_buf[64];
        time_t temp_time = time_v.tv_sec;
        struct tm* l_time = localtime(&temp_time);
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

        INFO("Metadata for %s[%d]: FPGA Seq: %" PRIu64
                ", stream ID = {create ID: %d, slot ID: %d, link ID: %d, freq ID: %d}, lost samples: %" PRIu64
                 ", time stamp: %ld.%06ld (%s.%06ld)",
                buf->buffer_name, frame_id, fpga_seq,
                stream_id.crate_id, stream_id.slot_id,
                stream_id.link_id, stream_id.unused, lost_samples,
                time_v.tv_sec, time_v.tv_usec, time_buf, time_v.tv_usec);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }
}