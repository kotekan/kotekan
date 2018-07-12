#include "zeroSamples.hpp"
#include "nt_memset.h"
#include "chimeMetadata.h"

REGISTER_KOTEKAN_PROCESS(zeroSamples);

zeroSamples::zeroSamples(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&zeroSamples::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    lost_samples_buf = get_buffer("lost_samples_buf");
    register_consumer(lost_samples_buf, unique_name.c_str());

    sample_size = config.get_int_default(unique_name.c_str(), "sample_size", 2048);
}

zeroSamples::~zeroSamples() {
}

void zeroSamples::main_thread() {

    uint32_t zero_location;
    int64_t lost_samples;

    while (!stop_thread) {

        lost_samples = 0;

        uint8_t * data_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_frame_id);
        if (data_frame == NULL) break;

        uint8_t * flag_frame = wait_for_full_frame(lost_samples_buf,
                                                   unique_name.c_str(), lost_samples_buf_frame_id);
        if (flag_frame == NULL) break;

        for (int32_t i = 0; i < lost_samples_buf->frame_size; ++i) {
            zero_location = i * sample_size;
            // Check array bounds
            assert((zero_location + sample_size) <= (uint32_t)out_buf->frame_size);
            if (flag_frame[i] == 1) {
                nt_memset((void *)(&data_frame[zero_location]), 0x88, sample_size);
                lost_samples++;
            }
        }

        atomic_add_lost_timesamples(out_buf, out_buf_frame_id, lost_samples);

        mark_frame_empty(lost_samples_buf, unique_name.c_str(), lost_samples_buf_frame_id);
        lost_samples_buf_frame_id = (lost_samples_buf_frame_id + 1) % lost_samples_buf->num_frames;

        mark_frame_full(out_buf, unique_name.c_str(), out_buf_frame_id);
        out_buf_frame_id = (out_buf_frame_id + 1) % out_buf->num_frames;
    }
}