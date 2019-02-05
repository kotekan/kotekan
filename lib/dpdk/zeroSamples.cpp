#include "zeroSamples.hpp"

#include "chimeMetadata.h"
#include "nt_memset.h"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(zeroSamples);

zeroSamples::zeroSamples(Config& config, const string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&zeroSamples::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    _duplicate_ls_buffer = config.get_default<bool>(unique_name, "duplicate_ls_buffer", false);
    // Register as producer for all desired multiplied lost samples buffers
    if (_duplicate_ls_buffer) {
        json in_bufs = config.get_value(unique_name, "out_lost_sample_buffers");
        for (json::iterator it = in_bufs.begin(); it != in_bufs.end(); ++it) {
            struct Buffer* buf = buffer_container.get_buffer(it.value());
            out_lost_sample_bufs.push_back(buf);
            register_producer(buf, unique_name.c_str());
        }
    }

    lost_samples_buf = get_buffer("lost_samples_buf");
    register_consumer(lost_samples_buf, unique_name.c_str());

    sample_size = config.get_default<uint32_t>(unique_name.c_str(), "sample_size", 2048);
    zero_value = config.get_default<uint8_t>(unique_name.c_str(), "zero_value", 0x88);
}

zeroSamples::~zeroSamples() {}

void zeroSamples::main_thread() {

    uint32_t zero_location;
    int64_t lost_samples;

    while (!stop_thread) {

        lost_samples = 0;

        uint8_t* data_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_frame_id);
        if (data_frame == NULL)
            break;

        uint8_t* flag_frame =
            wait_for_full_frame(lost_samples_buf, unique_name.c_str(), lost_samples_buf_frame_id);
        if (flag_frame == NULL)
            break;

        for (int32_t i = 0; i < lost_samples_buf->frame_size; ++i) {
            zero_location = i * sample_size;
            // Check array bounds
            assert((zero_location + sample_size) <= (uint32_t)out_buf->frame_size);
            if (flag_frame[i] == 1) {
                nt_memset((void*)(&data_frame[zero_location]), zero_value, sample_size);
                lost_samples++;
            }
        }
        if (_duplicate_ls_buffer) {
            for (size_t i = 0; i < out_lost_sample_bufs.size(); i++) {
                uint8_t* new_flag_frame = wait_for_empty_frame(
                    out_lost_sample_bufs[i], unique_name.c_str(), lost_samples_buf_frame_id);
                if (new_flag_frame == NULL)
                    break;
                memcpy(new_flag_frame, flag_frame, lost_samples_buf->frame_size);
                mark_frame_full(out_lost_sample_bufs[i], unique_name.c_str(),
                                lost_samples_buf_frame_id);
            }
        }
        atomic_add_lost_timesamples(out_buf, out_buf_frame_id, lost_samples);

        mark_frame_empty(lost_samples_buf, unique_name.c_str(), lost_samples_buf_frame_id);
        lost_samples_buf_frame_id = (lost_samples_buf_frame_id + 1) % lost_samples_buf->num_frames;

        mark_frame_full(out_buf, unique_name.c_str(), out_buf_frame_id);
        out_buf_frame_id = (out_buf_frame_id + 1) % out_buf->num_frames;
    }
}
