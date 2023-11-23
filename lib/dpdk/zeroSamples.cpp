#include "zeroSamples.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_full, register_producer, wait_for_empt...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for atomic_add_lost_timesamples
#include "nt_memset.h"         // for nt_memset

#include "json.hpp" // for json, basic_json, basic_json<>::iterator, iter_impl

#include <algorithm>  // for max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy, size_t
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(zeroSamples);

zeroSamples::zeroSamples(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&zeroSamples::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _duplicate_ls_buffer = config.get_default<bool>(unique_name, "duplicate_ls_buffer", false);
    // Register as producer for all desired multiplied lost samples buffers
    if (_duplicate_ls_buffer) {
        json in_bufs = config.get_value(unique_name, "out_lost_sample_buffers");
        for (json::iterator it = in_bufs.begin(); it != in_bufs.end(); ++it) {
            Buffer* buf = buffer_container.get_buffer(it.value());
            out_lost_sample_bufs.push_back(buf);
            buf->register_producer(unique_name);
        }
    }

    lost_samples_buf = get_buffer("lost_samples_buf");
    lost_samples_buf->register_consumer(unique_name);

    sample_size = config.get_default<uint32_t>(unique_name.c_str(), "sample_size", 2048);
    zero_value = config.get_default<uint8_t>(unique_name.c_str(), "zero_value", 0x88);
}

zeroSamples::~zeroSamples() {}

void zeroSamples::main_thread() {

    uint32_t zero_location;
    int64_t lost_samples;

    while (!stop_thread) {

        lost_samples = 0;

        uint8_t* data_frame = out_buf->wait_for_empty_frame(unique_name, out_buf_frame_id);
        if (data_frame == nullptr)
            break;

        uint8_t* flag_frame =
            lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_buf_frame_id);
        if (flag_frame == nullptr)
            break;

        for (size_t i = 0; i < lost_samples_buf->frame_size; ++i) {
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
                uint8_t* new_flag_frame = out_lost_sample_bufs[i]->wait_for_empty_frame(
                    unique_name, lost_samples_buf_frame_id);
                if (new_flag_frame == nullptr)
                    break;
                memcpy(new_flag_frame, flag_frame, lost_samples_buf->frame_size);
                out_lost_sample_bufs[i]->mark_frame_full(unique_name, lost_samples_buf_frame_id);
            }
        }
        atomic_add_lost_timesamples(out_buf, out_buf_frame_id, lost_samples);

        lost_samples_buf->mark_frame_empty(unique_name, lost_samples_buf_frame_id);
        lost_samples_buf_frame_id = (lost_samples_buf_frame_id + 1) % lost_samples_buf->num_frames;

        out_buf->mark_frame_full(unique_name, out_buf_frame_id);
        out_buf_frame_id = (out_buf_frame_id + 1) % out_buf->num_frames;
    }
}
