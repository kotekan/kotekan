#include "compressLostSamples.hpp"

#include "StageFactory.hpp"  // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"        // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp" // for atomic_add_lost_timesamples, zero_lost_samples

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <visUtil.hpp> // for frameID, modulo


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(compressLostSamples);

compressLostSamples::compressLostSamples(Config& config_, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&compressLostSamples::main_thread, this)) {

    // Apply config.
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _compression_factor = config.get_default<uint32_t>(unique_name, "compression_factor", 256);
    _zero_all_in_group = config.get<bool>(unique_name, "zero_all_in_group");

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    if (_samples_per_data_set != (uint32_t)in_buf->frame_size) {
        throw std::runtime_error("compressLostSamples in_frame has the wrong size.");
    }

    if ((_samples_per_data_set / _compression_factor)
        != (uint32_t)out_buf->frame_size / sizeof(uint32_t)) {
        throw std::runtime_error("compressLostSamples out_frame has the wrong size.");
    }
}

compressLostSamples::~compressLostSamples() {}

void compressLostSamples::main_thread() {

    frameID in_buffer_ID(in_buf);
    uint8_t* in_frame;
    frameID out_buffer_ID(out_buf);
    uint32_t* out_frame;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            break;

        out_frame = (uint32_t*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
        if (out_frame == nullptr)
            break;

        uint32_t total_lost_samples = 0;

        // Compress lost samples buffer by checking each sample for a flag
        for (uint32_t sample = 0; sample < _samples_per_data_set; sample += _compression_factor) {
            // assert(sample/_compression_factor < (uint32_t)out_buf->frame_size);
            out_frame[sample / _compression_factor] = 0;
            for (uint32_t sub_index = 0; sub_index < _compression_factor; sub_index++) {
                if (_zero_all_in_group && in_frame[sample + sub_index]) {
                    out_frame[sample / _compression_factor] = 1;
                    total_lost_samples += _compression_factor;
                    break;
                } else {
                    out_frame[sample / _compression_factor] += in_frame[sample + sub_index];
                }
            }
            if (!_zero_all_in_group) {
                total_lost_samples += out_frame[sample / _compression_factor];
            }
        }

        // Create new metadata
        allocate_new_metadata_object(out_buf, out_buffer_ID);
        copy_metadata(in_buf, in_buffer_ID, out_buf, out_buffer_ID);
        zero_lost_samples(out_buf, out_buffer_ID);
        atomic_add_lost_timesamples(out_buf, out_buffer_ID, total_lost_samples);

        mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);

        out_buffer_ID++;
        in_buffer_ID++;
    }
}
