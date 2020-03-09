#include "compressLostSamples.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"         // for wait_for_empty_frame, Buffer, allocate_new_metadata_object
#include "chimeMetadata.h"

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <string>      // for string
#include <sys/types.h> // for uint


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

#define NUM_SETS_OF_SUB_FREQS 3

REGISTER_KOTEKAN_STAGE(compressLostSamples);

compressLostSamples::compressLostSamples(Config& config_, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&compressLostSamples::main_thread, this)) {

    // Apply config.
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");

    in_buf = get_buffer("lost_samples_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("compressed_lost_samples_buf");
    register_producer(out_buf, unique_name.c_str());
}

compressLostSamples::~compressLostSamples() {}

void compressLostSamples::main_thread() {

    uint in_buffer_ID = 0;
    uint8_t* in_frame;
    int out_buffer_ID = 0;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == nullptr)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            goto end_loop;

        // Information on dropped packets
        uint8_t* lost_samples_frame = in_buf->frames[in_buffer_ID];
        uint8_t* compressed_lost_samples_frame = out_buf->frames[out_buffer_ID];
        uint32_t total_lost_samples = 0;

        // Compress lost samples buffer by checking each sample for a flag
        for (uint sample = 0; sample < _samples_per_data_set;
             sample += NUM_SETS_OF_SUB_FREQS * _factor_upchan) {
            compressed_lost_samples_frame[sample / (NUM_SETS_OF_SUB_FREQS * _factor_upchan)] = 0;
            for (uint freq = 0; freq < NUM_SETS_OF_SUB_FREQS * _factor_upchan; freq++) {
                if (lost_samples_frame[sample + freq]) {
                    compressed_lost_samples_frame[sample
                                                  / (NUM_SETS_OF_SUB_FREQS * _factor_upchan)] = 1;
                    total_lost_samples += NUM_SETS_OF_SUB_FREQS * _factor_upchan;
                    break;
                }
            }
        }

        // Create new metadata
        allocate_new_metadata_object(out_buf, out_buffer_ID);
        copy_metadata(in_buf, in_buffer_ID, out_buf, out_buffer_ID);
        zero_lost_samples(out_buf, out_buffer_ID);
        atomic_add_lost_timesamples(out_buf, out_buffer_ID, total_lost_samples);

        mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);

        // Get a new output buffer
        out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;
        out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
        if (out_frame == nullptr)
            goto end_loop;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;

    } // end stop thread
end_loop:;
}
