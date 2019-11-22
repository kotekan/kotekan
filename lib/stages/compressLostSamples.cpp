#include <string>

using std::string;

#include "chimeMetadata.h"
#include "compressLostSamples.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(compressLostSamples);

compressLostSamples::compressLostSamples(Config& config_, const string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&compressLostSamples::main_thread, this)) {

    // Apply config.
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _num_sub_freqs = config.get<uint32_t>(unique_name, "num_sub_freqs");

    in_buf = get_buffer("lost_samples_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("compressed_lost_samples_buf");
    register_producer(out_buf, unique_name.c_str());
}

compressLostSamples::~compressLostSamples() {}

void compressLostSamples::main_thread() {

    uint in_buffer_ID = 0; // Process only 1 GPU buffer, cycle through buffer depth
    uint8_t* in_frame;
    int out_buffer_ID = 0;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == NULL)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == NULL)
            goto end_loop;

        // Information on dropped packets
        uint8_t* lost_samples_frame = in_buf->frames[in_buffer_ID];
        uint8_t* compressed_lost_samples_frame = out_buf->frames[out_buffer_ID];

        // Compress lost samples buffer by checking each sample for a flag
        for (uint sample = 0; sample < _samples_per_data_set; sample += 3 * _num_sub_freqs) {
            compressed_lost_samples_frame[sample / (3 * _num_sub_freqs)] = 0;
            for (uint freq = 0; freq < 3 * _num_sub_freqs; freq++) {
                if (lost_samples_frame[sample + freq]) {
                    compressed_lost_samples_frame[sample / (3 * _num_sub_freqs)] = 1;
                    break;
                }
            }
        }
        
        mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);

        // Get a new output buffer
        out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;
        out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
        if (out_frame == NULL)
            goto end_loop;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;

    } // end stop thread
end_loop:;
}
