#include "invalidateVDIFframes.hpp"

#include "chimeMetadata.h"
#include "vdif_functions.h"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(invalidateVDIFframes);

invalidateVDIFframes::invalidateVDIFframes(Config& config, const string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&invalidateVDIFframes::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    lost_samples_buf = get_buffer("lost_samples_buf");
    register_consumer(lost_samples_buf, unique_name.c_str());
}

invalidateVDIFframes::~invalidateVDIFframes() {}

void invalidateVDIFframes::main_thread() {

    uint32_t frame_location;
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
            frame_location = i * vdif_frame_size * num_elements;
            // Check array bounds
            assert((frame_location + vdif_frame_size * num_elements)
                   <= (uint32_t)out_buf->frame_size);
            if (flag_frame[i] == 1) {
                // There is one VDIF frame generated for each element extracted from a sample.  So
                // losing one sample results in losing one or more (default 2) VDIF frames.
                for (uint32_t j = 0; j < num_elements; ++j) {
                    uint32_t sub_frame_location = frame_location + j * vdif_frame_size;
                    struct VDIFHeader* header = (struct VDIFHeader*)&data_frame[sub_frame_location];
                    header->invalid = 1;
                }
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
