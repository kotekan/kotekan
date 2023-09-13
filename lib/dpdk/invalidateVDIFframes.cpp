#include "invalidateVDIFframes.hpp"

#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for Buffer, mark_frame_empty, mark_frame_full, register_con...
#include "chimeMetadata.hpp"     // for atomic_add_lost_timesamples
#include "prometheusMetrics.hpp" // for Metrics, Counter
#include "vdif_functions.h"      // for VDIFHeader

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stddef.h>   // for size_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(invalidateVDIFframes);

invalidateVDIFframes::invalidateVDIFframes(Config& config, const std::string& unique_name,
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

    auto& lost_frame_total =
        Metrics::instance().add_counter("kotekan_vdif_lost_frames_total", unique_name);

    while (!stop_thread) {

        lost_samples = 0;

        uint8_t* data_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_frame_id);
        if (data_frame == nullptr)
            break;

        uint8_t* flag_frame =
            wait_for_full_frame(lost_samples_buf, unique_name.c_str(), lost_samples_buf_frame_id);
        if (flag_frame == nullptr)
            break;

        for (size_t i = 0; i < lost_samples_buf->frame_size; ++i) {
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
        lost_frame_total.inc(lost_samples);

        mark_frame_empty(lost_samples_buf, unique_name.c_str(), lost_samples_buf_frame_id);
        lost_samples_buf_frame_id = (lost_samples_buf_frame_id + 1) % lost_samples_buf->num_frames;

        mark_frame_full(out_buf, unique_name.c_str(), out_buf_frame_id);
        out_buf_frame_id = (out_buf_frame_id + 1) % out_buf->num_frames;
    }
}
