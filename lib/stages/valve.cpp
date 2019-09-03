#include "valve.hpp"

#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <cstring>
#include <pthread.h>
#include <signal.h>
#include <string>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(Valve);

Valve::Valve(Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&Valve::main_thread, this)) {

    _buf_in = get_buffer("in_buf");
    register_consumer(_buf_in, unique_name.c_str());
    _buf_out = get_buffer("out_buf");
    register_producer(_buf_out, unique_name.c_str());
}

void Valve::main_thread() {
    frameID frame_id_in(_buf_in);
    frameID frame_id_out(_buf_out);

    /// Metric to track the number of dropped frames.
    auto& dropped_total =
        Metrics::instance().add_counter("kotekan_valve_dropped_frames_total", unique_name);

    while (!stop_thread) {
        // Fetch a new frame and get its sequence id
        uint8_t* frame_in = wait_for_full_frame(_buf_in, unique_name.c_str(), frame_id_in);
        if (frame_in == nullptr)
            break;

        // check if there is space for it in the output buffer
        if (is_frame_empty(_buf_out, frame_id_out)) {
            // This call cannot block because of the check above.
            uint8_t* frame_out = wait_for_empty_frame(_buf_out, unique_name.c_str(), frame_id_out);
            if (frame_out == nullptr)
                break;
            try {
                copy_frame(_buf_in, frame_id_in, _buf_out, frame_id_out);
            } catch (std::exception& e) {
                FATAL_ERROR("Failure copying frame: {:s}\nExiting...", e.what());
                break;
            }
            mark_frame_full(_buf_out, unique_name.c_str(), frame_id_out++);
        } else {
            WARN("Output buffer full. Dropping incoming frame {:d}.", (int)frame_id_in);
            dropped_total.inc();
        }
        mark_frame_empty(_buf_in, unique_name.c_str(), frame_id_in++);
    }
}

// mostly copied from visFrameView
void Valve::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest) {
    allocate_new_metadata_object(buf_dest, frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size));
    }

    // Metadata sizes must match exactly
    if (buf_src->metadata[frame_id_src]->metadata_size
        != buf_dest->metadata[frame_id_dest]->metadata_size) {
        throw std::runtime_error(
            fmt::format(fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->metadata[frame_id_src]->metadata_size,
                        buf_dest->metadata[frame_id_dest]->metadata_size));
    }

    int num_consumers = get_num_consumers(buf_src);

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Transfer frame contents with directly...
        swap_frames(buf_src, frame_id_src, buf_dest, frame_id_dest);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        std::memcpy(buf_dest->frames[frame_id_dest], buf_src->frames[frame_id_src],
                    buf_src->frame_size);
    }

    // Copy over the metadata
    std::memcpy(buf_dest->metadata[frame_id_dest]->metadata,
                buf_src->metadata[frame_id_src]->metadata,
                buf_src->metadata[frame_id_src]->metadata_size);
}
