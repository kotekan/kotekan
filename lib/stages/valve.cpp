#include "valve.hpp"

#include "Config.hpp"
#include "Stage.hpp"        // for Stage
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"       // for Buffer, allocate_new_metadata_object, get_num_consumers
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN
#include "metadata.hpp"          // for metadataContainer
#include "prometheusMetrics.hpp" // for Metrics, Counter
#include "visUtil.hpp"           // for frameID, modulo

#include "fmt.hpp" // for format, fmt

#include <atomic>     // for atomic_bool
#include <cstring>    // for memcpy
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint8_t
#include <string>     // for string, allocator


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(Valve);

Valve::Valve(Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&Valve::main_thread, this)) {

    _buf_in = get_buffer("in_buf");
    _buf_in->register_consumer(unique_name);
    _buf_out = get_buffer("out_buf");
    _buf_out->register_producer(unique_name);
}

void Valve::main_thread() {
    frameID frame_id_in(_buf_in);
    frameID frame_id_out(_buf_out);

    /// Metric to track the number of dropped frames.
    auto& dropped_total =
        Metrics::instance().add_counter("kotekan_valve_dropped_frames_total", unique_name);

    while (!stop_thread) {
        // Fetch a new frame and get its sequence id
        uint8_t* frame_in = _buf_in->wait_for_full_frame(unique_name, frame_id_in);
        if (frame_in == nullptr)
            break;

        // check if there is space for it in the output buffer
        if (_buf_out->is_frame_empty(frame_id_out)) {
            // This call cannot block because of the check above.
            uint8_t* frame_out = _buf_out->wait_for_empty_frame(unique_name, frame_id_out);
            if (frame_out == nullptr)
                break;
            try {
                copy_frame(_buf_in, frame_id_in, _buf_out, frame_id_out);
            } catch (std::exception& e) {
                FATAL_ERROR("Failure copying frame: {:s}\nExiting...", e.what());
                break;
            }
            _buf_out->mark_frame_full(unique_name, frame_id_out++);
        } else {
            WARN("Output buffer full. Dropping incoming frame {:d}.", frame_id_in);
            dropped_total.inc();
        }
        _buf_in->mark_frame_empty(unique_name, frame_id_in++);
    }
}

// mostly copied from VisFrameView
void Valve::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest) {
    buf_dest->allocate_new_metadata_object(frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size));
    }

    // Metadata sizes must match exactly
    if (buf_src->metadata[frame_id_src]->get_object_size()
        != buf_dest->metadata[frame_id_dest]->get_object_size()) {
        throw std::runtime_error(
            fmt::format(fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->metadata[frame_id_src]->get_object_size(),
                        buf_dest->metadata[frame_id_dest]->get_object_size()));
    }

    int num_consumers = buf_src->get_num_consumers();

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Transfer frame contents with directly...
        buf_src->swap_frames(frame_id_src, buf_dest, frame_id_dest);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        std::memcpy(buf_dest->frames[frame_id_dest], buf_src->frames[frame_id_src],
                    buf_src->frame_size);
    }

    // Copy over the metadata
    *buf_dest->metadata[frame_id_dest] = *buf_src->metadata[frame_id_src];
}
