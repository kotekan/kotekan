#include "valve.hpp"

#include <stdint.h>               // for uint8_t
#include <cstring>                // for memcpy
#include <exception>              // for exception
#include <functional>             // for bind, function
#include <memory>                 // for shared_ptr, __shared_ptr_access
#include <stdexcept>              // for runtime_error
#include <string>                 // for allocator, basic_string, string
#include <vector>                 // for vector

#include "Config.hpp"             // for Config
#include "Stage.hpp"              // for Stage
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "kotekanLogging.hpp"     // for FATAL_ERROR, WARN
#include "metadata.hpp"           // for metadataObject
#include "prometheusMetrics.hpp"  // for Metrics, Counter
#include "visUtil.hpp"            // for frameID, modulo
#include "fmt.hpp"                // for compile_string_to_view, format, fmt


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
    // ...and the DENOMINATOR. A drop count alone cannot be read: 2909 is catastrophic on a
    // 1-minute run and negligible on an 8-hour one. With both counters any consumer (the
    // viewer's stream-health strip, a prometheus rule) gets the fraction of the stream that
    // was silently lost without needing to know the frame period. (2026-07-27: the L5 peel
    // spent a day looking like broken arithmetic because this loss was invisible -- the
    // dropped frames became sample_seq gaps, which the ring zero-filled, which shredded
    // every coherent window that touched them.)
    auto& passed_total =
        Metrics::instance().add_counter("kotekan_valve_passed_frames_total", unique_name);
    uint64_t n_dropped = 0;

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
            passed_total.inc();
        } else {
            // Rate-limited: one line per frame buried the 2026-07-27 soak log under 4642
            // WARNs, which is how a real signal becomes noise nobody greps for. The ring
            // frame id was never the useful number anyway -- the RUNNING TOTAL is.
            ++n_dropped;
            if (n_dropped == 1 || n_dropped % 100 == 0)
                WARN("Output buffer full, dropping frames: {:d} lost so far (downstream "
                     "cannot keep up; each loss is a gap the consumer must zero-fill).",
                     n_dropped);
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
    buf_dest->metadata[frame_id_dest]->deepCopy(buf_src->metadata[frame_id_src]);
}
