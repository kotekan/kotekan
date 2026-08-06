#include "countCheck.hpp"

#include "BeamMetadata.hpp"    // for BeamMetadata
#include "Config.hpp"          // for Config
#include "HFBMetadata.hpp"     // for HFBMetadata
#include "N2Metadata.hpp"      // for N2Metadata
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"       // for Telescope
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG
#include "metadata.hpp"        // for metadataObject, metadataPool
#include "visBuffer.hpp"       // for VisMetadata

#include "fmt.hpp" // for compile_string_to_view

#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast
#include <stdlib.h>   // for llabs
#include <sys/time.h> // for timeval
#include <time.h>     // for timespec
#include <utility>    // for pair
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(countCheck);


countCheck::countCheck(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&countCheck::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    // Fetch tolerance from config.
    start_time_tolerance = config.get_default<int>(unique_name, "start_time_tolerance", 3);

    // Initialize the start_time to zero:
    start_time_ns = 0;

    tick_len_ns = Telescope::instance().seq_length_nsec();
    if (tick_len_ns == 0) {
        FATAL_ERROR("countCheck: telescope tick length must be > 0.");
    }

    tolerance_ns = (int64_t)start_time_tolerance * 1000000000LL;
}

std::pair<int64_t, int64_t> countCheck::extract_timing_info(int frame_id) {
    auto meta = in_buf->metadata[frame_id];
    if (!meta) {
        FATAL_ERROR("countCheck: Frame {:d} has no metadata", frame_id);
        return {0, 0};
    }

    // Check metadata type and extract timing information accordingly
    std::string pool_type = meta->parent_pool.lock()->type_name;

    if (pool_type == "VisMetadata") {
        auto vis_meta = std::dynamic_pointer_cast<VisMetadata>(meta);
        int64_t fpga_seq = (int64_t)vis_meta->fpga_seq_start;
        int64_t time_ns = (int64_t)vis_meta->ctime.tv_sec * 1000000000LL + vis_meta->ctime.tv_nsec;
        DEBUG("countCheck: VisMetadata fpga_seq={:d}, time_ns={:d}", fpga_seq, time_ns);
        return {fpga_seq, time_ns};
    }

    if (pool_type == "N2Metadata") {
        auto n2_meta = std::dynamic_pointer_cast<N2Metadata>(meta);
        int64_t fpga_seq = (int64_t)n2_meta->fpga_start_tick;
        int64_t time_ns = (int64_t)n2_meta->frame_start_time_ns;
        DEBUG("countCheck: N2Metadata fpga_seq={:d}, time_ns={:d}", fpga_seq, time_ns);
        return {fpga_seq, time_ns};
    }

    if (pool_type == "HFBMetadata") {
        auto hfb_meta = std::dynamic_pointer_cast<HFBMetadata>(meta);
        int64_t fpga_seq = hfb_meta->fpga_seq_start;
        int64_t time_ns = (int64_t)hfb_meta->ctime.tv_sec * 1000000000LL + hfb_meta->ctime.tv_nsec;
        DEBUG("countCheck: HFBMetadata fpga_seq={:d}, time_ns={:d}", fpga_seq, time_ns);
        return {fpga_seq, time_ns};
    }

    if (pool_type == "BeamMetadata") {
        auto beam_meta = std::dynamic_pointer_cast<BeamMetadata>(meta);
        int64_t fpga_seq = beam_meta->fpga_seq_start;
        int64_t time_ns =
            (int64_t)beam_meta->ctime.tv_sec * 1000000000LL + beam_meta->ctime.tv_nsec;
        DEBUG("countCheck: BeamMetadata fpga_seq={:d}, time_ns={:d}", fpga_seq, time_ns);
        return {fpga_seq, time_ns};
    }

    if (pool_type == "chordMetadata") {
        auto chord_meta = get_chord_metadata(meta);
        if (!chord_meta) {
            FATAL_ERROR("countCheck: Failed to get chordMetadata from frame {:d}", frame_id);
            return {0, 0};
        }
        if (!chord_meta->has_fpga_seq_num()) {
            FATAL_ERROR("countCheck: chordMetadata on frame {:d} does not have fpga_seq_num set",
                        frame_id);
            return {0, 0};
        }
        if (!chord_meta->has_first_packet_recv_time()) {
            FATAL_ERROR(
                "countCheck: chordMetadata on frame {:d} does not have first_packet_recv_time set",
                frame_id);
            return {0, 0};
        }
        int64_t fpga_seq = chord_meta->get_fpga_seq_num();
        // Use first_packet_recv_time (wall clock) rather than get_gps_time() which is
        // computed from fpga_seq. This allows restart detection since first_packet_recv_time
        // continues to advance even when fpga_seq resets.
        timeval tv = chord_meta->get_first_packet_recv_time();
        int64_t time_ns = (int64_t)tv.tv_sec * 1000000000LL + (int64_t)tv.tv_usec * 1000LL;
        DEBUG("countCheck: chordMetadata fpga_seq={:d}, time_ns={:d}", fpga_seq, time_ns);
        return {fpga_seq, time_ns};
    }

    FATAL_ERROR("countCheck: Metadata type '{:s}' does not support timing information. "
                "Supported types: VisMetadata, N2Metadata, HFBMetadata, BeamMetadata, "
                "chordMetadata.",
                pool_type);
    return {0, 0};
}

void countCheck::main_thread() {

    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }

        // Extract timing from metadata (works with any supported metadata type)
        auto [fpga_seq, time_ns] = extract_timing_info(input_frame_id);
        int64_t new_start_time = time_ns - fpga_seq * (int64_t)tick_len_ns;

        DEBUG("countCheck: fpga_seq={:d}, time_ns={:d}, start_time_ns={:d}, time_diff={:d}",
              fpga_seq, time_ns, start_time_ns, llabs(start_time_ns - new_start_time));

        // If this is the first frame, store start time
        if (start_time_ns == 0) {
            start_time_ns = new_start_time;
            // Else, test that start time is still the same
        } else if (llabs(start_time_ns - new_start_time) > tolerance_ns) {
            // Shut Kotekan down
            FATAL_ERROR("Found wrong start time. Possible acquisition re-start occurred.");
            break;
        }

        // Mark the buffers and move on
        in_buf->mark_frame_empty(unique_name, input_frame_id);

        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
