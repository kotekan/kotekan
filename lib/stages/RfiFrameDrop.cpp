#include "RfiFrameDrop.hpp"

#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "chimeMetadata.h"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <cstring>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <string>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(RfiFrameDrop);

RfiFrameDrop::RfiFrameDrop(Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RfiFrameDrop::main_thread, this)),
    _dropped_frame_counter(Metrics::instance().add_counter("kotekan_rfiframedrop_dropped_frame_total",
                                                           unique_name,
                                                           {"freq_id", "threshold", "fraction"})),
    _frame_counter(Metrics::instance().add_counter("kotekan_rfiframedrop_frame_total",
                                                   unique_name, {"freq_id"}))
{

    _buf_in_vis = get_buffer("in_buf_vis");
    _buf_in_sk = get_buffer("in_buf_sk");
    _buf_out = get_buffer("out_buf");
    register_consumer(_buf_in_vis, unique_name.c_str());
    register_consumer(_buf_in_sk, unique_name.c_str());
    register_producer(_buf_out, unique_name.c_str());

    auto num_samples = config.get<size_t>(unique_name, "samples_per_data_set");
    sk_step = config.get<size_t>(unique_name, "sk_step");
    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_sub_frames = config.get<size_t>(unique_name, "num_sub_frames");
    sk_samples_per_frame = num_samples / sk_step / num_sub_frames;
    enable_rfi_zero = config.get_default<bool>(unique_name, "enable_rfi_zero", false);

    assert((size_t)_buf_in_sk->frame_size == sizeof(float) * sk_samples_per_frame * num_sub_frames);

    auto j = config.get<nlohmann::json>(unique_name, "thresholds");

    if (!parse_thresholds(j)) {
        FATAL_ERROR("Error parsing thresholds.");
        return;
    }
}

void RfiFrameDrop::main_thread() {
    frameID frame_id_in_vis(_buf_in_vis);
    frameID frame_id_in_sk(_buf_in_sk);
    frameID frame_id_out(_buf_out);

    size_t num_thresholds = _thresholds.size();
    std::vector<float> sk_delta(sk_samples_per_frame);
    std::vector<size_t> sk_exceeds(num_thresholds, 0);

    // Calculate the scaling to turn kurtosis value into sigma
    // TODO: this should really use the actual number of bad_inputs
    // In theory this could be pre-applied to the threshold, but that's more
    // difficult if we dynamically receive the bad input list
    size_t num_inputs = num_elements;
    float sigma_scale = sqrt((num_inputs * (sk_step - 1) * (sk_step + 2) * (sk_step + 3)) / (4.0 * sk_step * sk_step));


    while (!stop_thread) {
        // Fetch the input buffers
        uint8_t* frame_in_vis = wait_for_full_frame(_buf_in_vis, unique_name.c_str(), frame_id_in_vis);
        float* frame_in_sk = (float*)wait_for_full_frame(_buf_in_sk, unique_name.c_str(), frame_id_in_sk);

        // Test to ensure we actually got valid buffers back
        if (frame_in_vis == nullptr || frame_in_sk == nullptr)
            break;

        auto* metadata_vis = (chimeMetadata*)get_metadata(_buf_in_vis, frame_id_in_vis);
        auto* metadata_sk = (chimeMetadata*)get_metadata(_buf_in_sk, frame_id_in_sk);

        // Set the frequency index from the stream id of the metadata
        auto stream_id = extract_stream_id(metadata_vis->stream_ID);
        uint32_t freq_id = bin_number_chime(&stream_id);

        // Try and synchronize up the frames. Even though they arrive at
        // different rates, this should eventually sync them up.
        if (metadata_vis->fpga_seq_num < metadata_sk->fpga_seq_num) {
            mark_frame_empty(_buf_in_vis, unique_name.c_str(), frame_id_in_vis++);
            continue;
        }
        if (metadata_sk->fpga_seq_num < metadata_vis->fpga_seq_num) {
            mark_frame_empty(_buf_in_sk, unique_name.c_str(), frame_id_in_sk++);
            continue;
        }

        for (size_t ii = 0; ii < num_sub_frames; ii++) {

            bool skip = false;

            // Process all the SK values to their deltas and for each threshold
            // we need to count how many samples exceed that threshold
            for (size_t jj = 0; jj < sk_samples_per_frame; jj++) {
                float sk_sig = fabs(sigma_scale * frame_in_sk[ii * sk_samples_per_frame + jj] - 1.0f);

                for (size_t kk = 0; kk < num_thresholds; kk++) {
                    sk_exceeds[kk] += (sk_sig > std::get<0>(_thresholds[kk]));
                }
            }

            for (size_t kk = 0; kk < num_thresholds; kk++) {
                if (sk_exceeds[kk] > std::get<1>(_thresholds[kk])) {
                    skip = true;
                    _dropped_frame_counter.labels({
                        std::to_string(freq_id), std::to_string(std::get<2>(_thresholds[kk])), std::to_string(std::get<2>(_thresholds[kk]))
                    }).inc();
                }
                // TODO: log prometheus output

                // Reset counters for the next sub_frame
                sk_exceeds[kk] = 0;
            }

            // If no frame exceeded it's threshold then we should transfer the
            // frame over to the output and release it. If we want to drop the
            // incoming frame then we leave the output as is.
            if (!skip || !enable_rfi_zero) {

                if (wait_for_empty_frame(_buf_out, unique_name.c_str(), frame_id_out) == nullptr) {
                    break;
                }
                copy_frame(_buf_in_vis, frame_id_in_vis, _buf_out, frame_id_out);
                mark_frame_full(_buf_out, unique_name.c_str(), frame_id_out++);
            }
            //
            mark_frame_empty(_buf_in_vis, unique_name.c_str(), frame_id_in_vis++);
            _frame_counter.labels({std::to_string(freq_id)}).inc();
        }
    }
}


// mostly copied from visFrameView
void RfiFrameDrop::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest) {
    allocate_new_metadata_object(buf_dest, frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size));
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

    pass_metadata(buf_src, frame_id_src, buf_dest, frame_id_dest);
}



bool RfiFrameDrop::parse_thresholds(nlohmann::json j) {

    if (!j.is_array()) {
        ERROR("Could not parse thresholds: entry is not a list : {}", j.dump());
        return false;
    }

    for (const auto& t : j) {

        if (!t.is_object()) {
            ERROR("Could not parse thresholds item: element is not a dict : {}", t.dump());
            return false;
        }

        if (t.count("threshold") == 0) {
            ERROR("Required key `threshold` not present in item: {}", t.dump());
            return false;
        }

        if (t.count("fraction") == 0) {
            ERROR("Required key `fraction` not present in item: {}", t.dump());
            return false;
        }

        float threshold = t["threshold"].get<float>();
        float fraction = t["fraction"].get<float>();

        _thresholds.push_back({threshold, (size_t)(fraction * sk_samples_per_frame), fraction});
    }
    return true;
}