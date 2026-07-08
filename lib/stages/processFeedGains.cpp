#include "processFeedGains.hpp"

#include "Config.hpp"         // for Config
#include "DataType.hpp"       // for float16_t
#include "N2Util.hpp"         // for frameID
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for WARN, INFO, DEBUG
#include "visUtil.hpp"        // for current_time, double_to_ts

#include <algorithm>   // fpr copy, copy_n
#include <functional>  // for bind, function, _1
#include <memory>      // for __shared_ptr_access, shared_ptr
#include <stdio.h>     // for fclose, fopen, fread, snprintf, FILE
#include <sys/types.h> // for uint


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(processFeedGains);

processFeedGains::processFeedGains(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&processFeedGains::main_thread, this)) {
    // Apply config.
    num_elements = config.get<uint32_t>(unique_name, "num_elements");
    num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    num_beams = config.get_default<uint32_t>(unique_name, "num_beams", 1);
    upchan_factor = config.get_default<uint32_t>(unique_name, "upchan_factor", 1);
    num_components = config.get<uint32_t>(unique_name, "num_components");
    scaling_factor = config.get_default<float>(unique_name, "scaling_factor", 1.0);

    // Input gain buffers
    json in_buf_list = config.get_value(unique_name, "gain_buffers");
    if (in_buf_list.size() != num_beams) {
        FATAL_ERROR("Expected {:d} gain buffers to match num_beams, got {:d}", num_beams,
                    in_buf_list.size());
    }

    for (size_t beam_id = 0; beam_id < num_beams; beam_id++) {
        const std::string buf_name = in_buf_list.at(beam_id).get<std::string>();
        Buffer* buf = buffer_container.get_buffer(buf_name);

        gain_buffers.push_back(buf);
        buf->register_consumer(unique_name);

        if (buf->frame_size != gain_buffers.at(0)->frame_size) {
            FATAL_ERROR("Input buffers have different frame sizes: {:d} != {:d}", buf->frame_size,
                        gain_buffers.at(0)->frame_size);
        }

        if (buf->frame_size != num_local_freq * num_elements * num_components * sizeof(float)) {
            FATAL_ERROR("Input buffer does not have the expected size. Expected {:d}, got {:d}",
                        num_local_freq * num_elements * num_components * sizeof(float),
                        buf->frame_size);
        }
    }

    // Input mask buffer
    in_mask_buf = get_buffer("in_mask_buf");
    in_mask_buf->register_consumer(unique_name);

    // Output buffer
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Check that buffers are as-expected
    auto in_mask_size = num_elements * sizeof(uint8_t);
    if (in_mask_buf->frame_size != in_mask_size) {
        FATAL_ERROR("Input mask buffer does not have the expected size. Expected {:d}, got {:d}",
                    in_mask_size, in_mask_buf->frame_size);
    }

    out_num_values = num_beams * num_elements * num_local_freq * upchan_factor * num_components;
    auto out_buf_size = out_num_values * sizeof(float16_t);
    if (out_buf->frame_size != out_buf_size) {
        FATAL_ERROR("Output buffer does not have the expected shape. Expected {:d}, got {:d}",
                    out_buf_size, out_buf->frame_size);
    }

    // Set (originate) the output buffer frame description. out_buf is declared
    // `standard` here: this gains subsystem (chime_upgrade_test_gains, DPDK-only)
    // hasn't been migrated to ndarray, so the stage attaches the descriptor itself
    // via ensure_frame_desc. This works equally if out_buf is later declared
    // ndarray, in which case ensure_frame_desc validates against the config
    // descriptor instead of attaching one.
    set_frame_desc(out_buf);

    // Allocate permanent buffers to hold the loaded gains in memory. These will only
    // be updated whenever the gains are updated, and get repeatedly copied
    // into received kotekan buffers
    gain_store_buf =
        std::vector<float16_t>(out_num_values, std::numeric_limits<float16_t>::quiet_NaN());
    mask_store_buf = std::vector<uint8_t>(num_elements, 1u);
}

processFeedGains::~processFeedGains() {}

void processFeedGains::set_frame_desc(Buffer* buf) {
    buf->ensure_frame_desc(kotekan::NDArray<kotekan::GetType_t<kotekan::float16>, 4>::describe(
        "W",
        {static_cast<ptrdiff_t>(num_beams), static_cast<ptrdiff_t>(num_local_freq * upchan_factor),
         static_cast<ptrdiff_t>(num_elements), static_cast<ptrdiff_t>(num_components)},
        {"R", "Fbar", "E", "C"}, {1, 1, 1, 1}));

    freq_upchan_factor = std::vector<int>(num_local_freq * upchan_factor, upchan_factor);
    freq_upchan_index = std::vector<int>(num_local_freq * upchan_factor);
    coarse_freq = std::vector<int>(num_local_freq * upchan_factor, -1);

    // set the actual frequency upchan indices. Assume increasing
    // upchannelized index
    // TODO: this needs to be consistent with the upchannelizer, and
    // potentially configurable
    for (uint64_t f = 0; f < num_local_freq * upchan_factor; ++f) {
        freq_upchan_index[f] = static_cast<int>(f % upchan_factor);
    }
}

void processFeedGains::copy_upchannelize_f(const float* src_f, float16_t* dst_f, size_t fid) {
    (void)fid; // not used in the default implementation
    auto scaling_factor = this->scaling_factor;
    for (size_t u = 0; u < upchan_factor; ++u) {
        // copy ell elements from the source into each fine channel
        float16_t* u_ptr = dst_f + u * num_elements * num_components;
        // apply the constant scaling factor
        std::transform(src_f, src_f + num_components * num_elements, u_ptr,
                       [scaling_factor](float v) { return float16_t(v * scaling_factor); });
    }
}

void processFeedGains::copy_upchannelize(float* frame, size_t beam_id) {
    size_t in_fstride = num_elements * num_components;
    // duplicate the gains at each upchan = 0 bin for each freq
    size_t out_fstride = upchan_factor * num_elements * num_components;

    // data starting at this beam_id
    float16_t* out_ptr = gain_store_buf.data() + beam_id * num_local_freq * out_fstride;

    for (size_t f = 0; f < num_local_freq; ++f) {
        // pointer to this frequency in the input frame
        const float* src = frame + f * in_fstride;
        // pointer to the first fine channel of each coarse freq
        float16_t* dst = out_ptr + f * out_fstride;
        // upchannelize and copy
        copy_upchannelize_f(src, dst, f);
    }
}

void processFeedGains::main_thread() {
    // make frame IDs for each gain buffer, and the output buffers
    N2::frameID in_mask_frame_id(in_mask_buf);
    N2::frameID out_buf_frame_id(out_buf);
    std::vector<N2::frameID> gain_buffer_frame_ids;
    for (size_t iter = 0; iter < gain_buffers.size(); iter++) {
        gain_buffer_frame_ids.emplace_back(gain_buffers.at(iter));
    }

    bool set_coarse_freqs_once = true;

    while (!stop_thread) {
        // Poll all possible producing buffers
        for (size_t beam_id = 0; beam_id < gain_buffers.size(); beam_id++) {
            Buffer* buf = gain_buffers.at(beam_id);
            N2::frameID& frame_id = gain_buffer_frame_ids.at(beam_id);

            // check if this buffer has an available frame, using a short
            // timeout to avoid blocking
            timespec timeout;
            if (set_coarse_freqs_once) {
                // first frame - need to wait until we get something
                timeout = double_to_ts(60 * 60 * 24);
            } else {
                timeout = double_to_ts(current_time());
            }
            int status = buf->wait_for_full_frame_timeout(unique_name, frame_id, timeout);

            if (status == 0) {
                DEBUG("Received gain update for beam {:d}", beam_id);
                // frame available
                float* frame = (float*)buf->frames.at(frame_id);
                // copy gains into the permanent buffer and upchannelize
                copy_upchannelize(frame, beam_id);
                DEBUG("Copied upchannelized gains for beam {:d}", beam_id);

                // set coarse freq metadata once
                if (set_coarse_freqs_once) {
                    auto meta_in = get_chord_metadata(buf, frame_id);
                    auto coarse_freq_in = meta_in->get_coarse_freq();

                    for (uint64_t f = 0; f < num_local_freq * upchan_factor; ++f) {
                        int idx = f / upchan_factor;
                        coarse_freq[f] = coarse_freq_in[idx];
                    }
                    set_coarse_freqs_once = false;
                }

                buf->mark_frame_empty(unique_name, frame_id);
                frame_id++;
            } else if (status == -1) {
                // thread exit signal
                break;
            }
        }
        // Check for mask updates and copy to the mask buffer
        timespec timeout = double_to_ts(current_time());
        int status =
            in_mask_buf->wait_for_full_frame_timeout(unique_name, in_mask_frame_id, timeout);
        if (status == 0) {
            DEBUG("Received bad input mask update");
            // frame available, so update
            uint8_t* in_mask_frame = (uint8_t*)in_mask_buf->frames.at(in_mask_frame_id);
            // copy into the permanent buffer
            std::copy_n(in_mask_frame, num_elements, mask_store_buf.begin());

            in_mask_buf->mark_frame_empty(unique_name, in_mask_frame_id);
            in_mask_frame_id++;
        } else if (status == -1) {
            break;
        }

        // Get an output buffer
        float16_t* out_frame =
            (float16_t*)out_buf->wait_for_empty_frame(unique_name, out_buf_frame_id);
        if (out_frame == nullptr) {
            return;
        }

        // Set metadata from the frame desc, and other metadata
        out_buf->allocate_new_metadata_object(out_buf_frame_id);
        auto meta = get_chord_metadata(out_buf, out_buf_frame_id);
        meta->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());
        meta->set_name("W");
        // Set the frequency upchannelization metadata
        meta->set_freq_upchan_factor(freq_upchan_factor);
        meta->set_freq_upchan_index(freq_upchan_index);
        meta->set_coarse_freq(coarse_freq);
        // Verify that frame desc and metadata match
        meta->check_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

        // copy from permanent buffer to output buffer
        // NB: this needs to happen every time, since the mask and gain buffers
        // can change at a different cadence and thus can't be stored together
        std::copy_n(gain_store_buf.begin(), out_num_values, out_frame);
        // apply the element (feed) mask
        // assume that there are far more good feed than bad feeds, so the mask
        // as the outer loop should minimize the amount of data being touched
        for (size_t e = 0; e < num_elements; e++) {
            if (mask_store_buf[e] == 0) {
                // zero out this element for all other indices
                for (size_t k = num_components * e; k < out_num_values;
                     k += num_components * num_elements) {
                    for (size_t j = 0; j < num_components; ++j) {
                        out_frame[k + j] = float16_t(0.0);
                    }
                }
            }
        }

        out_buf->mark_frame_full(unique_name, out_buf_frame_id);
        out_buf_frame_id++;
    }
}
