#include "timeDownsample.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(timeDownsample);

timeDownsample::timeDownsample(Config& config, const string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&timeDownsample::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the number of time samples to combine
    nsamp = config.get_default<int>(unique_name, "num_samples", 2);

    nprod = num_elements * (num_elements + 1) / 2;
}

void timeDownsample::main_thread() {

    unsigned int frame_id = 0;
    unsigned int nframes = 0; // the number of frames accumulated so far
    unsigned int wdw_pos = 0; // the current position within the accumulation window
    uint64_t wdw_end = 0;     // the end of the accumulation window in FPGA counts
    unsigned int wdw_len = 0; // the length of the accumulation window
    uint64_t fpga_seq_start = 0;
    unsigned int output_frame_id = 0;
    int32_t freq_id = -1; // needs to be set by first frame

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }

        auto frame = visFrameView(in_buf, frame_id);
        fpga_seq_start = std::get<0>(frame.time);

        // The first frame
        if (freq_id == -1) {
            // Enforce starting on an even sample to help with synchronisation
            if (fpga_seq_start % (nsamp * frame.fpga_seq_length) != 0) {
                mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
                frame_id = (frame_id + 1) % in_buf->num_frames;
                continue;
            }
            // Get parameters from first frame
            freq_id = frame.freq_id;
            nprod = frame.num_prod;
            num_elements = frame.num_elements;
            num_eigenvectors = frame.num_ev;
            // Set the parameters of the accumulation window
            wdw_len = nsamp * frame.fpga_seq_length;
            wdw_end = fpga_seq_start + wdw_len;
        } else if (frame.freq_id != (unsigned)freq_id) {
            throw std::runtime_error("Cannot downsample stream with more than one frequency.");
        }

        // Get position within accumulation window
        wdw_pos = (fpga_seq_start % wdw_len) / frame.fpga_seq_length;
        DEBUG("wdw_pos: %d; wdw_end: %d", wdw_pos, wdw_end);

        // Don't start accumulating unless at the start of window
        if (nframes == 0 and wdw_pos != 0) {
            // Skip this frame
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;
            continue;
        } else if (nframes == 0) { // Start accumulating frames

            // Update window
            wdw_end = fpga_seq_start + wdw_len;

            // Wait for an empty frame
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                break;
            }

            // Copy frame into output buffer
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id, frame);

            // Increase the total frame length
            output_frame.fpga_seq_length *= nsamp;

            // We will accumulate inverse weights, i.e. variance
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] = 1. / output_frame.weight[i];
            }

            // Go to next frame
            nframes += 1;
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;
            continue;
        }

        auto output_frame = visFrameView(out_buf, output_frame_id);

        // Check we are still in accumulation window
        if (fpga_seq_start < wdw_end) {
            // Accumulate contents of buffer
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] += frame.vis[i];
                // average inverse weights, i.e. variance
                output_frame.weight[i] += 1. / frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] += frame.eval[i];
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] += frame.evec[k];
                }
            }
            output_frame.erms += frame.erms;

            // Accumulate integration totals
            output_frame.fpga_seq_total += frame.fpga_seq_total;

            // Move to next frame
            nframes += 1;
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;

        } else {
            // Otherwise, stop accumulating
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] /= nframes;
                // extra factor of nsamp for sample variance
                output_frame.weight[i] = nframes * nframes / output_frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] /= nframes;
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] /= nframes;
                }
            }
            output_frame.erms /= nframes;
            // mark as full
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
            // reset accumulation and move on, starting with this frame
            nframes = 0;
        }
    }
}
