#include "timeDownsample.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(timeDownsample);

timeDownsample::timeDownsample(Config &config,
                               const string& unique_name,
                               bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&timeDownsample::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the number of time samples to combine
    nsamp = config.get_int_default(unique_name, "num_samples", 2);

    nprod = num_elements * (num_elements + 1) / 2;

}

void timeDownsample::apply_config(uint64_t fpga_seq) {

}

// TODO: Currently there is nothing preventing sets of combined  timestamps from
//       being misaligned between frequencies. Could enforce starting on even/odd
//       number of frames.
//       There is also no mechanism to report or deal with missing frames.
void timeDownsample::main_thread() {

    unsigned int frame_id = 0;
    unsigned int nframes = 0;
    unsigned int output_frame_id = 0;
    int32_t freq_id = -1;  // needs to be set by first frame

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }

        auto frame = visFrameView(in_buf, frame_id);

        // Should only ever read one frequency
        if (freq_id == -1) {
            // Get parameters from first frame
            freq_id = frame.freq_id;
            nprod = frame.num_prod;
            num_elements = frame.num_elements;
            num_eigenvectors = frame.num_ev;
        } else if (frame.freq_id != freq_id) {
            throw std::runtime_error("Cannot downsample stream with more than one frequency.");
        }

        if (nframes == 0) { // Start accumulating frames
            // Wait for an empty frame
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);

            // Copy frame into output buffer
            auto output_frame = visFrameView(out_buf, output_frame_id, frame);

            // Go to next frame
            nframes = (nframes + 1) % nsamp;
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % in_buf->num_frames;
            continue;
        }

        auto output_frame = visFrameView(out_buf, output_frame_id);

        // Accumulate contents of buffer
        for (size_t i = 0; i < nprod; i++) {
            output_frame.vis[i] += frame.vis[i];
            // average inverse weights, i.e. variance
            output_frame.weight[i] += 1. / frame.weight[i];
        }
        for (int i = 0; i < num_eigenvectors; i++) {
            output_frame.eval[i] += frame.eval[i];
            for (int j = 0; j < num_elements; j++) {
                int k = i * num_elements + j;
                output_frame.evec[k] += frame.evec[k];
            }
        }
        output_frame.erms += frame.erms;

        if (nframes == nsamp - 1) { // Reached the end, average contents of buffer
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] /= nsamp;
                // extra factor of nsamp for sample variance
                output_frame.weight[i] = nsamp*nsamp / output_frame.weight[i];
            }
            for (int i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] /= nsamp;
                for (int j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] /= nsamp;
                }
            }
            output_frame.erms /= nsamp;
            // mark as full
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;

            // TODO: would be good to verify timestamps are concurrent.
            //       Would need to know integration time.
        }

        // Move to next frame
        nframes = (nframes + 1) % nsamp;
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;

    }
}

