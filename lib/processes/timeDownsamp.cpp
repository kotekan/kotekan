#include "timeDownsamp.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(timeDownsamp);

timeDownsamp::timeDownsamp(Config &config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&timeDownsamp::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the number of time samples to combine
    nsamp = config.get_int_default(unique_name, "num_samples", 2);

    nprod = num_elements * (num_elements + 1) / 2;

}

void timeDownsamp::apply_config(uint64_t fpga_seq) {

}

// TODO: Currently there is nothing preventing sets of combined  timestamps from
//       being misaligned between frequencies. Could enforce starting on even/odd
//       number of frames.
//       There is also no mechanism to report or deal with missing frames.
void timeDownsamp::main_thread() {

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

        auto frame = visFrameView(in_buf, frame_id, num_elements, num_eigenvectors);

        // Should only ever read one frequency
        if (freq_id == -1)
            freq_id = frame.freq_id;
        else if (frame.freq_id != freq_id)
            throw std::runtime_error("Cannot downsample stream with more than one frequency.");

        if (nframes == 0) { // Start accumulating frames
            // Wait for an empty frame
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);

            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            // Transfer over the metadata from the first frame
            output_frame.fill_chime_metadata((const chimeMetadata *)in_buf->metadata[frame_id]->metadata);

            // Zero out existing data
            std::fill(output_frame.vis.begin(), output_frame.vis.end(), 0.0);
            std::fill(output_frame.weight.begin(), output_frame.weight.end(), 0.0);
            std::fill(output_frame.eigenvectors.begin(), output_frame.eigenvectors.end(), 0.0);
            std::fill(output_frame.eigenvalues.begin(), output_frame.eigenvalues.end(), 0.0);
            output_frame.rms = 0.0;
        }

        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         num_elements, num_eigenvectors);
        // Accumulate contents of buffer
        for (size_t i = 0; i < nprod; i++) {
            output_frame.vis[i] += frame.vis[i];
            // average inverse weights, i.e. variance
            output_frame.weight[i] += 1. / frame.weight[i];
        }
        for (int i = 0; i < num_eigenvectors; i++) {
            output_frame.eigenvalues[i] += frame.eigenvalues[i];
            for (int j = 0; j < num_elements; j++) {
                int k = i * num_elements + j;
                output_frame.eigenvectors[k] += frame.eigenvectors[k];
            }
        }
        output_frame.rms += frame.rms;

        if (nframes == nsamp - 1) { // Reached the end, average contents of buffer
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] /= nsamp;
                // extra factor of nsamp for sample variance
                output_frame.weight[i] = nsamp*nsamp / output_frame.weight[i];
            }
            for (int i = 0; i < num_eigenvectors; i++) {
                output_frame.eigenvalues[i] /= nsamp;
                for (int j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.eigenvectors[k] /= nsamp;
                }
            }
            output_frame.rms /= nsamp;
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

