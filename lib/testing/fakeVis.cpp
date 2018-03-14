#include "fakeVis.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include <csignal>
#include <time.h>
#include <math.h>


REGISTER_KOTEKAN_PROCESS(fakeVis);
REGISTER_KOTEKAN_PROCESS(replaceVis);


fakeVis::fakeVis(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&fakeVis::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_ev");

    // Get the output buffer
    std::string buffer_name = config.get_string(unique_name, "out_buf");

    // Fetch the buffer, register it
    out_buf = buffer_container.get_buffer(buffer_name);
    register_producer(out_buf, unique_name.c_str());

    // Get frequency IDs from config
    for (auto f : config.get_int_array(unique_name, "freq_ids")) {
        freq.push_back((uint32_t) f);
    }

    // Get fill type
    mode = config.get_string_default(unique_name, "mode", "default");

    // Get timing and frame params
    cadence = config.get_float(unique_name, "cadence");
    num_frames = config.get_int_default(unique_name, "num_frames", -1);
    wait = config.get_bool_default(unique_name, "wait", true);
}

void fakeVis::apply_config(uint64_t fpga_seq) {

}

void fakeVis::main_thread() {

    unsigned int output_frame_id = 0, frame_count = 0;
    uint64_t fpga_seq = 0;

    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // Calculate the time increments in seq and ctime
    uint64_t delta_seq = (uint64_t)(800e6 / 2048 * cadence);
    uint64_t delta_ns = (uint64_t)(cadence * 1000000000);


    while (!stop_thread) {

        double start = current_time();

        for (auto f : freq) {

            DEBUG("Making fake visBuffer for freq=%i, fpga_seq=%i", f, fpga_seq);

            // Wait for the buffer frame to be free
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), 
                                     output_frame_id) == nullptr) {
                break;
            }

            // Below adapted from visWriter

            // Allocate metadata and get frame
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            // TODO: dataset ID properly when we have gated data
            output_frame.dataset_id = 0;

            // Set the frequency index
            output_frame.freq_id = f;

            // Set the time
            output_frame.time = std::make_tuple(fpga_seq, ts);

            // Set the length and total data
            output_frame.fpga_seq_length = delta_seq;
            output_frame.fpga_seq_total = delta_seq;

            // Insert values into vis array to help with debugging
            auto out_vis = output_frame.vis;

            if (mode == "default") {
                // Set diagonal elements to (0, row)
                for (uint32_t i = 0; i < num_elements; i++) {
                    uint32_t pi = cmap(i, i, num_elements);
                    out_vis[pi] = {0., (float) i};
                }
                // Save metadata in first few cells
                if ( sizeof(out_vis) < 4 ) {
                    WARN("Number of elements (%d) is too small to encode \
                          debugging values in fake visibilities", num_elements);
                } else {
                    // For simplicity overwrite diagonal if needed
                    out_vis[0] = {(float) fpga_seq, 0.};
                    out_vis[1] = {(float) (ts.tv_sec + 1e-9 * ts.tv_nsec), 0.};
                    out_vis[2] = {(float) f, 0.};
                    out_vis[3] = {(float) output_frame_id, 0.};
                }
            } else if(mode == "fill_ij") {
                int ind = 0;
                for(uint32_t i = 0; i < num_elements; i++) {
                    for(uint32_t j = i; j < num_elements; j++) {
                        out_vis[ind] = {(float)i, (float)j};
                        ind++;
                    }
                }
            } else if(mode == "phase_ij") {
                int ind = 0;
                for(uint32_t i = 0; i < num_elements; i++) {
                    for(uint32_t j = i; j < num_elements; j++) {
                        float phase = (float) i - (float) j;
                        out_vis[ind] = {cos(phase), sin(phase)};
                        ind++;
                    }
                }
            } else {
                ERROR("Invalid visibility filling mode: %s.", mode.c_str());
                break;
            }

            // Insert values into eigenvectors, eigenvalues and rms
            for (int i = 0; i < num_eigenvectors; i++) {
                for (int j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] = {(float)i, (float)j};
                }
                output_frame.eval[i] = i;
            }
            output_frame.erms = 1.;

            // Set the weights to 1.
            auto out_wei = output_frame.weight;
            int ind = 0;
            for(uint32_t i = 0; i < num_elements; i++) {
                for(uint32_t j = i; j < num_elements; j++) {
                    out_wei[ind] = 1.;
                    ind++;
                }
            }


            // Mark the buffers and move on
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // Increment time
        fpga_seq += delta_seq;
        frame_count++;  // NOTE: frame count increase once for all freq

        // Increment the timespec
        ts.tv_sec += ((ts.tv_nsec + delta_ns) / 1000000000);
        ts.tv_nsec = (ts.tv_nsec + delta_ns) % 1000000000;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if(num_frames > 0 && frame_count >= num_frames) {
            INFO("Reached frame limit [%i frames]. Exiting kotekan...", num_frames);
            std::raise(SIGINT);
            return;
        }

        // If requested sleep for the extra time required to produce a fake vis
        // at the correct cadence
        if(this->wait) {
            double diff = cadence - (current_time() - start);
            timespec ts_diff = double_to_ts(diff);
            nanosleep(&ts_diff, nullptr);
        }
    }
}


replaceVis::replaceVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&replaceVis::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void replaceVis::apply_config(uint64_t fpga_seq) {
}

void replaceVis::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                output_frame_id) == nullptr) {
            break;
        }
        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // Copy input frame to output frame and create view
        allocate_new_metadata_object(out_buf, output_frame_id);
        auto output_frame = visFrameView(out_buf,
                                         output_frame_id, input_frame);

        for(int i = 0; i < output_frame.num_prod; i++) {
            float real = (i % 2 == 0 ?
                          output_frame.freq_id :
                          std::get<0>(output_frame.time));
            float imag = i;

            output_frame.vis[i] = {real, imag};
        }


        // Mark the output buffer and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
