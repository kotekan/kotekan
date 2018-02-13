#include "freqSlicer.hpp"
#include "visFile.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "util.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "gpuPostProcess.hpp"
#include "errors.h"
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>


freqSplit::freqSplit(Config& config,
                     const string& unique_name,
                     bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&freqSplit::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> output_buffer_names =
        config.get_string_array(unique_name, "out_bufs");

    // Setup the input buffer
    input_buffer = get_buffer("in_buf");
    register_consumer(input_buffer, unique_name.c_str());

    // Fetch the output buffers, register them, and store them in our buffer vector
    for(auto name : output_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_producer(buf, unique_name.c_str());
        output_buffers.push_back({buf, 0});
    }
}

void freqSplit::apply_config(uint64_t fpga_seq) {

}

void freqSplit::main_thread() {

    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(input_buffer,
                    unique_name.c_str(),input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(input_buffer, input_frame_id,
                                       num_elements, num_eigenvectors);

        // frequency index of this frame
        freq = input_frame.freq_id();

        // Choose output buffer for this frequency:
        if(freq < 512) {
            buf_ind=0;
        } else {
            buf_ind=1;
        }
        auto& buffer_pair = output_buffers[buf_ind];
        std::tie(buf, frame_id) = buffer_pair;

        INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(buf, unique_name.c_str(),
                                        frame_id) == nullptr) {
            break;
        }

        allocate_new_metadata_object(buf, frame_id);

        // Copy frame and create view
        auto frame = visFrameView(buf, frame_id, input_frame);

        // Mark the buffers and move on
        mark_frame_empty(input_buffer, unique_name.c_str(),
                                                input_frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
    }

}


freqSubset::freqSubset(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&freqSubset::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int("/", "num_elements");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    // Get list of frequencies to subset from config
    for (auto ff : config.get_int_array(unique_name, "subset_list")) {
        subset_list.push_back((uint16_t) ff);
    }

    // Setup the input buffer
    input_buffer = get_buffer("in_buf");
    register_consumer(input_buffer, unique_name.c_str());

    // Setup the output buffer
    output_buffer = get_buffer("out_buf");
    register_producer(output_buffer, unique_name.c_str());

}

void freqSubset::apply_config(uint64_t fpga_seq) {

}

void freqSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(input_buffer,
                    unique_name.c_str(),input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(input_buffer, input_frame_id,
                                       num_elements, num_eigenvectors);

        // frequency index of this frame
        freq = input_frame.freq_id();

        // If this frame is part of subset
        // TODO: Aparently std::set can be used to speed up this search
        if (std::find(subset_list.begin(), subset_list.end(), freq)
                                                != subset_list.end()) {

            // Wait for the output buffer to be empty of data
            if(wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                            output_frame_id) == nullptr) {
                break;
            }

            allocate_new_metadata_object(output_buffer, output_frame_id);

            // Copy frame and create view
            auto output_frame = visFrameView(output_buffer,
                                            output_frame_id, input_frame);

            // Mark the output buffer and move on
            mark_frame_full(output_buffer, unique_name.c_str(),
                                                        output_frame_id);
            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
        }
        // Mark the input buffer and move on
        mark_frame_empty(input_buffer, unique_name.c_str(),
                                                input_frame_id);
        // Advance the current input frame id
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
    }
}
