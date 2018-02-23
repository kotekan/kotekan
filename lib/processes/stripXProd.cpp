#include "stripXProd.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include <algorithm>

stripXProd::stripXProd(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&stripXProd::main_thread, this)) {

// TODO: delete
//    // Get list of frequencies to subset from config
//    for (auto ff : config.get_int_array(unique_name, "subset_list")) {
//        subset_list.push_back((uint32_t) ff);
//    }

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

}

void stripXProd::apply_config(uint64_t fpga_seq) {

}

void stripXProd::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

// TODO: delete
//    unsigned int freq;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

// TODO: delete
//        // frequency index of this frame
//        freq = input_frame.freq_id;

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                output_frame_id) == nullptr) {
            break;
        }

        allocate_new_metadata_object(out_buf, output_frame_id);

        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         input_frame.num_elements,
                                         input_frame.num_elements,// n_prod==n_elements
                                         input_frame.num_eigenvectors);

        // Copy auto correlations:
        int idx = 0;
        for (int ii=0; ii<input_frame.num_elements; ii++) {
            for (int jj=ii; jj<input_frame.num_elements; jj++) {
                //
                if (jj==ii) {
                    output_frame.vis[ii] = input_frame.vis[idx];
                }
                idx++;
            }
        }

        std::cout << "Copied autocorrelations" << std::endl;

        // Copy the remaining parts of the buffer:
        std::copy(input_frame.eigenvalues.begin(), input_frame.eigenvalues.end(), 
                  output_frame.eigenvalues.begin());
        std::copy(input_frame.eigenvectors.begin(), input_frame.eigenvectors.end(), 
                  output_frame.eigenvectors.begin());
        std::copy(input_frame.weight.begin(), input_frame.weight.end(), 
                  output_frame.weight.begin());
        output_frame.rms = input_frame.rms;

        std::cout << "Copied buffer" << std::endl;

        // Copy metadata
        output_frame.copy_nonconst_metadata(input_frame);

        std::cout << "Copied metadata" << std::endl;

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;

        std::cout << "Marked and advanced" << std::endl;

    }
}

