#include "fakeVis.hpp"
#include "visBuffer.hpp"
#include "chimeMetadata.h"
#include <time.h>

fakeVis::fakeVis(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&fakeVis::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int("/", "num_elements");
    block_size = config.get_int("/", "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    // Is num_prod needed?
    num_prod = config.get_int("/", "num_prod");

    // Get the output buffer
    std::string buffer_name = config.get_string(unique_name, "output_buffer");

    // Fetch the buffer, register it
    output_buffer = buffer_container.get_buffer(buffer_name);
    register_producer(output_buffer, unique_name.c_str());

    DEBUG("Buffer size %d", output_buffer->num_frames);

    // Get frequency IDs from config
    for (auto f : config.get_int_array(unique_name, "freq")) {
        freq.push_back((uint16_t) f);
    }

}

void fakeVis::apply_config(uint64_t fpga_seq) {

}

void fakeVis::main_thread() {

    unsigned int output_frame_id = 0;
uint64_t fpga_seq = 0;

    while (!stop_thread) {

        // Get current time TODO: does it matter if cadence is not regular?
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        for (uint16_t f : freq) {
            // Wait for the buffer frame to be free
            wait_for_empty_frame(output_buffer, unique_name.c_str(), output_frame_id);

            // Below adapted from visWriter

            // Allocate metadata and get frame
            // TODO: how do I populate metadata?
            allocate_new_metadata_object(output_buffer, output_frame_id);
            auto output_frame = visFrameView(output_buffer, output_frame_id,
                                             num_elements, num_prod,
                                             num_eigenvectors);

            // TODO: dataset ID properly when we have gated data
            output_frame.dataset_id() = 0;

            // Set the frequency index
            output_frame.freq_id() = f;

            // Set the time
            // TODO: will this work even if bufer has been created from scratch?
            //       seems to always be 0
            //uint64_t fpga_seq = get_fpga_seq_num(output_buffer, frame_id);
            output_frame.time() = std::make_tuple(fpga_seq, ts);

            // Copy the visibility data into a proper triangle and write into
            // the file
            // TODO: generate fake visibilities
            //       in the meantime visibility array and others just point to 
            //       uninitialized area of memory
            //copy_vis_triangle((int32_t *)frame, input_remap, block_size,
            //                  num_elements, output_frame.vis();

            // Mark the buffers and move on
            mark_frame_full(output_buffer, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
        }
        // TODO: at some this point this should roll over I think
        fpga_seq = fpga_seq + 1;
    }
}
