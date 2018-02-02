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
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Get the output buffer
    std::string buffer_name = config.get_string(unique_name, "out_buf");

    // Fetch the buffer, register it
    out_buf = buffer_container.get_buffer(buffer_name);
    register_producer(out_buf, unique_name.c_str());

    // Get frequency IDs from config
    for (auto f : config.get_int_array(unique_name, "freq")) {
        freq.push_back((uint16_t) f);
    }

    // Get cadence
    cadence = config.get_float(unique_name, "cadence");

    // Get fill type
    fill_ij = config.get_bool_default(unique_name, "fill_ij", false);
}

void fakeVis::apply_config(uint64_t fpga_seq) {

}

void fakeVis::main_thread() {

    unsigned int output_frame_id = 0;
    uint64_t fpga_seq = 0;
    unsigned int fpga_seq_i = 800e6 / 2048 * cadence;
    timespec ts;
    timespec now;
    clock_gettime(CLOCK_REALTIME, &ts);

    while (!stop_thread) {

        for (uint16_t f : freq) {
            // Wait for the buffer frame to be free
            wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id);

            // Below adapted from visWriter

            // Allocate metadata and get frame
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            // TODO: dataset ID properly when we have gated data
            output_frame.dataset_id() = 0;

            // Set the frequency index
            output_frame.freq_id() = f;

            // Set the time
            output_frame.time() = std::make_tuple(fpga_seq, ts);

            // Insert values into vis array to help with debugging
            std::complex<float> * out_vis = output_frame.vis();

            if(fill_ij) {
                int ind = 0;
                for(int i = 0; i < num_elements; i++) {
                    for(int j = i; j < num_elements; j++) {
                        out_vis[ind] = {(float)i, (float)j};
                        ind++;
                    }
                }
            } else {
                // Set diagonal elements to (0, row)
                for (int i = 0; i < num_elements; i++) {
                    uint32_t pi = cmap(i, i, num_elements);
                    out_vis[pi] = {0., (float) i};
                }
                // Save metadata in first few cells
                if ( sizeof(out_vis) < 4 ) {
                    INFO("Number of elements (%d) is too small to encode \
                          debugging values in fake visibilities", num_elements);
                } else {
                    // For simplicity overwrite diagonal if needed
                    out_vis[0] = {(float) fpga_seq, 0.};
                    out_vis[1] = {(float) (ts.tv_sec + 1e-9 * ts.tv_nsec), 0.};
                    out_vis[2] = {(float) f, 0.};
                    out_vis[3] = {(float) output_frame_id, 0.};
                }
            }

            // Mark the buffers and move on
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // Get current time, delaying to satisfy cadence
        clock_gettime(CLOCK_REALTIME, &now);
        if (float delay = now.tv_sec - ts.tv_sec + 1e-9 * (now.tv_nsec - ts.tv_nsec) < cadence) {
            sleep(cadence - delay + 1);  // delay always > 0
        }
        clock_gettime(CLOCK_REALTIME, &ts);

        // TODO: at some point this should roll over I think?
        fpga_seq += fpga_seq_i;
    }
}
