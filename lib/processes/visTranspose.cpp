#include "visTranspose.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include "transpose.hpp"

REGISTER_KOTEKAN_PROCESS(visTranspose);

visTranspose::visTranspose(Config &config, const string& unique_name, bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&visTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Chunk dimensions for write
    chunk_t = config.get_int(unique_name, "chunk_dim_time");
    chunk_f = config.get_int(unique_name, "chunk_dim_freq");

    // Get file path to write to
    // TODO: communicate this from reader
    filename = config.get_string(unique_name, "filename")

    // TODO: Get metadata from reader somehow
    // times, freqs, inputs, prods, attributes
}

void visTranspose::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

visTranspose::~visTranspose() {
    // Flush up to frames_sofar
}

void visTranspose::main_thread() {

    uint32_t frame_id = 0;
    uint32_t frames_sofar = 0;

    // Create HDF5 file
    //      Create datasets and attributes
    //      Should make a new class for transposed files or extend visFile
    mk_file();

    while (!stop_thread && frames_sofar < num_time * num_freq) {
        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        // Collect frames until a chunk is filled
        //      The number of frames is specified in the config
        //      Ensure they are coming in the right order
        // format time
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};
        time[frames_sofar] = t;
        // copy data
        std::copy(frame.vis.begin(), frame.vis.end(), vis.begin() + frames_sofar * num_prod);
        std::copy(frame.weight.begin(), frame.weight.end(), vis_weight.begin() + frames_sofar * num_prod);
        std::fill(gain_coeff.begin() + frames_sofar * inputs.size(),
                  gain_coeff.begin() + (frames_sofar+1) * inputs.size(), {1, 0});
        std::fill(gain_exp.begin() + frames_sofar * inputs.size(),
                  gain_exp.begin() + (frames_sofar+1) * inputs.size(), 0);
        // TODO: are sizes of eigenvectors always the number of inputs?
        std::copy(frame.eval.begin(), frame.eval.end(), eval.begin() + frames_sofar * num_input;
        std::copy(frame.evec.begin(), frame.evec.end(), evec.begin() + frames_sofar * num_input * num_ev);
        erms[frames_sofar] = frame.erms;

        frames_sofar++;
        if (frames_sofar == chunk_t * chunk_f) {
            transpose_write();
            frames_sofar = 0;
        }

        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

void visTranspose::mk_file() {
    file = visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev);
}

void visTranspose::transpose_write() {
    // transpose and write data to file
}
