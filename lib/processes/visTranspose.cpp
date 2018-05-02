#include "visTranspose.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include "transpose.c"

REGISTER_KOTEKAN_PROCESS(visTranspose);

const uint BLOCK_SIZE = 32;

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

    // Allocate the memory for write buffer
    char[chunk_f * chunk_t * num_prod * sizeof(cfloat)] write_buf;

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
    file = visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev);

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
        std::copy(frame.eval.begin(), frame.eval.end(), eval.begin() + frames_sofar * num_input);
        std::copy(frame.evec.begin(), frame.evec.end(), evec.begin() + frames_sofar * num_input * num_ev);
        erms[frames_sofar] = frame.erms;

        frames_sofar++;
        if (frames_sofar == chunk_t * chunk_f) {
            // TODO: handle blocks that don't fit exactly in chunk_t*chunk_f
            transpose_write();
            frames_sofar = 0;
        }

        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

void visTranspose::transpose_write() {
    // loop over frequency and transpose
    for (f = 0; f < chunk_f; f++){
        blocked_transpose((void*) (vis.begin() + f * chunk_t * num_prod), (void*) write_buf,
                          chunk_t, num_prod, BLOCK_SIZE, sizeof(cfloat));
        file.write_block("vis", f_ind, t_ind, chunk_f, chunk_t, (cfloat*) write_buf);

        blocked_transpose((void*) (vis_weight.begin() + f * chunk_t * num_prod), (void*) write_buf,
                          chunk_t, num_prod, BLOCK_SIZE, sizeof(float));
        file.write_block("vis_weight", f_ind, t_ind, chunk_f, chunk_t, (float*) write_buf);

        // TODO: for now should bypass this since we are just filling it with ones
        blocked_transpose((void*) (gain_coeff.begin() + f * chunk_t * num_prod), (void*) write_buf,
                          chunk_t, num_prod, BLOCK_SIZE, sizeof(cfloat));
        file.write_block("gain_coeff", f_ind, t_ind, chunk_f, chunk_t, (cfloat*) write_buf);

        blocked_transpose((void*) (eval.begin() + f * chunk_t * num_ev), (void*) write_buf,
                          chunk_t, num_ev, 4, sizeof(float));
        file.write_block("eval", f_ind, t_ind, chunk_f, chunk_t, (float*) write_buf);

        blocked_transpose((void*) (evec.begin() + f * chunk_t * num_ev * num_input), (void*) write_buf,
                          chunk_t, num_input, BLOCK_SIZE, num_ev * sizeof(cfloat));
        file.write_block("evec", f_ind, t_ind, chunk_f, chunk_t, (cfloat*) write_buf);

        file.write_block("erms", f_ind, t_ind, chunk_f, chunk_t, erms.data());
    }

    blocked_transpose((void*) gain_exp.data(), (void*) write_buf,
                      chunk_t, num_input, BLOCK_SIZE, sizeof(int));
    file.write_block("gain_exp", f_ind, t_ind, chunk_f, chunk_t, (int*) write_buf);
}
