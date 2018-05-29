#include "visTranspose.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include "transpose.c"
#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <csignal>
#include <stdexcept>
#include "fmt.hpp"

REGISTER_KOTEKAN_PROCESS(visTranspose);

const size_t BLOCK_SIZE = 32;

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
    filename = config.get_string(unique_name, "filename");

    // TODO: Get metadata from reader somehow
    // For now read from file    // Read the metadata
    std::string md_filename = config.get_string(unique_name, "md_filename");
    INFO("Reading metadata file: %s", md_filename.c_str());
    struct stat st;
    stat(md_filename.c_str(), &st);
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    metadata_file.read((char *)&packed_json[0], filesize);
    std::cout << packed_json.size() << std::endl;
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    metadata = _t["attributes"];
    times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
    prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
    ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();

    num_time = times.size();
    num_freq = freqs.size();
    num_input = inputs.size();
    num_prod = prods.size();
    num_ev = ev.size();

    // Ensure chunk_size not too large
    write_t = std::min(chunk_t, num_time);
    write_f = std::min(chunk_f, num_freq);

    // Allocate the memory for write buffer
    write_buf.reserve(chunk_f * chunk_t * num_prod * sizeof(cfloat));

    // Allocate memory for collecting frames
    vis.reserve(chunk_t*chunk_f*num_prod);
    vis_weight.reserve(chunk_t*chunk_f*num_prod);
    // TODO: fill these at this point?
    gain_coeff.reserve(chunk_t*chunk_f*num_prod);
    gain_exp.reserve(chunk_t*num_input);
    eval.reserve(chunk_t*chunk_f*num_ev);
    evec.reserve(chunk_t*chunk_f*num_ev*num_input);
    erms.reserve(chunk_t*chunk_f);
}

void visTranspose::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

visTranspose::~visTranspose() {
    // Flush up to frames_sofar
}

void visTranspose::main_thread() {

    uint32_t frame_id = 0;
    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0;
    uint32_t ti = 0;
    // offset for copying into buffer
    uint32_t offset = 0;

    // Create HDF5 file
    //      Create datasets and attributes
    //      Should make a new class for transposed files or extend visFile
    file = std::unique_ptr<visFileArchive>(
        new visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev)
    );

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);
        DEBUG("Frames so far %d", frames_so_far);

        // Collect frames until a chunk is filled
        // Fastest varying is time (needs to be consistent with reader!)
        // TODO: could this be streamlined upstream?
        offset = fi * write_t + ti;
        std::copy(frame.vis.begin(), frame.vis.end(), vis.begin() + offset * num_prod);
        std::copy(frame.weight.begin(), frame.weight.end(), vis_weight.begin() + offset * num_prod);
        // TODO: just fill until these are populated in the frames
        std::fill(gain_coeff.begin() + offset * num_prod,
                  gain_coeff.begin() + (offset+1) * num_prod, (cfloat) {1, 0});
        if (fi == 0) {
            std::fill(gain_exp.begin() + offset * inputs.size(),
                      gain_exp.begin() + (offset+1) * inputs.size(), 0);
        }
        // TODO: are sizes of eigenvectors always the number of inputs?
        std::copy(frame.eval.begin(), frame.eval.end(), eval.begin() + offset * num_ev);
        std::copy(frame.evec.begin(), frame.evec.end(), evec.begin() + offset * num_input * num_ev);
        erms[offset] = frame.erms;

        // Increment within a chunk
        ti = (ti + 1) % write_t;
        if (ti == 0)
            fi++;
        if (fi == write_f) {
            transpose_write();
            fi = 0;
            ti = 0;
        }

        frames_so_far++;
        // Exit when all frames have been written
        if (frames_so_far == num_time * num_freq)
            std::raise(SIGINT);

        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

void visTranspose::transpose_write() {
    DEBUG("Writing at freq %d and time %d", f_ind, t_ind);
    DEBUG("Writing block of %d times and %d freqs", write_t, write_f);
    // adjust block size for small dimensions
    // TODO: Just use fixed size for now.
    //size_t block_size = std::min(BLOCK_SIZE, write_t);
    size_t block_size = BLOCK_SIZE;
    size_t ev_block_size = num_ev % 4 == 0 ? std::min(block_size, num_ev) : block_size;

    // Reused parameters for loops
    size_t n_val;
    int err;

    // loop over frequency and transpose
    for (size_t f = 0; f < write_f; f++) {
        n_val = f * write_t * num_prod;
        err = blocked_transpose(&*(vis.begin() + n_val),
                          &*(write_buf.begin() + n_val * sizeof(cfloat)),
                          write_t, num_prod, block_size, sizeof(cfloat));
        if (err != 0)
            throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    }

    //FILE * f = fopen("bit_Table_File", "wb");
    //for (size_t i = 0; i < write_t * write_f * num_prod; i++) {
    //    fprintf(f, "%f,", ((cfloat*) write_buf.data())[i].real());
    //    fprintf(f, "%f\n", ((cfloat*) write_buf.data())[i].imag());
    //}
    //fclose(f);

    DEBUG("transposed vis");
    file->write_block("vis", f_ind, t_ind, write_f, write_t, (const cfloat*) write_buf.data());
    DEBUG("wrote vis.");

    for (size_t f = 0; f < write_f; f++) {
        n_val = f * write_t * num_prod;
        err = blocked_transpose(&*(vis_weight.begin() + n_val),
                          &*(write_buf.begin() + n_val * sizeof(float)),
                          write_t, num_prod, block_size, sizeof(float));
        if (err != 0)
            throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    }
    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t, (const float*) write_buf.data());
    DEBUG("wrote vis_weight");

    for (size_t f = 0; f < write_f; f++) {
        n_val = f * write_t * num_prod;
        // TODO: for now should bypass this since we are just filling it with ones
        err = blocked_transpose(&*(gain_coeff.begin() + n_val),
                          &*(write_buf.begin() + n_val * sizeof(cfloat)),
                          write_t, num_prod, block_size, sizeof(cfloat));
        if (err != 0)
            throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    }
    file->write_block("gain_coeff", f_ind, t_ind, write_f, write_t, (const cfloat*) write_buf.data());
    DEBUG("wrote gain_coeff");

    for (size_t f = 0; f < write_f; f++) {
        n_val = f * write_t * num_ev;
        err = blocked_transpose(&*(eval.begin() + n_val),
                          &*(write_buf.begin() + n_val * sizeof(float)),
                          write_t, num_ev, ev_block_size, sizeof(float));
        if (err != 0)
            throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    }
    DEBUG("transposed eval");
    file->write_block("eval", f_ind, t_ind, write_f, write_t, (const float*) write_buf.data());
    DEBUG("wrote eval");

    size_t i_in, i_out;
    for (size_t f = 0; f < write_f; f++) {
        //n_val = f * write_t * num_ev * num_input;
        //blocked_transpose(&*(evec.begin() + n_val),
        //                  &*(write_buf.begin() + n_val * sizeof(cfloat)),
        //                  write_t, num_ev, ev_block_size, num_input * sizeof(cfloat));
        // TODO: need to stride over evals
        //       for now just do the slow loop transpose.
        for (size_t t = 0; t < write_t; t++) {
            for (size_t e = 0; e < num_ev; e++) {
                for (size_t i = 0; i < num_input; i++) {
                    i_in = num_input * (num_ev * (f * write_t + t) + e) + i;
                    i_out = write_t * (num_input * (f * num_ev + e) + i) + t;
                    ((cfloat*) write_buf.data())[i_out] = evec[i_in];
                }
            }
        }
    }
    DEBUG("transposed evec.");
    file->write_block("evec", f_ind, t_ind, write_f, write_t, (const cfloat*) write_buf.data());

    err = blocked_transpose(erms.data(), write_buf.data(), write_t, write_f,
                      block_size, sizeof(float));
    if (err != 0)
        throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    file->write_block("erms", f_ind, t_ind, write_f, write_t, (const float*) write_buf.data());

    err = blocked_transpose(gain_exp.data(), write_buf.data(),
                      write_t, num_input, block_size, sizeof(int));
    if (err != 0)
        throw std::runtime_error(fmt::format("Blocked transpose raised error code {}", err));
    file->write_block("gain_exp", f_ind, t_ind, write_f, write_t, (const int*) write_buf.data());

    DEBUG("wrote all");
    increment_chunk();
}

// TODO: might be better to include same function as used by Reader
void visTranspose::increment_chunk() {
    // Figure out where the next chunk starts
    f_ind = f_edge ? 0 : (f_ind + chunk_f) % num_freq;
    if (f_ind == 0) {
        f_edge = false;  // reset incomplete chunk flag
        t_ind += chunk_t;
        if (num_time - t_ind < chunk_t) {
            // Reached an incomplete chunk
            t_edge = true;
        }
    } else if (num_freq - f_ind < chunk_f) {
        // Reached an incomplete chunk
        f_edge = true;
    }
    // Determine size of next chunk
    write_f = f_edge ? num_freq - f_ind : chunk_f;
    write_t = t_edge ? num_time - t_ind : chunk_t;
}
