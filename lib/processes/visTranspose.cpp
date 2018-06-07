#include "visTranspose.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <csignal>
#include <stdexcept>
#include "fmt.hpp"
#include "visUtil.hpp"

REGISTER_KOTEKAN_PROCESS(visTranspose);

const size_t BLOCK_SIZE = 32;

visTranspose::visTranspose(Config &config, const string& unique_name, bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&visTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Chunk dimensions for write
    chunk = config.get_int_array(unique_name, "chunk_size");
    if (chunk.size() != 3)
        throw std::runtime_error("Chunk size needs exactly three elements.");
    chunk_t = chunk[2];
    chunk_f = chunk[0];

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
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

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
    double total_time = current_time() - start_time;
    DEBUG("total time %f", total_time);
    DEBUG("wait time %f", wait_time);
    DEBUG("copy time %f", copy_time);
    DEBUG("write time %f", write_time);
}

void visTranspose::main_thread() {

    uint32_t frame_id = 0;
    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0;
    uint32_t ti = 0;
    // offset for copying into buffer
    uint32_t offset = 0;

    start_time = current_time();

    // Create HDF5 file
    file = std::unique_ptr<visFileArchive>(
        new visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev, chunk)
    );

    while (!stop_thread) {
        last_time = current_time();
        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        wait_time += current_time() - last_time;
        last_time = current_time();

        //DEBUG("Frames so far %d", frames_so_far);

        // Collect frames until a chunk is filled
        // Time-transpose as frames come in
        // Fastest varying is time (needs to be consistent with reader!)
        offset = fi * write_t;
        strided_copy(frame.vis.data(), vis.data(), offset*num_prod + ti, write_t, num_prod);
        strided_copy(frame.weight.data(), vis_weight.data(), offset*num_prod + ti, write_t, num_prod);
        // TODO: just fill until these are populated in the frames
        std::fill(gain_coeff.begin() + (offset+ti) * num_prod,
                  gain_coeff.begin() + (offset+ti+1) * num_prod, (cfloat) {1, 0});
        if (fi == 0) {
            std::fill(gain_exp.begin() + (offset+ti) * inputs.size(),
                      gain_exp.begin() + (offset+ti+1) * inputs.size(), 0);
        }
        // TODO: are sizes of eigenvectors always the number of inputs?
        strided_copy(frame.eval.data(), eval.data(), fi*num_ev*write_t + ti, write_t, num_ev);
        strided_copy(frame.evec.data(), evec.data(), fi*num_ev*num_input*write_t + ti,
                     write_t, num_ev*num_input);
        erms[offset + ti] = frame.erms;

        copy_time += current_time() - last_time;

        // Increment within a chunk
        ti = (ti + 1) % write_t;
        if (ti == 0)
            fi++;
        if (fi == write_f) {
            last_time = current_time();
            write();
            write_time += current_time() - last_time;
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

void visTranspose::write() {
    DEBUG("Writing at freq %d and time %d", f_ind, t_ind);
    DEBUG("Writing block of %d freqs and %d times", write_f, write_t);

    file->write_block("vis", f_ind, t_ind, write_f, write_t, vis.data());
    //DEBUG("wrote vis.");

    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t, vis_weight.data());
    //DEBUG("wrote vis_weight");

    file->write_block("gain_coeff", f_ind, t_ind, write_f, write_t, gain_coeff.data());
    //DEBUG("wrote gain_coeff");

    file->write_block("eval", f_ind, t_ind, write_f, write_t, eval.data());
    //DEBUG("wrote eval");

    file->write_block("evec", f_ind, t_ind, write_f, write_t, evec.data());

    file->write_block("erms", f_ind, t_ind, write_f, write_t, erms.data());

    file->write_block("gain_exp", f_ind, t_ind, write_f, write_t, gain_exp.data());

    //DEBUG("wrote all");
    increment_chunk();
}

// TODO: might be better to include same function as used by Reader
void visTranspose::increment_chunk() {
    // Figure out where the next chunk starts
    f_ind = f_edge ? 0 : (f_ind + chunk_f) % num_freq;
    if (f_ind == 0) {
        // set incomplete chunk flag
        f_edge = (num_freq < chunk_f);
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
