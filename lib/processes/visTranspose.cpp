#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <csignal>
#include <stdexcept>

#include "errors.h"
#include "visBuffer.hpp"
#include "fmt.hpp"
#include "visUtil.hpp"
#include "visTranspose.hpp"
#include "prometheusMetrics.hpp"

REGISTER_KOTEKAN_PROCESS(visTranspose);

visTranspose::visTranspose(Config &config, const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
            std::bind(&visTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // get chunk dimensions for write from config file
    chunk = config.get_int_array(unique_name, "chunk_size");
    if (chunk.size() != 3)
        throw std::invalid_argument("Chunk size needs exactly three elements " \
                "(has " + std::to_string(chunk.size()) + ").");
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument("visTranspose: Config: Chunk size needs " \
                "to be equal to or greater than one.");
    chunk_t = chunk[2];
    chunk_f = chunk[0];

    // Get file path to write to
    // TODO: communicate this from reader
    filename = config.get<std::string>(unique_name, "outfile");

    // TODO: Get metadata from reader somehow
    // For now read from file    // Read the metadata
    std::string md_filename = config.get<std::string>(unique_name, "infile")
            + ".meta";

    INFO("Reading metadata file: %s", md_filename.c_str());
    struct stat st;
    if (stat(md_filename.c_str(), &st) == -1)
        throw std::ios_base::failure("visRawReader: Error reading from " \
                                "metadata file: " + md_filename);
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);
    std::string version;

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // read only if no error
        metadata_file.read((char *)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("visRawReader: Error reading from " \
                                "metadata file: " + md_filename);
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps from metadata
    metadata = _t["attributes"];
    times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
    prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
    ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();

    // Check if this is baseline-stacked data
    if (_t["index_map"].find("stack") != _t["index_map"].end()) {
        stack = _t["index_map"]["stack"].get<std::vector<stack_ctype>>();
        // TODO: verify this is where it gets stored
        reverse_stack = _t["reverse_map"]["stack"].get<std::vector<rstack_ctype>>();
    }

    num_time = times.size();
    num_freq = freqs.size();
    num_input = inputs.size();
    num_prod = prods.size();
    num_ev = ev.size();

    // the dimension of the visibilities is different for stacked data
    eff_prod_dim = (stack.size() > 0) ? stack.size() : num_prod;

    // change archive version: remove "NT_" prefix (not transposed)
    version = metadata["archive_version"];
    if (version.length() > 3) {
        if (version.substr(0, 3) == "NT_")
            metadata["archive_version"] = version.erase(0, 3);
        else
            DEBUG("visTranspose: NT_ prefix not found in archive_version" \
                   " attribute (%s) in metadata file: %s", version.c_str(),
                   md_filename.c_str());
    } else
            DEBUG("visTranspose: found a very short archive_version" \
                   " attribute (%s) in metadata file: %s", version.c_str(),
                   md_filename.c_str());

    DEBUG("File has %d times, %d frequencies, %d products",
                  num_time, num_freq, eff_prod_dim);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    vis.resize(chunk_t*chunk_f*eff_prod_dim);
    vis_weight.resize(chunk_t*chunk_f*eff_prod_dim);
    eval.resize(chunk_t*chunk_f*num_ev);
    evec.resize(chunk_t*chunk_f*num_ev*num_input);
    erms.resize(chunk_t*chunk_f);
    gain.resize(chunk_t*chunk_f*num_input);
    frac_lost.resize(chunk_t*chunk_f);
    input_flags.resize(chunk_t*num_input);
}

void visTranspose::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void visTranspose::main_thread() {

    uint32_t frame_id = 0;
    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0;
    uint32_t ti = 0;
    // offset for copying into buffer
    uint32_t offset = 0;

    uint64_t frame_size = 0;

    // Create HDF5 file
    if (stack.size() > 0) {
        file = std::unique_ptr<visFileArchive>(new visFileArchive(filename,
                    metadata, times, freqs, inputs, prods,
                    stack, reverse_stack, num_ev, chunk)
        );
    } else {
        file = std::unique_ptr<visFileArchive>(new visFileArchive(filename,
                    metadata, times, freqs, inputs, prods, num_ev, chunk)
        );
    }

    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        // Collect frames until a chunk is filled
        // Time-transpose as frames come in
        // Fastest varying is time (needs to be consistent with reader!)
        offset = fi * write_t;
        strided_copy(frame.vis.data(), vis.data(), offset*eff_prod_dim + ti,
                write_t, eff_prod_dim);
        strided_copy(frame.weight.data(), vis_weight.data(),
                offset*eff_prod_dim + ti, write_t, eff_prod_dim);
        strided_copy(frame.eval.data(), eval.data(), fi*num_ev*write_t + ti,
                write_t, num_ev);
        strided_copy(frame.evec.data(), evec.data(),
                fi*num_ev*num_input*write_t + ti, write_t, num_ev*num_input);
        erms[offset + ti] = frame.erms;
        frac_lost[offset + ti] = frame.fpga_seq_length == 0 ?
                1. : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;
        strided_copy(frame.gain.data(), gain.data(), offset*num_input + ti,
                write_t, num_input);
        strided_copy(frame.flags.data(), input_flags.data(), ti,
                write_t, num_input);

        // Increment within read chunk
        ti = (ti + 1) % write_t;
        if (ti == 0)
            fi++;
        if (fi == write_f) {
            // chunk is complete
            write();
            // increment between chunks
            increment_chunk();
            fi = 0;
            ti = 0;

            // export prometheus metric
            if (frame_size == 0)
                frame_size = frame.calculate_buffer_layout(num_input, num_prod,
                        num_ev)["_struct"].second;
            prometheusMetrics::instance().add_process_metric(
                "kotekan_vistranspose_data_transposed_bytes", unique_name,
                        frame_size * frames_so_far);
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

    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t,
            vis_weight.data());
    //DEBUG("wrote vis_weight");

    if (num_ev > 0) {
        file->write_block("eval", f_ind, t_ind, write_f, write_t, eval.data());
        file->write_block("evec", f_ind, t_ind, write_f, write_t, evec.data());
        file->write_block("erms", f_ind, t_ind, write_f, write_t, erms.data());
    }

    file->write_block("gain", f_ind, t_ind, write_f, write_t,
            gain.data());

    file->write_block("flags/inputs", f_ind, t_ind, write_f, write_t,
            input_flags.data());

    file->write_block("flags/frac_lost", f_ind, t_ind, write_f, write_t,
            frac_lost.data());
}

// increment between chunks
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
