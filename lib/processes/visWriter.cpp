#include "visWriter.hpp"
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


visTransform::visTransform(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int("/", "num_elements");
    block_size = config.get_int("/", "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get_string_array(unique_name, "input_buffers");

    // Fetch the input buffers, register them, and store them in our buffer vector
    for(auto name : input_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_consumer(buf, unique_name.c_str());
        input_buffers.push_back({buf, 0});
    }

    // Setup the output vector
    output_buffer = get_buffer("output_buffer");
    register_producer(output_buffer, unique_name.c_str());

    // Get the indices for reordering
    input_remap = std::get<0>(parse_reorder_default(config, unique_name));
}

void visTransform::apply_config(uint64_t fpga_seq) {

}

void visTransform::main_thread() {

    uint8_t * frame = nullptr;
    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    while (!stop_thread) {

        // This is where all the main set of work happens. Iterate over the
        // available buffers, wait for data to appear and then attempt to write
        // the data into a file
        unsigned int buf_ind = 0;
        for(auto& buffer_pair : input_buffers) {
            std::tie(buf, frame_id) = buffer_pair;

            INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);

            // Wait for the buffer to be filled with data
            if((frame = wait_for_full_frame(buf, unique_name.c_str(),
                                            frame_id)) == nullptr) {
                break;
            }

            // Wait for the buffer to be filled with data
            if(wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }
            allocate_new_metadata_object(output_buffer, output_frame_id);

            auto output_frame = visFrameView(output_buffer, output_frame_id,
                                             num_elements, num_eigenvectors);

            // TODO: set the dataset ID properly when we have gated data
            output_frame.dataset_id() = 0;

            // Set the frequency index from the stream id of this buffer
            stream_id_t stream_id = get_stream_id_t(buf, frame_id);
            output_frame.freq_id() = bin_number_chime(&stream_id);

            // Set the time
            // TODO: get the GPS time instead
            uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
            timeval tv = get_first_packet_recv_time(buf, frame_id);
            timespec ts;
            TIMEVAL_TO_TIMESPEC(&tv, &ts);
            output_frame.time() = std::make_tuple(fpga_seq, ts);

            // TODO: do something with the list timesamples data
            // uint64_t lost_samples = get_lost_timesamples(buf, frame_id);

            // Copy the visibility data into a proper triangle and write into
            // the file
            copy_vis_triangle((int32_t *)frame, input_remap, block_size,
                              num_elements, output_frame.vis());

            // Mark the buffers and move on
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            mark_frame_full(output_buffer, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
            output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
            buf_ind++;
        }

    }

}


visDebug::visDebug(Config& config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visDebug::main_thread, this)) {

    // Setup the input vector
    buffer = get_buffer("buffer");
    register_consumer(buffer, unique_name.c_str());
}

void visDebug::apply_config(uint64_t fpga_seq) {

}

void visDebug::main_thread() {

    unsigned int frame_id = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(buffer, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        auto frame = visFrameView(buffer, frame_id);
        INFO("%s", frame.summary().c_str());

        // Mark the buffers and move on
        mark_frame_empty(buffer, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % buffer->num_frames;

    }

}


visWriter::visWriter(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visWriter::main_thread, this)) {

    // Fetch any simple configuration
    // num_freq = config.get_int(unique_name, "num_freq");
    root_path = config.get_string_default(unique_name, "root_path", ".");

    // Get the list of buffers that this process shoud connect to
    buffer = get_buffer("buffer");
    register_consumer(buffer, unique_name.c_str());

    // Get the input labels
    inputs = std::get<1>(parse_reorder_default(config, unique_name));

    // TODO: dynamic setting of instrument name, shouldn't be hardcoded here, At
    // the moment this either uses chime, or if set to use a per_node_instrument
    // it uses the hostname of the current node
    node_mode = config.get_bool_default(unique_name, "separate_nodes", true);

    if(node_mode) {

        // Set the instrument_name from the hostname
        char temp[256];
        gethostname(temp, 256);
        std::string t = temp;
        // Here we trim the hostname to the first alphanumeric segment only.
        instrument_name = t.substr(0, (t + ".").find_first_of(".-"));

        node_mode = true;
        num_freq = 4;

    } else {
        instrument_name = config.get_string_default(unique_name, "instrument_name", "chime");
        freq_id_list = config.get_int_array(unique_name, "freq_ids");
    }
}

void visWriter::apply_config(uint64_t fpga_seq) {

}

void visWriter::main_thread() {


    unsigned int frame_id = 0;


    // Look over the current buffers for information to setup the acquisition
    init_acq();

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(buffer, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(buffer, frame_id);

        // Construct the new time
        auto ftime = frame.time();
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

        // Check if the frequency we are receiving is on the list of frequencies we are processing
        if(freq_map.count(frame.freq_id()) == 0) {
            WARN("Frequency id=%i is not enabled for visWriter, discarding frame", frame.freq_id());
        } else {

            INFO("Writing frequency id=%i", frame.freq_id());

            // Lookup the frequency index if reordering, otherwise write out in buffer order
            uint32_t freq_ind = freq_map[frame.freq_id()];

            // Create fake entries to fill out the gain and weight datasets with
            // because these don't correctly make it through kotekan yet
            std::vector<std::complex<float>> vis(frame.vis(), frame.vis() + frame.num_prod());
            std::vector<uint8_t> vis_weight(vis.size(), 255);
            std::vector<std::complex<float>> gain_coeff(inputs.size(), {1, 0});
            std::vector<int32_t> gain_exp(inputs.size(), 0);

            // Add all the new information to the file.
            file_bundle->addSample(t, freq_ind, vis, vis_weight,
                                   gain_coeff, gain_exp);
        }

        // Mark the buffers and move on
        mark_frame_empty(buffer, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % buffer->num_frames;

    }
}



void visWriter::init_acq() {

    // Fetch out information from the buffers that are needed for setting  up
    // the acq. For the moment just read the first frame.
    unsigned int frame_id = 0;
    std::vector<uint32_t> freq_ids;
    //std::vector<timespec> start_times;

    wait_for_full_frame(buffer, unique_name.c_str(), frame_id);

    auto frame = visFrameView(buffer, frame_id);
    freq_ids.push_back(frame.freq_id());
    //start_times.push_back(std::get<1>(frame.time()));

    // Use the per buffer info to setup the acqusition properties
    setup_freq(freq_ids);

    // Create the visFileBundle. This will not create any files until addSample
    // is called
    // TODO: connect up notes
    std::string notes = "";
    file_bundle = std::unique_ptr<visFileBundle>(
         new visFileBundle(
             root_path, chunk_id, instrument_name, notes, freqs, inputs
         )
    );
}


void visWriter::setup_freq(const std::vector<uint32_t>& freq_ids) {
    // TODO: this function needs to do three things: set the frequency input map
    // (freqs), set the frequency ordering (freq_map) and set the chunk ID
    // (chunk_id).

    if(node_mode) {
        // Output all the frequencies that we have found
        std::string s;
        for(auto id : freq_ids) {
            char t[32];
            snprintf(t, 32, "%i [%.2f MHz] ", id, freq_from_bin(id));
            s += t;
        }
        INFO("Frequency bins found: %s", s.c_str());

        // TODO: this uses the hacky way of deriving the chunk ID
        unsigned int node_id = freq_ids[0] % 256;

        // Set the list of frequency ids that this node will deal with
        for(unsigned int i = 0; i < 4; i++) {
            freq_id_list.push_back(256 * i + node_id);
        }
    }

    // Set the chunk_id
    // TODO: eventually this should be set properly, but at the moment as we are
    // not merging into a single dir, chunk_id=0 should always be fine
    chunk_id = 0;

    // Sort the streams into bin order, this will give the order in which they
    // are written out
    unsigned int fpos = 0;
    for(auto id : freq_id_list) {
        freq_map[id] = fpos;
        freqs.push_back({freq_from_bin(id), (400.0 / 1024)});
        fpos++;
    }
}
