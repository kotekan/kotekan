#include "visProcess.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"
#include <time.h>
#include <iomanip>
#include <iostream>


visTransform::visTransform(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get_string_array(unique_name, "in_bufs");

    // Fetch the input buffers, register them, and store them in our buffer vector
    for(auto name : input_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_consumer(buf, unique_name.c_str());
        in_bufs.push_back({buf, 0});
    }

    // Setup the output vector
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

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
        for(auto& buffer_pair : in_bufs) {
            std::tie(buf, frame_id) = buffer_pair;

            INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);

            // Wait for the buffer to be filled with data
            if((frame = wait_for_full_frame(buf, unique_name.c_str(),
                                            frame_id)) == nullptr) {
                break;
            }

            // Wait for the buffer to be filled with data
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);

            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            // Transfer over the metadata
            output_frame.fill_chime_metadata((const chimeMetadata *)buf->metadata[frame_id]->metadata);

            // Copy the visibility data into a proper triangle and write into
            // the file
            copy_vis_triangle((int32_t *)frame, input_remap, block_size,
                              num_elements, output_frame.vis);

            // Mark the buffers and move on
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
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
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

void visDebug::apply_config(uint64_t fpga_seq) {

}

void visDebug::main_thread() {

    unsigned int frame_id = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        auto frame = visFrameView(in_buf, frame_id);
        INFO("%s", frame.summary().c_str());

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % in_buf->num_frames;

    }

}



visAccumulate::visAccumulate(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&visAccumulate::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    float int_time = config.get_float_default(unique_name, "integration_time", -1.0);

    // If the integration time was set then calculate the number of GPU frames
    // we need to integrate for.
    if(int_time >= 0.0) {
        // TODO: don't hard code the sample time length
        float frame_length = _samples_per_data_set * 2.56e-6;

        // Calculate nearest *even* number of frames
        _num_gpu_frames = 2 * ((int)(int_time / frame_length) / 2);

        INFO("Integrating for %i gpu frames (~%.1f s)", _num_gpu_frames, int_time);
    } else {
        _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
        INFO("Integrating for %i gpu frames.", _num_gpu_frames);
    }

}

visAccumulate::~visAccumulate() {
}

void visAccumulate::apply_config(uint64_t fpga_seq) {
}

void visAccumulate::main_thread() {
/*
    int in_frame_id = 0;
    int out_frame_id = 0;
    int64_t frame_id = 0;
    int32_t * input;
    int32_t * output;
    uint64_t seq_num;

    uint8_t * in_frame;

    while (!stop_thread) {

        if(in_frame = wait_for_full_frame(in_buf, unique_name.c_str(),
                                          in_frame_id) == nullptr) {
            break;
        }

        input = (int32_t *)in_frame;

        seq_num = get_fpga_seq_num(in_buf, in_frame_id);

        // We've started a new frame, initialisation
        if (frame_id % _num_gpu_frames == 0) {
            uint8_t * out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
            if (out_frame == NULL) break;
            output = (int32_t *)out_frame;

            allocate_new_metadata_object(out_buf, out_frame_id);

            // Copy over metadata

            // Copy over initial data

        } else {
            // Add up the number of lost samples from each input frame.
            uint64_t lost_samples = get_lost_timesamples(in_buf, in_frame_id);
            atomic_add_lost_timesamples(out_buf, out_frame_id, lost_samples);

            for (int i = 0; i < in_buf->frame_size/sizeof(int32_t); ++i) {
                output[i] += input[i];
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        frame_id++;

        if (frame_id % _num_gpu_frames == 0) {
            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        }
    }
    */
}
