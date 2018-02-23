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

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    // Get the indices for reordering
    input_remap = std::get<0>(parse_reorder_default(config, unique_name));

    float int_time = config.get_float_default(unique_name, "integration_time", -1.0);

    // If the integration time was set then calculate the number of GPU frames
    // we need to integrate for.
    if(int_time >= 0.0) {
        // TODO: don't hard code the sample time length
        float frame_length = samples_per_data_set * 2.56e-6;

        // Calculate nearest *even* number of frames
        num_gpu_frames = 2 * ((int)(int_time / frame_length) / 2);

        INFO("Integrating for %i gpu frames (=%.2f s  ~%.2f s)",
             num_gpu_frames, frame_length * num_gpu_frames, int_time);
    } else {
        num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
        INFO("Integrating for %i gpu frames.", num_gpu_frames);
    }

}

visAccumulate::~visAccumulate() {
}

void visAccumulate::apply_config(uint64_t fpga_seq) {
}

void visAccumulate::main_thread() {

    int in_frame_id = 0;
    int out_frame_id = 0;
    int64_t frame_id = 0;
    int32_t * input;
    int32_t * output;
    uint64_t seq_num;

    uint8_t * in_frame;

    size_t nprod = num_elements * (num_elements + 1) / 2;

    std::vector<cfloat> vis(nprod);
    std::vector<cfloat> vis_even(nprod);

    while (!stop_thread) {

        if((in_frame = wait_for_full_frame(in_buf, unique_name.c_str(),
                                           in_frame_id)) == nullptr) {
            break;
        }

        input = (int32_t *)in_frame;

        seq_num = get_fpga_seq_num(in_buf, in_frame_id);

        // We've started a new frame, start the initialisation
        if (frame_id % num_gpu_frames == 0) {

            if (wait_for_empty_frame(out_buf, unique_name.c_str(),
                                     out_frame_id) == nullptr) {
                break;
            }

            allocate_new_metadata_object(out_buf, out_frame_id);
            auto output_frame = visFrameView(out_buf, out_frame_id, num_elements, num_eigenvectors);

            // Transfer over the metadata
            output_frame.fill_chime_metadata((const chimeMetadata *)in_buf->metadata[in_frame_id]->metadata);
            
            // Zero out existing data
            std::fill(output_frame.vis.begin(), output_frame.vis.end(), 0.0);
            std::fill(output_frame.weight.begin(), output_frame.weight.end(), 0.0);
            std::fill(output_frame.eigenvectors.begin(), output_frame.eigenvectors.end(), 0.0);
            std::fill(output_frame.eigenvalues.begin(), output_frame.eigenvalues.end(), 0.0);
        }

        // Copy out the visibilities from the blocked representation and reorder
        // them. This is done for simplicity, now we can just use them how we
        // want without any remapping
        copy_vis_triangle((const int32_t *)in_frame, input_remap, block_size, num_elements, vis);

        auto output_frame = visFrameView(out_buf, out_frame_id);

        // First, divide through by the number of accumulations done in the GPUs
        // themselves. Then accumulate the weighted vis into the main vis buffer
        // to progressively calculate the average
        for(size_t i = 0; i < nprod; i++) {
            vis[i] /= (float)(samples_per_data_set);

            output_frame.vis[i] += vis[i] / (float)(num_gpu_frames);
        }

        // We are calculating the weights by differencing even and odd samples.
        // Every even sample we save the set of visibilities...
        if(frame_id % 2 == 0) {
            std::swap(vis, vis_even);  // Swap the vis into a separate vector to save it
        }
        // ... every odd sample we accumulate the squared differences into the weight dataset
        // NOTE: this incrementally calculates the variance, but eventually
        // output_frame.weight will hold the *inverse* variance
        else {
            for(size_t i = 0; i < nprod; i++) {
                auto t = abs(vis[i] - vis_even[i]) / (float)(num_gpu_frames);
                output_frame.weight[i] += pow(t, 2);
            }
        }
        
        // TODO: do something with the lost packet counts

        // TODO: gating should go in here. Gates much be created such that the
        // squared sum of the weights is equal to 1.

        // Move the input buffer on one step
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        
        // Increment the frame counter which counts the number of GPU frames we've seen
        frame_id++;

        // Once we've integrated over the required number of frames, then do any
        // final cleanups and release the data...
        if (frame_id % num_gpu_frames == 0) {

            // Invert everything in the weight dataset
            for(size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] = 1.0 / output_frame.weight[i];
            }
            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        }
    }
}
