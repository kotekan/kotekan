#include "visProcess.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "fmt.hpp"

#include <time.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <complex.h>

REGISTER_KOTEKAN_PROCESS(visTransform);
REGISTER_KOTEKAN_PROCESS(visDebug);
REGISTER_KOTEKAN_PROCESS(visAccumulate);
REGISTER_KOTEKAN_PROCESS(visMerge);


visTransform::visTransform(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_ev");

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
        // available buffers, wait for data to appear and transform into
        // visBuffer style data
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

            // Copy over the metadata
            output_frame.fill_chime_metadata((const chimeMetadata *)buf->metadata[frame_id]->metadata);

            // Copy the visibility data into a proper triangle and write into
            // the file
            copy_vis_triangle((int32_t *)frame, input_remap, block_size,
                              num_elements, output_frame.vis);

            // Fill other datasets with reasonable values
            std::fill(output_frame.weight.begin(), output_frame.weight.end(), 1.0);
            std::fill(output_frame.evec.begin(), output_frame.evec.end(), 0.0);
            std::fill(output_frame.eval.begin(), output_frame.eval.end(), 0.0);
            output_frame.erms = 0;

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

        // Update the frame count for prometheus
        fd_pair key {frame.freq_id, frame.dataset_id};
        frame_counts[key]++;  // Relies on the fact that insertion zero intialises
        std::string labels = fmt::format("freq_id=\"{}\",dataset_id=\"{}\"",
                                         frame.freq_id, frame.dataset_id);
        prometheusMetrics::instance().add_process_metric(
            "kotekan_visdebug_frame_total", unique_name, frame_counts[key], labels
        );

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

    // NOTE: requires null check
    if (config.exists(unique_name, "pulsar_buf")) {
        pulsar_buf = get_buffer("pulsar_buf");
        register_consumer(pulsar_buf, unique_name.c_str());
    } else {
        pulsar_buf = nullptr;
    }

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    block_size = config.get_int(unique_name, "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_ev");
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
    int pulsar_frame_id = 0;
    int out_frame_id = 0;

    uint last_frame_count = 0;
    uint frames_in_this_cycle = 0;

    size_t nb = num_elements / block_size;
    // nprod_gpu is the size of the correlation triangle
    size_t nprod_gpu = nb * (nb + 1) * block_size * block_size / 2;

    // Temporary arrays for storing intermediates
    int32_t* vis_even = new int32_t[2 * nprod_gpu];
    cfloat* vis1 = new cfloat[nprod_gpu];
    float* vis2 = new float[nprod_gpu];

    // Have we initialised a frame for writing yet
    bool init = false;

    // Temporary holding thing for pulsar vis
    int32_t* vis_pulsar_even = new int32_t[2 * nprod_gpu];
    cfloat* vis_pulsar = new cfloat[nprod_gpu];
    float* vis_pulsar2 = new float[nprod_gpu];

    while (!stop_thread) {

        // fetch a new frame and get its sequence id
        uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(),
                                                in_frame_id);
        if(in_frame == nullptr) break;

        int32_t* input = (int32_t *)in_frame;
        uint frame_count = get_fpga_seq_num(in_buf, in_frame_id) / samples_per_data_set;
        INFO("acc: frame_count = %d", frame_count);

        // If we have wrapped around we need to write out any frames that have
        // been filled in previous iterations. In here we need to reorder the
        // accumulates and do any final manipulations. `last_frame_count` is
        // initially set to UINT_MAX to ensure this doesn't happen immediately.
        bool wrapped = (last_frame_count / num_gpu_frames) < (frame_count / num_gpu_frames);
        if (init && wrapped) {
            auto output_frame = visFrameView(out_buf, out_frame_id);

            // Unpack the main visibilities
            float w1 = 1.0 / (num_gpu_frames * samples_per_data_set);

            if (pulsar_buf != nullptr) {
                uint8_t* pulsar_frame;
                // Fetch a new pulsar frame
                pulsar_frame = wait_for_full_frame(pulsar_buf, unique_name.c_str(),
                                                   pulsar_frame_id);

                // Zero out the accumulation arrays
                std::fill(vis_pulsar, vis_pulsar + nprod_gpu, 0);
                std::fill(vis_pulsar2, vis_pulsar2 + nprod_gpu, 0);

                int pframe_count = 0;
                uint64_t pulsar_samples = 0;

                // TODO: Find a better way to avoid the first pulsar frame being null
                if (pulsar_frame != nullptr) {
                    int32_t* pulsar = (int32_t *)pulsar_frame;

                    while (pulsar_frame != nullptr && pframe_count < num_gpu_frames/2) {
                        // Storing the sample num to compute weights
                        uint64_t lost_samples = get_lost_timesamples(pulsar_buf, pulsar_frame_id);
                        pulsar_samples += samples_per_data_set - lost_samples;

                        // Accumulate pulsar vis like normal vis
                        for(size_t i = 0; i < nprod_gpu; i++) {
                            cfloat t = {(float)pulsar[2*i+1], (float)pulsar[2*i]};
                            vis_pulsar[i] += t;
                        }

                        if (get_is_last_pulsar_frame(pulsar_buf, pulsar_frame_id)) break;

                        // Fetch the next pulsar frame
                        mark_frame_empty(pulsar_buf, unique_name.c_str(), pulsar_frame_id);
                        pulsar_frame_id = (pulsar_frame_id + 1) % pulsar_buf->num_frames;
                        pulsar_frame = wait_for_full_frame(pulsar_buf, unique_name.c_str(),
                                                           pulsar_frame_id);
                        pframe_count++;
                    }
                }

                float N = num_gpu_frames * samples_per_data_set;
                float P = (float) pulsar_samples;
                double alpha = sqrt(P / (N + P));

                map_vis_triangle(input_remap, block_size, num_elements,
                    [&](int32_t pi, int32_t bi, bool conj) {
                        cfloat t1 = !conj ? vis1[bi] : std::conj(vis1[bi]);
                        cfloat tp = !conj ? vis_pulsar[bi] : std::conj(vis_pulsar[bi]);
                        output_frame.vis[pi] = (float) alpha * ((float)(1./P) * tp - (float)(1./N) * t1);
                    }
                );

                // Unpack and invert the weights
                map_vis_triangle(input_remap, block_size, num_elements,
                    [&](int32_t pi, int32_t bi, bool conj) {
                        float t = vis2[bi];
                        output_frame.weight[pi] = 1.0 / (w1 * w1 * t);
                    }
                );
            } else {
                INFO("\n\n\n raw visibilities");
                for (int k = 0; k < 5; k++) {
                     INFO("vis1[%d]: ", k);
                     std::cout << vis1[k] << std::endl;
                }

                map_vis_triangle(input_remap, block_size, num_elements,
                    [&](int32_t pi, int32_t bi, bool conj) {
                        cfloat t = !conj ? vis1[bi] : std::conj(vis1[bi]);
                        output_frame.vis[pi] = w1 * t;
                    }
                );

                // Unpack and invert the weights
                map_vis_triangle(input_remap, block_size, num_elements,
                    [&](int32_t pi, int32_t bi, bool conj) {
                        float t = vis2[bi];
                        output_frame.weight[pi] = 1.0 / (w1 * w1 * t);
                    }
                );
            }

            // Set the actual amount of time we accumulated for 
            output_frame.fpga_seq_total = frames_in_this_cycle * samples_per_data_set;

            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
            init = false;
            frames_in_this_cycle = 0;
        }

        // We've started accumulating a new frame. Initialise the output and
        // copy over any metadata.
        if (frame_count % num_gpu_frames == 0) {

            if (wait_for_empty_frame(out_buf, unique_name.c_str(),
                                     out_frame_id) == nullptr) {
                break;
            }

            allocate_new_metadata_object(out_buf, out_frame_id);
            auto output_frame = visFrameView(out_buf, out_frame_id, num_elements, num_eigenvectors);

            // Copy over the metadata
            output_frame.fill_chime_metadata((const chimeMetadata *)in_buf->metadata[in_frame_id]->metadata);

            // Set the length of time this frame will cover
            output_frame.fpga_seq_length = samples_per_data_set * num_gpu_frames;
            
            // Zero out existing data
            std::fill(output_frame.evec.begin(), output_frame.evec.end(), 0.0);
            std::fill(output_frame.eval.begin(), output_frame.eval.end(), 0.0);
            output_frame.erms = 0;

            // Zero out accumulation arrays
            std::fill(vis1, vis1 + nprod_gpu, 0);
            std::fill(vis2, vis2 + nprod_gpu, 0);

            init = true;
        }

        // Perform primary accumulation
        for(int i = 0; i < nprod_gpu; i++) {
            cfloat t = {(float)input[2*i+1], (float)input[2*i]};
            vis1[i] += t;
        }

        // We are calculating the weights by differencing even and odd samples.
        // Every even sample we save the set of visibilities...
        if(frame_count % 2 == 0) {
            std::memcpy(vis_even, input, 8 * nprod_gpu);
        }
        // ... every odd sample we accumulate the squared differences into the weight dataset
        // NOTE: this incrementally calculates the variance, but eventually
        // output_frame.weight will hold the *inverse* variance
        else {
            for(size_t i = 0; i < nprod_gpu; i++) {
                // NOTE: avoid using the slow std::complex routines in here
                float di = input[2 * i    ] - vis_even[2 * i    ];
                float dr = input[2 * i + 1] - vis_even[2 * i + 1];
                vis2[i] += (dr * dr + di * di);
            }
        }

        // TODO: do something with the lost packet counts

        // TODO: gating should go in here. Gates much be created such that the
        // squared sum of the weights is equal to 1.
        get_fpga_seq_num(in_buf, in_frame_id);
        // Move the input buffer on one step
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        last_frame_count = frame_count;
        frames_in_this_cycle++;
    }

    // Cleanup 
    delete[] vis_even;
    delete[] vis1;
    delete[] vis2;
    delete[] vis_pulsar_even;
    delete[] vis_pulsar;
    delete[] vis_pulsar2;
}


visMerge::visMerge(Config& config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visMerge::main_thread, this)) {
                    
    // Setup the output vector
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get_string_array(unique_name, "in_bufs");

    // Fetch the input buffers, register them, and store them in our buffer vector
    for(auto name : input_buffer_names) {
        auto buf = buffer_container.get_buffer(name);

        if(buf->frame_size > out_buf->frame_size) {
            throw std::invalid_argument("Input buffer [" + name + 
                                        "] larger that output buffer size.");
        }

        register_consumer(buf, unique_name.c_str());
        in_bufs.push_back({buf, 0});
    }

}

void visMerge::apply_config(uint64_t fpga_seq) {

}

void visMerge::main_thread() {

    uint8_t * frame = nullptr;
    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    while (!stop_thread) {

        // This is where all the main set of work happens. Iterate over the
        // available buffers, wait for data to appear and then attempt to write
        // the data into a file
        for(auto& buffer_pair : in_bufs) {
            std::tie(buf, frame_id) = buffer_pair;

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

            DEBUG("Merging buffer %s[%i] into %s[%i]",
                  buf->buffer_name, frame_id,
                  out_buf->buffer_name, output_frame_id);
        
            // Transfer metadata
            pass_metadata(buf, frame_id, out_buf, output_frame_id);

            // Copy the frame data here:
            std::memcpy(out_buf->frames[output_frame_id],
                        buf->frames[frame_id], 
                        buf->frame_size);

            // Mark the buffers and move on
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

    }

}
