#include "visProcess.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "fmt.hpp"
#include "datasetManager.hpp"

#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

REGISTER_KOTEKAN_PROCESS(visTransform);
REGISTER_KOTEKAN_PROCESS(visDebug);
REGISTER_KOTEKAN_PROCESS(visAccumulate);
REGISTER_KOTEKAN_PROCESS(visMerge);
REGISTER_KOTEKAN_PROCESS(visTestPattern);
REGISTER_KOTEKAN_PROCESS(registerInitialDatasetState);

visTransform::visTransform(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get<size_t>(unique_name, "num_elements");
    block_size = config.get<size_t>(unique_name, "block_size");
    num_eigenvectors =  config.get<size_t>(unique_name, "num_ev");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "in_bufs");

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
            std::fill(output_frame.flags.begin(), output_frame.flags.end(), 1.0);
            std::fill(output_frame.evec.begin(), output_frame.evec.end(), 0.0);
            std::fill(output_frame.eval.begin(), output_frame.eval.end(), 0.0);
            output_frame.erms = 0;
            std::fill(output_frame.gain.begin(), output_frame.gain.end(), 1.0);

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

    uint64_t num_frames = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        if ((num_frames % 1000) == 0)
            INFO("Got frame number %lli", num_frames);
        auto frame = visFrameView(in_buf, frame_id);
        DEBUG("%s", frame.summary().c_str());

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
        num_frames++;
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
    num_elements = config.get<size_t>(unique_name, "num_elements");
    block_size = config.get<size_t>(unique_name, "block_size");
    num_eigenvectors =  config.get<size_t>(unique_name, "num_ev");
    samples_per_data_set = config.get<size_t>(unique_name, "samples_per_data_set");
    minimum_fraction = config.get_default<double>(unique_name, "minimum_fraction", 0.01);

    // Get the indices for reordering
    input_remap = std::get<0>(parse_reorder_default(config, unique_name));

    float int_time = config.get_default<float>(unique_name, "integration_time", -1.0);

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
        num_gpu_frames = config.get<size_t>(unique_name, "num_gpu_frames");
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

    uint32_t last_frame_count = 0;
    uint32_t frames_in_this_cycle = 0;
    uint32_t total_samples = 0;

    size_t nb = num_elements / block_size;
    size_t nprod_gpu = nb * (nb + 1) * block_size * block_size / 2;

    // Temporary arrays for storing intermediates
    int32_t* vis_even = new int32_t[2 * nprod_gpu];
    cfloat* vis1 = new cfloat[nprod_gpu];
    float* vis2 = new float[nprod_gpu];

    // Have we initialised a frame for writing yet
    bool init = false;

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(),
                                                in_frame_id);
        if(in_frame == nullptr) break;

        int32_t* input = (int32_t *)in_frame;
        uint frame_count = get_fpga_seq_num(in_buf, in_frame_id) / samples_per_data_set;

        // If we have wrapped around we need to write out any frames that have
        // been filled in previous iterations. In here we need to reorder the
        // accumulates and do any final manipulations. `last_frame_count` is
        // initially set to UINT_MAX to ensure this doesn't happen immediately.
        bool wrapped = (last_frame_count / num_gpu_frames) < (frame_count / num_gpu_frames);
        if (init && wrapped) {

            auto output_frame = visFrameView(out_buf, out_frame_id);

            // Set the actual amount of time we accumulated for
            output_frame.fpga_seq_total = total_samples;
            DEBUG("Total samples accumulate %i", total_samples);

            // Reset the cycle
            init = false;
            frames_in_this_cycle = 0;
            total_samples = 0;

            // Check if we have exceeded the minimum number of required samples,
            // if not immediately skip without releasing the frame
            size_t min_length = minimum_fraction * output_frame.fpga_seq_length;
            if (output_frame.fpga_seq_total < min_length) {
                DEBUG("Too few samples in frame %i (minimum %i)",
                      output_frame.fpga_seq_total, min_length);
                continue;
            }

            // Unpack the main visibilities
            float w1 = 1.0 / output_frame.fpga_seq_total;

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
                    //std::cout << t << std::endl;
                    output_frame.weight[pi] = 1.0 / (w1 * w1 * t);
                }
            );

            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
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

            // Fill other datasets with reasonable values
            std::fill(output_frame.flags.begin(), output_frame.flags.end(), 1.0);
            std::fill(output_frame.evec.begin(), output_frame.evec.end(), 0.0);
            std::fill(output_frame.eval.begin(), output_frame.eval.end(), 0.0);
            output_frame.erms = 0;
            std::fill(output_frame.gain.begin(), output_frame.gain.end(), 1.0);

            // Zero out accumulation arrays
            std::fill(vis1, vis1 + nprod_gpu, 0);
            std::fill(vis2, vis2 + nprod_gpu, 0);

            init = true;
        }

        // Perform primary accumulation
        for(size_t i = 0; i < nprod_gpu; i++) {
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
        // TODO: we might need to account for packet loss in here too, but it
        // would require some awkward rescalings
        else {
            for(size_t i = 0; i < nprod_gpu; i++) {
                // NOTE: avoid using the slow std::complex routines in here
                float di = input[2 * i    ] - vis_even[2 * i    ];
                float dr = input[2 * i + 1] - vis_even[2 * i + 1];
                vis2[i] += (dr * dr + di * di);
            }
        }

        // Accumulate the total number of samples, accounting for lost ones
        total_samples += samples_per_data_set - get_lost_timesamples(in_buf, in_frame_id);

        // TODO: gating should go in here. Gates much be created such that the
        // squared sum of the weights is equal to 1.

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
        config.get<std::vector<std::string>>(unique_name, "in_bufs");

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


visTestPattern::visTestPattern(Config& config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTestPattern::main_thread, this)) {

    // Setup the buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // get config
    mode = config.get<std::string>(unique_name, "mode");

    INFO("visCheckTestPattern: mode = %s", mode.c_str());
    if (mode == "test_pattern_simple") {
        exp_val = config.get_default<cfloat>(unique_name,
                                             "default_val", {1.,0});
    } else if (mode == "test_pattern_freq") {
        num_freq = config.get<size_t>(unique_name,"num_freq");

        cfloat default_val = config.get_default<cfloat>(unique_name,
                                                 "default_val", {128., 0.});
        std::vector<uint32_t> bins = config.get<std::vector<uint32_t>>(
                       unique_name, "frequencies");
        std::vector<cfloat> bin_values = config.get<std::vector<cfloat>>(
                       unique_name, "freq_values");
        if (bins.size() != bin_values.size()) {
            throw std::invalid_argument("fakeVis: lengths of frequencies ("
                                        + std::to_string(bins.size())
                                        + ") and freq_value ("
                                        + std::to_string(bin_values.size())
                                        + ") arrays have to be equal.");
        }
        if (bins.size() > num_freq) {
            throw std::invalid_argument(
                        "fakeVis: length of frequencies array ("
                        + std::to_string(bins.size()) + ") can not be larger " \
                        "than num_freq (" + std::to_string(num_freq)
                        + ").");
        }

        exp_val_freq = std::vector<cfloat>(num_freq);
        for (size_t i = 0; i < num_freq; i++) {
            size_t j;
            for (j = 0; j < bins.size(); j++) {
                if (bins.at(j) == i)
                    break;
            }
            if (j == bins.size())
                exp_val_freq[i] = default_val;
            else
                exp_val_freq[i] = bin_values.at(j);
        }
    } else
        throw std::invalid_argument("visCheckTestpattern: unknown mode: " +
                                    mode);

    tolerance = config.get_default<float>(unique_name, "tolerance", 1e-6);
    report_freq = config.get_default<uint64_t>(unique_name, "report_freq", 1000);

    outfile_name = config.get<std::string>(unique_name, "out_file");

    if (tolerance < 0)
        throw std::invalid_argument("visCheckTestPattern: tolerance has to be" \
               " positive (is " + std::to_string(tolerance) + ").");

    outfile.open (outfile_name);
    if (!outfile.is_open()) {
        throw std::ios_base::failure("visCheckTestPattern: Failed to open " \
                                     "out file " + outfile_name);
    }
    outfile << "fpga_count,time,freq_id,num_bad,avg_err,min_err,max_err"
        << std::endl;
}

void visTestPattern::apply_config(uint64_t fpga_seq) {

}

void visTestPattern::main_thread() {

    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    // number of bad elements in frame and totally
    size_t num_bad, num_bad_tot = 0;

    // average error of the bad values in frame and totally
    float avg_err, avg_err_tot = 0;

    // greatest errors in frame and totally
    float min_err, max_err;
    float min_err_tot = 0;
    float max_err_tot = 0;

    // timestamp of frame
    uint64_t fpga_count;
    timespec time;

    // frequency ID of frame
    uint32_t freq_id;

    uint64_t i_frame = 0;

    // Comparisons will be against tolerance^2
    float t2 = tolerance * tolerance;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
         auto frame = visFrameView(in_buf, frame_id);
         //INFO("%s", frame.summary().c_str());

        num_bad = 0;
        avg_err = 0.0;
        min_err = 0.0;
        max_err = 0.0;

        cfloat expected;

        if (mode == "test_pattern_simple")
            expected = exp_val;
        else if (mode == "test_pattern_freq") {
            expected = exp_val_freq.at(frame.freq_id);
        }

	    // Iterate over covariance matrix
	    for (size_t i = 0; i < frame.num_prod; i++) {

            // Calculate the error^2 and compared this to the tolerance as it's
            // much faster than taking the square root where we don't need to.
            float r2 = fast_norm(frame.vis[i] - expected);

            // check for bad values
            if (r2 > t2) {
                num_bad++;

                // Calculate the error here, this square root is then
                // evalulated only when there is bad data.
                float error = sqrt(r2);
                avg_err += error;

                if (error > max_err)
                    max_err = error;
                if (error < min_err || min_err == 0.0)
                    min_err = error;
            }
        }


        if (num_bad) {
            avg_err /= (float)num_bad;
            time = std::get<1>(frame.time);
            fpga_count = std::get<0>(frame.time);
            freq_id = frame.freq_id;

            // write frame report to outfile
            outfile << fpga_count << ",";
            outfile << time.tv_sec << "." << time.tv_nsec << ",";
            outfile << freq_id << ",";
            outfile << num_bad << ",";
            outfile << avg_err << ",";
            outfile << min_err << ",";
            outfile << max_err << std::endl;

            // report errors in this frame
            DEBUG("%d bad elements", num_bad);
            DEBUG("mean error: %f", avg_err);
            DEBUG("min error: %f", min_err);
            DEBUG("max error: %f", max_err);
            DEBUG("time: %d, %lld.%d", fpga_count, (long long)time.tv_sec,
                    time.tv_nsec);
            DEBUG("freq id: %d", freq_id);
            DEBUG("expected: (%f,%f)", expected.real(), expected.imag());

            // gather data for report after many frames
            num_bad_tot += num_bad;
            avg_err_tot += avg_err * (float)num_bad;
            if (min_err < min_err_tot || min_err_tot == 0)
                min_err_tot = min_err;
            if (max_err > max_err_tot)
                max_err_tot = max_err;


            // pass this bad frame to the output buffer:

            // Wait for an empty frame in the output buffer
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }

            // Transfer metadata
            pass_metadata(in_buf, frame_id, out_buf, output_frame_id);

            // Copy the frame data here:
            std::memcpy(out_buf->frames[output_frame_id],
                        in_buf->frames[frame_id],
                        in_buf->frame_size);


            mark_frame_full(out_buf, unique_name.c_str(),
                                output_frame_id);

            // Advance output frame id
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // print report some times
        if (++i_frame == report_freq) {
            i_frame = 0;

            avg_err_tot /= (float)num_bad_tot;
            if (num_bad_tot == 0)
                avg_err_tot = 0;

            INFO("Summary from last %d frames: num bad values: %d, mean " \
                    "error: %f, min error: %f, max error: %f", report_freq,
                    num_bad_tot, avg_err_tot, min_err_tot, max_err_tot);
            avg_err_tot = 0.0;
            num_bad_tot = 0;
            min_err_tot = 0;
            max_err_tot = 0;
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance input frame id
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}


registerInitialDatasetState::registerInitialDatasetState(Config& config,
    const string& unique_name, bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&registerInitialDatasetState::main_thread, this))
{
    // Fetch any needed config.
    apply_config(0);
    // Setup the buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void registerInitialDatasetState::apply_config(uint64_t fpga_seq)
{
    std::vector<uint32_t> freq_ids;

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    }
    else {
        freq_ids.resize(1024);
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(_freqs),
                   [] (uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
                   });

    // Extract the input specification from the config
    _inputs = std::get<1>(parse_reorder_default(config, unique_name));

    size_t num_elements = _inputs.size();

    // Create the product specification
    _prods.reserve(num_elements);
    for(uint16_t i = 0; i < num_elements; i++) {
        for(uint16_t j = i; j < num_elements; j++) {
            _prods.push_back({i, j});
        }
    }

}


void registerInitialDatasetState::main_thread() {

    // In case we have multiple processes all registering different datasets we
    // need to make sure that they all get distinct roots, use this for
    // co-ordination.
    static std::atomic<int> root_dataset_id(-1);

    uint32_t frame_id_in = 0;
    uint32_t frame_id_out = 0;

    auto& dm = datasetManager::instance();

    // Construct a nested description of the initial state
    state_uptr freq_state = std::make_unique<freqState>(_freqs);
    state_uptr input_state = std::make_unique<inputState>(
        _inputs, std::move(freq_state));
    state_uptr prod_state = std::make_unique<prodState>(
        _prods, std::move(input_state));

    // Register the initial state with the manager
    auto s = dm.add_state(std::move(prod_state));
    state_id initial_state = s.first;

    // Get the new dataset ID, this uses the current root ID and then decrements
    // it for any other instance of this process.
    dset_id output_dataset = dm.add_dataset(initial_state, root_dataset_id--);

    while (!stop_thread) {
        // Wait for an input frame
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id_in) == nullptr) {
            break;
        }
        //wait for an empty output frame
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                frame_id_out) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto frame_out = visFrameView::copy_frame(in_buf, frame_id_in,
                                                  out_buf, frame_id_out);

        // Assign the frame the correct dataset ID
        frame_out.dataset_id = output_dataset;

        // Mark output frame full and input frame empty
        mark_frame_full(out_buf, unique_name.c_str(), frame_id_out);
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id_in);
        // Move forward one frame
        frame_id_out = (frame_id_out + 1) % out_buf->num_frames;
        frame_id_in = (frame_id_in + 1) % in_buf->num_frames;
    }
}
