#include "visAccumulate.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "chimeMetadata.h"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "fmt.hpp"
#include "datasetManager.hpp"
#include "configUpdater.hpp"

#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <mutex>


using namespace std::placeholders;

REGISTER_KOTEKAN_PROCESS(visAccumulate);


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
    num_freq_in_frame = config.get_default<size_t>(unique_name, "num_freq_in_frame", 1);
    block_size = config.get<size_t>(unique_name, "block_size");
    num_eigenvectors =  config.get<size_t>(unique_name, "num_ev");
    samples_per_data_set = config.get<size_t>(unique_name, "samples_per_data_set");

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

    // Register gating update callbacks
    std::map<std::string, std::function<bool(nlohmann::json &)>> callbacks;

    nlohmann::json gating_modes = config.get_value(unique_name, "gating");

    for (nlohmann::json::iterator it = gating_modes.begin(); it < gating_modes.end(); ++it) {
        // Get gating mode and initialize the correct spec
        std::string mode = it.value().at("mode").get<std::string>();
        std::string key = it.key();
        if (mode == "pulsar") {
            gating_specs[key] = new pulsarSpec(samples_per_data_set * 2.56e-6);
        } else {
            throw std::runtime_error("Could not find gating mode " + mode);
        }
        callbacks[key] = std::bind(&pulsarSpec::update_spec, gating_specs[key], _1);
    }

    configUpdater::instance().subscribe(this, callbacks);

}

visAccumulate::~visAccumulate() {}

void visAccumulate::apply_config(uint64_t fpga_seq) {
}

void visAccumulate::main_thread() {

    int in_frame_id = 0;
    int out_frame_id = 0;

    uint32_t last_frame_count = 0;
    uint32_t frames_in_this_cycle = 0;
    uint32_t total_samples = 0;

    size_t nb = num_elements / block_size;
    size_t nprod_gpu = num_freq_in_frame * nb * (nb + 1) * block_size * block_size / 2;

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

            DEBUG("Total samples accumulate %i", total_samples);

            // Loop over the frequencies in the frame and unpack the accumulates
            // into the output frame...
            for(uint32_t freq_ind = 0; freq_ind < num_freq_in_frame; freq_ind++) {

                finalise_output(out_buf, out_frame_id, vis1, vis2,
                                freq_ind, total_samples);

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
                out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
            }

            init = false;
            frames_in_this_cycle = 0;
            total_samples = 0;
        }

        // We've started accumulating a new frame. Initialise the output and
        // copy over any metadata.
        if (frame_count % num_gpu_frames == 0) {

            // Iterate over the set of output frames we will be using and
            // initialise them...
            for(uint32_t freq_ind = 0; freq_ind < num_freq_in_frame; freq_ind++) {

                uint32_t frame_id = (out_frame_id + freq_ind) % out_buf->num_frames;

                if (wait_for_empty_frame(out_buf, unique_name.c_str(),
                                         frame_id) == nullptr) {
                    break;
                }

                initialise_output(out_buf, frame_id, in_frame_id, freq_ind);
            }

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


void visAccumulate::initialise_output(
    Buffer* out_buf, int out_frame_id, int in_frame_id, int freq_ind
)
{

    allocate_new_metadata_object(out_buf, out_frame_id);
    auto frame = visFrameView(out_buf, out_frame_id, num_elements,
                              num_eigenvectors);

    // Copy over the metadata
    // TODO: CHIME
    frame.fill_chime_metadata(
        (const chimeMetadata *)in_buf->metadata[in_frame_id]->metadata);

    // TODO: set frequency id in some sensible generic manner
    frame.freq_id += freq_ind;

    // Set the length of time this frame will cover
    frame.fpga_seq_length = samples_per_data_set * num_gpu_frames;

    // Fill other datasets with reasonable values
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.evec.begin(), frame.evec.end(), 0.0);
    std::fill(frame.eval.begin(), frame.eval.end(), 0.0);
    frame.erms = 0;
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}


void visAccumulate::finalise_output(Buffer* out_buf, int out_frame_id,
                                    cfloat* vis1, float* vis2, int freq_ind,
                                    uint32_t total_samples)
{
    // Unpack the main visibilities
    float w1 = 1.0 / total_samples;

    auto output_frame = visFrameView(out_buf, out_frame_id);

    // Copy the visibilities into place
    map_vis_triangle(input_remap, block_size, num_elements, freq_ind,
        [&](int32_t pi, int32_t bi, bool conj) {
            cfloat t = !conj ? vis1[bi] : std::conj(vis1[bi]);
            output_frame.vis[pi] = w1 * t;
        }
    );

    // Unpack and invert the weights
    map_vis_triangle(input_remap, block_size, num_elements, freq_ind,
        [&](int32_t pi, int32_t bi, bool conj) {
            float t = vis2[bi];
            output_frame.weight[pi] = 1.0 / (w1 * w1 * t);
        }
    );

    // Set the actual amount of time we accumulated for
    output_frame.fpga_seq_total = total_samples;
}


gateSpec::gateSpec(double width) : gpu_frame_width(width) {}


bool pulsarSpec::update_spec(nlohmann::json &json) {

    try {
        enabled = json.at("enabled").get<bool>();
    } catch (std::exception& e) {
        WARN("Failure reading 'enabled' from update: %s", e.what());
        return false;
    }

    if (!enabled)
        return true;

    std::vector<float> coeff;
    try {
        // Get gating specifications from config
        coeff = json.at("coeff").get<std::vector<float>>();
        dm = json.at("dm").get<float>();
        tmid = json.at("tmid").get<double>();
        phase_ref = json.at("phase_ref").get<double>();
        rot_freq = json.at("rot_freq").get<double>();
        pulse_width = json.at("pulse_width").get<float>();
    } catch (std::exception& e) {
        WARN("Failure reading pulsar parameters from update: %s", e.what());
        return false;
    }
    polyco = * new Polyco(tmid, dm, phase_ref, rot_freq, coeff);
}


std::function<float(timespec, float)> pulsarSpec::get_gating_func() {

    // capture the variables needed to calculate timing
    return [
        p = polyco, f0 = rot_freq, fw = gpu_frame_width,
        pw = pulse_width
    ](timespec t, float freq) mutable {
        // Calculate nearest pulse times of arrival
        double toa = p.next_toa(t, freq);
        double last_toa = toa - 1. / f0;

        // Weights are on/off for now
        if (toa < fw || last_toa + pw > 0) {
            return 1.;
        } else {
            return 0.;
        }
    };
}


bool gateInternalState::reset() {

    // Reset the internal counters
    weighted_samples = 0;
    // ... zero out the accumulation array

    // Acquire the lock so we don't get confused by any changes made via the
    // REST callback
    std::lock_guard<std::mutex> lock(gate_mtx);

    if (!spec.enabled) {
        weightfunc = nullptr;
        return false;
    }

    weightfunc = spec.weight_function();
    return true;
}