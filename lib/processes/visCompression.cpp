#include "visCompression.hpp"

#include "StageFactory.hpp"
#include "datasetManager.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include "fmt.hpp"
#include "gsl-lite.hpp"

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <future>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <pthread.h>
#include <regex>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::prometheusMetrics;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(baselineCompression);


baselineCompression::baselineCompression(Config& config, const string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&baselineCompression::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    frame_id_in(in_buf),
    frame_id_out(out_buf) {

    register_consumer(in_buf, unique_name.c_str());
    register_producer(out_buf, unique_name.c_str());

    // Fill out the map of stack types
    stack_type_defs["diagonal"] = stack_diagonal;
    stack_type_defs["chime_in_cyl"] = stack_chime_in_cyl;

    // Apply config.
    std::string stack_type = config.get<std::string>(unique_name, "stack_type");
    if (stack_type_defs.count(stack_type) == 0) {
        ERROR("unknown stack type %s", stack_type.c_str());
        return;
    }
    INFO("using stack type: %s", stack_type.c_str());
    calculate_stack = stack_type_defs.at(stack_type);

    if (config.exists(unique_name, "exclude_inputs")) {
        exclude_inputs = config.get<std::vector<uint32_t>>(unique_name, "exclude_inputs");
    }

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    if (num_threads == 0)
        throw std::invalid_argument("baselineCompression: "
                                    "num_threads has to be at least 1.");
    if (in_buf->num_frames % num_threads != 0 || out_buf->num_frames % num_threads != 0)
        throw std::invalid_argument("baselineCompression: both "
                                    "the size of the input and output buffer"
                                    "have to be multiples of num_threads.");
}

void baselineCompression::main_thread() {

    // Create the threads
    thread_handles.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i] = std::thread(&baselineCompression::compress_thread, this);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i].join();
    }
}

dset_id_t baselineCompression::change_dataset_state(dset_id_t input_ds_id) {
    auto& dm = datasetManager::instance();
    state_id_t stack_state_id;

    // Get input & prod states synchronoulsy
    auto input_state = std::async(&datasetManager::dataset_state<inputState>, &dm, input_ds_id);
    auto prod_state = std::async(&datasetManager::dataset_state<prodState>, &dm, input_ds_id);

    const inputState* input_state_ptr = input_state.get();
    prod_state_ptr = prod_state.get();
    if (input_state_ptr == nullptr || prod_state_ptr == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find "
              "freqState ancestor of dataset 0x%" PRIx64 ". Make sure there "
              "is a stage upstream in the config, that adds a freqState.\n"
              "Exiting...",
              input_ds_id);
        raise(SIGINT);
    }

    auto sspec = calculate_stack(input_state_ptr->get_inputs(), prod_state_ptr->get_prods());
    auto sstate = std::make_unique<stackState>(sspec.first, std::move(sspec.second));

    std::tie(stack_state_id, stack_state_ptr) = dm.add_state(std::move(sstate));

    return dm.add_dataset(input_ds_id, stack_state_id);
}

void baselineCompression::compress_thread() {

    int input_frame_id;
    int output_frame_id;

    dset_id_t input_dset_id;
    dset_id_t output_dset_id = 0;

    // Get the current values of the shared frame IDs.
    {
        std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
        output_frame_id = frame_id_out++;
        input_frame_id = frame_id_in++;
    }

    // Wait for the input buffer to be filled with data
    // in order to get dataset ID
    if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
        return;
    }
    auto input_frame = visFrameView(in_buf, input_frame_id);
    input_dset_id = input_frame.dataset_id;
    auto future_output_dset_id =
        std::async(&baselineCompression::change_dataset_state, this, input_dset_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        double start_time = current_time();

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // If the input dataset has changed construct a new stack spec for the
        // datasetManager
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;
            future_output_dset_id =
                std::async(&baselineCompression::change_dataset_state, this, input_dset_id);
        }

        // Are we waiting for a new dataset ID?
        if (future_output_dset_id.valid())
            output_dset_id = future_output_dset_id.get();

        const auto& stack_map = stack_state_ptr->get_rstack_map();
        const auto& prods = prod_state_ptr->get_prods();
        auto num_stack = stack_state_ptr->get_num_stack();

        std::vector<float> stack_norm(stack_state_ptr->get_num_stack(), 0.0);
        std::vector<float> stack_v2(stack_state_ptr->get_num_stack(), 0.0);

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);
        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame.num_elements,
                                         num_stack, input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {visField::vis, visField::weight});
        output_frame.dataset_id = output_dset_id;

        // Zero the output frame
        std::fill(std::begin(output_frame.vis), std::end(output_frame.vis), 0.0);
        std::fill(std::begin(output_frame.weight), std::end(output_frame.weight), 0.0);

        // Flag out the excluded inputs
        for (auto& input : exclude_inputs) {
            output_frame.flags[input] = 0.0;
        }

        auto in_vis = input_frame.vis.data();
        auto out_vis = output_frame.vis.data();
        auto in_weight = input_frame.weight.data();
        auto out_weight = output_frame.weight.data();
        auto flags = output_frame.flags.data();

        // Iterate over all the products and average together
        for (uint32_t prod_ind = 0; prod_ind < prods.size(); prod_ind++) {
            // TODO: if the weights are ever different from 0 or 1, we will
            // definitely need to rewrite this.

            // Alias the parts of the data we are going to stack
            cfloat vis = in_vis[prod_ind];
            float weight = in_weight[prod_ind];

            auto& p = prods[prod_ind];
            auto& s = stack_map[prod_ind];

            // If the weight is zero, completey skip this iteration
            if (weight == 0 || flags[p.input_a] == 0 || flags[p.input_b] == 0)
                continue;

            vis = s.conjugate ? conj(vis) : vis;

            // First summation of the visibilities (dividing by the total weight will be done later)
            out_vis[s.stack] += vis;

            // Accumulate the square for variance calculation
            stack_v2[s.stack] += fast_norm(vis);

            // Accumulate the weighted *variances*. Normalising and inversion
            // will be done later
            out_weight[s.stack] += (1.0 / weight);

            // Accumulate the weights so we can normalize correctly
            stack_norm[s.stack] += 1.0;
        }

        // Loop over the stacks and normalise (and invert the variances)
        float vart = 0.0;
        float normt = 0.0;
        for (uint32_t stack_ind = 0; stack_ind < num_stack; stack_ind++) {

            // Calculate the mean and accumulate weight and place in the frame
            float norm = stack_norm[stack_ind];

            // Invert norm if set, otherwise use zero to set data to zero.
            float inorm = (norm != 0.0) ? (1.0 / norm) : 0.0;

            output_frame.vis[stack_ind] *= inorm;
            output_frame.weight[stack_ind] = norm * norm / output_frame.weight[stack_ind];

            // Accumulate to calculate the variance of the residuals
            vart += stack_v2[stack_ind] - std::norm(output_frame.vis[stack_ind]) * norm;
            normt += norm;
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Get the current values of the shared frame IDs and increment them.
        {
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            output_frame_id = frame_id_out++;
            input_frame_id = frame_id_in++;
        }

        // Calculate residuals (return zero if no data for this freq)
        float residual = (normt != 0.0) ? (vart / normt) : 0.0;

        // Update prometheus metrics
        double elapsed = current_time() - start_time;
        std::string labels = fmt::format("freq_id=\"{}\",dataset_id=\"{}\",thread_id=\"{}\"",
                                         output_frame.freq_id, output_frame.dataset_id,
                                         std::this_thread::get_id());
        prometheusMetrics::instance().add_stage_metric("kotekan_baselinecompression_residuals",
                                                       unique_name, residual, labels);
        prometheusMetrics::instance().add_stage_metric("kotekan_baselinecompression_time_seconds",
                                                       unique_name, elapsed);

        DEBUG("Compression time %.4f", elapsed);
    }
}

// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_diagonal(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods) {
    uint32_t num_elements = inputs.size();
    std::vector<rstack_ctype> stack_def;

    for (auto& p : prods) {
        uint32_t stack_ind = abs(p.input_b - p.input_a);
        bool conjugate = p.input_a > p.input_b;

        stack_def.push_back({stack_ind, conjugate});
    }

    return {num_elements, stack_def};
}


chimeFeed chimeFeed::from_input(input_ctype input) {
    chimeFeed feed;

    if (input.chan_id >= 2048) {
        throw std::invalid_argument("Channel ID is not a valid CHIME feed.");
    }

    feed.cylinder = (input.chan_id / 512);
    feed.polarisation = ((input.chan_id / 256 + 1) % 2);
    feed.feed_location = input.chan_id % 256;

    return feed;
}


std::ostream& operator<<(std::ostream& os, const chimeFeed& f) {
    char cyl_name[4] = {'A', 'B', 'C', 'D'};
    char pol_name[2] = {'X', 'Y'};
    return os << fmt::format("{}{:03}{}", cyl_name[f.cylinder], f.feed_location,
                             pol_name[f.polarisation]);
}

using feed_diff = std::tuple<int8_t, int8_t, int8_t, int8_t, int16_t>;

// Calculate the baseline parameters and whether the product must be
// conjugated to get canonical ordering
std::pair<feed_diff, bool> calculate_chime_vis(const prod_ctype& p,
                                               const std::vector<input_ctype>& inputs) {

    chimeFeed fa = chimeFeed::from_input(inputs[p.input_a]);
    chimeFeed fb = chimeFeed::from_input(inputs[p.input_b]);

    bool is_wrong_cylorder = (fa.cylinder > fb.cylinder);
    bool is_same_cyl_wrong_feed_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location > fb.feed_location));
    bool is_same_feed_wrong_pol_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location == fb.feed_location)
         && (fa.polarisation > fb.polarisation));

    bool conjugate = false;

    // Check if we need to conjugate/transpose to get the correct order
    if (is_wrong_cylorder || is_same_cyl_wrong_feed_order || is_same_feed_wrong_pol_order) {

        chimeFeed t = fa;
        fa = fb;
        fb = t;
        conjugate = true;
    }

    return {std::make_tuple(fa.polarisation, fb.polarisation, fa.cylinder, fb.cylinder,
                            fb.feed_location - fa.feed_location),
            conjugate};
}

// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_chime_in_cyl(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods) {
    // Calculate the set of baseline properties
    std::vector<std::pair<feed_diff, bool>> bl_prop;
    std::transform(std::begin(prods), std::end(prods), std::back_inserter(bl_prop),
                   std::bind(calculate_chime_vis, _1, inputs));

    // Create an index array for doing the sorting
    std::vector<uint32_t> sort_ind(prods.size());
    std::iota(std::begin(sort_ind), std::end(sort_ind), 0);

    auto sort_fn = [&](const uint32_t& ii, const uint32_t& jj) -> bool {
        return (bl_prop[ii].first < bl_prop[jj].first);
    };
    std::sort(std::begin(sort_ind), std::end(sort_ind), sort_fn);

    std::vector<rstack_ctype> stack_map(prods.size());

    feed_diff cur = bl_prop[sort_ind[0]].first;
    uint32_t cur_stack_ind = 0;

    for (auto& ind : sort_ind) {
        if (bl_prop[ind].first != cur) {
            cur = bl_prop[ind].first;
            cur_stack_ind++;
        }
        stack_map[ind] = {cur_stack_ind, bl_prop[ind].second};
    }

    return {++cur_stack_ind, stack_map};
}
