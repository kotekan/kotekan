#include "visCompression.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "VisFrameView.hpp"      // for VisFrameView, VisField, VisField::vis, VisField::weight
#include "buffer.h"              // for wait_for_full_frame, allocate_new_metadata_object, mark...
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, state_id_t, datasetManager
#include "datasetState.hpp"      // for stackState, prodState, inputState
#include "kotekanLogging.hpp"    // for INFO, DEBUG, ERROR, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Gauge, Counter, Metrics, MetricFamily
#include "visUtil.hpp"           // for rstack_ctype, prod_ctype, current_time, modulo, input_c...

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>    // for copy, max, fill, copy_backward, equal, sort, transform
#include <atomic>       // for atomic_bool
#include <complex>      // for complex, norm
#include <cstdlib>      // for abs
#include <cxxabi.h>     // for __forced_unwind
#include <deque>        // for deque
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function, _1, placeholders
#include <future>       // for async, future
#include <iterator>     // for begin, end, back_insert_iterator, back_inserter
#include <math.h>       // for abs
#include <memory>       // for allocator_traits<>::value_type
#include <numeric>      // for iota
#include <pthread.h>    // for pthread_setaffinity_np
#include <regex>        // for match_results<>::_Base_type
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>    // for invalid_argument, out_of_range, runtime_error
#include <system_error> // for system_error
#include <tuple>        // for tuple, make_tuple, operator!=, operator<, tie
#include <vector>       // for vector, __alloc_traits<>::value_type


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(baselineCompression);


baselineCompression::baselineCompression(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&baselineCompression::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    frame_id_in(in_buf),
    frame_id_out(out_buf),
    compression_residuals_metric(Metrics::instance().add_gauge(
        "kotekan_baselinecompression_residuals", unique_name, {"freq_id"})),
    compression_time_seconds_metric(Metrics::instance().add_gauge(
        "kotekan_baselinecompression_time_seconds", unique_name, {"thread_id"})),
    compression_frame_counter(Metrics::instance().add_counter(
        "kotekan_baselinecompression_frame_total", unique_name, {"thread_id"})) {

    register_consumer(in_buf, unique_name.c_str());
    register_producer(out_buf, unique_name.c_str());

    // Fill out the map of stack types
    stack_type_defs["diagonal"] = stack_diagonal;
    stack_type_defs["chime_in_cyl"] = stack_chime_in_cyl;

    // Apply config.
    std::string stack_type = config.get<std::string>(unique_name, "stack_type");
    if (stack_type_defs.count(stack_type) == 0) {
        ERROR("unknown stack type {:s}", stack_type);
        return;
    }
    INFO("using stack type: {:s}", stack_type);
    calculate_stack = stack_type_defs.at(stack_type);

    if (config.exists(unique_name, "exclude_inputs")) {
        exclude_inputs = config.get<std::vector<uint32_t>>(unique_name, "exclude_inputs");
    }

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    if (num_threads == 0)
        throw std::invalid_argument("baselineCompression: num_threads has to be at least 1.");
}

void baselineCompression::main_thread() {

    // Create the threads
    thread_handles.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i] = std::thread(&baselineCompression::compress_thread, this, i);

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

void baselineCompression::change_dataset_state(dset_id_t input_ds_id) {

    double start_time = current_time();

    auto& dm = datasetManager::instance();
    auto fprint = dm.fingerprint(input_ds_id, {"inputs", "products"});

    if (state_map.count(fprint) == 0) {
        // Get input & prod states synchronously
        auto istate = std::async(&datasetManager::dataset_state<inputState>, &dm, input_ds_id);
        auto pstate = std::async(&datasetManager::dataset_state<prodState>, &dm, input_ds_id);

        auto istate_ptr = istate.get();
        auto pstate_ptr = pstate.get();

        if (istate_ptr == nullptr || pstate_ptr == nullptr) {
            FATAL_ERROR("Couldn't find inputState or prodState ancestor of dataset "
                        "{}. Make sure there is a stage upstream in the config, that adds a "
                        "freqState.\nExiting...",
                        input_ds_id);
        }

        // Calculate stack description and register the state
        auto sspec = calculate_stack(istate_ptr->get_inputs(), pstate_ptr->get_prods());
        auto [state_id, sstate_ptr] =
            dm.create_state<stackState>(sspec.first, std::move(sspec.second));

        // Insert state into map
        state_map[fprint] = {state_id, sstate_ptr, pstate_ptr};
    }


    auto [state_id, sstate, pstate] = state_map.at(fprint);
    auto new_ds_id = dm.add_dataset(state_id, input_ds_id);

    dset_id_map[input_ds_id] = {new_ds_id, sstate, pstate};

    INFO("Created new stack update and registering. Took {:.2f}s", current_time() - start_time);
}

void baselineCompression::compress_thread(uint32_t thread_id) {

    int input_frame_id;
    int output_frame_id;

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
    auto input_frame = VisFrameView(in_buf, input_frame_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        double start_time = current_time();

        // Get a view of the current frame
        auto input_frame = VisFrameView(in_buf, input_frame_id);

        dset_id_t new_dset_id;
        const stackState* sstate_ptr = nullptr;
        const prodState* pstate_ptr = nullptr;

        // If the input dataset has changed construct a new stack spec for the
        // datasetManager
        {
            std::lock_guard<std::mutex> _lock(m_dset_map);
            if (dset_id_map.count(input_frame.dataset_id) == 0) {
                change_dataset_state(input_frame.dataset_id);
            }
            std::tie(new_dset_id, sstate_ptr, pstate_ptr) = dset_id_map.at(input_frame.dataset_id);
        }

        const auto& stack_map = sstate_ptr->get_rstack_map();
        const auto& prods = pstate_ptr->get_prods();
        auto num_stack = sstate_ptr->get_num_stack();

        std::vector<float> stack_norm(sstate_ptr->get_num_stack(), 0.0);
        std::vector<float> stack_v2(sstate_ptr->get_num_stack(), 0.0);

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);
        VisFrameView::set_metadata((VisMetadata*)out_buf->metadata[output_frame_id]->metadata,
                                   input_frame.num_elements, num_stack, input_frame.num_ev);

        // Create view to output frame
        auto output_frame = VisFrameView(out_buf, output_frame_id);

        // Copy over the data we won't modify
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {VisField::vis, VisField::weight});
        output_frame.dataset_id = new_dset_id;

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

            // If the weight is zero, completely skip this iteration
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
            float iwgt = (output_frame.weight[stack_ind] != 0.0)
                             ? (1.0 / output_frame.weight[stack_ind])
                             : 0.0;

            output_frame.vis[stack_ind] *= inorm;
            output_frame.weight[stack_ind] = norm * norm * iwgt;

            // Accumulate to calculate the variance of the residuals
            vart += stack_v2[stack_ind] - std::norm(output_frame.vis[stack_ind]) * norm;
            normt += norm;
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Calculate residuals (return zero if no data for this freq)
        float residual = (normt != 0.0) ? (vart / normt) : 0.0;

        // Update prometheus metrics
        double elapsed = current_time() - start_time;
        compression_residuals_metric.labels({std::to_string(output_frame.freq_id)}).set(residual);
        compression_time_seconds_metric.labels({std::to_string(thread_id)}).set(elapsed);
        compression_frame_counter.labels({std::to_string(thread_id)}).inc();

        // Get the current values of the shared frame IDs and increment them.
        {
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            output_frame_id = frame_id_out++;
            input_frame_id = frame_id_in++;
        }

        DEBUG("Compression time {:.4f}", elapsed);
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
    return os << fmt::format(fmt("{:c}{:03d}{:c}"), cyl_name[f.cylinder], f.feed_location,
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
