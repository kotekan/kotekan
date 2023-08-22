#include "visCompression.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator<
#include "Stack.hpp"             // for stack_chime_in_cyl, stack_diagonal
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"              // for wait_for_full_frame, mark_frame_empty, mark_frame_full
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, state_id_t, datasetManager
#include "datasetState.hpp"      // for stackState, prodState, inputState
#include "kotekanLogging.hpp"    // for INFO, DEBUG, ERROR, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Gauge, Counter, Metrics, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView, VisField, VisField::vis, VisField::weight
#include "visUtil.hpp"           // for current_time, modulo, rstack_ctype, cfloat, frameID

#include "gsl-lite.hpp" // for span

#include <algorithm>    // for copy, max, fill, copy_backward, equal
#include <atomic>       // for atomic_bool
#include <complex>      // for complex, norm
#include <cxxabi.h>     // for __forced_unwind
#include <deque>        // for deque
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function, placeholders
#include <future>       // for async, future
#include <iterator>     // for begin, end
#include <memory>       // for allocator_traits<>::value_type
#include <pthread.h>    // for pthread_setaffinity_np
#include <regex>        // for match_results<>::_Base_type
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>    // for invalid_argument, out_of_range, runtime_error
#include <system_error> // for system_error
#include <tuple>        // for tuple, tie
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
    in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")), frame_id_in(in_buf),
    frame_id_out(out_buf), compression_residuals_metric(Metrics::instance().add_gauge(
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

        // Create view to output frame
        auto output_frame = VisFrameView::create_frame_view(
            out_buf, output_frame_id, input_frame.num_elements, num_stack, input_frame.num_ev);

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
