#include "applyGains.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, wait_for_full_frame, allocate_new...
#include "bufferContainer.hpp"   // for bufferContainer
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for dset_id_t, datasetManager, state_id_t
#include "datasetState.hpp"      // for gainState, freqState, inputState
#include "kotekanLogging.hpp"    // for WARN, FATAL_ERROR, DEBUG, INFO, ERROR
#include "modp_b64.hpp"          // for modp_b64_decode, modp_b64_decode_len
#include "prometheusMetrics.hpp" // for Metrics, Counter, Gauge
#include "restClient.hpp"        // for restClient::restReply, restClient
#include "visBuffer.hpp"         // for visFrameView, visField, visField::vis, visField::we...
#include "visFileH5.hpp"         // IWYU pragma: keep
#include "visUtil.hpp"           // for cfloat, modulo, double_to_ts, ts_to_double, frameID

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>                 // for max, copy, copy_backward
#include <assert.h>                  // for assert
#include <chrono>                    // for operator""s, chrono_literals
#include <cmath>                     // for abs, pow
#include <complex>                   // for operator*, operator+, complex, operator""if, operat...
#include <cstdint>                   // for uint64_t, uint32_t, uint8_t
#include <exception>                 // for exception
#include <functional>                // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <highfive/H5DataSet.hpp>    // for DataSet
#include <highfive/H5File.hpp>       // for File, NodeTraits::getDataSet, File::File, File::Rea...
#include <highfive/H5Object.hpp>     // for HighFive
#include <highfive/H5Selection.hpp>  // for SliceTraits::read
#include <highfive/bits/H5Utils.hpp> // for type_of_array<>::type
#include <memory>                    // for operator==, __shared_ptr_access, allocator_traits<>...
#include <pthread.h>                 // for pthread_setaffinity_np
#include <regex>                     // for match_results<>::_Base_type
#include <sched.h>                   // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>                 // for runtime_error, invalid_argument, out_of_range
#include <string.h>                  // for memcpy
#include <sys/stat.h>                // for stat
#include <thread>                    // for thread, sleep_for
#include <tuple>                     // for get, tie, tuple


using nlohmann::json;
using namespace HighFive;
using namespace std::placeholders;
using namespace std::chrono_literals;


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(applyGains);


applyGains::applyGains(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&applyGains::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    frame_id_in(in_buf),
    frame_id_out(out_buf),
    update_age_metric(
        Metrics::instance().add_gauge("kotekan_applygains_update_age_seconds", unique_name)),
    late_update_counter(
        Metrics::instance().add_counter("kotekan_applygains_late_update_count", unique_name)),
    late_frames_counter(
        Metrics::instance().add_counter("kotekan_applygains_late_frame_count", unique_name)),
    client(restClient::instance()) {

    // Setup the input buffer
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    register_producer(out_buf, unique_name.c_str());

    // Apply config.
    // Number of gain versions kept. Default is 5.
    num_kept_updates = config.get_default<uint64_t>(unique_name, "num_kept_updates", 5);
    if (num_kept_updates < 1)
        throw std::invalid_argument("applyGains: config: num_kept_updates has"
                                    "to be equal or greater than one (is "
                                    + std::to_string(num_kept_updates) + ").");

    // Get the parameters for how to fetch gains from the cal_broker
    read_from_file = config.get_default<bool>(unique_name, "read_from_file", false);
    if (read_from_file) {
        gains_dir = config.get<std::string>(unique_name, "gains_dir");
    } else {
        broker_host = config.get<std::string>(unique_name, "broker_host");
        broker_port = config.get<unsigned int>(unique_name, "broker_port");
    }

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    if (num_threads == 0)
        throw std::invalid_argument("applyGains: num_threads has to be at least 1.");

    // FIFO for gains and weights updates
    gains_fifo.resize(num_kept_updates);

    // subscribe to gain timestamp updates
    configUpdater::instance().subscribe(this, std::bind(&applyGains::receive_update, this, _1));
}

bool applyGains::fexists(const std::string& filename) const {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

bool applyGains::receive_update(json& json) {

    double new_ts;
    std::string gains_path;
    std::string update_id;
    double transition_interval;
    bool new_state;

    // receive new gains timestamp ("start_time" might move to "start_time")
    try {
        if (!json.at("start_time").is_number())
            throw std::invalid_argument(fmt::format(fmt("applyGains: received bad gains "
                                                        "timestamp: {:s}"),
                                                    json.at("start_time").dump()));
        if (json.at("start_time") < 0)
            throw std::invalid_argument(fmt::format(fmt("applyGains: received negative gains "
                                                        "timestamp: {:s}"),
                                                    json.at("start_time").dump()));
        new_ts = json.at("start_time");
    } catch (std::exception& e) {
        WARN("Failure reading 'start_time' from update: {:s}", e.what());
        return false;
    }
    if (ts_frame.load() > double_to_ts(new_ts)) {
        WARN("applyGains: Received update with a timestamp that is older "
             "than the current frame (The difference is {:f} s).",
             ts_to_double(ts_frame.load()) - new_ts);
        late_update_counter.inc();
    }

    // receive new gains update
    try {
        if (!json.at("update_id").is_string())
            throw std::invalid_argument(
                fmt::format(fmt("applyGains: received bad gains update_id: {:s}"),
                            json.at("update_id").dump()));
        update_id = json.at("update_id").get<std::string>();
    } catch (std::exception& e) {
        WARN("Failure reading 'update_id' from update: {:s}", e.what());
        return false;
    }

    // Read the interval to blend over
    try {
        transition_interval = json.at("transition_interval").get<double>();
    } catch (std::exception& e) {
        WARN("Failure reading 'transition_interval' from update: {:s}", e.what());
        return false;
    }

    // Should we register a new dataset?
    try {
        if (!json.at("new_state").is_boolean())
            throw std::invalid_argument(fmt::format(fmt("applyGains: received bad gains "
                                                        "new_state: {:s}"),
                                                    json.at("new_state").dump()));
        new_state = json.at("new_state").get<bool>();
    } catch (std::exception& e) {
        WARN("Failure reading 'new_state' from update: {:s}", e.what());
        return false;
    }

    // Signal to fetch thread to get gains from broker
    update_fetch_queue.put({update_id, transition_interval, new_ts, new_state});

    INFO("Received gain update with tag {:s}.", update_id);

    return true;
}


void applyGains::main_thread() {

    std::thread fetch_thread;
    std::vector<std::thread> apply_thread_handles;

    // Create a cpu_set for the cores we want to bind the worker threads onto
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
        CPU_SET(i, &cpuset);

    // Create a thread for fetching gains from cal broker
    DEBUG("Creating worker fetch thread id");
    fetch_thread = std::thread(&applyGains::fetch_thread, this);
    pthread_setaffinity_np(fetch_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    // Fetch the first frame to initialise various parameters
    initialise();

    // Sleep briefly here. This is to give the fetch_thread chance to spin up
    // and read the initial gains
    std::this_thread::sleep_for(0.5s);

    for (uint32_t i = 0; i < num_threads; i++) {
        DEBUG("Creating worker thread id={}", i);
        auto& thread = apply_thread_handles.emplace_back(&applyGains::apply_thread, this);
        pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (auto& thread : apply_thread_handles)
        thread.join();

    // Join fetch thread
    update_fetch_queue.cancel();
    fetch_thread.join();
}

void applyGains::apply_thread() {

    auto& dm = datasetManager::instance();

    int output_frame_id;
    int input_frame_id;

    // Get the current values of the shared frame IDs and increment them.
    {
        std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
        output_frame_id = frame_id_out++;
        input_frame_id = frame_id_in++;
    }

    // Vectors for storing gains computed on the fly, keep out here so they are
    // only allocated once
    std::vector<cfloat> gain(num_elements.value());
    std::vector<cfloat> gain_conj(num_elements.value());
    std::vector<float> weight_factor(num_elements.value());

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // Check that the input frame has the right sizes
        if (!validate_frame(input_frame))
            return;

        // get the frames timestamp
        ts_frame.store(std::get<1>(input_frame.time));

        // Get the frequency index of this ID. The map will have been set by initialise
        // Also get the UNIX timestamp
        auto freq_ind = freq_map.at(input_frame.freq_id);
        auto frame_time = ts_to_double(std::get<1>(input_frame.time));

        // Calculate the gain factors we need to apply to this frame
        auto [late, age, state_id] =
            calculate_gain(frame_time, freq_ind, gain, gain_conj, weight_factor);

        // Report number of frames received late and skip the frame entirely
        if (late) {
            late_frames_counter.inc();
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
            input_frame_id = frame_id_in++;
            continue;
        }

        // Wait for the output buffer to be empty of data
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(out_buf, output_frame_id);

        // Copy frame and create view
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame.num_elements,
                                         input_frame.num_prod, input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {visField::vis, visField::weight});

        // Check if we have already registered this gain update against this
        // input dataset, do so if we haven't, and then label the output data
        // with the new id. This must be done while locked as the underlying map
        // could change
        {
            std::scoped_lock<std::mutex> lock_frame_ids(m_frame_ids);
            std::pair<state_id_t, dset_id_t> key = {state_id, input_frame.dataset_id};
            if (output_dataset_ids.count(key) == 0) {
                output_dataset_ids[key] = dm.add_dataset(state_id, input_frame.dataset_id);
            }
            output_frame.dataset_id = output_dataset_ids[key];
        }

        // Get raw pointers to avoid the gsl::span bounds checking
        cfloat* out_vis = output_frame.vis.data();
        cfloat* in_vis = input_frame.vis.data();
        float* out_weight = output_frame.weight.data();
        float* in_weight = input_frame.weight.data();

        // For now this doesn't try to do any type of check on the
        // ordering of products in vis and elements in gains.
        // Also assumes the ordering of freqs in gains is standard
        uint32_t idx = 0;
        for (uint32_t ii = 0; ii < input_frame.num_elements; ii++) {
            for (uint32_t jj = ii; jj < input_frame.num_elements; jj++) {
                // Gains are to be multiplied to vis
                out_vis[idx] = in_vis[idx] * gain[ii] * gain_conj[jj];
                // Update the weights.
                out_weight[idx] = in_weight[idx] * weight_factor[ii] * weight_factor[jj];
                idx++;
            }
            // Update the gains.
            output_frame.gain[ii] = input_frame.gain[ii] * gain[ii];
        }

        // Report how old the gains being applied to the current data are.
        update_age_metric.set(age);

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Get the current values of the shared frame IDs.
        {
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            output_frame_id = frame_id_out++;
            input_frame_id = frame_id_in++;
        }
    }
}


void applyGains::fetch_thread() {

    auto& dm = datasetManager::instance();

    while (!stop_thread) {

        // Wait until we've start receiving data before fetching any updates. By
        // waiting until then before reading the gains we can test for
        // consistency with the incoming data
        if (!started) {
            continue;
        }

        // Block for an update, and check if we should just exit
        auto update = update_fetch_queue.get();
        if (!update)
            break;

        auto [update_id, transition_interval, new_ts, new_state] = *update;

        std::optional<applyGains::GainData> gain_data;
        if (read_from_file) {
            DEBUG("Reading gains from file...");
            gain_data = read_gain_file(update_id);
        } else {
            DEBUG("Fetching gains...");
            gain_data = fetch_gains(update_id);
        }
        if (!gain_data) {
            WARN("Ignoring gain update with update_id {:s}.", update_id);
            continue;
        }

        // update gains
        state_id_t state_id;
        if (new_state) {
            state_id = dm.create_state<gainState>(update_id, transition_interval).first;
        }
        // Use the current dataset ID
        else {
            const auto update = gains_fifo.get_update(double_to_ts(new_ts)).second;
            state_id = update->state_id;
        }
        GainUpdate gain_update{std::move(gain_data.value()), transition_interval, state_id};

        {
            // Lock mutex exclusively while we update FIFO
            std::lock_guard<std::shared_mutex> lock(gain_mtx);
            gains_fifo.insert(double_to_ts(new_ts), std::move(gain_update));
        }
        INFO("Updated gains to {:s}.", update_id);
    }
}


std::optional<applyGains::GainData> applyGains::read_gain_file(std::string update_id) const {

    // Define the output arrays
    std::vector<std::vector<cfloat>> gain_read;
    std::vector<std::vector<float>> weight_read;

    // Get the gains for this timestamp
    // TODO: For now, assume the update_id is the gain file name.
    std::string gains_path = fmt::format(fmt("{:s}/{:s}.h5"), gains_dir, update_id);
    // Check if file exists
    if (!fexists(gains_path)) {
        // Try a different extension
        gains_path = fmt::format(fmt("{:s}/{:s}.hdf5"), gains_dir, update_id);
        if (!fexists(gains_path)) {
            WARN("Could not update gains. File not found: {:s}", gains_path)
            return std::nullopt;
        }
    }

    // Read the gains from file
    HighFive::File gains_fl(gains_path, HighFive::File::ReadOnly);
    // Read the dataset and allocates it to the most recent entry of the gain vector
    HighFive::DataSet gains_ds = gains_fl.getDataSet("/gain");
    HighFive::DataSet gain_weight_ds = gains_fl.getDataSet("/weight");
    // Read the gain and weight datasets
    gains_ds.read(gain_read);
    gain_weight_ds.read(weight_read);

    GainData g{std::move(gain_read), std::move(weight_read)};

    // Check that the gains read are the correct sizes and if not, reject the update
    if (!validate_gain(g))
        return std::nullopt;

    return g;
}


template<typename T>
std::pair<std::vector<T>, std::vector<uint32_t>> json_base64_to_array(const json& j) {

    auto shape = j.at("shape").get<std::vector<uint32_t>>();
    const auto& data_json = j.at("data");

    if (!data_json.is_string()) {
        throw std::runtime_error("Value for data key must be a string.");
    }

    auto string_data = data_json.get_ref<const json::string_t&>();
    auto enc_size = string_data.size();

    // Must be a multiple of 4 bytes to be base64 decodable
    if (enc_size % 4 != 0) {
        throw std::runtime_error(fmt::format(
            "base64 encoded data must be a multiple of four bytes. Length: {}.", enc_size));
    }

    // Decode the base64 data into a temporary buffer
    std::vector<char> decoded_data(modp_b64_decode_len(enc_size));
    auto dec_len = modp_b64_decode(decoded_data.data(), string_data.c_str(), string_data.size());

    if (dec_len < 0) {
        throw std::runtime_error("Could not base64 decode the array data.");
    }

    // Calculate the total array size
    uint64_t size = 1;
    for (const auto& axsize : shape)
        size *= axsize;
    auto size_bytes = size * sizeof(T);

    if (dec_len != size_bytes) {
        throw std::runtime_error(fmt::format("Size of decoded data different from expectation."
                                             "Got {}, expected {}",
                                             dec_len, size_bytes));
    }

    // Copy data into a vector of the correct length and alignment,
    // (decoded_data likely has padding bytes at the end)
    std::vector<T> arr(size);
    std::memcpy(arr.data(), decoded_data.data(), size_bytes);

    return {std::move(arr), std::move(shape)};
}

template<typename T, typename U = T>
std::vector<std::vector<U>> unpack_2d(const std::vector<T>& arr1d, uint32_t nr, uint32_t nc) {

    if (nr * nc != arr1d.size()) {
        throw std::runtime_error(fmt::format(
            "Size of arr1d ({}) does not match number of rows and columns ({} x {}).", nr, nc));
    }

    std::vector<std::vector<U>> arr2d;

    auto b = arr1d.cbegin();

    for (uint32_t i = 0; i < nr; i++) {
        auto br = b + i * nc;
        arr2d.push_back(std::vector<U>(br, br + nc));
    }
    return arr2d;
}


std::optional<applyGains::GainData> applyGains::fetch_gains(std::string update_id) const {

    // query cal broker
    json json_request;
    json_request["update_id"] = update_id;
    restClient::restReply reply =
        client.make_request_blocking("/gain", json_request, broker_host, broker_port);
    if (!reply.first) {
        WARN("Failed to retrieve gains from calibration broker. ({})", reply.second);
        return std::nullopt;
    }
    INFO("Got reply from cal broker.");

    // parse reply
    json js_reply;
    try {
        js_reply = json::parse(reply.second);
    } catch (std::exception& e) {
        WARN("Failed to parse calibration broker gains reponse.: {}", e.what());
        return std::nullopt;
    }

    std::vector<uint32_t> g_shape, w_shape;
    std::vector<cfloat> gains_1d;
    std::vector<uint8_t> weights_1d;

    try {
        const auto& gain = js_reply.at("gain");
        const auto& weight = js_reply.at("weight");

        std::tie(gains_1d, g_shape) = json_base64_to_array<cfloat>(gain);
        std::tie(weights_1d, w_shape) = json_base64_to_array<uint8_t>(weight);

    } catch (std::exception& e) {
        WARN("Failed to read gain and weight from cal broker response.: {}", e.what());
        return std::nullopt;
    }

    GainData g{unpack_2d(gains_1d, g_shape[0], g_shape[1]),
               unpack_2d<uint8_t, float>(weights_1d, w_shape[0], w_shape[1])};

    // Check that the gains read are the correct sizes and if not, reject the update
    if (!validate_gain(g))
        return std::nullopt;

    return g;
}


void applyGains::initialise() {

    if (started) {
        FATAL_ERROR("We should not end up here if we've already started processing.");
    }

    // Wait for the input buffer to be filled with data
    if (wait_for_full_frame(in_buf, unique_name.c_str(), 0) == nullptr) {
        return;
    }

    // Create view to input frame
    auto frame = visFrameView(in_buf, 0);

    auto& dm = datasetManager::instance();
    auto* fstate = dm.dataset_state<freqState>(frame.dataset_id);
    auto* istate = dm.dataset_state<inputState>(frame.dataset_id);

    if (istate == nullptr || fstate == nullptr) {
        FATAL_ERROR("Stream with dataset_id={} must have both frequency"
                    "and input states registered.",
                    frame.dataset_id);
        return;
    }

    auto& freqs = fstate->get_freqs();

    // Construct the frequency mapping determining which
    uint32_t freq_ind = 0;
    for (auto& f : freqs) {
        freq_map[f.first] = freq_ind;
        freq_ind++;
    }

    num_freq = freqs.size();
    num_elements = istate->get_inputs().size();
    num_prod = num_elements.value() * (num_elements.value() + 1) / 2;

    started = true;
}


bool applyGains::validate_frame(const visFrameView& frame) const {

    // TODO: this should validate that the hashes of the input and frequencies
    // dataset states have not changed whenever the dataset_id changes

    // Check that the number of inputs has not changed
    if (frame.num_elements != num_elements.value()) {
        FATAL_ERROR("Number of elements cannot change during lifetime "
                    "of task. Should be {}, just received {}.",
                    num_elements.value(), frame.num_elements);
        return false;
    }

    // Check that the number of products is still the full upper triangle.
    if (frame.num_prod != num_prod.value()) {
        FATAL_ERROR("Must have full triangle of products. Should be {}"
                    ", just received {}.",
                    num_prod.value(), frame.num_prod);
        return false;
    }
    return true;
}


bool applyGains::validate_gain(const applyGains::GainData& data) const {

    // If we haven't got the freq and elements numbers set we can't actually
    // check. This should not happen.
    if (!num_freq || !num_elements) {
        WARN("Can't validate gain data as num_freq and num_elements not yet known.");
        return false;
    }

    // Check that the components have consistent sizes
    if ((data.gain.size() != data.weight.size())
        || (data.gain.size() && data.gain[0].size() != data.weight[0].size())) {
        return false;
    }

    if (data.gain.size() != num_freq.value()) {
        FATAL_ERROR("Number of frequencies in gain update does not match the number in the "
                    "incoming datastream. Expected {}, gain data has {}.",
                    num_freq.value(), data.gain.size());
        return false;
    }

    if (data.gain.size() > 0 && data.gain[0].size() != num_elements.value()) {
        FATAL_ERROR("Number of elements in gain update does not match the number in the incoming "
                    "datastream. Expected {}, gain data has {}.",
                    num_elements.value(), data.gain[0].size());
        return false;
    }
    return true;
}

std::tuple<bool, double, state_id_t>
applyGains::calculate_gain(double timestamp, uint32_t freq_ind, std::vector<cfloat>& gain,
                           std::vector<cfloat>& gain_conj,
                           std::vector<float>& weight_factor) const {
    using namespace std::complex_literals;

    size_t num_elem = num_elements.value();

    // Check that the input vectors are large enough
    assert(gain.size() == num_elem);
    assert(gain_conj.size() == num_elem);
    assert(weight_factor.size() == num_elem);

    // Pointers to which ever set of gains we are using
    const cfloat* gain_ptr = nullptr;

    // Check to see if any gains are available at all.
    if (gains_fifo.size() == 0) {
        ERROR("No initial gains available.");
        return {true, -1, state_id_t::null};
    }

    const auto [ts_new, update_new] = gains_fifo.get_update(double_to_ts(timestamp));

    // Calculate how much has time has passed between the last gain update and this frame
    double age = timestamp - ts_to_double(ts_new);

    if (update_new == nullptr) {
        WARN("No gains update is as old as the currently processed frame.");
        return {true, -1, state_id_t::null};
    }

    // Now we know how long to combine over, we can see if there's
    // another gain update within that time window
    const auto update_old =
        gains_fifo.get_update(double_to_ts(timestamp - update_new->transition_interval)).second;

    const auto& new_gain = update_new->data.gain.at(freq_ind);
    const auto& weights = update_new->data.weight.at(freq_ind);
    assert(new_gain.size() == num_elem);
    assert(weights.size() == num_elem);

    if (update_old == nullptr || update_new == update_old) {
        gain_ptr = new_gain.data();
    } else {

        auto& old_gain = update_old->data.gain.at(freq_ind);
        assert(old_gain.size() == num_elem);

        float coeff_new = age / update_new->transition_interval;
        float coeff_old = 1 - coeff_new;

        for (uint32_t ii = 0; ii < num_elem; ii++) {
            gain[ii] = coeff_new * new_gain[ii] + coeff_old * old_gain[ii];
        }
        gain_ptr = gain.data();
    }

    // Compute weight factors and conjugate gains
    for (uint32_t ii = 0; ii < num_elem; ii++) {
        bool zero_weight = (weights[ii] == 0.0) || (gain_ptr[ii] == (0.0f + 0.0if));

        weight_factor[ii] = zero_weight ? 0.0 : pow(abs(gain_ptr[ii]), -2.0);
        gain[ii] = zero_weight ? (1.0f + 0.0if) : gain_ptr[ii];
        gain_conj[ii] = std::conj(gain[ii]);
    }

    return {false, age, update_new->state_id};
}
