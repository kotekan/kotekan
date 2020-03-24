#include "applyGains.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, allocate_new_metadata_object
#include "bufferContainer.hpp"   // for bufferContainer
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for dset_id_t, datasetManager, state_id_t
#include "datasetState.hpp"      // for gainState, freqState, inputState
#include "kotekanLogging.hpp"    // for WARN, FATAL_ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics, Counter, Gauge
#include "visBuffer.hpp"         // for visFrameView, visField, visField::vis, visField...
#include "visFileH5.hpp"         // IWYU pragma: keep
#include "visUtil.hpp"           // for cfloat, modulo, double_to_ts, ts_to_double, fra...

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span
#include "libbase64.h"

#include <algorithm>                 // for copy, max, copy_backward
#include <cmath>                     // for abs, pow
#include <complex>                   // for operator*, operator+, complex, operator""if
#include <cstdint>                   // for uint64_t
#include <exception>                 // for exception
#include <functional>                // for _Bind_helper<>::type, _Placeholder, bind, _1
#include <highfive/H5DataSet.hpp>    // for DataSet, DataSet::getSpace
#include <highfive/H5DataSpace.hpp>  // for DataSpace, DataSpace::getDimensions
#include <highfive/H5File.hpp>       // for File, NodeTraits::getDataSet, File::File, File:...
#include <highfive/H5Object.hpp>     // for HighFive
#include <highfive/H5Selection.hpp>  // for SliceTraits::read
#include <highfive/bits/H5Utils.hpp> // for type_of_array<>::type
#include <memory>                    // for allocator_traits<>::value_type
#include <pthread.h>                 // for pthread_setaffinity_np
#include <regex>                     // for match_results<>::_Base_type
#include <sched.h>                   // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>                 // for invalid_argument, out_of_range, runtime_error
#include <sys/stat.h>                // for stat
#include <tuple>                     // for get


using namespace HighFive;
using namespace std::placeholders;


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
        Metrics::instance().add_counter("kotekan_applygains_late_frame_count", unique_name)) {

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

    // Get the calibration broker gains endpoint
    gains_dir = config.get<std::string>(unique_name, "gains_dir");
    broker_host = config.get<std::string>(unique_name, "broker_host");
    broker_port = config.get<unsigned int>(unique_name, "broker_port");
    read_from_file = config.get_default<bool>(unique_name, "read_from_file", false);

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    if (num_threads == 0)
        throw std::invalid_argument("applyGains: num_threads has to be at least 1.");

    // FIFO for gains and weights updates
    gains_fifo = updateQueue<GainUpdate>(num_kept_updates);

    // Initialise update tuple
    {
        std::unique_lock<std::mutex> lk(update_mtx);
        new_update = {"", -1., -1.};
    }

    // subscribe to gain timestamp updates
    configUpdater::instance().subscribe(this, std::bind(&applyGains::receive_update, this, _1));
}

bool applyGains::fexists(const std::string& filename) const {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

bool applyGains::receive_update(nlohmann::json& json) {

    auto& dm = datasetManager::instance();

    double new_ts;
    std::string gains_path;
    std::string update_id;
    double transition_interval;

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

    // Signal to fetch thread to get gains from broker
    {
        std::unique_lock<std::mutex> lk(update_mtx);
        new_update = {gtag, t_combine, new_ts};
    }
    received_update_cv.notify_one();

    INFO("Received gain update with tag {:s}.", gtag);

    return true;
}


void applyGains::main_thread() {

    // Create the threads
    thread_handles.resize(num_threads + 1);

    // Create a thread for fetching gains from cal broker
    thread_handles[0] = std::thread(&applyGains::fetch_thread, this);

    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i + 1] = std::thread(&applyGains::apply_thread, this);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i + 1].join();
    }
    // Join fetch thread
    received_update_cv.notify_one();
    thread_handles[0].join();
}

void applyGains::apply_thread() {
    using namespace std::complex_literals;

    auto& dm = datasetManager::instance();

    int output_frame_id;
    int input_frame_id;
    double frame_time;

    // Get the current values of the shared frame IDs and increment them.
    {
        std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
        output_frame_id = frame_id_out++;
        input_frame_id = frame_id_in++;
    }

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

        // Get the frequency index of this ID. The map will have been set in the first
        // validate_frame
        auto freq_ind = freq_map.at(input_frame.freq_id);

        // Unix time
        frame_time = ts_to_double(std::get<1>(input_frame.time));

        // Vectors for storing gains and weights
        std::vector<cfloat> gain(input_frame.num_elements);
        std::vector<cfloat> gain_conj(input_frame.num_elements);
        std::vector<float> weight_factor(input_frame.num_elements);

        state_id_t state_id;
        double age;
        bool skip = false;

        // Check to see if any gains are available at all.
        if (gains_fifo.size() == 0) {
            FATAL_ERROR("No gains available.");
        }

        {
            // Request shared lock for reading from FIFO
            std::shared_lock<std::shared_mutex> lock(gain_mtx);
            auto [ts_new, update_new] = gains_fifo.get_update(double_to_ts(frame_time));

            // Calculate how much has time has passed between the last gain update and this frame
            age = frame_time - ts_to_double(ts_new);

            if (update_new == nullptr) {
                WARN("No gains update is as old as the currently processed frame.");
                // Report number of frames received late
                late_frames_counter.inc();

                skip = true;

                // TODO: should figure out how to skip data or use the oldest
            } else {

                // Check that the gains have the right size
                if (!validate_gain(update_new->data))
                    return;

                // Now we know how long to combine over, we can see if there's
                // another gain update within that time window
                auto update_old =
                    gains_fifo
                        .get_update(double_to_ts(frame_time - update_new->transition_interval))
                        .second;

                auto& new_gain = update_new->data.gain.at(freq_ind);
                if (update_old == nullptr || update_new == update_old) {
                    gain = new_gain;
                } else {

                    // Check that the gains have the right size
                    if (!validate_gain(update_old->data))
                        return;

                    auto& old_gain = update_old->data.gain.at(freq_ind);
                    float coeff_new = age / update_new->transition_interval;
                    float coeff_old = 1 - coeff_new;

                    for (uint32_t ii = 0; ii < input_frame.num_elements; ii++) {
                        gain[ii] = coeff_new * new_gain[ii] + coeff_old * old_gain[ii];
                    }
                }

                // Copy weights TODO: should we combine weights somehow?
                weight_factor = update_new->data.weight.at(freq_ind);

                state_id = update_new->state_id;
            }
        }

        // If the data is too old we should skip the frame entirely
        if (skip) {
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
            input_frame_id = frame_id_in++;
            continue;
        }

        // Compute weight factors and conjugate gains
        for (uint32_t ii = 0; ii < input_frame.num_elements; ii++) {
            bool zero_weight = (weight_factor[ii] == 0.0) || (gain[ii] == (0.0f + 0.0if));

            weight_factor[ii] = zero_weight ? 0.0 : pow(abs(gain[ii]), -2.0);
            gain[ii] = zero_weight ? (1.0f + 0.0if) : gain[ii];
            gain_conj[ii] = std::conj(gain[ii]);
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
            std::lock_guard<std::mutex> lock_frame_ids(m_frame_ids);
            std::pair<state_id_t, dset_id_t> key = {state_id, input_frame.dataset_id};
            if (output_dataset_ids.count(key) == 0) {
                output_dataset_ids[key] = dm.add_dataset(state_id, input_frame.dataset_id);
            }
            output_frame.dataset_id = output_dataset_ids[key];
        }


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

    bool first = true;

    while (!stop_thread) {
        double test = -1;
        // we may have missed the initial gain update
        if (first) {
            {
                std::unique_lock<std::mutex> lk(update_mtx);
                test = std::get<1>(new_update);
            }
        }
        if (first && test == -1){
            // We can go straight to the wait step
            first = false;
        } else {
            auto& dm = datasetManager::instance();

            // Extract the update specs
            std::string gtag;
            double t_combine, new_ts;
            {
                std::unique_lock<std::mutex> lk(update_mtx);
                gtag = std::get<0>(new_update);
                t_combine = std::get<1>(new_update);
                new_ts = std::get<2>(new_update);
            }

            std::optional<applyGains::GainData> gain_data;
            if (read_from_file) {
                INFO("Reading gains from file...");
                gain_data = read_gain_file(gtag);
            } else {
                INFO("Fetching gains...");
                gain_data = fetch_gains(gtag);
            }
            if (!gain_data) {
                WARN("Ignoring gain update with tag {:s}.", gtag);
                continue;
            }

            // update gains
            state_id_t state_id = dm.create_state<gainState>(gtag, t_combine).first;
            GainUpdate update = {std::move(gain_data.value()), t_combine, state_id};

            {
                // Lock mutex exclusively while we update FIFO
                std::lock_guard<std::shared_mutex> lock(gain_mtx);
                gains_fifo.insert(double_to_ts(new_ts), std::move(update));
            }
            INFO("Updated gains to {:s}.", gtag);
        }

        // wait for conditional variable signal
        std::unique_lock<std::mutex> lk(update_mtx);
        received_update_cv.wait(lk);
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
            return {};
        }
    }

    // Read the gains from file
    HighFive::File gains_fl(gains_path, HighFive::File::ReadOnly);
    // Read the dataset and allocates it to the most recent entry of the gain vector
    HighFive::DataSet gains_ds = gains_fl.getDataSet("/gain");
    HighFive::DataSet gain_weight_ds = gains_fl.getDataSet("/weight");

    // Check the size of the datasets.
    // This first test will be skipped loading the initial gains as we don't yet
    // know what num_freq and num_elements will be.
    auto gain_size = gains_ds.getSpace().getDimensions();
    auto weight_size = gain_weight_ds.getSpace().getDimensions();
    if ((num_freq && gain_size[0] != num_freq.value())
        || (num_elements && gain_size[1] != num_elements.value())) {
        WARN("Gain dataset does not have the right shape. "
             "Got ({}, {}), expected ({}, {})). Skipping update.",
             gain_size[0], gain_size[1], num_freq.value(), num_elements.value());
        return {};
    }
    // This consistency test will work regardless
    if (gain_size[0] != weight_size[0] || gain_size[1] != weight_size[1]) {
        WARN("Gain and weight datasets do not match shapes. "
             "Gain ({}, {}), weight ({}, {})). Skipping update.",
             gain_size[0], gain_size[1], weight_size[0], weight_size[1]);
        return {};
    }

    // Read the gain and weight datasets
    gains_ds.read(gain_read);
    gain_weight_ds.read(weight_read);

    GainData g{std::move(gain_read), std::move(weight_read)};

    return g;
}


std::optional<applyGains::GainData> applyGains::fetch_gains(std::string tag) const {

    // query cal broker
    nlohmann::json json;
    json["update_id"] = tag;
    restReply reply = client.make_request_blocking("/gain", json, broker_host, broker_port);
    if (!reply.first) {
        WARN("Failed to retrieve gains from calibration broker. ({})", reply.second);
        return {};
    }
    INFO("Got reply from cal broker.");

    // parse reply
    nlohmann::json js_reply;
    try {
        js_reply = json::parse(reply.second);
    } catch (std::exception& e) {
        WARN("Failed to parse calibration broker gains reponse.: {}", e.what());
        return {};
    }

    std::string gain_str, wgt_str;
    std::vector<uint32_t> g_shape, w_shape;
    try {
        auto gain = js_reply.at("gain");
        auto weight = js_reply.at("weight");
        g_shape = gain.at("shape").get<std::vector<uint32_t>>();
        w_shape = weight.at("shape").get<std::vector<uint32_t>>();
        // read into a string
        gain_str = gain.at("data").get<std::string>();
        wgt_str = weight.at("data").get<std::string>();
    } catch (std::exception& e) {
        WARN("Failed to read gain and weight from cal broker response.: {}", e.what());
        return {};
    }
    INFO("Gain shape: {:d}, {:d}", g_shape[0], g_shape[1]);

    // TODO how to load initial gains?
    // Check the size of the datasets.
    // This first test will be skipped loading the initial gains as we don't yet
    // know what num_freq and num_elements will be.
    if ((num_freq && g_shape[0] != num_freq.value())
        || (num_elements && g_shape[1] != num_elements.value())) {
        WARN("Gain dataset does not have the right shape. "
             "Got ({}, {}), expected ({}, {})). Skipping update.",
             g_shape[0], g_shape[1], num_freq.value(), num_elements.value());
        return {};
    }
    // This consistency test will work regardless
    if (g_shape[0] != w_shape[0] || g_shape[1] != w_shape[1]) {
        WARN("Gain and weight datasets do not match shapes. "
             "Gain ({}, {}), weight ({}, {})). Skipping update.",
             g_shape[0], g_shape[1], w_shape[0], w_shape[1]);
        return {};
    }

    // decode base64
    size_t gain_size = g_shape[0] * g_shape[1];
    cfloat gain[gain_size];
    size_t gain_len, wgt_len;
    bool weight_bool[gain_size];
    if (base64_decode(gain_str.c_str(), gain_str.length(), (char*) gain, &gain_len, 0) != 1) {
        WARN("Failed to decode gains.");
        return {};
    }
    if (base64_decode(wgt_str.c_str(), wgt_str.length(), (char*) weight_bool, &wgt_len, 0) != 1) {
        WARN("Failed to decode gain weights.");
        return {};
    }
    if (gain_len != gain_size * sizeof(cfloat) || wgt_len != gain_size * sizeof(bool)) {
        WARN("Decoded gains/weights do not have expected length. Got (gain: {:d}, weight: {:d}), expected {:d}.",
             gain_len / sizeof(cfloat), wgt_len / sizeof(bool), gain_size);
        return {};
    }

    // Reshape
    // TODO: I don't know if it's possible to turn this into vectors without copying
    std::vector<std::vector<cfloat>> gain_vec;
    std::vector<std::vector<float>> weight_vec;
    for (size_t fi = 0; fi < g_shape[0]; fi++) {
        gain_vec.push_back(std::vector<cfloat>(gain, gain + g_shape[1]));
        weight_vec.push_back(std::vector<float>(weight_bool, weight_bool + g_shape[1]));
        for (size_t j = 0; j < g_shape[1]; j++) {
            INFO("weight[{:d}, {:d}] = {:f}", fi, j, weight_vec[fi][j]);
            INFO("gain[{:d}, {:d}] = {}", fi, j, gain_vec[fi][j]);
        }
    }

    GainData g{std::move(gain_vec), std::move(weight_vec)};

    return g;
}


bool applyGains::validate_frame(const visFrameView& frame) {

    // TODO: this should validate that the hashes of the input and frequencies
    // dataset states have not changed whenever the dataset_id changes

    std::unique_lock _lock(freqmap_mtx);

    if (!num_freq || !num_elements) {
        auto& dm = datasetManager::instance();
        auto* fstate = dm.dataset_state<freqState>(frame.dataset_id);
        auto* istate = dm.dataset_state<inputState>(frame.dataset_id);

        if (istate == nullptr || fstate == nullptr) {
            FATAL_ERROR("Stream with dataset_id={} must have both frequency"
                        "and input states registered.",
                        frame.dataset_id);
            return false;
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
    }

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
        WARN("Can't validate gain data as num_freq and num_elements not yet known.")
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
