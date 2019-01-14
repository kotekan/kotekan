#include "applyGains.hpp"

#include "configUpdater.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"
#include "visFileH5.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <csignal>
#include <exception>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <sys/stat.h>

using namespace HighFive;
using namespace std::placeholders;


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::prometheusMetrics;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(applyGains);


applyGains::applyGains(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&applyGains::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Apply config.
    // Number of gain versions kept. Default is 5.
    num_kept_updates = config.get_default<uint64_t>(unique_name, "num_kept_updates", 5);
    if (num_kept_updates < 1)
        throw std::invalid_argument("applyGains: config: num_kept_updates has"
                                    "to be equal or greater than one (is "
                                    + std::to_string(num_kept_updates) + ").");
    // Time to blend old and new gains in seconds. Default is 5 minutes.
    tcombine = config.get_default<float>(unique_name, "combine_gains_time", 5 * 60);
    if (tcombine < 0)
        throw std::invalid_argument("applyGains: config: combine_gains_time has"
                                    "to be positive (is "
                                    + std::to_string(tcombine) + ").");

    // Get the path to gains directory
    gains_dir = config.get<std::string>(unique_name, "gains_dir");

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    if (num_threads == 0)
        throw std::invalid_argument("applyGains: num_threads has to be at least 1.");
    if (in_buf->num_frames % num_threads != 0 || out_buf->num_frames % num_threads != 0)
        throw std::invalid_argument("applyGains: both the size of the input "
                                    "and output buffer have to be multiples "
                                    "of num_threads.");

    // FIFO for gains and weights updates
    gains_fifo = updateQueue<gainUpdate>(num_kept_updates);

    // subscribe to gain timestamp updates
    configUpdater::instance().subscribe(this, std::bind(&applyGains::receive_update, this, _1));
}

bool applyGains::fexists(const std::string& filename) {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

bool applyGains::receive_update(nlohmann::json& json) {
    double new_ts;
    std::string gains_path;
    std::string gtag;
    std::vector<std::vector<cfloat>> gain_read;
    std::vector<std::vector<float>> weight_read;
    // receive new gains timestamp ("start_time" might move to "start_time")
    try {
        if (!json.at("start_time").is_number())
            throw std::invalid_argument("applyGains: received bad gains "
                                        "timestamp: "
                                        + json.at("start_time").dump());
        if (json.at("start_time") < 0)
            throw std::invalid_argument("applyGains: received negative gains "
                                        "timestamp: "
                                        + json.at("start_time").dump());
        new_ts = json.at("start_time");
    } catch (std::exception& e) {
        WARN("Failure reading 'start_time' from update: %s", e.what());
        return false;
    }
    if (ts_frame.load() > double_to_ts(new_ts)) {
        WARN("applyGains: Received update with a timestamp that is older "
             "than the current frame (The difference is %f s).",
             ts_to_double(ts_frame.load()) - new_ts);
        num_late_updates++;
    }

    // receive new gains tag
    try {
        if (!json.at("tag").is_string())
            throw std::invalid_argument("applyGains: received bad gains tag:"
                                        + json.at("tag").dump());
        gtag = json.at("tag");
    } catch (std::exception& e) {
        WARN("Failure reading 'tag' from update: %s", e.what());
        return false;
    }
    // Get the gains for this timestamp
    // TODO: For now, assume the tag is the gain file name.
    gains_path = gains_dir + "/" + gtag + ".h5";
    // Check if file exists
    if (!fexists(gains_path)) {
        // Try a different extension
        gains_path = gains_dir + "/" + gtag + ".hdf5";
        if (!fexists(gains_path)) {
            WARN("Could not update gains. File not found: %s", gains_path.c_str())
            return false;
        }
    }

    // Read the gains from file
    HighFive::File gains_fl(gains_path, HighFive::File::ReadOnly);
    // Read the dataset and alocates it to the most recent entry of the gain vector
    HighFive::DataSet gains_ds = gains_fl.getDataSet("/gain");
    gains_ds.read(gain_read);
    // Read the gains weight dataset
    HighFive::DataSet gain_weight_ds = gains_fl.getDataSet("/weight");
    gain_weight_ds.read(weight_read);
    // Check dimensions are consistent
    if (weight_read.size() != gain_read.size()) {
        WARN("Gain and weight frequency axes are different lengths. ", "Skipping update.");
        return false;
    }
    for (uint i = 0; i < gain_read.size(); i++) {
        if (weight_read[i].size() != gain_read[i].size()) {
            WARN("Gain and weight time axes are different lengths. ", "Skipping update.");
            return false;
        }
    }
    gainUpdate gain_update = {gain_read, weight_read};
    {
        // Lock mutex exclusively while we update FIFO
        std::lock_guard<std::shared_timed_mutex> lock(gain_mtx);
        gains_fifo.insert(double_to_ts(new_ts), std::move(gain_update));
    }
    INFO("Updated gains to %s.", gtag.c_str());

    return true;
}

void applyGains::main_thread() {

    num_late_frames = 0;
    num_late_updates = 0;

    // Create the threads
    thread_handles.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handles[i] = std::thread(&applyGains::apply_thread, this, i);

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

void applyGains::apply_thread(int thread_id) {

    unsigned int output_frame_id = thread_id;
    unsigned int input_frame_id = thread_id;
    unsigned int freq;
    double tpast;
    double frame_time;

    while (!stop_thread) {


        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // get the frames timestamp
        ts_frame.store(std::get<1>(input_frame.time));

        // frequency index of this frame
        freq = input_frame.freq_id;
        // Unix time
        frame_time = ts_to_double(std::get<1>(input_frame.time));
        // Vector for storing gains
        std::vector<cfloat> gain(input_frame.num_elements);
        std::vector<cfloat> gain_conj(input_frame.num_elements);
        // Vector for storing weight factors
        std::vector<float> weight_factor(input_frame.num_elements);


        std::pair<timespec, const gainUpdate*> gainpair_new;
        std::pair<timespec, const gainUpdate*> gainpair_old;

        {
            // Request shared lock for reading from FIFO
            std::shared_lock<std::shared_timed_mutex> lock(gain_mtx);
            gainpair_new = gains_fifo.get_update(double_to_ts(frame_time));
            if (gainpair_new.second == NULL) {
                WARN("No gains available.\nKilling kotekan");
                std::raise(SIGINT);
            }
            tpast = frame_time - ts_to_double(gainpair_new.first);

            // Determine if we need to combine gains:
            bool combine_gains = (tpast >= 0) && (tpast < tcombine);
            if (combine_gains) {
                gainpair_old = gains_fifo.get_update(double_to_ts(frame_time - tcombine));
                // If we are not using the very first set of gains, do gains interpolation:
                combine_gains = combine_gains && !(gainpair_new.first == gainpair_old.first);
            }

            try {
                // Combine gains if needed:
                if (combine_gains) {
                    float coef_new = tpast / tcombine;
                    float coef_old = 1 - coef_new;
                    for (uint32_t ii = 0; ii < input_frame.num_elements; ii++) {
                        gain[ii] = coef_new * gainpair_new.second->gain.at(freq)[ii]
                                   + coef_old * gainpair_old.second->gain.at(freq)[ii];
                    }
                } else {
                    gain = gainpair_new.second->gain.at(freq);
                    if (tpast < 0) {
                        WARN("(Thread %d) No gains update is as old as the currently processed "
                             "frame. Using oldest gains available."
                             "Time difference is: %f seconds.",
                             thread_id, tpast);
                        num_late_frames++;
                    }
                }
                // Copy weights TODO: should we combine weights somehow?
                weight_factor = gainpair_new.second->weight.at(freq);
            } catch (std::out_of_range& e) {
                WARN("Freq ID %d is out of range in gain array: %s", freq, e.what());
                continue;
            }
        }
        try {
            // Compute weight factors and conjugate gains
            for (uint32_t ii = 0; ii < input_frame.num_elements; ii++) {
                if (weight_factor.at(ii) == 0.0) {
                    // If gain_weight is zero, make gains = 1 and weights = 0
                    gain.at(ii) = 1. + 0i;
                } else if (gain.at(ii) == (cfloat){0.0, 0.0}) {
                    // If gain is zero make the weight be zero
                    weight_factor.at(ii) = 0.0;
                    gain.at(ii) = {1., 0.};
                } else {
                    weight_factor.at(ii) = pow(abs(gain.at(ii)), -2.0);
                }
                gain_conj.at(ii) = std::conj(gain.at(ii));
            }
        } catch (std::out_of_range& e) {
            WARN("Input out of range in gain array: %s", e.what());
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
        prometheusMetrics::instance().add_process_metric("kotekan_applygains_update_age_seconds",
                                                         unique_name, tpast);

        // Report number of updates received too late
        prometheusMetrics::instance().add_process_metric("kotekan_applygains_late_update_count",
                                                         unique_name, num_late_updates.load());

        // Report number of frames received late
        prometheusMetrics::instance().add_process_metric("kotekan_applygains_late_frame_count",
                                                         unique_name, num_late_frames.load());

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        // Advance the current frame ids
        input_frame_id = (input_frame_id + num_threads) % in_buf->num_frames;
        output_frame_id = (output_frame_id + num_threads) % out_buf->num_frames;
    }
}
