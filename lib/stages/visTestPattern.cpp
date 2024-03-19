#include "visTestPattern.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator!=
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for wait_for_full_frame, Buffer, mark_frame_empty, mark_fra...
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for datasetManager, dset_id_t
#include "datasetState.hpp"      // for freqState, inputState, prodState
#include "kotekanLogging.hpp"    // for INFO, DEBUG, DEBUG2, FATAL_ERROR, WARN
#include "prometheusMetrics.hpp" // for Metrics, Gauge, MetricFamily
#include "restClient.hpp"        // for restClient, restClient::restReply
#include "restServer.hpp"        // for connectionInstance, restServer, HTTP_RESPONSE, HTTP_RES...
#include "visBuffer.hpp"         // for VisFrameView

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>    // for max
#include <atomic>       // for atomic_bool
#include <cmath>        // for log10, sqrt
#include <complex>      // for operator*, operator-, complex
#include <cstdint>      // for uint64_t
#include <cstring>      // for memcpy, strerror
#include <cxxabi.h>     // for __forced_unwind
#include <dirent.h>     // for opendir
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <future>       // for async, future
#include <iomanip>      // for operator<<, setprecision
#include <memory>       // for allocator_traits<>::value_type
#include <mutex>        // for mutex, lock_guard, unique_lock
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument, out_of_range
#include <sys/stat.h>   // for mkdir, S_IRGRP, S_IROTH, S_IRWXU
#include <system_error> // for system_error
#include <time.h>       // for timespec
#include <tuple>        // for get


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::HTTP_RESPONSE;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(visTestPattern);


visTestPattern::visTestPattern(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visTestPattern::main_thread, this)) {

    // Setup the buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _tolerance = config.get_default<double>(unique_name, "tolerance", 1e-6);
    _report_freq = config.get_default<uint64_t>(unique_name, "report_freq", 1000);

    if (_tolerance <= 0)
        throw std::invalid_argument("visCheckTestPattern: tolerance has to be positive (is "
                                    + std::to_string(_tolerance) + ").");

    // report precision: a bit more than error tolerance
    precision = log10(1. / _tolerance) + 2;

    if (precision < 0) {
        throw std::invalid_argument("visCheckTestPattern: invalid value for tolerance: %f "
                                    "(resultet in negative report precision)"
                                    + std::to_string(_tolerance));
    }
    INFO("Using report precision {:d}", precision);

    write_dir = config.get<std::string>(unique_name, "write_dir");
    if (opendir(write_dir.c_str()) == nullptr) {
        // Create directory
        if (mkdir(write_dir.c_str(), S_IRWXU | S_IRGRP | S_IROTH) < 0) {
            throw std::runtime_error(fmt::format(fmt("Failure creating directory {:s}: {:s}"),
                                                 write_dir, std::strerror(errno)));
        }
    }
    INFO("Writing report to '{:s}'.", write_dir);

    endpoint_name = config.get_default<std::string>(unique_name, "endpoint_name", "run_test");

    // Don't run any tests until update received.
    num_frames = 0;

    expected_data_ready = false;

    // Subscribe to the dynamic config update: used to start a test for a number of frames.
    kotekan::restServer::instance().register_post_callback(
        endpoint_name, std::bind(&visTestPattern::receive_update, this, std::placeholders::_1,
                                 std::placeholders::_2));
}

visTestPattern::~visTestPattern() {
    kotekan::restServer::instance().remove_json_callback(endpoint_name);
}

void visTestPattern::main_thread() {

    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);

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

    // Wait for the first frame
    if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr) {
        INFO("No frames in input buffer on start.");
        if (outfile.is_open())
            outfile.close();
        return;
    }
    dset_id_t ds_id = VisFrameView(in_buf, frame_id).dataset_id;
    get_dataset_state(ds_id);

    // Comparisons will be against tolerance^2
    float t2 = _tolerance * _tolerance;

    auto& bad_values_counter = Metrics::instance().add_gauge(
        "kotekan_vistestpattern_bad_values_total", unique_name, {"name", "freq_id"});
    auto& avg_error_metric = Metrics::instance().add_gauge("kotekan_vistestpattern_avg_error",
                                                           unique_name, {"name", "freq_id"});
    auto& min_error_metric = Metrics::instance().add_gauge("kotekan_vistestpattern_min_error",
                                                           unique_name, {"name", "freq_id"});
    auto& max_error_metric = Metrics::instance().add_gauge("kotekan_vistestpattern_max_error",
                                                           unique_name, {"name", "freq_id"});
    auto& fpga_sequence_number_metric = Metrics::instance().add_gauge(
        "kotekan_vistestpattern_fpga_sequence_number", unique_name, {"name", "freq_id"});
    auto& ctime_seconds_metric = Metrics::instance().add_gauge(
        "kotekan_vistestpattern_ctime_seconds", unique_name, {"name", "freq_id"});

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr) {
            break;
        }

        // Check if the dataset ID changed.
        auto frame = VisFrameView(in_buf, frame_id);
        if (frame.dataset_id != ds_id) {
            std::string error_msg = fmt::format(fmt("Expected dataset id {}, got {}.\nNot "
                                                    "supported. Exiting..."),
                                                ds_id, frame.dataset_id);
            std::lock_guard<std::mutex> thread_lck(mtx_update);
            exit_failed_test(error_msg);
            return;
        }

        {
            // Wait, if there is an update.
            std::lock_guard<std::mutex> thread_lck(mtx_update);
            // Did someone send a config to our endpoint recently? Otherwise skip this frame...
            if (num_frames) {
                if (!expected_data_ready) {
                    try {
                        compute_expected_data();
                    } catch (std::exception& e) {
                        exit_failed_test(
                            fmt::format(fmt("Failure computing expected data. Received "
                                            "FPGA buffer data format doesn't match data "
                                            "stream: {:s}. Exiting..."),
                                        e.what()));
                        return;
                    }
                }

                num_bad = 0;
                avg_err = 0.0;
                min_err = 0.0;
                max_err = 0.0;

                // Iterate over covariance matrix
                for (size_t i = 0; i < frame.num_prod; i++) {

                    // Calculate the error^2 and compared this to the tolerance as it's
                    // much faster than taking the square root where we don't need to.
                    float r2 = fast_norm(frame.vis[i] - expected.at(frame.freq_id).at(i));

                    // check for bad values
                    if (r2 > t2) {
                        DEBUG2("Bad value (product {:d}): Expected {:f} + {:f}j, but found {:f} + "
                               "{:f}j in frame {:d} with freq_id {:d}.",
                               i, (float)expected.at(frame.freq_id).at(i).real(),
                               (float)expected.at(frame.freq_id).at(i).imag(),
                               (float)frame.vis[i].real(), (float)frame.vis[i].imag(), frame_id,
                               frame.freq_id);
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


                time = std::get<1>(frame.time);
                fpga_count = std::get<0>(frame.time);
                freq_id = frame.freq_id;

                // update Prometheus metrics
                const std::string freq_id_label = std::to_string(freq_id);
                bad_values_counter.labels({test_name, freq_id_label}).set(num_bad);
                fpga_sequence_number_metric.labels({test_name, freq_id_label}).set(fpga_count);
                ctime_seconds_metric.labels({test_name, freq_id_label}).set(ts_to_double(time));

                if (num_bad) {
                    avg_err /= (float)num_bad;

                    // write frame report to outfile
                    outfile << fpga_count << ",";
                    outfile << time.tv_sec << "." << time.tv_nsec << ",";
                    outfile << freq_id << ",";
                    outfile << num_bad << ",";
                    outfile << avg_err << ",";
                    outfile << min_err << ",";
                    outfile << max_err << std::endl;

                    // report errors in this frame
                    DEBUG("{:d} bad elements", num_bad);
                    DEBUG("mean error: {:f}", avg_err);
                    DEBUG("min error: {:f}", min_err);
                    DEBUG("max error: {:f}", max_err);
                    DEBUG("time: {:d}, {:d}.{:d}", fpga_count, (long long)time.tv_sec,
                          time.tv_nsec);
                    DEBUG("freq id: {:d}", freq_id);

                    // update error stats
                    avg_error_metric.labels({test_name, freq_id_label}).set(avg_err);
                    min_error_metric.labels({test_name, freq_id_label}).set(min_err);
                    max_error_metric.labels({test_name, freq_id_label}).set(max_err);

                    // gather data for report after many frames
                    num_bad_tot += num_bad;
                    avg_err_tot += avg_err * (float)num_bad;
                    if (min_err < min_err_tot || min_err_tot == 0)
                        min_err_tot = min_err;
                    if (max_err > max_err_tot)
                        max_err_tot = max_err;


                    // pass this bad frame to the output buffer:

                    // Wait for an empty frame in the output buffer
                    if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
                        break;
                    }

                    // Transfer metadata
                    in_buf->pass_metadata(frame_id, out_buf, output_frame_id);

                    // Copy the frame data here:
                    std::memcpy(out_buf->frames[output_frame_id], in_buf->frames[frame_id],
                                in_buf->frame_size);


                    out_buf->mark_frame_full(unique_name, output_frame_id);

                    // Advance output frame id
                    output_frame_id++;
                } else {
                    avg_error_metric.labels({test_name, freq_id_label}).set(0);
                    min_error_metric.labels({test_name, freq_id_label}).set(0);
                    max_error_metric.labels({test_name, freq_id_label}).set(0);
                }

                // print report some times
                if (++i_frame == _report_freq) {
                    i_frame = 0;

                    avg_err_tot /= (float)num_bad_tot;
                    if (num_bad_tot == 0)
                        avg_err_tot = 0;

                    INFO("Summary from last {:d} frames: num bad values: {:d}, mean "
                         "error: {:f}, min error: {:f}, max error: {:f}",
                         _report_freq, num_bad_tot, avg_err_tot, min_err_tot, max_err_tot);
                    avg_err_tot = 0.0;
                    num_bad_tot = 0;
                    min_err_tot = 0;
                    max_err_tot = 0;
                }

                num_frames--;

                // Test done?
                if (num_frames == 0) {
                    // Compute expected data at beginning of next test.
                    expected_data_ready = false;

                    // Report back.
                    if (test_done_host != "none") {
                        json data;
                        data["result"] = "OK";
                        data["name"] = test_name;
                        restClient::restReply reply = restClient::instance().make_request_blocking(
                            test_done_path, data, test_done_host, test_done_port);
                        if (!reply.first) {
                            FATAL_ERROR("Failed to report back test completion: {:s}",
                                        reply.second);
                        }
                    }

                    INFO("Test '{:s}' done.", test_name);
                }
            }
        }

        in_buf->mark_frame_empty(unique_name, frame_id);

        // Advance input frame id
        frame_id++;
    }
    if (outfile.is_open())
        outfile.close();

    if (num_frames) {
        std::string error_msg = "Kotekan exited before test was done.";
        WARN("{:s}", error_msg);
        std::lock_guard<std::mutex> thread_lck(mtx_update);
        json data;
        data["result"] = error_msg;
        data["name"] = test_name;
        restClient::restReply reply = restClient::instance().make_request_blocking(
            test_done_path, data, test_done_host, test_done_port);
        if (!reply.first) {
            FATAL_ERROR("Failed to report back test completion: {:s}", reply.second);
        }
    }
}

void visTestPattern::reply_failure(kotekan::connectionInstance& conn, std::string& msg) {
    WARN("{:s}", msg);
    conn.send_error(msg, HTTP_RESPONSE::REQUEST_FAILED);
}

void visTestPattern::receive_update(kotekan::connectionInstance& conn, json& data) {
    std::unique_lock<std::mutex> thread_lck(mtx_update);

    if (inputs.empty()) {
        std::string msg =
            fmt::format("Received update before receiving first frame. Try again later.");
        reply_failure(conn, msg);
        return;
    }

    if (num_frames) {
        std::string msg = fmt::format(
            fmt("Received update, but not done with the last test ({:d} frames remaining)."),
            num_frames);
        reply_failure(conn, msg);
        return;
    }

    try {
        test_done_host = data.at("reply_host").get<std::string>();
        test_done_path = data.at("reply_path").get<std::string>();
        test_done_port = data.at("reply_port").get<unsigned short>();
    } catch (std::exception& e) {
        std::string msg =
            fmt::format(fmt("Failure reading reply-endpoint from update: {:s}."), e.what());
        DEBUG2("This was the update: {:s}", data.dump(4));
        reply_failure(conn, msg);
        return;
    }

    try {
        num_frames = data.at("num_frames").get<uint32_t>();
    } catch (std::exception& e) {
        std::string msg =
            fmt::format(fmt("Failure reading 'num_frames' from update: {:s}."), e.what());
        DEBUG2("This was the update: {:s}", data.dump(4));
        reply_failure(conn, msg);
        return;
    }
    if (num_frames == 0) {
        INFO("Received update: DISABLE");
        conn.send_empty_reply(HTTP_RESPONSE::OK);
        return;
    }

    try {
        test_name = data.at("name").get<std::string>();
        fpga_buf_pattern =
            data.at("test_pattern").get<std::map<std::string, std::vector<cfloat>>>();
    } catch (std::exception& e) {
        std::string msg =
            fmt::format(fmt("Failure reading test pattern data from update: {:s}."), e.what());
        DEBUG2("This was the update: {:s}", data.dump(4));
        num_frames = 0;
        reply_failure(conn, msg);
        return;
    }

    if (fpga_buf_pattern.size() != inputs.size()) {
        std::string msg =
            fmt::format(fmt("Failure reading test pattern data from update: Number of "
                            "inputs ({:d}) does not match data stream ({:d} inputs)."),
                        fpga_buf_pattern.size(), inputs.size());
        num_frames = 0;
        reply_failure(conn, msg);
        return;
    }

    for (auto f : fpga_buf_pattern) {
        if (f.second.size() != freqs.size()) {
            std::string msg =
                fmt::format(fmt("Failure reading test pattern data from update: Number "
                                "of frequencies ({:d}) does not match data stream "
                                "({:d} frequencies)."),
                            f.second.size(), freqs.size());
            num_frames = 0;
            reply_failure(conn, msg);
            return;
        }
    }

    INFO("Received update. Reply-endpoint set to {:s}:{:d}/{:s}", test_done_host, test_done_port,
         test_done_path);

    // Open a new report file for this test
    std::string file_name = fmt::format(fmt("{:s}/{:s}.csv"), write_dir, test_name);
    if (outfile.is_open())
        outfile.close();
    outfile.open(file_name);
    if (!outfile.is_open()) {
        std::string msg = fmt::format(fmt("Failed to open out file '{:s}'."), file_name);
        reply_failure(conn, msg);
        return;
    }
    outfile << "fpga_count,time,freq_id,num_bad,avg_err,min_err,max_err" << std::endl;

    // Set iostream decimal precision to a bit more than the configured tolerance
    outfile << std::setprecision(precision);
    outfile << std::fixed;

    conn.send_empty_reply(HTTP_RESPONSE::OK);

    INFO("Created new report file: {:s}", file_name);
    INFO("Running test '{:s}' for {:d} frames.", test_name, num_frames);
    DEBUG2("This was the update: {:s}", data.dump(4));
}

void visTestPattern::get_dataset_state(dset_id_t ds_id) {
    auto& dm = datasetManager::instance();

    // Get the frequency and input state asynchronously.
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);

    const freqState* fstate = fstate_fut.get();
    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();

    if (fstate == nullptr || istate == nullptr || pstate == nullptr) {
        if (outfile.is_open())
            outfile.close();
        FATAL_ERROR("Could not find all required states of dataset with ID {}.\nExiting...", ds_id);
    }

    freqs = fstate->get_freqs();
    inputs = istate->get_inputs();
    prods = pstate->get_prods();
}

void visTestPattern::compute_expected_data() {

    size_t num_freqs = freqs.size();
    size_t num_prods = prods.size();

    DEBUG("Computing expected data...");
    expected.resize(num_freqs);
    for (size_t f = 0; f < num_freqs; f++)
        expected[f].resize(num_prods);

        // Sort input feed values by frequency.
#ifdef _OPENMP
    DEBUG("Speeding up calculation of exptected data with OpenMP.");
// Extreme speed up on recv2 but much slower in unit tests...
#pragma omp parallel for
#endif
    for (size_t p = 0; p < num_prods; p++) {
        // Generate auto product.
        std::vector<cfloat> buf_a =
            fpga_buf_pattern.at(inputs.at(prods.at(p).input_a).correlator_input);
        std::vector<cfloat> buf_b =
            fpga_buf_pattern.at(inputs.at(prods.at(p).input_b).correlator_input);
        for (size_t f = 0; f < num_freqs; f++) {
            expected[f][p] = buf_a.at(f) * std::conj(buf_b.at(f));

            DEBUG2("For frequency {:d} and product {:d}, expecting {:f}, {:f}.", f, p,
                   expected[f][p].real(), expected[f][p].imag());
        }
    }
    DEBUG("Computing expected data done!");
    expected_data_ready = true;
}

void visTestPattern::exit_failed_test(std::string error_msg) {
    if (outfile.is_open())
        outfile.close();

    // tell orchestrator that we failed with this test
    json data;
    data["result"] = error_msg;
    restClient::instance().make_request_blocking(test_done_path, data, test_done_host,
                                                 test_done_port);
    FATAL_ERROR("{:s}", error_msg);
}
