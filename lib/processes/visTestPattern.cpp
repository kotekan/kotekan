#include "visTestPattern.hpp"

#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "processFactory.hpp"
#include "prometheusMetrics.hpp"
#include "restServer.hpp"
#include "visBuffer.hpp"

#include "fmt.hpp"
#include "gsl-lite.hpp"

#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iomanip>
#include <math.h>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <time.h>
#include <tuple>


REGISTER_KOTEKAN_PROCESS(visTestPattern);


visTestPattern::visTestPattern(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTestPattern::main_thread, this)) {

    // Setup the buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    _tolerance = config.get_default<float>(unique_name, "tolerance", 1e-6);
    _report_freq = config.get_default<uint64_t>(unique_name, "report_freq", 1000);

    if (_tolerance < 0)
        throw std::invalid_argument("visCheckTestPattern: tolerance has to be positive (is "
                                    + std::to_string(_tolerance) + ").");

    write_dir = config.get<std::string>(unique_name, "write_dir");
    INFO("Writing report to '%s'.", write_dir.c_str());

    endpoint_name = config.get_default<std::string>(unique_name, "endpoint_name", "run_test");

    // Don't run any tests until update received.
    num_frames = 0;
    no_update = true;
    test_done = true;

    expected_data_ready = false;

    // Subscribe to the dynamic config update: used to start a test for a number of frames.
    restServer::instance().register_post_callback(
        endpoint_name, std::bind(&visTestPattern::receive_update, this, std::placeholders::_1,
                                 std::placeholders::_2));
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
    if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
        if (outfile.is_open())
            outfile.close();
        return;
    }
    dset_id_t ds_id = visFrameView(in_buf, frame_id).dataset_id;
    get_dataset_state(ds_id);

    // Comparisons will be against tolerance^2
    float t2 = _tolerance * _tolerance;

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Check if the dataset ID changed.
        auto frame = visFrameView(in_buf, frame_id);
        if (frame.dataset_id != ds_id) {
            error_msg = fmt::format("Expected dataset id {:#x}, got {:#x}.\nNot supported. "
                                    "Exiting...",
                                    ds_id, frame.dataset_id);
            exit_failed_test();
            return;
        }

        // Wait, if there is an update.
        std::unique_lock<std::mutex> thread_lck(mtx_update);
        cv.wait(thread_lck, [&]() { return no_update; });

        // Did someone send a config to our endpoint recently? Otherwise skip this frame...
        if (num_frames) {
            if (!expected_data_ready) {
                try {
                    compute_expected_data();
                } catch (std::exception& e) {
                    error_msg = fmt::format("Failure computing expected data. Received "
                                            "FPGA buffer data format doesn't match data "
                                            "stream: {}\nExiting...",
                                            e.what());
                    exit_failed_test();
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
                    DEBUG2("Bad value (product %d): Expected %f + %fj, but found %f + %fj in "
                           "frame %d with freq_id %d.",
                           i, expected.at(frame.freq_id).at(i).real(),
                           expected.at(frame.freq_id).at(i).imag(), frame.vis[i].real(),
                           frame.vis[i].imag(), frame_id, frame.freq_id);
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
                DEBUG("time: %d, %lld.%d", fpga_count, (long long)time.tv_sec, time.tv_nsec);
                DEBUG("freq id: %d", freq_id);

                // Export results to prometheus.
                export_prometheus_metrics(num_bad, avg_err, min_err, max_err, fpga_count, time,
                                          freq_id);

                // gather data for report after many frames
                num_bad_tot += num_bad;
                avg_err_tot += avg_err * (float)num_bad;
                if (min_err < min_err_tot || min_err_tot == 0)
                    min_err_tot = min_err;
                if (max_err > max_err_tot)
                    max_err_tot = max_err;


                // pass this bad frame to the output buffer:

                // Wait for an empty frame in the output buffer
                if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id)
                    == nullptr) {
                    break;
                }

                // Transfer metadata
                pass_metadata(in_buf, frame_id, out_buf, output_frame_id);

                // Copy the frame data here:
                std::memcpy(out_buf->frames[output_frame_id], in_buf->frames[frame_id],
                            in_buf->frame_size);


                mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

                // Advance output frame id
                output_frame_id++;
            }

            // print report some times
            if (++i_frame == _report_freq) {
                i_frame = 0;

                avg_err_tot /= (float)num_bad_tot;
                if (num_bad_tot == 0)
                    avg_err_tot = 0;

                INFO("Summary from last %d frames: num bad values: %d, mean "
                     "error: %f, min error: %f, max error: %f",
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
                test_done = true;
                cv.notify_one();
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance input frame id
        frame_id++;
    }
    if (outfile.is_open())
        outfile.close();

    if (num_frames) {
        error_msg = "Kotekan exited before test was done.";
        WARN(error_msg.c_str());
        std::unique_lock<std::mutex> thread_lck(mtx_update);
        test_done = true;
        cv.notify_one();
    }
}

void visTestPattern::reply_failure(connectionInstance& conn, std::string& msg) {
    WARN(msg.c_str());
    conn.send_error(error_msg, HTTP_RESPONSE::REQUEST_FAILED);
}

void visTestPattern::receive_update(connectionInstance& conn, json& data) {
    std::unique_lock<std::mutex> thread_lck(mtx_update);
    no_update = false;

    if (num_frames) {
        std::string msg = fmt::format("Received update, but not done with the last test ({} frames "
                                      "remaining).",
                                      num_frames);
        reply_failure(conn, msg);
        return;
    }

    try {
        num_frames = data.at("num_frames").get<uint32_t>();
    } catch (std::exception& e) {
        std::string msg = fmt::format("Failure reading 'num_frames' from update: {}.", e.what());
        DEBUG2("This was the update: %s", data.dump().c_str());
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
            fmt::format("Failure reading test pattern data from update: %s.", e.what());
        DEBUG2("This was the update: %s", data.dump().c_str());
        num_frames = 0;
        reply_failure(conn, msg);
        return;
    }

    if (fpga_buf_pattern.size() != inputs.size()) {
        std::string msg = fmt::format("Failure reading test pattern data from update: Number of "
                                      "inputs (%d) does not match data stream (%d inputs).",
                                      fpga_buf_pattern.size(), inputs.size());
        num_frames = 0;
        reply_failure(conn, msg);
        return;
    }

    for (auto f : fpga_buf_pattern) {
        if (f.second.size() != freqs.size()) {
            std::string msg = fmt::format("Failure reading test pattern data from update: Number "
                                          "of frequencies (%d) does not match data stream "
                                          "(%d frequencies).",
                                          f.second.size(), freqs.size());
            num_frames = 0;
            reply_failure(conn, msg);
            return;
        }
    }

    // Open a new report file for this test
    std::string file_name = fmt::format("{}/{}.csv", write_dir, test_name);
    if (outfile.is_open())
        outfile.close();
    outfile.open(file_name);
    if (!outfile.is_open()) {
        WARN("Failed to open out file '%s'.", file_name.c_str());
    }
    outfile << "fpga_count,time,freq_id,num_bad,avg_err,min_err,max_err" << std::endl;

    // Set iostream decimal precision
    outfile << std::setprecision(REPORT_PRECISION);
    outfile << std::fixed;

    INFO("Created new report file: %s\nRunning test for %d frames.", file_name.c_str(), num_frames);
    DEBUG2("This was the update: %s", data.dump().c_str());

    // Wait until test is done...
    no_update = true;
    test_done = false;
    cv.notify_one();
    cv.wait(thread_lck, [&]() { return test_done; });

    if (error_msg.empty())
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    else {
        conn.send_error(error_msg, HTTP_RESPONSE::REQUEST_FAILED);
        error_msg = "";
    }
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
        ERROR("Could not find all required states of dataset with ID 0x%" PRIx64 ".\nExiting...",
              ds_id);
        if (outfile.is_open())
            outfile.close();
        raise(SIGINT);
    }

    freqs = fstate->get_freqs();
    inputs = istate->get_inputs();
    prods = pstate->get_prods();
}

void visTestPattern::compute_expected_data() {

    size_t num_freqs = freqs.size();
    size_t num_prods = prods.size();

    expected.resize(num_freqs);

    // Sort input feed values by frequency.
    for (size_t f = 0; f < num_freqs; f++) {
        // Generate auto product.
        expected[f].resize(num_prods);
        for (size_t p = 0; p < num_prods; p++) {
            uint16_t input_id_a = prods.at(p).input_a;
            uint16_t input_id_b = prods.at(p).input_b;
            cfloat input_a = fpga_buf_pattern.at(inputs.at(input_id_a).correlator_input).at(f);
            cfloat input_b = fpga_buf_pattern.at(inputs.at(input_id_b).correlator_input).at(f);
            expected[f][p] = input_a * std::conj(input_b);

            DEBUG("For frequency %d and product %d, expecting %f, %f.", f, p, expected[f][p].real(),
                  expected[f][p].imag());
        }
    }
    expected_data_ready = true;
}

void visTestPattern::exit_failed_test() {
    ERROR(error_msg.c_str());
    if (outfile.is_open())
        outfile.close();

    // tell orchestrator that we failed with this test
    test_done = true;
    cv.notify_one();

    raise(SIGINT);
}

void visTestPattern::export_prometheus_metrics(size_t num_bad, float avg_err, float min_err,
                                               float max_err, uint64_t fpga_count, timespec time,
                                               uint32_t freq_id) {
    prometheusMetrics& prometheus = prometheusMetrics::instance();
    std::string labels = fmt::format("name=\"{}\",freq_id=\"{}\"", test_name, freq_id);
    prometheus.add_process_metric("kotekan_vistestpattern_bad_values_total", unique_name, num_bad,
                                  labels);
    prometheus.add_process_metric("kotekan_vistestpattern_avg_error", unique_name, avg_err, labels);
    prometheus.add_process_metric("kotekan_vistestpattern_min_error", unique_name, min_err, labels);
    prometheus.add_process_metric("kotekan_vistestpattern_max_error", unique_name, max_err, labels);
    prometheus.add_process_metric("kotekan_vistestpattern_fpga_sequence_number", unique_name,
                                  fpga_count, labels);
    prometheus.add_process_metric("kotekan_vistestpattern_ctime_seconds", unique_name,
                                  ts_to_double(time), labels);
}
