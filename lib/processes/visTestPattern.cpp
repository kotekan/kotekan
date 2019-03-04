#include "visTestPattern.hpp"

#include "StageFactory.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "errors.h"
#include "visBuffer.hpp"

#include "gsl-lite.hpp"

#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <math.h>
#include <regex>
#include <stdexcept>
#include <time.h>
#include <tuple>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(visTestPattern);


visTestPattern::visTestPattern(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visTestPattern::main_thread, this)) {

    // Setup the buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // get config
    mode = config.get<std::string>(unique_name, "mode");

    INFO("visCheckTestPattern: mode = %s", mode.c_str());
    if (mode == "test_pattern_simple") {
        exp_val = config.get_default<cfloat>(unique_name, "default_val", {1., 0});
    } else if (mode == "test_pattern_freq") {
        num_freq = config.get<size_t>(unique_name, "num_freq");

        cfloat default_val = config.get_default<cfloat>(unique_name, "default_val", {128., 0.});
        std::vector<uint32_t> bins = config.get<std::vector<uint32_t>>(unique_name, "frequencies");
        std::vector<cfloat> bin_values =
            config.get<std::vector<cfloat>>(unique_name, "freq_values");
        if (bins.size() != bin_values.size()) {
            throw std::invalid_argument("fakeVis: lengths of frequencies ("
                                        + std::to_string(bins.size()) + ") and freq_value ("
                                        + std::to_string(bin_values.size())
                                        + ") arrays have to be equal.");
        }
        if (bins.size() > num_freq) {
            throw std::invalid_argument("fakeVis: length of frequencies array ("
                                        + std::to_string(bins.size())
                                        + ") can not be larger "
                                          "than num_freq ("
                                        + std::to_string(num_freq) + ").");
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
        throw std::invalid_argument("visCheckTestpattern: unknown mode: " + mode);

    tolerance = config.get_default<float>(unique_name, "tolerance", 1e-6);
    report_freq = config.get_default<uint64_t>(unique_name, "report_freq", 1000);

    outfile_name = config.get<std::string>(unique_name, "out_file");

    if (tolerance < 0)
        throw std::invalid_argument("visCheckTestPattern: tolerance has to be"
                                    " positive (is "
                                    + std::to_string(tolerance) + ").");

    outfile.open(outfile_name);
    if (!outfile.is_open()) {
        throw std::ios_base::failure("visCheckTestPattern: Failed to open "
                                     "out file "
                                     + outfile_name);
    }
    outfile << "fpga_count,time,freq_id,num_bad,avg_err,min_err,max_err" << std::endl;
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
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        auto frame = visFrameView(in_buf, frame_id);
        // INFO("%s", frame.summary().c_str());

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
            DEBUG("time: %d, %lld.%d", fpga_count, (long long)time.tv_sec, time.tv_nsec);
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
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                break;
            }

            // Transfer metadata
            pass_metadata(in_buf, frame_id, out_buf, output_frame_id);

            // Copy the frame data here:
            std::memcpy(out_buf->frames[output_frame_id], in_buf->frames[frame_id],
                        in_buf->frame_size);


            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            // Advance output frame id
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // print report some times
        if (++i_frame == report_freq) {
            i_frame = 0;

            avg_err_tot /= (float)num_bad_tot;
            if (num_bad_tot == 0)
                avg_err_tot = 0;

            INFO("Summary from last %d frames: num bad values: %d, mean "
                 "error: %f, min error: %f, max error: %f",
                 report_freq, num_bad_tot, avg_err_tot, min_err_tot, max_err_tot);
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