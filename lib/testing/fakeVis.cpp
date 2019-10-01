#include "fakeVis.hpp"

#include "StageFactory.hpp"
#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "version.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

#include <algorithm>
#include <atomic>
#include <complex>
#include <csignal>
#include <cstdint>
#include <exception>
#include <fmt.hpp>
#include <functional>
#include <iterator>
#include <math.h>
#include <memory>
#include <regex>
#include <stdexcept>
#include <sys/time.h>
#include <time.h>
#include <tuple>
#include <type_traits>
#include <utility>


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


REGISTER_KOTEKAN_STAGE(fakeVis);
REGISTER_KOTEKAN_STAGE(replaceVis);


fakeVis::fakeVis(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fakeVis::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get<size_t>(unique_name, "num_elements");
    block_size = config.get<size_t>(unique_name, "block_size");
    num_eigenvectors = config.get<size_t>(unique_name, "num_ev");
    sleep_time = config.get_default<float>(unique_name, "sleep_time", 2.0);

    // Get the output buffer
    std::string buffer_name = config.get<std::string>(unique_name, "out_buf");

    // Fetch the buffer, register it
    out_buf = buffer_container.get_buffer(buffer_name);
    register_producer(out_buf, unique_name.c_str());

    // Get frequency IDs from config
    freq = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");

    // Was a fixed dataset ID configured?
    if (config.exists(unique_name, "dataset_id")) {
        _dset_id = config.get<dset_id_t>(unique_name, "dataset_id");
        _fixed_dset_id = true;
    } else
        _fixed_dset_id = false;

    // Get fill type
    fill_map["default"] = std::bind(&fakeVis::fill_mode_default, this, _1);
    fill_map["fill_ij"] = std::bind(&fakeVis::fill_mode_fill_ij, this, _1);
    fill_map["fill_ij_missing"] = std::bind(&fakeVis::fill_mode_fill_ij_missing, this, _1);
    fill_map["phase_ij"] = std::bind(&fakeVis::fill_mode_phase_ij, this, _1);
    fill_map["chime"] = std::bind(&fakeVis::fill_mode_chime, this, _1);
    fill_map["test_pattern_simple"] = std::bind(&fakeVis::fill_mode_test_pattern_simple, this, _1);
    fill_map["test_pattern_freq"] = std::bind(&fakeVis::fill_mode_test_pattern_freq, this, _1);
    fill_map["test_pattern_inputs"] = std::bind(&fakeVis::fill_mode_test_pattern_inputs, this, _1);

    mode = config.get_default<std::string>(unique_name, "mode", "default");

    if (fill_map.count(mode) == 0) {
        throw std::invalid_argument("unknown fill type " + mode);
    }
    INFO("Using fill type: {:s}", mode);
    fill = fill_map.at(mode);

    // Get timing and frame params
    cadence = config.get<float>(unique_name, "cadence");
    num_frames = config.get_default<int32_t>(unique_name, "num_frames", -1);
    wait = config.get_default<bool>(unique_name, "wait", true);

    // Get zero_weight option
    zero_weight = config.get_default<bool>(unique_name, "zero_weight", false);

    if (mode == "test_pattern_simple") {
        test_pattern_value = std::vector<cfloat>(1);
        test_pattern_value[0] = config.get_default<cfloat>(unique_name, "default_val", {1., 0.});
    } else if (mode == "test_pattern_freq") {
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
        if (bins.size() > freq.size()) {
            throw std::invalid_argument("fakeVis: length of frequencies array ("
                                        + std::to_string(bins.size())
                                        + ") can not be larger "
                                          "than size of freq_ids array ("
                                        + std::to_string(freq.size()) + ").");
        }

        test_pattern_value = std::vector<cfloat>(freq.size());
        for (size_t i = 0; i < freq.size(); i++) {
            size_t j;
            for (j = 0; j < bins.size(); j++) {
                if (bins.at(j) == i)
                    break;
            }
            if (j == bins.size())
                test_pattern_value[i] = default_val * std::conj(default_val);
            else
                test_pattern_value[i] = bin_values.at(j) * std::conj(bin_values.at(j));
        }
        DEBUG(
            "Using test pattern mode {:s} with default value {:f}+{:f}j and {:d} frequency values",
            mode, default_val.real(), default_val.imag(), bins.size());
    } else if (mode == "test_pattern_inputs") {
        std::vector<cfloat> input_values =
            config.get<std::vector<cfloat>>(unique_name, "input_values");
        if (input_values.size() != num_elements) {
            throw std::invalid_argument("fakeVis: lengths of input values ("
                                        + std::to_string(input_values.size())
                                        + ") and number of elements ("
                                        + std::to_string(num_elements) + ") have to be equal.");
        }

        size_t num_prods = num_elements * (num_elements + 1) / 2;
        test_pattern_value = std::vector<cfloat>(num_prods);
        size_t ind = 0;
        for (size_t i = 0; i < num_elements; i++) {
            for (size_t j = 0; j <= i; j++) {
                test_pattern_value[ind] = input_values.at(j) * std::conj(input_values.at(i));
                ind++;
            }
        }
        DEBUG("Using test pattern mode {:s} with {:d} input values", mode, input_values.size());
    }
}

void fakeVis::main_thread() {

    unsigned int output_frame_id = 0, frame_count = 0;
    uint64_t fpga_seq = 0;

    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // Calculate the time increments in seq and ctime
    uint64_t delta_seq = (uint64_t)(800e6 / 2048 * cadence);
    uint64_t delta_ns = (uint64_t)(cadence * 1000000000);

    // Register datasetStates to describe the properties of the created stream
    dset_id_t ds_id = 0;
    auto& dm = datasetManager::instance();

    if (_fixed_dset_id) {
        ds_id = _dset_id;
    } else {
        std::vector<state_id_t> states;
        states.push_back(
            dm.create_state<metadataState>("not set", "fakeVis", get_git_commit_hash()).first);

        std::vector<std::pair<uint32_t, freq_ctype>> fspec;
        // TODO: CHIME specific
        std::transform(std::begin(freq), std::end(freq), std::back_inserter(fspec),
                       [](const uint32_t& id) -> std::pair<uint32_t, freq_ctype> {
                           return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
                       });
        states.push_back(dm.create_state<freqState>(fspec).first);

        std::vector<input_ctype> ispec;
        for (uint32_t i = 0; i < num_elements; i++)
            ispec.emplace_back((uint32_t)i, fmt::format(fmt("dm_input_{:d}"), i));
        states.push_back(dm.create_state<inputState>(ispec).first);

        std::vector<prod_ctype> pspec;
        for (uint16_t i = 0; i < num_elements; i++)
            for (uint16_t j = i; j < num_elements; j++)
                pspec.push_back({i, j});
        states.push_back(dm.create_state<prodState>(pspec).first);
        states.push_back(dm.create_state<eigenvalueState>(num_eigenvectors).first);

        // Register a root state
        ds_id = dm.add_dataset(states);
    }

    while (!stop_thread) {

        double start = current_time();

        for (auto f : freq) {

            DEBUG("Making fake visBuffer for freq={:d}, fpga_seq={:d}", f, fpga_seq);

            // Wait for the buffer frame to be free
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                break;
            }

            // Allocate metadata and get frame
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame =
                visFrameView(out_buf, output_frame_id, num_elements, num_eigenvectors);

            output_frame.dataset_id = ds_id;

            // Set the frequency index
            output_frame.freq_id = f;

            // Set the time
            output_frame.time = std::make_tuple(fpga_seq, ts);

            // Set the length and total data
            output_frame.fpga_seq_length = delta_seq;
            output_frame.fpga_seq_total = delta_seq;

            // Fill out the frame with debug info according to the given mode.
            fill(output_frame);

            // gains
            for (uint32_t i = 0; i < num_elements; i++) {
                output_frame.gain[i] = 1;
            }

            // Mark the buffers and move on
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // Increment time
        fpga_seq += delta_seq;
        frame_count++; // NOTE: frame count increase once for all freq

        // Increment the timespec
        ts.tv_sec += ((ts.tv_nsec + delta_ns) / 1000000000);
        ts.tv_nsec = (ts.tv_nsec + delta_ns) % 1000000000;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if (num_frames > 0 && frame_count >= (unsigned)num_frames) {
            INFO("Reached frame limit [{:d} frames]. Sleeping and then exiting kotekan...",
                 num_frames);
            timespec ts = double_to_ts(sleep_time);
            nanosleep(&ts, nullptr);
            exit_kotekan(ReturnCode::CLEAN_EXIT);
            return;
        }

        // If requested sleep for the extra time required to produce a fake vis
        // at the correct cadence
        if (this->wait) {
            double diff = cadence - (current_time() - start);
            timespec ts_diff = double_to_ts(diff);
            nanosleep(&ts_diff, nullptr);
        }
    }
}


void fakeVis::fill_mode_default(visFrameView& frame) {
    auto out_vis = frame.vis;
    // Set diagonal elements to (0, row)
    for (uint32_t i = 0; i < num_elements; i++) {
        uint32_t pi = cmap(i, i, num_elements);
        out_vis[pi] = {0., (float)i};
    }
    // Save metadata in first few cells
    if (out_vis.size() < 3) {
        FATAL_ERROR("Number of elements ({:d}) is too small to encode the 3 debugging values of "
                    "fill-mode 'default' in fake visibilities.\nExiting...",
                    num_elements);
    } else {
        // For simplicity overwrite diagonal if needed
        out_vis[0] = {(float)std::get<0>(frame.time), 0.0};
        out_vis[1] = {(float)ts_to_double(std::get<1>(frame.time)), 0.0};
        out_vis[2] = {(float)frame.freq_id, 0.};
        // out_vis[3] = {(float) output_frame_id, 0.};
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_fill_ij(visFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = {(float)i, (float)j};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_fill_ij_missing(visFrameView& frame) {
    fill_mode_fill_ij(frame);
    frame.fpga_seq_total = frame.fpga_seq_length - 2;
    frame.rfi_total = 1;
}

void fakeVis::fill_mode_phase_ij(visFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            float phase = (float)i - (float)j;
            frame.vis[ind] = {cosf(phase), sinf(phase)};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_chime(visFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            int cyl_i = i / 512;
            int cyl_j = j / 512;

            int pos_i = i % 256;
            int pos_j = j % 256;

            frame.vis[ind] = {(float)(cyl_j - cyl_i), (float)(pos_j - pos_i)};
            ind++;
        }
    }
    fill_non_vis(frame);
}

void fakeVis::fill_mode_test_pattern_simple(visFrameView& frame) {
    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = {1, 0};
            ind++;
        }
    }

    // Fill ev
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, 1};
        }
        frame.eval[i] = i;
    }
    frame.erms = 1;

    // Fill weights
    ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = 1.;
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}

void fakeVis::fill_mode_test_pattern_freq(visFrameView& frame) {
    cfloat fill_value = test_pattern_value.at(frame.freq_id);

    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = fill_value;
            ind++;
        }
    }

    // Fill ev
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, fill_value.real()};
        }
        frame.eval[i] = i;
    }
    frame.erms = fill_value.real();

    // Fill weights
    ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = fill_value.real();
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}

void fakeVis::fill_mode_test_pattern_inputs(visFrameView& frame) {
    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.vis[ind] = test_pattern_value[ind];
            ind++;
        }
    }
    // Fill ev
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, 1};
        }
        frame.eval[i] = i;
    }
    frame.erms = 1;

    // Fill weights
    ind = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = 1;
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}

void fakeVis::fill_non_vis(visFrameView& frame) {
    // Set ev section
    for (uint32_t i = 0; i < num_eigenvectors; i++) {
        for (uint32_t j = 0; j < num_elements; j++) {
            int k = i * num_elements + j;
            frame.evec[k] = {(float)i, (float)j};
        }
        frame.eval[i] = i;
    }
    frame.erms = 1.0;

    // Set weights
    int ind = 0;
    const float weight_fill = zero_weight ? 0.0 : 1.0;
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t j = i; j < num_elements; j++) {
            frame.weight[ind] = weight_fill;
            ind++;
        }
    }

    // Set flags and gains
    std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
}

replaceVis::replaceVis(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&replaceVis::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void replaceVis::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer to be empty of data
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }
        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // Copy input frame to output frame and create view
        allocate_new_metadata_object(out_buf, output_frame_id);
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame);

        for (uint32_t i = 0; i < output_frame.num_prod; i++) {
            float real = (i % 2 == 0 ? output_frame.freq_id : std::get<0>(output_frame.time));
            float imag = i;

            output_frame.vis[i] = {real, imag};
        }


        // Mark the output buffer and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
