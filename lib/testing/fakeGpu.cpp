#include "fakeGpu.hpp"

#include "Config.hpp"         // for Config
#include "Stage.hpp"          // for Stage
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "errors.h"           // for ReturnCode, exit_kotekan
#include "factory.hpp"        // for FACTORY
#include "fakeGpuPattern.hpp" // for FakeGpuPattern, _factory_aliasFakeGpuPattern
#include "geoUtil.hpp"        // for GeoFrame
#include "kotekanLogging.hpp" // for DEBUG, ERROR, INFO
#include "visUtil.hpp"        // for frameID, gpu_N2_size, modulo, operator+

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <assert.h>   // for assert
#include <csignal>    // for raise, SIGTERM
#include <functional> // for bind, function
#include <random>     // for random_device, uniform_real_distribution, mt19937
#include <string>     // for basic_string, string
#include <sys/time.h> // for TIMESPEC_TO_TIMEVAL, timeval
#include <time.h>     // for timespec, time_t, nanosleep
#include <vector>     // for vector


REGISTER_KOTEKAN_STAGE(FakeGpu);
REGISTER_TELESCOPE(FakeTelescope, "fake");

FakeGpu::FakeGpu(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container, std::bind(&FakeGpu::main_thread, this)) {

    freq = config.get<int>(unique_name, "freq");
    cadence = config.get_default<float>(unique_name, "cadence", 5.0);

    pre_accumulate = config.get_default<bool>(unique_name, "pre_accumulate", true);

    if (pre_accumulate) {
        samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    }

    block_size = config.get<int32_t>(unique_name, "block_size");
    num_elements = config.get<int32_t>(unique_name, "num_elements");
    num_frames = config.get_default<int32_t>(unique_name, "num_frames", -1);
    num_freq_in_frame = config.get_default<int32_t>(unique_name, "num_freq_in_frame", 1);
    wait = config.get_default<bool>(unique_name, "wait", true);
    drop_probability = config.get_default<float>(unique_name, "drop_probability", 0.0);
    dataset_id = config.get_default<dset_id_t>(
        unique_name, "dataset_id", dset_id_t::from_string("f65bec4949ca616fbeea62660351edcb"));

    // Fetch the correct fill function
    std::string pattern_name = config.get<std::string>(unique_name, "pattern");

    // Validate and create test pattern
    if (!FACTORY(FakeGpuPattern)::exists(pattern_name)) {
        ERROR("Test pattern \"%s\" does not exist.", pattern_name.c_str());
        std::raise(SIGTERM);
    }
    pattern = FACTORY(FakeGpuPattern)::create_unique(pattern_name, config, unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Check that the buffer is large enough
    auto expected_frame_size = num_freq_in_frame * gpu_N2_size(num_elements, block_size);
    if ((unsigned int)out_buf->frame_size < expected_frame_size) {
        ERROR("Buffer size too small (%i bytes, minimum required %i bytes)", out_buf->frame_size,
              expected_frame_size);
        std::raise(SIGTERM);
    }
}

FakeGpu::~FakeGpu() {}

void FakeGpu::main_thread() {

    auto& tel = Telescope::instance();

    int frame_count = 0;
    frameID frame_id(out_buf);
    timeval tv;
    timespec ts;

    uint64_t delta_seq, delta_ns;
    uint64_t fpga_seq = 0;
    const auto nprod_gpu = gpu_N2_size(num_elements, block_size);

    // Set the start time
    ts = tel.to_time(fpga_seq);

    // Calculate the increment in time between samples
    if (pre_accumulate) {
        delta_seq = samples_per_data_set;
        delta_ns = samples_per_data_set * tel.seq_length_nsec();
    } else {
        delta_seq = 1;
        delta_ns = (uint64_t)(cadence * 1000000000);
    }

    // Get the amount of time we need to sleep for.
    timespec delta_ts;
    delta_ts.tv_sec = delta_ns / 1000000000;
    delta_ts.tv_nsec = delta_ns % 1000000000;

    // Set up a random number generating for testing drops
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> drop(0.0, 1.0);

    while (!stop_thread) {

        // Test if this will be dropped or not, and only fill out the buffer
        // (and advance) if it won't
        if (drop(gen) < drop_probability) {
            DEBUG("Dropping frame.");

        } else {

            int32_t* output = (int*)out_buf->wait_for_empty_frame(unique_name, frame_id);
            if (output == nullptr)
                break;

            DEBUG("Simulating GPU buffer in {}[{}]", out_buf->buffer_name, frame_id);

            out_buf->allocate_new_metadata_object(frame_id);
            stream_t stream_id = {.id = static_cast<decltype(stream_id.id)>(freq)};
            get_chord_metadata(out_buf, frame_id)->set_stream_id(stream_id);
            get_chord_metadata(out_buf, frame_id)->set_fpga_seq_num(fpga_seq);

            // Set the two times
            TIMESPEC_TO_TIMEVAL(&tv, &ts);
            get_chord_metadata(out_buf, frame_id)->set_first_packet_recv_time(tv);
            get_chord_metadata(out_buf, frame_id)->set_gps_time(ts);
            get_chord_metadata(out_buf, frame_id)->set_dataset_id(dataset_id);

            // Fill the buffer with the specified mode
            auto metadata = get_chord_metadata(out_buf, frame_id).get();
            // have to set frequencies here since the patterns don't have all
            // information
            std::vector<int> freqs(num_freq_in_frame);
            for (int freq_ind = 0; freq_ind < num_freq_in_frame; freq_ind++) {
                gsl_lite::span<int32_t> data(output + 2 * freq_ind * nprod_gpu,
                                             output + 2 * (freq_ind + 1) * nprod_gpu);
                pattern->fill(data, metadata, frame_count, freq + freq_ind);
                freqs[freq_ind] = freq + freq_ind;
            }
            metadata->set_coarse_freq(freqs);

            // Mark full and move onto next frame...
            out_buf->mark_frame_full(unique_name, frame_id++);
        }

        // Increase total frame count
        frame_count++;

        // Increment time
        fpga_seq += delta_seq;

        // Increment the timespec
        ts = ts + delta_ts;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if (num_frames > 0 && frame_count > num_frames) {
            INFO("Reached frame limit [{} frames]. Exiting kotekan...", num_frames);
            exit_kotekan(ReturnCode::CLEAN_EXIT);
            return;
        }

        // TODO: only sleep for the extra time required, i.e. account for the
        // elapsed time each loop
        if (this->wait)
            nanosleep(&delta_ts, nullptr);
    }
}


FakeTelescope::FakeTelescope(const kotekan::Config& config, const std::string& path) :
    Telescope(path, config.get<std::string>(path, "log_level"),
              config.get_default<bool>(path, "require_eop", false),
              config.get_default<std::string>(path, "eop_updatable_config", ""),
              GeoFrame(config.get<std::string>(path, "log_level"), "grid", 0.0, 0.0, {0, 0, 0},
                       {1, 0, 0}, {0, 1, 0}, {0, 0, 1}),
              10.0, 10.0) {
    _num_local_freq = config.get_default<size_t>(path, "num_local_freq", 1);
}

freq_id_t FakeTelescope::to_freq_id(stream_t stream_id, uint32_t ind) const {
    return stream_id.id + ind;
}

double FakeTelescope::to_freq_MHz(freq_id_t freq_id) const {
    // Use CHIME frequencies
    return 800.0 - 400.0 / 1024 * freq_id;
}

double FakeTelescope::freq_width_MHz(freq_id_t /*freq_id*/) const {
    // Use CHIME frequencies
    return 400.0 / 1024;
}

size_t FakeTelescope::num_freq_per_stream() const {
    return _num_local_freq;
}

size_t FakeTelescope::num_freq() const {
    return 1024;
}

nyquist_zone_t FakeTelescope::nyquist_zone() const {
    return 2;
}

timespec FakeTelescope::to_time(uint64_t seq) const {
    uint64_t ns = seq_length_nsec() * seq;
    return {time_t(ns / 1000000000ULL), time_t(ns % 1000000000ULL)};
}

int64_t FakeTelescope::to_time_ns(uint64_t seq) const {
    uint64_t ns = seq_length_nsec() * seq;
    return int64_t(ns);
}

uint64_t FakeTelescope::to_seq(timespec time) const {
    uint64_t ns = time.tv_sec * time_t(1000000000ULL) + time.tv_nsec;
    assert(ns % seq_length_nsec() == 0);
    return ns / seq_length_nsec();
}

size_t FakeTelescope::seq_length_nsec() const {
    return 2560;
}

bool FakeTelescope::gps_time_enabled() const {
    return true;
}
