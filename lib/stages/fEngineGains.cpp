#include "fEngineGains.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "visUtil.hpp"           // for frameID

#include "fmt.hpp" // for format

#include <algorithm>  // for min
#include <errno.h>    // for errno
#include <functional> // for bind
#include <math.h>     // for sqrtf
#include <stdexcept>  // for invalid_argument
#include <stdio.h>    // for fopen, fread, fwrite, fclose
#include <string.h>   // for strerror
#include <unistd.h>   // for usleep
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(fEngineGains);

fEngineGains::fEngineGains(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fEngineGains::main_thread, this)),
    gain_buf(get_buffer("gain_buf")),
    stats_buf(config.exists(unique_name, "stats_buf") ? get_buffer("stats_buf") : nullptr),
    tables_counter(
        Metrics::instance().add_counter("kotekan_fengine_gains_tables_total", unique_name)) {

    fft_length = config.get<uint64_t>(unique_name, "fft_length");
    gain_mode = config.get_default<std::string>(unique_name, "gain_mode", "scalar");
    gain = config.get_default<float>(unique_name, "gain", 1.0f);
    gain_file = config.get_default<std::string>(unique_name, "gain_file", "");
    gain_file_out = config.get_default<std::string>(unique_name, "gain_file_out", "");
    target_rms = config.get_default<float>(unique_name, "target_rms", 2.0f);
    gain_max = config.get_default<float>(unique_name, "gain_max", 1e6f);
    stats_windows = config.get_default<uint32_t>(unique_name, "stats_windows", 64);

    if (gain_mode != "scalar" && gain_mode != "file" && gain_mode != "auto")
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: gain_mode must be \"scalar\", \"file\" or \"auto\", not "
                            "\"{:s}\""),
                        unique_name, gain_mode));
    if (gain_mode == "file" && gain_file.empty())
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: file mode needs gain_file"), unique_name));
    if (gain_mode == "auto" && stats_buf == nullptr)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: auto mode needs stats_buf"), unique_name));
    if (gain_buf->frame_size < fft_length / 2 * sizeof(float))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: gain_buf frames hold {:d} bytes but a table is {:d}"),
                        unique_name, gain_buf->frame_size, fft_length / 2 * sizeof(float)));

    gain_buf->register_producer(unique_name);
    if (stats_buf)
        stats_buf->register_consumer(unique_name);
}

bool fEngineGains::emit_table(const float* table) {
    uint8_t* frame = gain_buf->wait_for_empty_frame(unique_name, gain_frame_id);
    if (frame == nullptr)
        return false;
    memcpy(frame, table, fft_length / 2 * sizeof(float));
    gain_buf->mark_frame_full(unique_name, gain_frame_id);
    gain_frame_id = (gain_frame_id + 1) % gain_buf->num_frames;
    tables_counter.inc();
    return true;
}

void fEngineGains::main_thread() {
    const size_t m = fft_length / 2;
    std::vector<float> table(m, gain);

    if (gain_mode == "file") {
        FILE* f = fopen(gain_file.c_str(), "rb");
        if (f == nullptr)
            FATAL_ERROR("Cannot open gain_file {:s}: {:s}", gain_file, strerror(errno));
        const size_t got = fread(table.data(), sizeof(float), m, f);
        fclose(f);
        if (got != m)
            FATAL_ERROR("gain_file {:s} holds {:d} values, expected {:d}", gain_file, got, m);
    }
    if (!emit_table(table.data()))
        return;
    INFO("{:s} gain table of {:d} bins emitted", gain_mode, m);

    if (gain_mode != "auto") {
        while (!stop_thread)
            usleep(100000);
        return;
    }

    frameID stats_id(stats_buf);
    while (!stop_thread) {
        uint8_t* stats_frame = stats_buf->wait_for_full_frame(unique_name, stats_id);
        if (stats_frame == nullptr)
            return;
        const float* stats = reinterpret_cast<const float*>(stats_frame);
        for (size_t k = 0; k < m; k++) {
            // stats[k] sums |X[k]|^2 over stats_windows windows; per-component rms is
            // sqrt(mean / 2).
            const float rms = sqrtf(stats[k] / (2.0f * stats_windows));
            table[k] = rms > 0 ? std::min(gain_max, target_rms / rms) : gain_max;
        }
        stats_buf->mark_frame_empty(unique_name, stats_id);
        stats_id++;

        if (!gain_file_out.empty()) {
            FILE* f = fopen(gain_file_out.c_str(), "wb");
            if (f == nullptr)
                FATAL_ERROR("Cannot write gain_file_out {:s}: {:s}", gain_file_out,
                            strerror(errno));
            fwrite(table.data(), sizeof(float), m, f);
            fclose(f);
        }
        if (!emit_table(table.data()))
            return;
    }
}
