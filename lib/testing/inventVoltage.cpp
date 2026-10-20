#include "Config.hpp"            // for Config
#include "DataType.hpp"          // for string_to_type, DataType
#include "NDArray.hpp"           // for NDArray, GenericNDArray
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"            // for Symbol
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, metadata_is_chord, get_c...
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, INFO
#include "metadata.hpp"          // for metadataObject
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "testSkySignal.hpp"     // for testBadFeeds, testPacketLoss, testSkySignal, splitmix64
#include "visUtil.hpp"           // for current_time

#include "json.hpp" // for json

#include <algorithm>  // for copy, fill_n
#include <array>      // for array
#include <cassert>    // for assert
#include <cmath>      // for fmax, fmin, sqrt
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t, uint8_t
#include <cstring>    // for memcpy
#include <fmt.hpp>    // for compile_string_to_view
#include <functional> // for function
#include <iomanip>    // for operator<<, setfill, setw
#include <memory>     // for allocator, shared_ptr, __shared_ptr_access
#ifdef WITH_OMP
#include <omp.h> // for omp_get_wtime
#endif
#include <sstream>  // for basic_ostream, operator<<, basic_ostrin...
#include <string>   // for basic_string, char_traits, string, oper...
#include <unistd.h> // for gethostname, sleep
#include <vector>   // for vector

[[maybe_unused]] static std::uint32_t xorshift32(std::uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

class inventVoltage : public kotekan::Stage {
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");
    const int num_frequencies = config.get<int>(unique_name, "num_frequencies");
    const int num_times = config.get<int>(unique_name, "num_times");

    const std::string input_kind = config.get<std::string>(unique_name, "input_kind");
    // For `input_kind: box`: the FPGA samples [box_begin, box_end) hold a constant voltage, all
    // other samples are zero
    const std::int64_t box_begin =
        input_kind == "box" ? config.get<std::int64_t>(unique_name, "box_begin") : 0;
    const std::int64_t box_end =
        input_kind == "box" ? config.get<std::int64_t>(unique_name, "box_end") : 0;
    const bool reuse_frames = config.get_default<bool>(unique_name, "reuse_frames", false);
    // The voltage of lost samples (see `testPacketLoss`, the pattern of `inventPLMask`): `keep`
    // the signal, `zero` (0+0i, as the receivers write), or `nodata` (-8-8i)
    const std::string lost_voltage =
        config.get_default<std::string>(unique_name, "lost_voltage", "keep");
    const testPacketLoss loss{config, unique_name};
    // The voltage of bad inputs (see `testBadFeeds`, the schedule of `inventBFMask`, and the
    // dishes listed as Missing in `telescope.dish_inputs`, which `inventBFMask` mode `auto` marks
    // bad): `keep` the signal, or `random` bytes
    const std::string bad_voltage =
        config.get_default<std::string>(unique_name, "bad_voltage", "keep");
    const testBadFeeds bad_feeds{config, unique_name};
    // FPGA sequence number of the first sample; must agree between the voltage and mask
    // generators
    const std::int64_t first_fpga_seq_num =
        config.get_default<std::int64_t>(unique_name, "first_fpga_seq_num", 0);

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    const int out_frame_first = config.get_default<int>(unique_name, "out_frame_first", 10);
    const int out_frame_every = config.get_default<int>(unique_name, "out_frame_every", 10);

    Buffer* const buffer;

    // For `input_kind: sky`: the signal model, and a pool of precomputed frames holding the
    // periodic part of the signal (sources without a time window, and noise)
    std::unique_ptr<testSkySignal> sky;
    std::vector<std::vector<kotekan::int4x2_swapped_withoffset_t>> sky_pool;

    // For `bad_voltage: random`: the inputs [P][D] of Missing dishes
    std::vector<int> missing_inputs;

public:
    inventVoltage(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("voltage")) {
        assert(buffer);
        buffer->register_producer(unique_name);

        // Set metadata
        buffer->require_frame_desc(
            kotekan::NDArray<kotekan::int4x2_swapped_withoffset_t, 4>::describe(
                "E", {num_times, num_frequencies, num_polarizations, num_dishes},
                {"T", "F", "P", "D"}, {1, 1, 1, 1}));

        if (input_kind == "sky")
            sky = std::make_unique<testSkySignal>(
                config, unique_name, num_times, frequency_channels, num_polarizations, num_dishes);
        if (!(lost_voltage == "keep" || lost_voltage == "zero" || lost_voltage == "nodata"))
            FATAL_ERROR("Unknown value for `lost_voltage`: \"{:s}\"", lost_voltage);
        if (lost_voltage != "keep" && reuse_frames)
            FATAL_ERROR("`lost_voltage` requires `reuse_frames: false`");
        if (!(bad_voltage == "keep" || bad_voltage == "random"))
            FATAL_ERROR("Unknown value for `bad_voltage`: \"{:s}\"", bad_voltage);
        if (bad_voltage != "keep" && reuse_frames)
            FATAL_ERROR("`bad_voltage` requires `reuse_frames: false`");
        if (bad_voltage != "keep")
            for (const auto& dish : config.get<nlohmann::json>("/telescope", "dish_inputs"))
                if (dish.at("type").get<std::string>() == "Missing")
                    for (int p = 0; p < num_polarizations; ++p)
                        missing_inputs.push_back(p * num_dishes + dish.at("dish_idx").get<int>());
    }

    virtual ~inventVoltage() {}

    void main_thread() override {
        [[maybe_unused]] std::uint32_t rng_state = 0x0ad42ea5; // must not be 0

        double time_min = 1.0 / 0.0;
        double time_max = 0.0 / 0.0;
        double time_count = 0.0;
        double time_sum = 0.0;
        double time_sum2 = 0.0;

        double time0 = current_time();

        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

            if (stop_thread)
                return;

            // Wait for buffer
            DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
            kotekan::int4x2_swapped_withoffset_t* const frame =
                static_cast<kotekan::int4x2_swapped_withoffset_t*>(
                    static_cast<void*>(buffer->wait_for_empty_frame(unique_name, frame_id)));
            if (!frame)
                return;

            double time1 = current_time();
            // Skip the first few iterations to avoid the startup overhead
            if (frame_index > out_frame_first) {
                double time = time1 - time0;
                using std::fmax, std::fmin, std::sqrt;
                time_min = fmin(time_min, time);
                time_max = fmax(time_max, time);
                time_count += 1;
                time_sum += time;
                time_sum2 += time * time;
                if (frame_index % out_frame_every == 0) {
                    double time_avg = time_sum / time_count;
                    double time_sdv = sqrt(fmax(0.0, time_sum2 / time_count - time_avg * time_avg));
                    INFO("Timing statistics:\n"
                         "    frame: {:6d}\n"
                         "    count: {:6.0f} frames\n"
                         "    min:   {:10.3f} ms/frame   {:10.3f} μs/sample\n"
                         "    max:   {:10.3f} ms/frame   {:10.3f} μs/sample\n"
                         "    avg:   {:10.3f} ms/frame   {:10.3f} μs/sample\n"
                         "    sdv:   {:10.3f} ms/frame   {:10.3f} μs/sample\n",
                         frame_index,                                      //
                         time_count,                                       //
                         1.0e+3 * time_min, 1.0e+6 * time_min / num_times, //
                         1.0e+3 * time_max, 1.0e+6 * time_max / num_times, //
                         1.0e+3 * time_avg, 1.0e+6 * time_avg / num_times, //
                         1.0e+3 * time_sdv, 1.0e+6 * time_sdv / num_times  //
                    );
                }
            }
            time0 = time1;

            buffer->allocate_new_metadata_object(frame_id);
            const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
            meta->set_fpga_seq_num(first_fpga_seq_num + std::int64_t(frame_index) * num_times);
            meta->set_from_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
            meta->set_time_downsampling_fpga(1);

            // Set frequency information
            const std::vector<int>& coarse_freq = frequency_channels;
            const std::vector<int> freq_upchan_factor(coarse_freq.size(), 1);
            const std::vector<int> freq_upchan_index(coarse_freq.size(), 0);
            meta->set_coarse_freq(coarse_freq);
            meta->set_freq_upchan_factor(freq_upchan_factor);
            meta->set_freq_upchan_index(freq_upchan_index);

            // Fill buffer
            DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
            if (input_kind == "sky") {
                fill_sky(frame, first_fpga_seq_num + std::int64_t(frame_index) * num_times,
                         frame_index);
            } else if (!reuse_frames || frame_index < buffer->num_frames) {
                if (input_kind == "random") {
                    const std::ptrdiff_t str3 = 1;
                    const std::ptrdiff_t str2 = str3 * num_dishes;
                    const std::ptrdiff_t str1 = str2 * num_polarizations;
                    const std::ptrdiff_t str0 = str1 * num_frequencies;
                    for (int time = 0; time < num_times; ++time) {
                        for (int freq = 0; freq < num_frequencies; ++freq) {
                            for (int polr = 0; polr < num_polarizations; ++polr) {
                                for (int dish = 0; dish < num_dishes; ++dish) {
                                    const int idx3 = dish;
                                    const int idx2 = polr;
                                    const int idx1 = freq;
                                    const int idx0 = time;
                                    const std::ptrdiff_t idx =
                                        str3 * idx3 + str2 * idx2 + str1 * idx1 + str0 * idx0;
                                    assert(idx >= 0 && idx < std::ptrdiff_t(buffer->frame_size));
                                    const int val0 = int(rng_state % 15) - 7;
                                    const int val1 = int(rng_state / 16 % 15) - 7;
                                    frame[idx] = kotekan::int4x2_swapped_withoffset_t(val0, val1);
                                    rng_state = xorshift32(rng_state);
                                }
                            }
                        }
                    }
                } else if (input_kind == "pattern") {
                    std::uint64_t* __restrict__ const ptr =
                        static_cast<std::uint64_t*>(static_cast<void*>(frame));
                    const std::ptrdiff_t nmax = buffer->frame_size / sizeof *ptr;
                    // the signal:      +0 +4 +6 +4 +0 -4 -6 -4
                    // offset-encoded:   8  c  d  c  8  4  2  4
                    // use the signal twice, phase-shifted by 1/4 cycle
                    constexpr std::uint64_t signal1 = 0x80c0d0c080402040ULL;
                    constexpr std::uint64_t signal2 = 0x0d0c08040204080cULL;
                    constexpr std::uint64_t signal = signal1 | signal2;
                    for (std::ptrdiff_t n = 0; n < nmax; ++n)
                        ptr[n] = signal;
                } else if (input_kind == "memset") {
                    std::memset(frame, 0xcc, buffer->frame_size);
                } else if (input_kind == "box") {
                    const std::ptrdiff_t time_stride =
                        std::ptrdiff_t(num_frequencies) * num_polarizations * num_dishes;
                    const std::int64_t frame_begin =
                        first_fpga_seq_num + std::int64_t(frame_index) * num_times;
                    for (int time = 0; time < num_times; ++time) {
                        const std::int64_t seq = frame_begin + time;
                        const std::int8_t real = seq >= box_begin && seq < box_end ? 4 : 0;
                        // The arguments are (imaginary, real)
                        std::fill_n(frame + time * time_stride, time_stride,
                                    kotekan::int4x2_swapped_withoffset_t(0, real));
                    }
                } else {
                    FATAL_ERROR("Unknown value for `input_kind`: \"{:s}\"", input_kind);
                }
            }

            if (bad_voltage != "keep")
                overwrite_bad(frame, first_fpga_seq_num + std::int64_t(frame_index) * num_times);
            if (lost_voltage != "keep")
                overwrite_lost(frame, first_fpga_seq_num + std::int64_t(frame_index) * num_times);

            // Mark buffer as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
            buffer->mark_frame_full(unique_name, frame_id);
        }
    }

private:
    // Overwrite the lost samples in a [T, F, P, D] frame whose first sample is `frame_seq`
    void overwrite_lost(kotekan::int4x2_swapped_withoffset_t* const frame,
                        const std::int64_t frame_seq) const {
        // The arguments are (imaginary, real)
        const kotekan::int4x2_swapped_withoffset_t value =
            lost_voltage == "zero" ? kotekan::int4x2_swapped_withoffset_t(0, 0)
                                   : kotekan::int4x2_swapped_withoffset_t(-8, -8);
        const std::ptrdiff_t freq_stride = std::ptrdiff_t(num_polarizations) * num_dishes;
        const std::ptrdiff_t time_stride = num_frequencies * freq_stride;
        for (int time = 0; time < num_times; time += 2)
            for (int freq = 0; freq < num_frequencies; freq += 4)
                if (loss.lost(frame_seq + time, freq))
                    for (int t = time; t < time + 2; ++t)
                        std::fill_n(frame + t * time_stride + freq * freq_stride, 4 * freq_stride,
                                    value);
    }

    // Overwrite the bad inputs in a [T, F, P, D] frame whose first sample is `frame_seq` with
    // random bytes: every int4 pair, including -8 (no data), is equally likely
    void overwrite_bad(kotekan::int4x2_swapped_withoffset_t* const frame,
                       const std::int64_t frame_seq) const {
        const std::ptrdiff_t num_inputs = std::ptrdiff_t(num_polarizations) * num_dishes;
        const std::ptrdiff_t time_stride = num_frequencies * num_inputs;
#pragma omp parallel for
        for (int time = 0; time < num_times; ++time) {
            std::vector<int> bad = bad_feeds.bad_inputs(frame_seq + time);
            bad.insert(bad.end(), missing_inputs.begin(), missing_inputs.end());
            for (int freq = 0; freq < num_frequencies; ++freq) {
                for (const int input : bad) {
                    const std::ptrdiff_t element = freq * num_inputs + input;
                    const std::uint8_t byte =
                        splitmix64(std::uint64_t(frame_seq + time) * time_stride + element) >> 56;
                    std::memcpy(&frame[time * time_stride + element], &byte, 1);
                }
            }
        }
    }

    void fill_sky(kotekan::int4x2_swapped_withoffset_t* const frame, const std::int64_t frame_seq,
                  const int frame_index) {
        const std::size_t frame_elements = buffer->frame_size;
        if (sky_pool.empty()) {
            const int num_pool_frames =
                config.get_default<int>(unique_name, "noise_period_frames", 4);
            INFO("Precomputing {:d} sky frames...", num_pool_frames);
            sky_pool.resize(num_pool_frames);
            for (int n = 0; n < num_pool_frames; ++n) {
                sky_pool[n].resize(frame_elements);
                sky->fill(sky_pool[n].data(), first_fpga_seq_num + std::int64_t(n) * num_times, 0,
                          num_times, true);
            }
        }
        std::memcpy(frame, sky_pool[frame_index % sky_pool.size()].data(), frame_elements);
        // Recompute the samples covered by a source's time window
        for (const auto& src : sky->get_sources()) {
            if (!src.windowed)
                continue;
            const std::int64_t begin = std::max(src.window_begin, frame_seq);
            const std::int64_t end = std::min(src.window_end, frame_seq + num_times);
            if (begin < end)
                sky->fill(frame, frame_seq, int(begin - frame_seq), int(end - frame_seq), false);
        }
    }
};

REGISTER_KOTEKAN_STAGE(inventVoltage);
