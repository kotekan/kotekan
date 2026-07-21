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
#include "visUtil.hpp"           // for current_time

#include <algorithm>  // for copy
#include <array>      // for array
#include <cassert>    // for assert
#include <cmath>      // for fmax, fmin, sqrt
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t, uint8_t
#include <cstring>    //
#include <fmt.hpp>    // for compile_string_to_view
#include <functional> // for function
#include <iomanip>    // for operator<<, setfill, setw
#include <memory>     // for allocator, shared_ptr, __shared_ptr_access
#ifdef WITH_OMP
#include <omp.h> // for omp_get_wtime
#endif
#include <sstream>    // for basic_ostream, operator<<, basic_ostrin...
#include <string>     // for basic_string, char_traits, string, oper...
#include <unistd.h>   // for gethostname, sleep
#include <vector>     // for vector

[[maybe_unused]] static std::uint32_t xorshift32(std::uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

class inventVoltageCHIME : public kotekan::Stage {
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");
    const int num_frequencies = config.get<int>(unique_name, "num_frequencies");
    const int num_times = config.get<int>(unique_name, "num_times");

    const std::string input_kind = config.get<std::string>(unique_name, "input_kind");
    const bool reuse_frames = config.get_default<bool>(unique_name, "reuse_frames", false);

    const std::vector<std::string> startup_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "startup_buffer_names");

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    const int out_frame_first = config.get_default<int>(unique_name, "out_frame_first", 10);
    const int out_frame_every = config.get_default<int>(unique_name, "out_frame_every", 10);

    std::vector<Buffer*> startup_buffers;

    std::vector<Buffer*> buffers;

public:
    inventVoltageCHIME(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {
        for (const auto& startup_buffer_name : startup_buffer_names) {
            auto buffer = buffer_container.get_buffer(startup_buffer_name);
            startup_buffers.push_back(buffer);
            buffer->register_consumer(unique_name);
        }

        for (int frequency = 0; frequency < num_frequencies; ++frequency) {
            std::ostringstream buf;
            buf << "voltage_" << frequency;
            const auto name = buf.str();
            Buffer* const buffer = get_buffer(name);
            assert(buffer);
            buffers.push_back(buffer);
            buffer->register_producer(unique_name);

            // Set metadata
            buffer->require_frame_desc(
                kotekan::NDArray<kotekan::int4x2_swapped_withoffset_t, 3>::describe(
                    "E", {1, num_times, num_polarizations * num_dishes}, {"F", "T", "E"},
                    {1, 1, 1}));
        }
    }

    virtual ~inventVoltageCHIME() {}

    void main_thread() override {
        [[maybe_unused]] std::uint32_t rng_state = 0x0ad42ea5; // must not be 0

        // Wait for startup to finish
        INFO("Waiting for startup to finish:");
        for (const auto& startup_buffer : startup_buffers) {
            INFO("Waiting for startup buffer {:s}...", startup_buffer->buffer_name);
            startup_buffer->wait_for_full_frame(unique_name, 0);
            startup_buffer->mark_frame_empty(unique_name, 0);
        }
        INFO("Done waiting for startup to finish.");

        double time_min = 1.0 / 0.0;
        double time_max = 0.0 / 0.0;
        double time_count = 0.0;
        double time_sum = 0.0;
        double time_sum2 = 0.0;

        double time0 = current_time();

        for (int frame_index = 0;; ++frame_index) {

            if (stop_thread)
                return;

            for (int frequency = 0; frequency < num_frequencies; ++frequency) {
                const auto buffer = buffers.at(frequency);
                const int frame_id = frame_index % buffer->num_frames;

                // Wait for buffer
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                kotekan::int4x2_swapped_withoffset_t* const frame =
                    static_cast<kotekan::int4x2_swapped_withoffset_t*>(
                        static_cast<void*>(buffer->wait_for_empty_frame(unique_name, frame_id)));
                if (!frame)
                    return;

                if (frequency == 0) {
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
                            double time_sdv =
                                sqrt(fmax(0.0, time_sum2 / time_count - time_avg * time_avg));
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
                }

                buffer->allocate_new_metadata_object(frame_id);
                const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
                meta->set_fpga_seq_num(frame_index * num_times);
                meta->set_from_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
                meta->set_time_downsampling_fpga(1);

                // Set frequency information
                const std::vector<int> coarse_freq{frequency_channels.at(frequency)};
                const std::vector<int> freq_upchan_factor{1};
                const std::vector<int> freq_upchan_index{0};
                meta->set_coarse_freq(coarse_freq);
                meta->set_freq_upchan_factor(freq_upchan_factor);
                meta->set_freq_upchan_index(freq_upchan_index);

                // Fill buffer
                DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
                if (!reuse_frames || frame_index < buffer->num_frames) {
                    if (input_kind == "random") {
                        const std::ptrdiff_t str2 = 1;
                        const std::ptrdiff_t str1 = str2 * num_dishes * num_polarizations;
                        const std::ptrdiff_t str0 = str1 * 1;
                        for (int freq = 0; freq < 1; ++freq) {
                            for (int time = 0; time < num_times; ++time) {
                                for (int elem = 0; elem < num_polarizations * num_dishes; ++elem) {
                                    const int idx2 = elem;
                                    const int idx1 = time;
                                    const int idx0 = freq;
                                    const std::ptrdiff_t idx =
                                        str2 * idx2 + str1 * idx1 + str0 * idx0;
                                    assert(idx >= 0 && idx < std::ptrdiff_t(buffer->frame_size));
                                    const int val0 = int(rng_state % 15) - 7;
                                    const int val1 = int(rng_state / 16 % 15) - 7;
                                    frame[idx] = kotekan::int4x2_swapped_withoffset_t(val0, val1);
                                    rng_state = xorshift32(rng_state);
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
                    } else {
                        FATAL_ERROR("Unknown value for `input_kind`: \"{:s}\"", input_kind);
                    }
                }

                // Mark buffer as full
                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);
            } // for buffer
        } // for frame
    }
};

REGISTER_KOTEKAN_STAGE(inventVoltageCHIME);
