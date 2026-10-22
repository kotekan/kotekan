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

static std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/**
 * @brief Invent CHIME voltages, one buffer per frequency, and optionally the lost samples.
 *
 * @conf lost_samples       List of buffers. Optional. One buffer per group of 4 frequencies,
 *                          with one byte per time sample, 1 if lost, as dpdkCore writes them.
 *                          A group's frame is marked full after the voltages of its 4
 *                          frequencies are written, so that `zeroSamples` can then zero the
 *                          lost samples.
 * @conf lost_sample_rate   Double. Default 0. Probability that a packet of a group is lost,
 *                          from a hash of `lost_sample_seed`, the packet and the group. A
 *                          packet holds 2 time samples, as from the ICEboards.
 * @conf lost_sample_seed   Int. Default 0.
 * @conf lost_sample_boxes  List of [begin, end) FPGA sample ranges that are lost in every group.
 *                          Default empty.
 */
class inventVoltageCHIME : public kotekan::Stage {
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");
    const int num_frequencies = config.get<int>(unique_name, "num_frequencies");
    const int num_times = config.get<int>(unique_name, "num_times");

    const std::string input_kind = config.get<std::string>(unique_name, "input_kind");
    const bool reuse_frames = config.get_default<bool>(unique_name, "reuse_frames", false);

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    const int out_frame_first = config.get_default<int>(unique_name, "out_frame_first", 10);
    const int out_frame_every = config.get_default<int>(unique_name, "out_frame_every", 10);

    const double lost_sample_rate =
        config.get_default<double>(unique_name, "lost_sample_rate", 0.0);
    const std::uint64_t lost_sample_seed =
        config.get_default<std::uint64_t>(unique_name, "lost_sample_seed", 0);
    const std::vector<std::vector<std::int64_t>> lost_sample_boxes =
        config.get_default<std::vector<std::vector<std::int64_t>>>(unique_name, "lost_sample_boxes",
                                                                   {});

    std::vector<Buffer*> buffers;
    std::vector<Buffer*> lost_samples_buffers;

    // Whether sample `seq` of frequency group `group` is lost
    bool lost(const std::int64_t seq, const int group) const {
        for (const auto& box : lost_sample_boxes)
            if (seq >= box.at(0) && seq < box.at(1))
                return true;
        // Compare the top 53 bits of the hash of the packet with the rate
        const std::uint64_t hash =
            splitmix64(lost_sample_seed ^ splitmix64(std::uint64_t(seq / 2) * 4 + group));
        return double(hash >> 11) * 0x1.0p-53 < lost_sample_rate;
    }

    // Write and mark full the lost samples of group `group` for frame `frame_index`
    bool write_lost_samples(const int group, const int frame_index) {
        Buffer* const buffer = lost_samples_buffers.at(group);
        const int frame_id = frame_index % buffer->num_frames;
        std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
        if (!frame)
            return false;
        const std::int64_t frame_seq = std::int64_t(frame_index) * num_times;

        buffer->allocate_new_metadata_object(frame_id);
        const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
        meta->set_fpga_seq_num(frame_seq);
        meta->set_from_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
        meta->set_time_downsampling_fpga(1);
        meta->set_coarse_freq(std::vector<int>(frequency_channels.begin() + 4 * group,
                                               frequency_channels.begin() + 4 * group + 4));
        meta->set_freq_upchan_factor(std::vector<int>(4, 1));
        meta->set_freq_upchan_index(std::vector<int>(4, 0));

        for (int time = 0; time < num_times; ++time)
            frame[time] = lost(frame_seq + time, group);

        buffer->mark_frame_full(unique_name, frame_id);
        return true;
    }

public:
    inventVoltageCHIME(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {
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

        if (config.exists(unique_name, "lost_samples")) {
            lost_samples_buffers = get_buffer_array("lost_samples");
            if (4 * int(lost_samples_buffers.size()) != num_frequencies)
                FATAL_ERROR("`lost_samples` lists {:d} buffers for {:d} frequencies",
                            lost_samples_buffers.size(), num_frequencies);
            for (Buffer* const buffer : lost_samples_buffers)
                buffer->register_producer(unique_name);
        }
        // `zeroSamples` zeroes lost samples in place, which a reused frame would keep
        if (reuse_frames && (lost_sample_rate > 0 || !lost_sample_boxes.empty()))
            FATAL_ERROR("Lost samples require `reuse_frames: false`");
        for (const auto& box : lost_sample_boxes)
            if (box.size() != 2)
                FATAL_ERROR("`lost_sample_boxes` entries must be [begin, end)");
    }

    virtual ~inventVoltageCHIME() {}

    void main_thread() override {
        [[maybe_unused]] std::uint32_t rng_state = 0x0ad42ea5; // must not be 0

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
                // `zeroSamples` adds the samples it zeroes
                if (!lost_samples_buffers.empty())
                    meta->set_lost_timesamples(0);

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

                // The voltages of a group of 4 frequencies are written
                if (!lost_samples_buffers.empty() && frequency % 4 == 3)
                    if (!write_lost_samples(frequency / 4, frame_index))
                        return;
            } // for buffer
        } // for frame
    }
};

REGISTER_KOTEKAN_STAGE(inventVoltageCHIME);
