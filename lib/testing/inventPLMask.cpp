#include "Config.hpp"            // for Config
#include "DataType.hpp"          // for string_to_type, DataType
#include "NDArray.hpp"           // for NDArray, GenericNDArray
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"            // for Symbol
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, metadata_is_chord, get_c...
#include "div.hpp"               // for div_noremainder
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, INFO
#include "metadata.hpp"          // for metadataObject
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "testSkySignal.hpp"     // for testPacketLoss
#include "visUtil.hpp"           // for current_time

#include <algorithm>  // for copy
#include <array>      // for array
#include <cassert>    // for assert
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t, uint8_t
#include <fmt.hpp>    // for compile_string_to_view
#include <functional> // for function
#include <iomanip>    // for operator<<, setfill, setw
#include <memory>     // for allocator, shared_ptr, __shared_ptr_access
#include <sstream>    // for basic_ostream, operator<<, basic_ostrin...
#include <string>     // for basic_string, char_traits, string, oper...
#include <unistd.h>   // for gethostname, sleep
#include <vector>     // for vector

using kotekan::div_noremainder;

/**
 * @brief Generate a packet loss (PL) mask for tests.
 *
 * The FPGA samples [box_begin, box_end) are lost, and those that `testPacketLoss` describes
 * (random packet loss and loss boxes, the pattern whose voltages `inventVoltage` can overwrite).
 * Lost samples are masked for all dishes and polarizations.
 *
 * @conf first_fpga_seq_num  Int. FPGA sequence number of the first sample. Default 0.
 * @conf box_begin, box_end  Int. FPGA samples, multiples of 2. Default: no loss.
 * @conf loss_boxes, loss_rate, loss_seed  See `testPacketLoss`.
 */
class inventPLMask : public kotekan::Stage {
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");
    const int num_frequencies = config.get<int>(unique_name, "num_frequencies");
    const int num_times = config.get<int>(unique_name, "num_times");

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    // FPGA sequence number of the first sample; must agree between the voltage and mask
    // generators
    const std::int64_t first_fpga_seq_num =
        config.get_default<std::int64_t>(unique_name, "first_fpga_seq_num", 0);

    // The FPGA samples [box_begin, box_end) are lost, and those that `loss` says
    const std::int64_t box_begin = config.get_default<std::int64_t>(unique_name, "box_begin", 0);
    const std::int64_t box_end = config.get_default<std::int64_t>(unique_name, "box_end", 0);
    const testPacketLoss loss{config, unique_name};

    Buffer* const buffer;

public:
    inventPLMask(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("pl_mask")) {
        assert(buffer);
        buffer->register_producer(unique_name);
        if (frequency_channels.size() != static_cast<size_t>(num_frequencies))
            FATAL_ERROR("inventPLMask: frequency_channels has {:d} entries, expected "
                        "num_frequencies={:d}",
                        frequency_channels.size(), num_frequencies);
        // Each bit of the mask covers 2 FPGA samples
        if (!(box_begin % 2 == 0 && box_end % 2 == 0))
            FATAL_ERROR("box_begin={:d} and box_end={:d} must be even", box_begin, box_end);
    }

    virtual ~inventPLMask() {}

    void main_thread() override {
        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

            if (stop_thread)
                return;

            // Wait for buffer
            DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
            std::int8_t* const frame = static_cast<std::int8_t*>(
                static_cast<void*>(buffer->wait_for_empty_frame(unique_name, frame_id)));
            if (!frame)
                return;

            // Set metadata
            buffer->require_frame_desc(kotekan::NDArray<kotekan::uint1x8_t, 5>::describe(
                "pl_mask",
                {div_noremainder(num_times, 2 * 64), div_noremainder(num_frequencies, 4),
                 num_polarizations, div_noremainder(num_dishes, 8), 64 / 8},
                {"T2hi64", "F4", "P", "D8", "T2lo64"}, {128, 4, 1, 8, 16}));
            buffer->allocate_new_metadata_object(frame_id);
            const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
            meta->set_from_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
            meta->set_fpga_seq_num(first_fpga_seq_num + std::int64_t(frame_index) * num_times);
            meta->set_time_downsampling_fpga(2 * 64);

            // Set frequency information. As in `testLostSamplesToPLMask`, this lists every
            // frequency, not one entry per `F4` bin; consumers such as `N2Accumulate` compare
            // it against the voltage frequencies.
            const std::vector<int>& coarse_freq = frequency_channels;
            const std::vector<int> freq_upchan_factor(coarse_freq.size(), 1);
            const std::vector<int> freq_upchan_index(coarse_freq.size(), 0);
            meta->set_coarse_freq(coarse_freq);
            meta->set_freq_upchan_factor(freq_upchan_factor);
            meta->set_freq_upchan_index(freq_upchan_index);

            // Fill buffer
            DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
            const std::ptrdiff_t str4 = 1;
            const std::ptrdiff_t str3 = str4 * (64 / 8);
            const std::ptrdiff_t str2 = str3 * div_noremainder(num_dishes, 8);
            const std::ptrdiff_t str1 = str2 * num_polarizations;
            const std::ptrdiff_t str0 = str1 * div_noremainder(num_frequencies, 4);
            // Each byte holds 8 bits, the first sample pair in the least significant bit
            for (int time = 0; time < num_times; time += 16) {
                for (int freq = 0; freq < num_frequencies; freq += 4) {
                    std::uint8_t valid = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const std::int64_t seq = first_fpga_seq_num
                                                 + std::int64_t(frame_index) * num_times + time
                                                 + 2 * bit;
                        const bool lost =
                            (seq >= box_begin && seq < box_end) || loss.lost(seq, freq);
                        valid |= std::uint8_t(!lost) << bit;
                    }
                    for (int polr = 0; polr < num_polarizations; ++polr) {
                        for (int dish = 0; dish < num_dishes; dish += 8) {
                            const int idx4 = time / 2 % 64 / 8;
                            const int idx3 = dish / 8;
                            const int idx2 = polr;
                            const int idx1 = freq / 4;
                            const int idx0 = time / 2 / 64;
                            const std::ptrdiff_t idx =
                                str4 * idx4 + str3 * idx3 + str2 * idx2 + str1 * idx1 + str0 * idx0;
                            assert(idx >= 0 && idx < std::ptrdiff_t(buffer->frame_size));
                            frame[idx] = valid;
                        }
                    }
                }
            }

            // Mark buffer as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
            buffer->mark_frame_full(unique_name, frame_id);
        }
    }
};

REGISTER_KOTEKAN_STAGE(inventPLMask);
