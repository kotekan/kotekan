#include "Config.hpp"   // for Config
#include "DataType.hpp" // for float16_t
#include "NDArray.hpp"
#include "Stage.hpp"              // for Stage
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "chordMetadata.hpp"      // for chordMetadata, get_chord_metadata
#include "frb1IntensityBound.hpp" // for frb1_intensity_bound, frb1_intensity_limit
#include "kotekanLogging.hpp"     // for DEBUG, FATAL_ERROR, INFO

#include "fmt.hpp" // for compile_string_to_view, format

#include <algorithm>  // for copy
#include <cassert>    // for assert
#include <cmath>      // for sqrt
#include <complex>    // for complex
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t
#include <functional> // for function
#include <limits>     // for numeric_limits
#include <memory>     // for allocator, __shared_ptr_access, shared_ptr
#include <optional>   // for optional
#include <string>     // for basic_string, string
#include <vector>     // for vector

/**
 * @class setFRB1Phase
 * @brief Produce the FRB1 beamforming weights `W` as a stream of frames.
 *
 * Each frame holds one set of weights, valid for `frb1_phase_lifetime_in_samples` FPGA samples.
 * Frame `k` is stamped with the FPGA sequence number `seq0 + k * frb1_phase_lifetime_in_samples`,
 * where `seq0` is the sequence number of the first frame of the `metadata_source` voltage
 * buffer(s): the FRB1 kernels locate weight element `k` at `k * lifetime` FPGA samples after the
 * logical beginning of their voltage ring buffer, so both streams have to share an origin.
 *
 * The frame has a leading length-1 time axis `TW` whose `dimscaling` is the lifetime; that is how
 * the GPU ring buffer learns how many FPGA samples one set of weights covers.
 *
 * The weights do not vary yet: every frame has the same content. Making them time dependent is
 * now a change to this stage alone.
 *
 * @par Buffers
 * @buffer frb1_phase      The weights, [TW=1][Fbar][P][dishN][dishM][C], float16
 * @buffer metadata_source A voltage buffer, or a list of them. Only the first frame of each is
 *                         read, for its `fpga_seq_num`; then this stage stops consuming it.
 *
 * @conf frb1_phase_lifetime_in_samples Int. How many FPGA samples one set of weights covers.
 */
class setFRB1Phase : public kotekan::Stage {
    // Telescope layout
    const int num_components = config.get<int>(unique_name, "num_components");
    const int num_dishes_x = Telescope::instance().get_grid_size_x();
    const int num_dishes_y = Telescope::instance().get_grid_size_y();
    // See calcFRB2Weights.cpp
    const bool frb1_swap_MN = config.get_default<bool>(unique_name, "frb1_swap_MN", false);
    const int num_dishes_M = frb1_swap_MN ? num_dishes_y : num_dishes_x;
    const int num_dishes_N = frb1_swap_MN ? num_dishes_x : num_dishes_y;
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");
    const int upchan_factor = config.get<int>(unique_name, "upchan_factor");
    // Note: The "channel" numbers here are "channel indices", not "channel ids".
    const int upchan_min_channel = config.get<int>(unique_name, "upchan_min_channel");
    const int upchan_max_channel = config.get<int>(unique_name, "upchan_max_channel");
    const int upchan_num_channels = upchan_max_channel - upchan_min_channel;
    const int upchan_max_num_channels = config.get<int>(unique_name, "upchan_max_num_channels");
    // The FRB1 kernels normalize the weights themselves, so this should be about 1
    const float frb1_input_scale = config.get_default<double>(unique_name, "frb1_input_scale", 1);

    // Each set of weights is valid for this many FPGA samples. It is the cadence at which this
    // stage produces frames, and the `dimscaling` of the weights' leading time axis.
    const std::int64_t frb1_phase_lifetime_in_samples =
        config.get<std::int64_t>(unique_name, "frb1_phase_lifetime_in_samples");

    Buffer* const frb1_phase_buffer;
    const std::vector<Buffer*> metadata_sources;

public:
    setFRB1Phase(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        frb1_phase_buffer(get_buffer("frb1_phase")),
        metadata_sources(get_buffer_or_array("metadata_source"))
    //
    {
        assert(upchan_min_channel >= 0);
        assert(upchan_max_channel >= upchan_min_channel);
        assert(upchan_num_channels <= upchan_max_num_channels);
        assert(frb1_phase_buffer);
        // `time_downsampling_fpga` is an `int`
        if (frb1_phase_lifetime_in_samples <= 0
            || frb1_phase_lifetime_in_samples > std::numeric_limits<int>::max())
            FATAL_ERROR("frb1_phase_lifetime_in_samples {:d} must be positive and fit into an int",
                        frb1_phase_lifetime_in_samples);
        if (metadata_sources.empty())
            FATAL_ERROR("metadata_source must name at least one buffer");

        frb1_phase_buffer->register_producer(unique_name);
        for (Buffer* const metadata_source : metadata_sources)
            metadata_source->register_consumer(unique_name);

        // The leading axis has length 1 and carries the lifetime as its `dimscaling`. Same shape
        // as the bad feed mask's "Tbf" axis and the baseband phase's "Tbb" axis.
        frb1_phase_buffer->require_frame_desc(kotekan::NDArray<float16_t, 6>::describe(
            "W",
            {1, upchan_max_num_channels * upchan_factor, num_polarizations, num_dishes_N,
             num_dishes_M, num_components},
            {"TW", "Fbar", "P", "dishN", "dishM", "C"},
            {frb1_phase_lifetime_in_samples, 1, 1, 1, 1, 1}));
    }

    virtual ~setFRB1Phase() {}

    void main_thread() override {
        if (stop_thread)
            return;

        // Check buffer size
        const std::ptrdiff_t frame_num_values = std::ptrdiff_t(upchan_max_num_channels)
                                                * upchan_factor * num_polarizations * num_dishes_N
                                                * num_dishes_M;
        assert(std::ptrdiff_t(frb1_phase_buffer->frame_size)
               == std::ptrdiff_t(sizeof(float16_t)) * frame_num_values * num_components);

        // The FPGA sequence number of the first frame. See the class documentation.
        const std::optional<std::int64_t> seq0 = wait_for_first_fpga_seq_num();
        if (!seq0)
            return;
        INFO("FRB1 weight FPGA sequence numbers start at {:d}, from the first frame of {:s}", *seq0,
             metadata_sources.at(0)->buffer_name);

        // Frequency metadata
        std::vector<int> coarse_freq(upchan_num_channels * upchan_factor);
        std::vector<int> freq_upchan_factor(upchan_num_channels * upchan_factor);
        std::vector<int> freq_upchan_index(upchan_num_channels * upchan_factor);
        for (int freq = 0; freq < upchan_num_channels; ++freq) {
            for (int upchan_index = 0; upchan_index < upchan_factor; ++upchan_index) {
                const int idx = upchan_index + upchan_factor * freq;
                coarse_freq.at(idx) = frequency_channels.at(upchan_min_channel + freq);
                freq_upchan_factor.at(idx) = upchan_factor;
                freq_upchan_index.at(idx) = upchan_index;
            }
        }

        // The weights. They do not vary (yet), so calculate them once.
        const std::ptrdiff_t str_dish_M = 1;
        const std::ptrdiff_t str_dish_N = str_dish_M * num_dishes_M;
        const std::ptrdiff_t str_polr = str_dish_N * num_dishes_N;
        const std::ptrdiff_t str_freq = str_polr * num_polarizations;
        std::vector<std::complex<float16_t>> weights(frame_num_values);
        for (int freq = 0; freq < upchan_max_num_channels * upchan_factor; ++freq) {
            for (int polr = 0; polr < num_polarizations; ++polr) {
                for (int dish_N = 0; dish_N < num_dishes_N; ++dish_N) {
                    for (int dish_M = 0; dish_M < num_dishes_M; ++dish_M) {
                        const std::ptrdiff_t idx = str_dish_M * dish_M + str_dish_N * dish_N
                                                   + str_polr * polr + str_freq * freq;
                        weights.at(idx) =
                            float16_t(freq < upchan_num_channels * upchan_factor ? frb1_input_scale
                                                                                 : 0.0 / 0.0);
                    }
                }
            }
        }

        // Ensure that the FRB1 kernel cannot overflow for these weights. (The unused
        // frequencies have NaN weights and are skipped.)
        for (int freq = 0; freq < upchan_num_channels * upchan_factor; ++freq) {
            const double bound = kotekan::frb1_intensity_bound(
                reinterpret_cast<const float16_t*>(&weights.at(str_freq * freq)), num_polarizations,
                num_dishes_M, num_dishes_N);
            if (bound > kotekan::frb1_intensity_limit)
                FATAL_ERROR(
                    "The FRB1 weights can overflow Float16: frequency {:d} has a worst-case "
                    "intensity of {:g}, above the limit {:g}. Reduce frb1_input_scale by at "
                    "least a factor {:.3g}.",
                    freq, bound, kotekan::frb1_intensity_limit,
                    std::sqrt(bound / kotekan::frb1_intensity_limit));
        }

        const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
            frb1_phase_buffer->get_frame_desc<kotekan::GenericNDArray>();

        // `frame_index` counts all frames produced, not just the current slot: it must keep
        // increasing so that the weights form one continuous stream.
        for (std::int64_t frame_index = 0; !stop_thread; ++frame_index) {
            const int frame_id = frame_index % frb1_phase_buffer->num_frames;

            // Wait for buffer. Blocking here paces production to the consumers.
            DEBUG("[{:s}/{:d}] Waiting for buffer...", frb1_phase_buffer->buffer_name, frame_index);
            std::complex<float16_t>* const frb1_phase_frame = static_cast<std::complex<float16_t>*>(
                static_cast<void*>(frb1_phase_buffer->wait_for_empty_frame(unique_name, frame_id)));
            if (!frb1_phase_frame)
                return;

            // Set metadata
            frb1_phase_buffer->allocate_new_metadata_object(frame_id);
            const auto& frb1_phase_meta =
                get_chord_metadata(frb1_phase_buffer->get_metadata(frame_id));
            frb1_phase_meta->set_from_frame_desc(frame_desc);
            frb1_phase_meta->set_fpga_seq_num(*seq0 + frame_index * frb1_phase_lifetime_in_samples);
            frb1_phase_meta->set_time_downsampling_fpga(int(frb1_phase_lifetime_in_samples));
            frb1_phase_meta->set_coarse_freq(coarse_freq);
            frb1_phase_meta->set_freq_upchan_factor(freq_upchan_factor);
            frb1_phase_meta->set_freq_upchan_index(freq_upchan_index);
            frb1_phase_meta->check_frame_desc(frame_desc);

            // Set buffer
            std::copy(weights.begin(), weights.end(), frb1_phase_frame);

            // Mark buffer as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", frb1_phase_buffer->buffer_name,
                  frame_index);
            frb1_phase_buffer->mark_frame_full(unique_name, frame_id);
        }
    }

private:
    // Read the FPGA sequence number of the first frame of each metadata source, then stop being
    // a consumer so that the producers do not wait for us on the frames after it. The sources
    // must agree. Returns nothing if we are shutting down.
    std::optional<std::int64_t> wait_for_first_fpga_seq_num() {
        std::optional<std::int64_t> seq0;
        for (Buffer* const metadata_source : metadata_sources) {
            if (metadata_source->wait_for_full_frame(unique_name, 0) == nullptr)
                return std::nullopt;
            const std::shared_ptr<const chordMetadata> meta =
                get_chord_metadata(metadata_source, 0);
            if (!meta->has_fpga_seq_num())
                FATAL_ERROR("metadata_source {:s} has no fpga_seq_num, needed to start the FRB1 "
                            "weight sequence numbers",
                            metadata_source->buffer_name);
            const std::int64_t source_seq0 = meta->get_fpga_seq_num();
            if (seq0 && source_seq0 != *seq0)
                FATAL_ERROR("The metadata sources disagree on the first FPGA sequence number: "
                            "{:s} starts at {:d}, {:s} at {:d}",
                            metadata_sources.at(0)->buffer_name, *seq0,
                            metadata_source->buffer_name, source_seq0);
            seq0 = source_seq0;
            metadata_source->mark_frame_empty(unique_name, 0);
            metadata_source->unregister_consumer(unique_name);
        }
        return seq0;
    }
};

REGISTER_KOTEKAN_STAGE(setFRB1Phase);
