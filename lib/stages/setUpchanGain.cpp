#include "Config.hpp"   // for Config
#include "DataType.hpp" // for float16_t
#include "NDArray.hpp"
#include "Stage.hpp"                    // for Stage
#include "StageFactory.hpp"             // for REGISTER_KOTEKAN_STAGE
#include "UpchannelizationSchedule.hpp" // for UpchannelizationSchedule, wait_for_coarse_freq
#include "buffer.hpp"                   // for Buffer
#include "bufferContainer.hpp"          // for bufferContainer
#include "chordMetadata.hpp"            // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"           // for DEBUG, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view, format

#include <cassert>    // for assert
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t
#include <functional> // for function
#include <memory>     // for allocator, __shared_ptr_access, shared_ptr
#include <set>        // for operator!=, set
#include <string>     // for basic_string, string
#include <vector>     // for vector

/**
 * @class setUpchanGain
 * @brief Produce the upchannelizer's gains, one gain vector per gain lifetime.
 *
 * Emits one frame every @c upchan_gain_lifetime_in_samples FPGA samples. Frame @c k covers the
 * FPGA samples starting at @c seq0 + k * upchan_gain_lifetime_in_samples, where @c seq0 is the
 * sequence number of the first frame of the first @c metadata_source: the upchannelizer
 * locates gain element @c k at @c k * lifetime samples after the logical beginning of its
 * voltage ring buffer, so both streams have to share an origin. The frames have a leading
 * length-1 @c TG axis whose @c dimscaling is the lifetime; that is how the GPU ring buffer
 * learns how many samples one gain vector covers.
 *
 * The gains do not change yet, so every frame has the same content (apart from the optional
 * test knob @c upchan_gain_scale_cycle). The point is the plumbing: making them time dependent
 * is now a change to this stage alone.
 *
 * @conf upchannelization_schedule_name  String. Config path of the schedule, default "".
 * @conf upchan_factor                   Int. The upchannelization factor @c U.
 * @conf upchan_max_num_channels         Int. Coarse channels the gain buffer has room for.
 * @conf upchan_gain                     Vector of @c U doubles. Gain per fine channel.
 * @conf upchan_gain_lifetime_in_samples Int. FPGA samples one gain vector covers.
 * @conf upchan_gain_scale_cycle         Vector of doubles, default [1]. For testing: the gains
 *                                       of frame @c k are multiplied by element
 *                                       @c k % size of this vector, so that a consumer
 *                                       applying the wrong gain element is detectable.
 * @conf upchan_gain_buffer              Buffer. The output.
 * @conf metadata_source                 Buffer or list of buffers. Voltage buffers providing
 *                                       the coarse frequencies and the stream origin.
 */
class setUpchanGain : public kotekan::Stage {
    const std::string upchannelization_schedule_name =
        config.get_default<std::string>(unique_name, "upchannelization_schedule_name", "");
    const int upchan_factor = config.get<int>(unique_name, "upchan_factor");
    const int upchan_max_num_channels = config.get<int>(unique_name, "upchan_max_num_channels");
    const std::vector<double> upchan_gain =
        config.get<std::vector<double>>(unique_name, "upchan_gain");
    const std::int64_t upchan_gain_lifetime_in_samples =
        config.get<std::int64_t>(unique_name, "upchan_gain_lifetime_in_samples");
    const std::vector<double> upchan_gain_scale_cycle = config.get_default<std::vector<double>>(
        unique_name, "upchan_gain_scale_cycle", std::vector<double>{1.0});

    Buffer* const upchan_gain_buffer;
    const std::vector<Buffer*> metadata_sources;

public:
    setUpchanGain(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        upchan_gain_buffer(get_buffer("upchan_gain_buffer")),
        metadata_sources(get_buffer_or_array("metadata_source"))
    //
    {
        // We use the same gain for all channels
        assert(std::ptrdiff_t(upchan_gain.size()) == upchan_factor);
        assert(upchan_gain_buffer);

        if (upchan_gain_lifetime_in_samples <= 0)
            FATAL_ERROR("upchan_gain_lifetime_in_samples must be positive, not {:d}",
                        upchan_gain_lifetime_in_samples);
        if (upchan_gain_scale_cycle.empty())
            FATAL_ERROR("upchan_gain_scale_cycle must not be empty");

        // Check buffer size
        assert(upchan_gain_buffer->frame_size
               == sizeof(float16_t) * upchan_max_num_channels * upchan_factor);

        upchan_gain_buffer->register_producer(unique_name);
        for (Buffer* const metadata_source : metadata_sources)
            metadata_source->register_consumer(unique_name);

        // The leading axis has length 1 and carries the lifetime as its `dimscaling`. Same
        // shape as the baseband phase matrix's "Tbb" axis.
        upchan_gain_buffer->require_frame_desc(kotekan::NDArray<float16_t, 2>::describe(
            "G", {1, upchan_max_num_channels * upchan_factor}, {"TG", "Fbar"},
            {upchan_gain_lifetime_in_samples, 1}));
    }

    virtual ~setUpchanGain() {}

    void main_thread() override {
        if (stop_thread)
            return;

        // Upchannelization schedule
        // The coarse frequency channels handled by this GPU. These are local to
        // a GPU and thus cannot come from the configuration, which is the same
        // for every GPU. The same frame also tells us where the voltage stream begins.
        std::int64_t seq0 = -1;
        const auto local_coarse_freq = wait_for_coarse_freq(metadata_sources, unique_name, &seq0);
        if (!local_coarse_freq)
            return;
        assert(seq0 >= 0);

        const UpchannelizationSchedule upchan_schedule(config, upchannelization_schedule_name,
                                                       *local_coarse_freq, unique_name);

        const auto& upchan_channels_set = upchan_schedule.get_upchan_channels(upchan_factor);
        const std::vector<int> upchan_channels(upchan_channels_set.begin(),
                                               upchan_channels_set.end());
        const int upchan_num_channels = int(upchan_channels.size());
        assert(upchan_num_channels <= upchan_max_num_channels);
        std::vector<int> coarse_freq(upchan_num_channels * upchan_factor);
        std::vector<int> freq_upchan_factor(upchan_num_channels * upchan_factor);
        std::vector<int> freq_upchan_index(upchan_num_channels * upchan_factor);
        for (int freq = 0; freq < upchan_num_channels; ++freq) {
            for (int upchan_index = 0; upchan_index < upchan_factor; ++upchan_index) {
                const int idx = upchan_index + upchan_factor * freq;
                coarse_freq.at(idx) = upchan_channels.at(freq);
                freq_upchan_factor.at(idx) = upchan_factor;
                freq_upchan_index.at(idx) = upchan_index;
            }
        }

        // `frame_index` counts all frames produced, not just the current slot: it must keep
        // increasing so that the gains form one continuous stream.
        for (std::int64_t frame_index = 0; !stop_thread; ++frame_index) {
            const int frame_id = frame_index % upchan_gain_buffer->num_frames;

            // Wait for buffer. Blocking here paces production to the consumers.
            DEBUG("[{:s}/{:d}] Waiting for buffer...", upchan_gain_buffer->buffer_name,
                  frame_index);
            float16_t* const upchan_gain_frame = static_cast<float16_t*>(static_cast<void*>(
                upchan_gain_buffer->wait_for_empty_frame(unique_name, frame_id)));
            if (!upchan_gain_frame)
                return;

            // Set metadata
            upchan_gain_buffer->allocate_new_metadata_object(frame_id);
            const auto& upchan_gain_meta =
                get_chord_metadata(upchan_gain_buffer->get_metadata(frame_id));
            upchan_gain_meta->set_from_frame_desc(
                upchan_gain_buffer->get_frame_desc<kotekan::GenericNDArray>());
            upchan_gain_meta->set_fpga_seq_num(seq0
                                               + frame_index * upchan_gain_lifetime_in_samples);
            upchan_gain_meta->set_time_downsampling_fpga(upchan_gain_lifetime_in_samples);
            upchan_gain_meta->set_coarse_freq(coarse_freq);
            upchan_gain_meta->set_freq_upchan_factor(freq_upchan_factor);
            upchan_gain_meta->set_freq_upchan_index(freq_upchan_index);
            upchan_gain_meta->check_frame_desc(
                upchan_gain_buffer->get_frame_desc<kotekan::GenericNDArray>());

            // Set buffer
            const double scale =
                upchan_gain_scale_cycle.at(frame_index % upchan_gain_scale_cycle.size());
            for (int freq = 0; freq < upchan_max_num_channels; ++freq) {
                for (int upchan_index = 0; upchan_index < upchan_factor; ++upchan_index) {
                    const int idx = upchan_index + upchan_factor * freq;
                    assert(idx >= 0
                           && idx < std::ptrdiff_t(upchan_gain_buffer->frame_size
                                                   / sizeof *upchan_gain_frame));
                    upchan_gain_frame[idx] =
                        float16_t(freq < upchan_num_channels ? scale * upchan_gain.at(upchan_index)
                                                             : 0.0 / 0.0);
                }
            }

            // Mark buffers as full
            DEBUG("[{:s}/{:d}] Marking buffer as full...", upchan_gain_buffer->buffer_name,
                  frame_index);
            upchan_gain_buffer->mark_frame_full(unique_name, frame_id);
        }
    }
};

REGISTER_KOTEKAN_STAGE(setUpchanGain);
