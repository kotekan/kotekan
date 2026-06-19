#include "Config.hpp"                   // for Config
#include "DataType.hpp"                 // for float16_t
#include "Stage.hpp"                    // for Stage
#include "StageFactory.hpp"             // for REGISTER_KOTEKAN_STAGE
#include "UpchannelizationSchedule.hpp" // for UpchannelizationSchedule
#include "buffer.hpp"                   // for Buffer
#include "bufferContainer.hpp"          // for bufferContainer
#include "chordMetadata.hpp"            // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"           // for DEBUG

#include "fmt.hpp" // for compile_string_to_view, format

#include <cassert>    // for assert
#include <cstddef>    // for ptrdiff_t
#include <functional> // for function
#include <memory>     // for allocator, __shared_ptr_access, shared_ptr
#include <set>        // for operator!=, set
#include <string>     // for basic_string, string
#include <unistd.h>   // for sleep
#include <vector>     // for vector

class setUpchanGain : public kotekan::Stage {
    const std::string upchannelization_schedule_name =
        config.get_default<std::string>(unique_name, "upchannelization_schedule_name", "");
    const int upchan_factor = config.get<int>(unique_name, "upchan_factor");
    const int upchan_max_num_channels = config.get<int>(unique_name, "upchan_max_num_channels");
    const std::vector<double> upchan_gain =
        config.get<std::vector<double>>(unique_name, "upchan_gain");

    Buffer* const upchan_gain_buffer;

public:
    setUpchanGain(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        upchan_gain_buffer(get_buffer("upchan_gain_buffer"))
    //
    {
        // We use the same gain for all channels
        assert(std::ptrdiff_t(upchan_gain.size()) == upchan_factor);
        assert(upchan_gain_buffer);

        // Check buffer size
        assert(upchan_gain_buffer->frame_size
               == sizeof(float16_t) * upchan_max_num_channels * upchan_factor);

        upchan_gain_buffer->register_producer(unique_name);
    }

    virtual ~setUpchanGain() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Upchannelization schedule
        const auto& upchan_schedule =
            UpchannelizationSchedule::instance(config, upchannelization_schedule_name);

        // Wait for buffer
        DEBUG("[{:s}/{:d}] Waiting for buffer...", upchan_gain_buffer->buffer_name, frame_index);
        float16_t* const upchan_gain_frame = static_cast<float16_t*>(
            static_cast<void*>(upchan_gain_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!upchan_gain_frame)
            return;

        // Set metadata
        upchan_gain_buffer->allocate_ndarray_frame_desc<float16_t, 1>(
            "G", {upchan_max_num_channels * upchan_factor}, {"Fbar"}, {1});
        upchan_gain_buffer->allocate_new_metadata_object(frame_id);
        const auto& upchan_gain_meta =
            get_chord_metadata(upchan_gain_buffer->get_metadata(frame_id));
        upchan_gain_meta->set_from_frame_desc(upchan_gain_buffer->get_ndarray_frame_desc());
        upchan_gain_meta->set_fpga_seq_num(0);           // ???
        upchan_gain_meta->set_time_downsampling_fpga(1); // ???
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
        upchan_gain_meta->set_coarse_freq(coarse_freq);
        upchan_gain_meta->set_freq_upchan_factor(freq_upchan_factor);
        upchan_gain_meta->set_freq_upchan_index(freq_upchan_index);

        // Set buffer
        for (int freq = 0; freq < upchan_max_num_channels; ++freq) {
            for (int upchan_index = 0; upchan_index < upchan_factor; ++upchan_index) {
                const int idx = upchan_index + upchan_factor * freq;
                assert(idx >= 0
                       && idx < std::ptrdiff_t(upchan_gain_buffer->frame_size
                                               / sizeof *upchan_gain_frame));
                upchan_gain_frame[idx] = float16_t(
                    freq < upchan_num_channels ? upchan_gain.at(upchan_index) : 0.0 / 0.0);
            }
        }

        // Mark buffers as full
        DEBUG("[{:s}/{:d}] Marking buffer as full...", upchan_gain_buffer->buffer_name,
              frame_index);
        upchan_gain_buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(setUpchanGain);
