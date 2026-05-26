#include <unistd.h>             // for sleep
#include <cassert>              // for assert
#include <string>               // for basic_string, string
#include <vector>               // for vector
#include <complex>              // for complex
#include <cstddef>              // for ptrdiff_t
#include <functional>           // for function
#include <memory>               // for allocator, __shared_ptr_access, shared_ptr

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"   // for DEBUG
#include "DataType.hpp"         // for float16_t
#include "fmt.hpp"              // for compile_string_to_view, format

class setFRB1Phase : public kotekan::Stage {
    // Telescope layout
    const int num_components = config.get<int>(unique_name, "num_components");
    const int num_dishes_x = config.get<int>(unique_name, "num_dishes_x");
    const int num_dishes_y = config.get<int>(unique_name, "num_dishes_y");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");
    const int upchan_factor = config.get<int>(unique_name, "upchan_factor");
    // Note: The "channel" numbers here are "channel indices", not "channel ids".
    const int upchan_min_channel = config.get<int>(unique_name, "upchan_min_channel");
    const int upchan_max_channel = config.get<int>(unique_name, "upchan_max_channel");
    const int upchan_num_channels = upchan_max_channel - upchan_min_channel;
    const int upchan_max_num_channels = config.get<int>(unique_name, "upchan_max_num_channels");
    const float frb1_input_scale = config.get<double>(unique_name, "frb1_input_scale");

    Buffer* const frb1_phase_buffer;

public:
    setFRB1Phase(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        frb1_phase_buffer(get_buffer("frb1_phase"))
    //
    {
        assert(upchan_min_channel >= 0);
        assert(upchan_max_channel >= upchan_min_channel);
        assert(upchan_num_channels <= upchan_max_num_channels);
        assert(frb1_phase_buffer);
        frb1_phase_buffer->register_producer(unique_name);
    }

    virtual ~setFRB1Phase() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Wait for buffer
        DEBUG("[{:s}/{:d}] Waiting for buffer...", frb1_phase_buffer->buffer_name, frame_index);
        std::complex<float16_t>* const frb1_phase_frame = static_cast<std::complex<float16_t>*>(
            static_cast<void*>(frb1_phase_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!frb1_phase_frame)
            return;

        // Check buffer size
        assert(frb1_phase_buffer->frame_size
               == sizeof(float16_t) * upchan_max_num_channels * upchan_factor * num_polarizations
                      * num_dishes_y * num_dishes_x * num_components);

        // Set metadata
        frb1_phase_buffer->allocate_ndarray_frame_desc<float16_t, 5>(
            "W",
            {upchan_max_num_channels * upchan_factor, num_polarizations, num_dishes_y, num_dishes_x,
             num_components},
            {"F", "P", "dishN", "dishM", "C"});
        frb1_phase_buffer->allocate_new_metadata_object(frame_id);
        const auto& frb1_phase_meta = get_chord_metadata(frb1_phase_buffer->get_metadata(frame_id));
        frb1_phase_meta->set_from_frame_desc(frb1_phase_buffer->get_ndarray_frame_desc());
        frb1_phase_meta->set_fpga_seq_num(0);           // ???
        frb1_phase_meta->set_time_downsampling_fpga(1); // ???
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
        frb1_phase_meta->set_coarse_freq(coarse_freq);
        frb1_phase_meta->set_freq_upchan_factor(freq_upchan_factor);
        frb1_phase_meta->set_freq_upchan_index(freq_upchan_index);

        // Set buffer
        const std::ptrdiff_t str_dish_x = 1;
        const std::ptrdiff_t str_dish_y = str_dish_x * num_dishes_x;
        const std::ptrdiff_t str_polr = str_dish_y * num_dishes_y;
        const std::ptrdiff_t str_freq = str_polr * num_polarizations;
        for (int freq = 0; freq < upchan_max_num_channels * upchan_factor; ++freq) {
            for (int polr = 0; polr < num_polarizations; ++polr) {
                for (int dish_y = 0; dish_y < num_dishes_y; ++dish_y) {
                    for (int dish_x = 0; dish_x < num_dishes_x; ++dish_x) {
                        const std::ptrdiff_t idx = str_dish_x * dish_x + str_dish_y * dish_y
                                                   + str_polr * polr + str_freq * freq;
                        assert(idx >= 0
                               && idx < std::ptrdiff_t(frb1_phase_buffer->frame_size
                                                       / sizeof *frb1_phase_frame));
                        frb1_phase_frame[idx] =
                            float16_t(freq < upchan_num_channels * upchan_factor ? frb1_input_scale
                                                                                 : 0.0 / 0.0);
                    }
                }
            }
        }

        // Mark buffers as full
        DEBUG("[{:s}/{:d}] Marking buffer as full...", frb1_phase_buffer->buffer_name, frame_index);
        frb1_phase_buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(setFRB1Phase);
