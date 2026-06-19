#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata
#include "kotekanLogging.hpp"  // for DEBUG

#include "fmt.hpp" // for compile_string_to_view, format

#include <cassert>    // for assert
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int32_t
#include <functional> // for function
#include <memory>     // for allocator, __shared_ptr_access, shared_ptr
#include <string>     // for basic_string, string
#include <unistd.h>   // for sleep
#include <vector>     // for vector

class setScatterIndices : public kotekan::Stage {
    // Telescope layout
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");

    const std::vector<int> scatter_indices =
        config.get<std::vector<int>>(unique_name, "scatter_indices");

    Buffer* const scatter_indices_buffer;

public:
    setScatterIndices(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        scatter_indices_buffer(get_buffer("scatter_indices_name"))
    //
    {
        assert(scatter_indices_buffer);
        scatter_indices_buffer->register_producer(unique_name);
    }

    virtual ~setScatterIndices() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Wait for buffer
        DEBUG("[{:s}/{:d}] Waiting for buffer...", scatter_indices_buffer->buffer_name,
              frame_index);
        std::int32_t* const scatter_indices_frame = static_cast<std::int32_t*>(static_cast<void*>(
            scatter_indices_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!scatter_indices_frame)
            return;

        // Check buffer size
        assert(scatter_indices_buffer->frame_size
               == sizeof(std::int32_t) * num_polarizations * num_dishes);

        // Set metadata
        scatter_indices_buffer->allocate_ndarray_frame_desc<std::int32_t, 2>(
            "scatter_indices", {num_polarizations, num_dishes}, {"P", "D"}, {1, 1});
        scatter_indices_buffer->allocate_new_metadata_object(frame_id);
        const auto& scatter_indices_meta =
            get_chord_metadata(scatter_indices_buffer->get_metadata(frame_id));
        scatter_indices_meta->set_from_frame_desc(scatter_indices_buffer->get_ndarray_frame_desc());
        // scatter_indices_meta->set_fpga_seq_num(0);           // ???
        // scatter_indices_meta->set_time_downsampling_fpga(1); // ???

        // Set buffer
        const std::ptrdiff_t str_dish = 1;
        const std::ptrdiff_t str_polr = str_dish * num_dishes;
        for (int polr = 0; polr < num_polarizations; ++polr) {
            for (int dish = 0; dish < num_dishes; ++dish) {
                const std::ptrdiff_t idx = str_dish * dish + str_polr * polr;
                assert(idx >= 0
                       && idx < std::ptrdiff_t(scatter_indices_buffer->frame_size
                                               / sizeof *scatter_indices_frame));
                scatter_indices_frame[idx] = scatter_indices.at(idx);
            }
        }

        // Mark buffers as full
        DEBUG("[{:s}/{:d}] Marking buffer as full...", scatter_indices_buffer->buffer_name,
              frame_index);
        scatter_indices_buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(setScatterIndices);
