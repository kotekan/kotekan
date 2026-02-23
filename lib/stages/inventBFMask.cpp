#include "Config.hpp"            // for Config
#include "DataType.hpp"          // for string_to_type, DataType
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

class inventBFMask : public kotekan::Stage {
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");

    Buffer* const buffer;

public:
    inventBFMask(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("bf_mask")) {
        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~inventBFMask() {}

    void main_thread() override {
        // Only invent a single frame
        const int frame_index = 0;
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
        buffer->allocate_ndarray_frame_desc<std::int8_t, 2>(
            "bf_mask", {num_polarizations, num_dishes}, {"P", "D"});
        buffer->allocate_new_metadata_object(frame_id);
        const auto& meta = get_chord_metadata(buffer->get_metadata(frame_id));
        meta->set_from_frame_desc(buffer->get_ndarray_frame_desc());
        meta->set_fpga_seq_num(0);           // ???
        meta->set_time_downsampling_fpga(1); // ???

        // We should probably set this...
        meta->ndishes = -1;
        meta->dish_index = nullptr;

        // Fill buffer
        DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
        for (int polr = 0; polr < num_polarizations; ++polr) {
            for (int dish = 0; dish < num_dishes; ++dish) {
                const int idx = dish + num_dishes * polr;
                assert(idx >= 0 && idx < std::ptrdiff_t(buffer->frame_size));
                frame[idx] = 1; // dish is active
            }
        }

        // Mark buffer as full
        DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
        buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(inventBFMask);
