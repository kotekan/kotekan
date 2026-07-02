#include "parseReorderDefault.hpp"

#include "DataType.hpp"        // for DataType
#include "NDArray.hpp"         // for GenericNDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"       // for Telescope, ElementOrder
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "visUtil.hpp"         // for get_cylinder_to_beamformer_reorder_table, parse_reorder_d...

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>  // for find
#include <array>      // for array
#include <assert.h>   // for assert
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <stdexcept>  // for invalid_argument
#include <stdint.h>   // for int32_t
#include <string>     // for basic_string, allocator, operator!=, operator==, string
#include <tuple>      // for get
#include <unistd.h>   // for sleep
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(parseReorderDefault);

parseReorderDefault::parseReorderDefault(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&parseReorderDefault::main_thread, this)),
    _out_buf(get_buffer("out_buf")), _name(config.get<std::string>(unique_name, "name")),
    _input_order(config.get_default<ElementOrder>(unique_name, "input_order",
                                                  ElementOrder::CHIMECorrelator)),
    _output_order(
        config.get_default<ElementOrder>(unique_name, "output_order", ElementOrder::CHIMECylinder)),
    _num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int>(unique_name, "num_dishes")) {
    _out_buf->register_producer(unique_name);

#if 0 // this is what it should be
    if (_input_order == ElementOrder::CHIMECorrelator) {
        _out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, _name, {_num_polarizations * _num_dishes}, {"E"}, {1}));
    } else if (_input_order == ElementOrder::CHIMECylinder) {
        _out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, _name,
            {_num_chime_cylinders, _num_polarizations, _num_dishes / _num_chime_cylinders},
            {"D", "P", "D"}, {_num_dishes / _num_chime_cylinders, 1, 1}));
    } else if (_input_order == ElementOrder::CHIMEBeamformer) {
        _out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, _name, {_num_polarizations, _num_dishes}, {"P", "D"}, {1, 1}));
    } else {
        FATAL_ERROR("Unexpected input_order {:s}", _input_order);
    }
#else
    // xpose2048 expects its input in correlator order {"E"} and will produce output in beamformer
    // order {"P", "D"}.
    _out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
        kotekan::int32, _name, {_num_polarizations, _num_dishes}, {"P", "D"}, {1, 1}));
#endif
}

void parseReorderDefault::main_thread() {
    int abs_frame_id = 0;

    const Telescope& tel = Telescope::instance();

    bool first_time = true;
    while (!stop_thread) {

        if (first_time && abs_frame_id > 0) {
            sleep(1);
            continue;
        }

        const int frame_id = abs_frame_id % _out_buf->num_frames;
        int32_t* const frame =
            reinterpret_cast<int32_t*>(_out_buf->wait_for_empty_frame(unique_name, frame_id));
        if (frame == nullptr)
            break;

        // TODO: This is probably somewhat inefficient. Might be better to build
        // a conversion table from the beginning.
        // TODO: This assumes the station IDs are consecutive and begin at 0,
        // which restricts this stage to CHIME orderings.
        for (int st_idx = 0; st_idx < _num_polarizations * _num_dishes; ++st_idx) {
            // indexed by input_index, returns output_index
            const station_id_t station_id = static_cast<station_id_t>(st_idx);
            const int input_idx = tel.station_id_to_element_index(station_id, _input_order);
            const int output_idx = tel.station_id_to_element_index(station_id, _output_order);
            frame[input_idx] = static_cast<int32_t>(output_idx);
        }

        _out_buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(_out_buf, frame_id);

        chordmeta->set_frame_counter(0); // these do not actually change with time

        chordmeta->set_from_frame_desc(_out_buf->get_frame_desc<kotekan::GenericNDArray>());
        chordmeta->check_frame_desc(_out_buf->get_frame_desc<kotekan::GenericNDArray>());

        _out_buf->mark_frame_full(unique_name, frame_id);

        abs_frame_id += 1;

        first_time = false;
    }
}
