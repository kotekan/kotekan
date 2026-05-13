#include "parseReorderDefault.hpp"

#include "DataType.hpp"        // for DataType, KOTEKAN_FLOAT16, float16_t
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR

#include <algorithm> // for find
#include <assert.h>  // for assert
#include <stdint.h>  // for int8_t, uint32_t, uint8_t, int16_t, int32_t, uint64_t
#include <string>    // for std::string
#include <strings.h> // for bzero
#include <unistd.h>  // for sleep
#include <vector>    // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(parseReorderDefault);

#define CHIME_ORDER_CORRELATOR "correlator" // not regular
#define CHIME_ORDER_CYLINDER "cylinder"     // cyl-pol-dish
#define CHIME_ORDER_BEAMFORMER "beamformer" // pol-dish

parseReorderDefault::parseReorderDefault(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&parseReorderDefault::main_thread, this)),
    _out_buf(get_buffer("out_buf")), _name(config.get<std::string>(unique_name, "name")),
    _input_reorder(std::get<0>(parse_reorder_default(config, unique_name))),
    _input_order(config.get_default<std::string>(unique_name, "input_order", CHIME_ORDER_CORRELATOR)),
    _output_order(config.get_default<std::string>(unique_name, "output_order", CHIME_ORDER_CYLINDER)),
    _num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int>(unique_name, "num_dishes")) {
    _out_buf->register_producer(unique_name);

    if (_out_buf->frame_size != _input_reorder.size() * sizeof(_input_reorder[0])) {
        throw std::invalid_argument("parseReorderDefault: incorrect frame size");
    }

    if(_input_order != CHIME_ORDER_CORRELATOR && _input_order != CHIME_ORDER_CYLINDER && _input_order != CHIME_ORDER_BEAMFORMER) {
      FATAL_ERROR("input_order must be one of {:s}, {:s}, or {:s}", CHIME_ORDER_CORRELATOR, CHIME_ORDER_CYLINDER, CHIME_ORDER_BEAMFORMER);
    }

    if(_output_order != CHIME_ORDER_CORRELATOR && _output_order != CHIME_ORDER_CYLINDER && _output_order != CHIME_ORDER_BEAMFORMER) {
      FATAL_ERROR("output_order must be one of {:s}, {:s}, or {:s}", CHIME_ORDER_CORRELATOR, CHIME_ORDER_CYLINDER, CHIME_ORDER_BEAMFORMER);
    }

#if 0 // this is what it should be
    if(_input_order == CHIME_ORDER_CORRELATOR) {
          _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {_num_polarizations*_num_dishes},
                                                {"E"});
    } else if(_input_order == CHIME_ORDER_CYLINDER) {
          _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {_num_chime_cylinders, _num_polarizations, _num_dishes/_num_chime_cylinders},
                                                {"C", "P", "D"});
    } else if(_input_order == CHIME_ORDER_BEAMFORMER) {
          _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {_num_polarizations, _num_dishes},
                                                {"P", "D"});
    } else {
        FATAL_ERROR("Unexpected input_order {:s}", _input_order);
    }
#else
    // this is what xpose2048 expects
    _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {_num_polarizations, _num_dishes},
                                          {"P", "D"});
#endif
}

parseReorderDefault::station_id_t parseReorderDefault::correlator_index_to_station_id(const int idx) const {
    assert(idx >= 0 && idx < _num_dishes * _num_polarizations);
    const auto id_it = std::find(_input_reorder.cbegin(), _input_reorder.cend(), idx);
    assert(id_it != _input_reorder.end());
    return static_cast<station_id_t>(id_it - _input_reorder.cbegin());
}

parseReorderDefault::station_id_t parseReorderDefault::cylinder_index_to_station_id(const int idx) const {
    assert(idx >= 0 && idx <= _num_dishes * _num_polarizations);
    return static_cast<station_id_t>(idx);
}

parseReorderDefault::station_id_t parseReorderDefault::beamformer_index_to_station_id(const int idx) const {
    assert(idx >= 0 && idx < _num_dishes * _num_polarizations); 
    constexpr auto cylinder_to_beamforrmer_table = get_cylinder_to_beamformer_reorder_table();
    const auto cylinder_idx_it = std::find(cylinder_to_beamforrmer_table.cbegin(), cylinder_to_beamforrmer_table.cend(), idx);
    assert(cylinder_idx_it != cylinder_to_beamforrmer_table.cend());
    return cylinder_index_to_station_id(*cylinder_idx_it);
}

int parseReorderDefault::station_id_to_correlator_index(const station_id_t id) const {
    assert(id >= 0 && id <= _num_dishes * _num_polarizations);
    return static_cast<int>(_input_reorder.at(id));
}

int parseReorderDefault::station_id_to_cylinder_index(const station_id_t id) const {
    assert(id >= 0 && id <= _num_dishes * _num_polarizations);
    return id;
}

int parseReorderDefault::station_id_to_beamformer_index(const station_id_t id) const {
    assert(id >= 0 && id < _num_dishes * _num_polarizations); 
    constexpr auto cylinder_to_beamforrmer_table = get_cylinder_to_beamformer_reorder_table();
    const int cylinder_idx = station_id_to_cylinder_index(id);
    return static_cast<int>(cylinder_to_beamforrmer_table.at(cylinder_idx));
}

void parseReorderDefault::main_thread() {
    int abs_frame_id = 0;

    station_id_t (parseReorderDefault::*input_index_to_station_id)(const int) const;
    if (_input_order == CHIME_ORDER_CORRELATOR)
      input_index_to_station_id = &parseReorderDefault::correlator_index_to_station_id;
    else if (_input_order == CHIME_ORDER_CYLINDER)
      input_index_to_station_id = &parseReorderDefault::cylinder_index_to_station_id;
    else if (_input_order == CHIME_ORDER_BEAMFORMER)
      input_index_to_station_id = &parseReorderDefault::beamformer_index_to_station_id;
    else
      FATAL_ERROR("Unexpected order {:s}", _input_order);

    int (parseReorderDefault::*station_id_to_output_index)(const station_id_t) const;
    if (_output_order == CHIME_ORDER_CORRELATOR)
      station_id_to_output_index = &parseReorderDefault::station_id_to_correlator_index;
    else if (_output_order == CHIME_ORDER_CYLINDER)
      station_id_to_output_index = &parseReorderDefault::station_id_to_cylinder_index;
    else if (_output_order == CHIME_ORDER_BEAMFORMER)
      station_id_to_output_index = &parseReorderDefault::station_id_to_beamformer_index;
    else
      FATAL_ERROR("Unexpected order {:s}", _output_order);

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

        for (int idx = 0; idx < static_cast<int>(_input_reorder.size()); ++idx) {
            // indexed by input_index, returns output_index
            frame[idx] = static_cast<int32_t>((this->*station_id_to_output_index)((this->*input_index_to_station_id)(idx)));
        }

        _out_buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(_out_buf, frame_id);

        chordmeta->set_frame_counter(0); // these do not actually change with time

        chordmeta->set_from_frame_desc(_out_buf->get_ndarray_frame_desc());
        chordmeta->check_frame_desc(_out_buf->get_ndarray_frame_desc());

        _out_buf->mark_frame_full(unique_name, frame_id);

        abs_frame_id += 1;

        first_time = false;
    }
}
