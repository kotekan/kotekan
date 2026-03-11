#include "parseReorderDefault.hpp"

#include "DataType.hpp"        // for DataType, KOTEKAN_FLOAT16, float16_t
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR

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

parseReorderDefault::parseReorderDefault(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&parseReorderDefault::main_thread, this)),
    _out_buf(get_buffer("out_buf")), _name(config.get<std::string>(unique_name, "name")),
    _input_reorder(std::get<0>(parse_reorder_default(config, unique_name))),
    _invert_mapping(config.get_default<bool>(unique_name, "invert_mapping", true)),
    _num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int>(unique_name, "num_dishes")) {
    _out_buf->register_producer(unique_name);

    if (_out_buf->frame_size != _input_reorder.size() * sizeof(_input_reorder[0])) {
        throw std::invalid_argument("parseReorderDefault: incorrect frame size");
    }

    // TODO: this is not quite correct. Really if going to cylinder order
    // the array is {4,2,256} {"C", "P", "D"}
    _out_buf->allocate_ndarray_frame_desc(kotekan::int32, _name, {_num_polarizations, _num_dishes},
                                          {"P", "D"});
}


void parseReorderDefault::main_thread() {
    int abs_frame_id = 0;

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

        for (size_t i = 0; i < _input_reorder.size(); ++i) {
            if (_invert_mapping) {
                // the (ptrdiff_t) cast is just there to pacify a warning about
                // comparing unsigned >= 0 being always true. The assert is to
                // future proof this in case anyone ever changes the type of
                // _input_reorder to be signed.
                assert((ptrdiff_t)_input_reorder.at(i) >= 0
                       && _input_reorder.at(i) < _input_reorder.size());
                frame[_input_reorder.at(i)] = static_cast<int32_t>(i);
            } else {
                frame[i] = _input_reorder.at(i);
            }
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
