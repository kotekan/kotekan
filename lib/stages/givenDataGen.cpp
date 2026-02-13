#include "givenDataGen.hpp"

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

REGISTER_KOTEKAN_STAGE(givenDataGen);

givenDataGen::givenDataGen(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&givenDataGen::main_thread, this)),
    _out_buf(get_buffer("out_buf")), _values(config.get<std::vector<int>>(unique_name, "values")),
    _name(config.get<kotekan::Symbol>(unique_name, "name")),
    _datatype(kotekan::string_to_type(config.get<std::string>(unique_name, "datatype"))),
    _array_shape(config.get<std::vector<ptrdiff_t>>(unique_name, "array_shape")),
    _dim_name(config.get<std::vector<kotekan::Symbol>>(unique_name, "dim_name")),
    _do_once(config.get<bool>(unique_name, "do_once")) {

    _out_buf->register_producer(unique_name);

    if (_datatype == kotekan::unknown_type) {
        throw std::invalid_argument("givenDataGen: unknown datatype for 'datatype' option");
    }

    if (_array_shape.size() != _dim_name.size()) {
        throw std::invalid_argument("givenDataGen: 'array_shape' and 'dim_name' config "
                                    "settings must be the same length!");
    }

    size_t num_elemns = 1;
    for (const auto sz : _array_shape)
        num_elemns *= sz;
    if (_out_buf->frame_size != num_elemns * kotekan::type_total_bytes(_datatype)) {
        throw std::invalid_argument("givenDataGen: product of 'array_shape' config setting must "
                                    "equal the buffer frame size");
    }

    if (num_elemns != _values.size()) {
        throw std::invalid_argument(
            "givenDataGen: size of 'values' config setting must equal the buffer frame size");
    }
}


void givenDataGen::main_thread() {

    int abs_frame_id = 0;
    while (!stop_thread) {

        if (_do_once && abs_frame_id > 0) {
            sleep(1);
            continue;
        }

        const int frame_id = abs_frame_id % _out_buf->num_frames;
        uint8_t* const frame = _out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        const size_t type_size = kotekan::type_total_bytes(_datatype);
        for (size_t i = 0; i < _values.size(); ++i) {
            // this really makes all kinds of asusmptions
            // only works for integers
            // only works on little endian machines
            assert((i + 1) * type_size <= _out_buf->frame_size && "Out of bounds access");
            std::memcpy(frame + i * type_size, &_values.at(i), type_size);
        }

        _out_buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(_out_buf, frame_id);

        chordmeta->set_frame_counter(abs_frame_id);

        const int name_written =
            std::snprintf(chordmeta->name, sizeof chordmeta->name, "%s", _name.get_c_string());
        if (!(name_written < (int)sizeof chordmeta->name)) {
            throw std::runtime_error("Name too long");
        }

        chordmeta->type = _datatype;

        chordmeta->dims = _array_shape.size();
        assert(chordmeta->dims <= CHORD_META_MAX_DIM);

        for (int d = 0; d < chordmeta->dims; ++d) {
            const int dim_written =
                std::snprintf(chordmeta->dim_name[d], sizeof chordmeta->dim_name[d], "%s",
                              _dim_name.at(d).get_c_string());
            if (!(dim_written < (int)sizeof chordmeta->dim_name[d])) {
                throw std::runtime_error("Dimension label too long");
            }
            chordmeta->dim[d] = _array_shape.at(d);
        }
        chordmeta->set_strides_simple();

        _out_buf->allocate_ndarray_frame_desc(chordmeta->type, _name, _array_shape, _dim_name);
        /* test that things are consistent */
        chordmeta->check_frame_desc(_out_buf->get_ndarray_frame_desc());

        _out_buf->mark_frame_full(unique_name, frame_id);

        abs_frame_id += 1;
    }
}
