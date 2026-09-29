#include "givenDataGen.hpp"

#include "DataType.hpp"        // for type_total_bytes, string_to_type, DataType
#include "NDArray.hpp"         // for GenericNDArray, Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_DIM

#include "fmt.hpp" // for format

#include <assert.h>   // for assert
#include <cstdint>    // for int64_t
#include <cstring>    // for memcpy
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <stdexcept>  // for invalid_argument, runtime_error
#include <stdint.h>   // for uint8_t
#include <string>     // for allocator, basic_string, string
#include <unistd.h>   // for sleep
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(givenDataGen);

givenDataGen::givenDataGen(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&givenDataGen::main_thread, this)),
    _out_buf(get_buffer("out_buf")),
    _values(config.get<std::vector<long double>>(unique_name, "values")),
    _name(config.get<kotekan::Symbol>(unique_name, "name")),
    _datatype(kotekan::string_to_type(config.get<std::string>(unique_name, "datatype"))),
    _array_shape(config.get<std::vector<ptrdiff_t>>(unique_name, "array_shape")),
    _dim_name(config.get<std::vector<kotekan::Symbol>>(unique_name, "dim_name")),
    _dim_scalings(config.get<std::vector<std::ptrdiff_t>>(unique_name, "dim_scalings")),
    _do_once(config.get<bool>(unique_name, "do_once")),
    _in_buf(config.exists(unique_name, "metadata_source") ? get_buffer("metadata_source")
                                                          : nullptr),
    _time_downsampling_fpga(
        _in_buf ? config.get<std::int64_t>(unique_name, "time_downsampling_fpga") : 0) {

    _out_buf->register_producer(unique_name);

    if (_in_buf) {
        if (_do_once)
            throw std::invalid_argument(
                "givenDataGen: 'metadata_source' requires 'do_once: false'");
        if (_time_downsampling_fpga <= 0)
            throw std::invalid_argument("givenDataGen: 'metadata_source' requires a positive "
                                        "'time_downsampling_fpga'");
        _in_buf->register_consumer(unique_name);
    }

    if (_datatype == kotekan::unknown_type) {
        throw std::invalid_argument("givenDataGen: unknown datatype for 'datatype' option");
    }

    if (_array_shape.size() != _dim_name.size()) {
        throw std::invalid_argument("givenDataGen: 'array_shape' and 'dim_name' config "
                                    "settings must be the same length!");
    }

    if (_array_shape.size() != _dim_scalings.size()) {
        throw std::invalid_argument("givenDataGen: 'array_shape' and 'dim_scalings' config "
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

    // If we have a clock, start the stream at its first sequence number
    std::int64_t seq0 = -1;
    if (_in_buf) {
        const int in_frame_id = 0;
        if (_in_buf->wait_for_full_frame(unique_name, in_frame_id) == nullptr)
            return;
        const std::shared_ptr<const chordMetadata> in_meta =
            get_chord_metadata(_in_buf, in_frame_id);
        if (!in_meta->has_fpga_seq_num())
            FATAL_ERROR("metadata_source {:s} has no fpga_seq_num, needed for setting clock.",
                        _in_buf->buffer_name);
        seq0 = in_meta->get_fpga_seq_num();
        _in_buf->mark_frame_empty(unique_name, in_frame_id);
        // Only the first frame is needed; stop being a consumer so that the producer does not
        // wait for us on the frames after it
        _in_buf->unregister_consumer(unique_name);
    }

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
        const bool is_float = kotekan::type_to_string(_datatype).find("float") != std::string::npos;
        const bool is_int = kotekan::type_to_string(_datatype).find("int") != std::string::npos;
        const bool is_complex = kotekan::type_to_string(_datatype)[0] == 'c';
        if (!(is_float + is_int == 1)) {
            FATAL_ERROR("Unexpected data type: {:s} which is not either int or float",
                        kotekan::type_to_string(_datatype));
        }
        if (is_complex) {
            FATAL_ERROR("Cannot currently handle complex type: {:s}",
                        kotekan::type_to_string(_datatype));
        }
        for (size_t i = 0; i < _values.size(); ++i) {
            assert((i + 1) * type_size <= _out_buf->frame_size && "Out of bounds access");
            if (is_float) {
                switch (_datatype) {
                    case kotekan::float16: {
                        const float16_t hval =
                            static_cast<float16_t>(static_cast<double>(_values[i]));
                        static_assert(sizeof(hval) == kotekan::type_total_bytes(kotekan::float16));
                        std::memcpy(frame + i * type_size, &hval, type_size);
                    } break;
                    case kotekan::float32: {
                        const float fval = static_cast<float>(_values[i]);
                        static_assert(sizeof(fval) == kotekan::type_total_bytes(kotekan::float32));
                        std::memcpy(frame + i * type_size, &fval, type_size);
                    } break;
                    case kotekan::float64: {
                        const double dval = static_cast<double>(_values[i]);
                        static_assert(sizeof(dval) == kotekan::type_total_bytes(kotekan::float64));
                        std::memcpy(frame + i * type_size, &dval, type_size);
                    } break;
                    default:
                        FATAL_ERROR("Unexpected data type: {:s}",
                                    kotekan::type_to_string(_datatype));
                        break;
                }
            } else if (is_int) {
                // this really makes all kinds of asusmptions
                // only works on little endian machines
                const long long ival = static_cast<long long>(_values[i]);
                std::memcpy(frame + i * type_size, &ival, type_size);
            } else {
                assert(0 && "Should not ever be reached");
            }
        }

        _out_buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(_out_buf, frame_id);

        chordmeta->set_frame_counter(abs_frame_id);
        if (_in_buf) {
            chordmeta->set_fpga_seq_num(seq0 + abs_frame_id * _time_downsampling_fpga);
            chordmeta->set_time_downsampling_fpga(_time_downsampling_fpga);
        }

        // The name field is not NUL-terminated, so all of it is usable
        if (_name.get_string().size() > size_t(CHORD_META_MAX_NAME)) {
            throw std::runtime_error("Name too long");
        }
        chordmeta->set_name(_name.get_string());

        chordmeta->type = _datatype;

        chordmeta->dims = _array_shape.size();
        assert(chordmeta->dims <= CHORD_META_MAX_DIM);

        for (int d = 0; d < chordmeta->dims; ++d) {
            if (_dim_name.at(d).get_string().size() > size_t(CHORD_META_MAX_DIMNAME)) {
                throw std::runtime_error("Dimension label too long");
            }
            chordmeta->set_array_dimension(d, _array_shape.at(d), _dim_name.at(d).get_string(),
                                           _dim_scalings.at(d));
        }
        chordmeta->set_strides_simple();

        _out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
            chordmeta->type, _name, _array_shape, _dim_name, _dim_scalings));
        /* test that things are consistent */
        chordmeta->check_frame_desc(_out_buf->get_frame_desc<kotekan::GenericNDArray>());

        _out_buf->mark_frame_full(unique_name, frame_id);

        abs_frame_id += 1;
    }
}
