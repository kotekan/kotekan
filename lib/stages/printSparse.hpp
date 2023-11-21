#ifndef PRINT_SPARSE_H
#define PRINT_SPARSE_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO
#include "oneHotMetadata.hpp"  // for get_onehot_frame_counter, metadata_is_onehot
#include "visUtil.hpp"         // for frameID, modulo

#include "fmt.hpp"

#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for bind
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for invalid_argument, runtime_error
#include <stdlib.h>   // for size_t
#include <string>     // for allocator, string, operator+, to_string, char_traits
#include <vector>     // for vector

/**
 * @class printSparse
 * @brief Prints out data in an (assumed sparse) buffer
 *
 * @par Buffers
 * @buffer input_buf Buffer to print
 *         @buffer_format (templated: uint8, uint32, float16)
 *         @buffer_metadata chimeMetadata or oneHotMetadata
 *
 * @conf  input_buf    String.  Input buffer name.
 * @conf  max_to_print Int.  Maximum number of array elements to print.
 * @conf  array_shape  Vector of int.  Shape of the matrix in the buffer.  When given,
 *               we will convert the flat index back into the N-dimensional matrix index.
 *
 * @author Dustin Lang
 */
template<typename A_Type>
class printSparse : public kotekan::Stage {
public:
    printSparse(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~printSparse();
    void main_thread() override;

private:
    Buffer* buf;
    int _max;
    std::vector<int> _array_shape;
};

template<typename A_Type>
printSparse<A_Type>::printSparse(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&printSparse::main_thread, this)) {
    buf = get_buffer("input_buf");
    buf->register_consumer(unique_name);

    _max = config.get_default<int>(unique_name, "max_to_print", 0);
    _array_shape =
        config.get_default<std::vector<int>>(unique_name, "array_shape", std::vector<int>());
    if (_array_shape.size()) {
        size_t sz = sizeof(A_Type);
        for (int s : _array_shape)
            sz *= s;
        if (sz != buf->frame_size)
            // clang-format off
            throw std::invalid_argument("printSparse: product of 'array_shape' config setting must equal the buffer frame size");
        // clang-format on
    }
}

template<typename A_Type>
printSparse<A_Type>::~printSparse() {}

template<typename A_Type>
void printSparse<A_Type>::main_thread() {
    frameID frame_id(buf);
    while (!stop_thread) {

        // Get frame
        const A_Type* frame = (A_Type*)buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        INFO("printSparse: checking {:s}[{:d}]", buf->buffer_name, frame_id);

        int frame_counter = 0;
        if (metadata_is_onehot(buf, frame_id)) {
            frame_counter = get_onehot_frame_counter(buf, frame_id);
            INFO("printSparse: got frame counter for {:s}[{:d}] = {:d}", buf->buffer_name, frame_id,
                 frame_counter);
        }
        if (frame_counter < int(frame_id))
            // HACK -- ugh, metadata copied from voltage (not phase) to baseband
            frame_counter = frame_id;

        int nset = 0;
        for (uint32_t i = 0; i < buf->frame_size / sizeof(A_Type); ++i) {
            if (!frame[i])
                continue;
            nset++;
            if ((_max > 0) && (nset > _max))
                continue;
            if (_array_shape.size()) {
                uint32_t j = i;
                bool first = true;
                std::string istring = "";
                for (auto it = _array_shape.rbegin(); it != _array_shape.rend(); it++) {
                    int n = (*it);
                    // prepend the index to the string
                    istring = std::to_string(j % n) + (first ? "" : ", ") + istring;
                    first = false;
                    j /= n;
                }
                INFO("printSparse: {:s}[{:d}] element {:s} ({:d} = 0x{:x}) has value {:s}",
                     buf->buffer_name, frame_id, istring, i, i, format_nice_string(frame[i]));
                if (nset == 1)
                    INFO("PY sparse[\"{:s}\"][{:d}] = (({:s}), {:s})", buf->buffer_name,
                         frame_counter, istring, format_python_string(frame[i]));
            } else {
                INFO("printSparse: {:s}[{:d}] element {:d} = 0x{:x} has value {:s}",
                     buf->buffer_name, frame_id, i, i, format_nice_string(frame[i]));
            }
        }

        INFO("printSparse: {:s}[{:d}] has {:d} elements set.", buf->buffer_name, frame_id, nset);
        if (nset == 0) {
            INFO("PY sparse[\"{:s}\"][{:d}] = (None, 0)", buf->buffer_name, frame_counter);
        }

        buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
    }
}

#endif
