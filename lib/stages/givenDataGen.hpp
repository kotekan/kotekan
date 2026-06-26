#ifndef GIVEN_DATA_GEN_H
#define GIVEN_DATA_GEN_H

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "Stage.hpp"           // for Stage
#include "Symbol.hpp"          // for Symbol
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstddef> // for ptrdiff_t
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class givenDataGen
 * @brief Present given data as a kotekan buffer.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill
 *         @buffer_format any format
 *         @buffer_metadata chordMetadata
 *
 * @conf  values                Vector of ints.
 * @conf  name                  String. Name of the quantity being set.
 * @conf  datatype              String. Kotekan datatype name.
 * @conf  array_shape           Vector of ints. Size of each dimension.
 * @conf  dim_name              Vector of strings. Name of each dimension.
 * @conf  do_once               Bool. Set data only once, for a single frame.
 *
 * @author Roland Haas, based on testDataGen
 */
class givenDataGen : public kotekan::Stage {
public:
    givenDataGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~givenDataGen() = default;
    void main_thread() override;

private:
    Buffer* const _out_buf;
    const std::vector<int> _values;
    const kotekan::Symbol _name;
    const kotekan::DataType _datatype;
    const std::vector<std::ptrdiff_t> _array_shape;
    const std::vector<kotekan::Symbol> _dim_name;
    const std::vector<std::ptrdiff_t> _dim_scalings;
    const bool _do_once;
};

#endif
