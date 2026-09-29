#ifndef GIVEN_DATA_GEN_H
#define GIVEN_DATA_GEN_H

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "Stage.hpp"           // for Stage
#include "Symbol.hpp"          // for Symbol
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstddef> // for ptrdiff_t
#include <cstdint> // for int64_t
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
 * @buffer metadata_source Optional. Any time-dependent buffer with an `fpga_seq_num`, typically
 *         the voltage buffer. Only its first frame is read. If given, the output is a stream
 *         clocked by it: frame `k` gets `fpga_seq_num = seq0 + k * time_downsampling_fpga`, where
 *         `seq0` is the `fpga_seq_num` of the source's first frame. Requires `do_once: false`.
 *         @buffer_format any format
 *         @buffer_metadata chordMetadata
 *
 * @conf  values                Vector of values.
 * @conf  name                  String. Name of the quantity being set.
 * @conf  datatype              String. Kotekan datatype name.
 * @conf  array_shape           Vector of ints. Size of each dimension.
 * @conf  dim_name              Vector of strings. Name of each dimension.
 * @conf  do_once               Bool. Set data only once, for a single frame.
 * @conf  time_downsampling_fpga  Int. Number of FPGA samples per output frame. Required (and
 *                              only used) if `metadata_source` is given.
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
    const std::vector<long double> _values;
    const kotekan::Symbol _name;
    const kotekan::DataType _datatype;
    const std::vector<std::ptrdiff_t> _array_shape;
    const std::vector<kotekan::Symbol> _dim_name;
    const std::vector<std::ptrdiff_t> _dim_scalings;
    const bool _do_once;
    // Optional clock
    Buffer* const _in_buf;
    const std::int64_t _time_downsampling_fpga;
};

#endif
