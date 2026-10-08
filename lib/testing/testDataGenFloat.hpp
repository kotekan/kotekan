#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for datatype_t
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string

/**
 * @class testDataGenFloat
 * @brief Generate floating-point test data as a stand-in for a GPU or network stage.
 *
 * Fills frames with a constant, a ramp, or random values in a floating-point type
 * (`float16`, `float32`, or `float64`) and attaches CHORD metadata describing the
 * array (name, dimensions, dimension names and scalings, coarse frequencies and
 * `fpga_seq_num`). One frame is produced roughly every 83 ms; the stage runs until
 * kotekan is stopped.
 *
 * @par Buffers
 * @buffer network_out_buf Buffer to fill
 *         @buffer_format any format, declared as `kotekan_buffer: ndarray`
 *         @buffer_metadata chordMetadata
 *
 * @conf  type                  String. "const", "ramp", or "random".
 * @conf  value                 Float. Required for type "const" and "ramp". For "ramp",
 *                              element j is set to `fmod(j * value, 256 * value)`.
 * @conf  seed                  Int. Required for type "random"; seeds the random number
 *                              generator for reproducible results.
 * @conf  rand_min              Float. Default: 0. Lower bound for type "random".
 * @conf  rand_max              Float. Default: 1. Upper bound for type "random".
 * @conf  value_type            String. Default: "float32". One of "float16", "float32",
 *                              or "float64"; sets the element type of the frames.
 * @conf  name                  String. Default: "E". Array name in the metadata.
 * @conf  array_shape           Vector of ints. Default: the whole frame as one dimension.
 *                              The product times the element size must equal the buffer
 *                              frame size.
 * @conf  dim_name              Vector of strings. Default: ["D"]. Must have the same
 *                              length as `array_shape`.
 * @conf  dim_scaling           Vector of ints. Default: all ones. Must have the same
 *                              length as `array_shape`.
 * @conf  samples_per_data_set  Int. Required. Number of time samples per frame; also the
 *                              increment of `fpga_seq_num` from frame to frame.
 * @conf  num_local_freq        Int. Default: 1. Number of frequencies in each frame.
 * @conf  manual_freq_ids       Vector of ints. Default: empty. If given, the coarse
 *                              frequency IDs to use (cycled); otherwise consecutive IDs
 *                              starting at 0, or at the telescope's minimum science
 *                              frequency ID on a CHORDTelescope.
 * @conf  meta_time_downsample_factor  Int. Default: 1. Currently only read, not written
 *                              to the metadata.
 * @conf  first_frame_index     Int. Default: 0. Index of the first frame, i.e.
 *                              `fpga_seq_num` starts at `first_frame_index *
 *                              samples_per_data_set`.
 * @conf  pathfinder_test_mode  Bool. Default: false. Advance `fpga_seq_num` only every
 *                              eighth frame, simulating eight pathfinder links.
 * @conf  gen_all_const_data    Bool. Default: false. For type "const", keep filling every
 *                              frame instead of only the first pass through the buffer.
 */
class testDataGenFloat : public kotekan::Stage {
public:
    testDataGenFloat(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~testDataGenFloat();
    void main_thread() override;

private:
    Buffer* buf;
    std::string type;
    int seed;
    float value;
    uint32_t _samples_per_data_set;
    bool _pathfinder_test_mode;
    uint32_t _first_frame_index;
    bool _gen_all_const_data;
    float _rand_min;
    float _rand_max;

    std::string _name;
    std::vector<int> _array_shape;
    std::vector<std::string> _dim_name;
    std::vector<ptrdiff_t> _dim_scaling;
    kotekan::DataType _value_type;
    size_t _num_freq_in_frame;
    std::vector<uint32_t> _manual_freq_ids;
    int _meta_time_downsample_factor;
};

#endif
