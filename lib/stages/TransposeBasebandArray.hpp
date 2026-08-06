#ifndef TRANSPOSE_BASEBAND_ARRAY_HPP
#define TRANSPOSE_BASEBAND_ARRAY_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint>  // for uint8_t, uint32_t
#include <stddef.h> // for size_t
#include <string>   // for string

/**
 * @class TransposeBasebandArray
 * @brief Transposes electric field data from DPDK format to a contiguous time-major layout
 *
 * This stage takes electric field voltage data in the format produced by DPDK
 * (with time and element dimensions split into long and short components) and
 * transposes it into a contiguous output format where time is the slowest varying
 * dimension.
 *
 * Input format:  E[time_long][frequency_local][element_long][time_short][element_short]
 * Output format: E'[time][frequency_local][element]
 *
 * where:
 *   - time = time_long * time_short
 *   - element = element_long * element_short
 *
 * @par Buffers
 * @buffer in_buf Input buffer containing electric field data
 *     @buffer_format E[time_long][frequency_local][element_long][time_short][element_short]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf Output buffer for transposed electric field data
 *     @buffer_format E[time][frequency_local][element]
 *     @buffer_metadata chordMetadata
 * @buffer pl_mask_buf Input buffer containing packet loss mask
 *     @buffer_format uint64_t[T/64][F][E/8] where each bit indicates valid (1) or lost (0) data
 *     @buffer_metadata none
 *
 * @conf timesamples_per_frame  Int. Total number of time samples per frame.
 * @conf num_local_freq         Int. Number of local frequency bins.
 * @conf num_elements           Int. Total number of elements.
 * @conf time_short             Int. Size of the short time dimension (default: 16).
 * @conf element_short          Int. Size of the short element dimension (default: 8).
 * @conf frame_mode             String. Which frames to process: "all" (default), "even", or "odd".
 *                              Use "even"/"odd" to parallelize two instances on alternate frames.
 * @conf check_for_zero_nibbles Bool. Enable checking for zero nibbles in the input data (default:
 * false).
 *
 * @author Kotekan Team
 */
class TransposeBasebandArray : public kotekan::Stage {
public:
    TransposeBasebandArray(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);
    ~TransposeBasebandArray();
    void main_thread() override;

private:
    /// Input buffer with DPDK format data
    Buffer* in_buf;

    /// Output buffer for transposed data
    Buffer* out_buf;

    /// Packet loss mask buffer
    Buffer* pl_mask_buf;

    /// Configuration parameter (runtime)
    uint32_t timesamples_per_frame;

    /// Frame processing mode: which frames this instance should handle
    enum class FrameMode { All, Even, Odd };
    FrameMode frame_mode;

    /// Derived dimension
    uint32_t time_long;

    /// Flag indicating if AVX512 fast path can be used
    bool use_avx512_fast_path;

    /// Flag to enable checking for zero nibbles in the input data.
    bool check_for_zero_nibbles;

#ifdef __AVX512F__
    /// AVX512 optimized transpose for a single (time_long, freq) block
    /// Input: 2048 bytes [element_long=16][time_short=16][element_short=8]
    /// Output: 16 rows of 128 bytes each, written with stride out_stride
    void transpose_block_avx512(const uint8_t* in, uint8_t* out, size_t out_stride);

    /// AVX512 optimized fill with 0x88 for a single (time_long, freq) block
    /// Writes 0x88 to all output locations that transpose_block_avx512 would write
    /// Output: 16 rows of 128 bytes each, written with stride out_stride
    void fill_block_0x88_avx512(uint8_t* out, size_t out_stride);
#endif
};

#endif // TRANSPOSE_BASEBAND_ARRAY_HPP
