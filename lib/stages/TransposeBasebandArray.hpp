#ifndef TRANSPOSE_BASEBAND_ARRAY_HPP
#define TRANSPOSE_BASEBAND_ARRAY_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstdint>
#include <string>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

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
 *
 * @conf timesamples_per_frame  Int. Total number of time samples per frame.
 * @conf num_local_freq         Int. Number of local frequency bins.
 * @conf num_elements           Int. Total number of elements.
 * @conf time_short             Int. Size of the short time dimension (default: 16).
 * @conf element_short          Int. Size of the short element dimension (default: 8).
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

    /// Frame dimensions
    uint32_t timesamples_per_frame;
    uint32_t num_local_freq;
    uint32_t num_elements;
    uint32_t time_short;
    uint32_t element_short;

    /// Derived dimensions
    uint32_t time_long;
    uint32_t element_long;

    /// Flag indicating if AVX512 fast path can be used
    bool use_avx512_fast_path;

#ifdef __AVX512F__
    /// AVX512 optimized transpose for a single (time_long, freq) block
    /// Input: 2048 bytes [element_long=16][time_short=16][element_short=8]
    /// Output: 16 rows of 128 bytes each, written with stride out_stride
    void transpose_block_avx512(const uint8_t* in, uint8_t* out, size_t out_stride);
#endif
};

#endif // TRANSPOSE_BASEBAND_ARRAY_HPP
