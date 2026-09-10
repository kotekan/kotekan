#ifndef TEST_DATA_GEN_FEW_HOT_H
#define TEST_DATA_GEN_FEW_HOT_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for freq_id_t
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stddef.h> // for ptrdiff_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class testDataGenFewHot
 * @brief Generate test data with a few "hot" elements, as a standin for DPDK.
 *
 * Fills an upchannelized "Ebar" frame of int4x2_swapped_withoffset samples with
 * dimensions (Tbar, Fbar, P, D). The frame layout is taken from the frame descriptor of
 * @c out_buf, so the buffer must be declared as ``kotekan_buffer: ndarray`` in the config;
 * the Tbar dimscaling is the upchannelization factor. Every sample is zero volts (byte
 * 0x88) except the listed elements, whose sample byte is 0x98 at every time and
 * frequency, so those elements can be traced through downstream reordering. An element
 * index addresses the flattened (P, D) axes: element ``el`` is polarization ``el / D`` of
 * dish ``el % D``.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill
 *         @buffer_format int4x2_swapped_withoffset ndarray, (Tbar, Fbar, P, D)
 *         @buffer_metadata chordMetadata
 *
 * @conf  type     String. Must be "fewhot".
 * @conf  freq_id  Vector of ints. Coarse frequency ids, one per group of upchannelized
 *                 frequencies; the first Fbar / upchan_factor entries are used.
 * @conf  elemns   Vector of ints. Element indices to set hot, each in [0, P * D).
 *
 */
class testDataGenFewHot : public kotekan::Stage {
public:
    testDataGenFewHot(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~testDataGenFewHot() = default;
    void main_thread() override;

private:
    Buffer* buf;
    const std::string type;
    const std::vector<freq_id_t> freq_id;
    const std::vector<int> elemns;

    // Frame layout, from the out_buf frame descriptor
    ptrdiff_t num_times;
    ptrdiff_t num_freq;
    ptrdiff_t num_elemns;
    ptrdiff_t upchan_factor;
};

#endif
