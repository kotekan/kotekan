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
 * Fills an upchan_U16-like "Ebar" frame of int4x2_swapped_withoffset samples with
 * dimensions (Tbar, Fbar, P, D) = (1024, 256, 2, 1024): 1024 time samples, 16 coarse
 * frequencies upchannelized by 16, two polarizations and 1024 dishes. Every sample is
 * zero volts (byte 0x88) except the listed elements, whose sample byte is 0x98 at every
 * time and frequency. The hot elements can then be traced through downstream reordering.
 * An element index addresses the flattened (P, D) axes: element ``el`` is polarization
 * ``el / 1024`` of dish ``el % 1024``.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill
 *         @buffer_format int4x2_swapped_withoffset, "Ebar" (Tbar, Fbar, P, D)
 *         @buffer_metadata chordMetadata
 *
 * @conf  type                  String. Must be "fewhot".
 * @conf  freq_id               Vector of ints. Coarse frequency ids, one per group of 16
 *                              upchannelized frequencies; the first 16 are used.
 * @conf  elemns                Vector of ints. Element indices to set hot, each in
 *                              [0, 2048).
 * @conf  samples_per_data_set  Int. How many time samples per frame.
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
    const ptrdiff_t samples_per_dataset;
};

#endif
