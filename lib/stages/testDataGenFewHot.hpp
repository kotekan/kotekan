#ifndef TEST_DATA_GEN_FEW_HOT_H
#define TEST_DATA_GEN_FEW_HOT_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for stream_t
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance
#include "visUtil.hpp"         // for StatTracker

#include "json.hpp" // for json

#include <memory>   // for shared_ptr
#include <stddef.h> // for ptrdiff_t
#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string
#include <vector>   // for vector

/**
 * @class testDataGenFewHot
 * @brief Generate test data as a standin for DPDK.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill
 *         @buffer_format any format
 *         @buffer_metadata chordMetadata
 *
 * @conf  type                  String. "single", "double"
 * @conf  freq_id               Int. Frequency to produce data for.
 * @conf  dish                  Int. Dish to set.
 * @conf  samples_per_data_set  Int. How often to produce data.
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
