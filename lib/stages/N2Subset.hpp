/*****************************************
@file
@brief Extract a subset of products from a N2 Buffer.
- N2Subset : public kotekan::Stage
*****************************************/
#ifndef N2_SUB_HPP
#define N2_SUB_HPP

#include "Config.hpp"
#include "N2FrameDesc.hpp" // for N2FrameDesc
#include "N2Util.hpp"      // for prod_ctype
#include "Stage.hpp"       // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstddef> // for size_t
#include <memory>  // for shared_ptr
#include <string>  // for string
#include <vector>  // for vector


/**
 * @class N2Subset
 * @brief ``kotekan::Stage`` that extracts a subset of the products.
 *
 * This task consumes a set of visibilities from an input ``N2Buffer`` and
 * passes a subset of products to an output ``N2Buffer``. The subset is
 * determined by the output buffer's frame descriptor, which is configured
 * via the buffer's YAML configuration.
 *
 * The stage validates that all products in the output buffer exist in the
 * input buffer, and builds an index mapping to efficiently copy data.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read.
 *     @buffer_format N2Buffer (any supported layout)
 *     @buffer_metadata N2Metadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format N2Buffer (layout must be a subset of in_buf products)
 *     @buffer_metadata N2Metadata
 *
 * @par Configuration
 * The subset of products is determined entirely by the output buffer's configuration.
 * See N2FrameDesc and bufferFactory for details on configuring N2 buffers with
 * different layouts (FullUpperTri, Autocorrelations, GeneralSubset, etc.) and
 * explicit product lists.
 */
class N2Subset : public kotekan::Stage {

public:
    /// Constructor. Validates buffer descriptors and builds index mapping.
    N2Subset(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    /// Primary loop: copies subset of products from input to output buffer.
    void main_thread() override;

private:
    /// Input buffer
    Buffer* in_buf;

    /// Output buffer to receive subset of visibilities
    Buffer* out_buf;

    /// Frame descriptor for input buffer
    std::shared_ptr<const kotekan::N2FrameDesc> in_desc;

    /// Frame descriptor for output buffer
    std::shared_ptr<const kotekan::N2FrameDesc> out_desc;

    /// Index mapping: for each output product index, the corresponding input product index
    std::vector<size_t> prod_index_map;
};


#endif
