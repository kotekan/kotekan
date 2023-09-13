#ifndef EXAMPLE_DOT_PRODUCT_STAGE_H
#define EXAMPLE_DOT_PRODUCT_STAGE_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string

/**
 * @class ExampleDotProduct
 * @brief A stage to compute the dot product between two vectors: A and B, which are represented by
 * two buffers. The result is written to an output buffer.
 *
 * @par Buffers
 * @buffer in_a_buf The input buffer representing vector A.
 *      @buffer_format Array of floats
 *      @buffer_metadata none
 * @buffer in_b_buf The input buffer representing vector B.
 *      @buffer_format Array of floats
 *      @buffer_metadata any
 * @buffer out_buf The output buffer to hold the result.
 *      @buffer_format Array of floats
 *      @buffer_metadata any
 */
class ExampleDotProduct : public kotekan::Stage {
public:
    ExampleDotProduct(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    virtual ~ExampleDotProduct();
    void main_thread() override;

private:
    // Input buffers
    Buffer* in_a_buf;
    Buffer* in_b_buf;

    // Output buffer
    Buffer* out_buf;
};

#endif /* EXAMPLE_DOT_PRODUCT_STAGE_H */
