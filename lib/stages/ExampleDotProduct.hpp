#ifndef EXAMPLE_DOT_PRODUCT_STAGE_H
#define EXAMPLE_DOT_PRODUCT_STAGE_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

/**
 * @class ExampleDotProduct
 * @brief A stage to compute the dot product between two vectors: A and B, which are represented by
 * two buffers. The result is written to an output buffer.
 *
 * @par Buffers
 * @buffer in_a_buf The input buffer representing vector A.
 *      @buffer_format standard
 *      @buffer_metadata any
 * @buffer in_b_buf The input buffer representing vector B.
 *      @buffer_format standard
 *      @buffer_metadata any
 * @buffer out_buf The output buffer to hold the result.
 *      @buffer_format standard
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

    // Length of vectors
    uint32_t _num_elements;
};

#endif /* EXAMPLE_DOT_PRODUCT_STAGE_H */
