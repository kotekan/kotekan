#ifndef EXAMPLE_CONSUMER_H
#define EXAMPLE_CONSUMER_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

/**
 * @class ExampleConsumer
 * @brief An example consumer stage to print the contents of a buffer.
 *
 * @par Buffers
 * @buffer in_buf The buffer to process the contents of.
 *      @buffer_format any
 *      @buffer_metadata any
 *
 */
class ExampleConsumer : public kotekan::Stage {
public:
    /**
     * @brief Constructor for the stage
     *   Note: you can use the macro STAGE_CONSTRUCTOR(ExampleConsumer)
     *   if your constructor does not need additional customisation
     *   and you wish to hide the complexity.
     */
    ExampleConsumer(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /**
     * @brief Deconstructor - what happens when Kotekan shuts down.
     */
    virtual ~ExampleConsumer();

    /**
     * @brief Framework managed pthread.
     */
    void main_thread() override;

private:
    // Input buffer
    Buffer* in_buf;

    // Length of frame
    uint32_t _num_elements;
};

#endif /* EXAMPLE_CONSUMER_H */
