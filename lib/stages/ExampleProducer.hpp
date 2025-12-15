#ifndef EXAMPLE_PRODUCER_H
#define EXAMPLE_PRODUCER_H
#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string

/**
 * @class ExampleProducer
 * @brief An example producer stage that sets each element of a buffer to a constant value.
 *
 * @par Buffers
 * @buffer out_buf The buffer to process the contents of.
 *      @buffer_format any
 *      @buffer_metadata any
 *
 * @conf init_value Float (default 0). The value to set each element to.
 *
 * @par Example
 * @code
 * example_producer:
 *   kotekan_stage: ExampleProducer
 *   out_buf: test_out
 *   init_value: 1.0
 * @endcode
 */
class ExampleProducer : public kotekan::Stage {
public:
    /**
     * @brief Constructor for the stage
     *   Note: you can use the macro STAGE_CONSTRUCTOR(ExampleProducer)
     *   if your constructor does not need additional customisation
     *   and you wish to hide the complexity.
     */
    ExampleProducer(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /**
     * @brief Deconstructor - Called on shutdown, after @c main_thread has exited.
     */
    virtual ~ExampleProducer();

    /**
     * @brief Framework managed pthread.
     */
    void main_thread() override;

private:
    // Output buffer
    Buffer* out_buf;

    // Initialised value
    float _init_value;
};

#endif /* EXAMPLE_PRODUCER_H */
