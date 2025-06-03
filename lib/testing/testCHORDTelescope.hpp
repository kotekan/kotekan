#ifndef TEST_CHORD_TELESCOPE_H
#define TEST_CHORD_TELESCOPE_H

#include <string>               // for string

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "bufferContainer.hpp"  // for bufferContainer

/**
 * @class TestCHORDTelescope
 * @brief An example consumer stage to build and print a CHORD Telescope.
 *
 * @par Buffers
 * @buffer in_buf The buffer to process the contents of.
 *      @buffer_format any
 *      @buffer_metadata any
 *
 */
class TestCHORDTelescope : public kotekan::Stage {
public:
    /**
     * @brief Constructor for the stage
     *   Note: you can use the macro STAGE_CONSTRUCTOR(ExampleConsumer)
     *   if your constructor does not need additional customisation
     *   and you wish to hide the complexity.
     */
    TestCHORDTelescope(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /**
     * @brief Deconstructor - what happens when Kotekan shuts down.
     */
    virtual ~TestCHORDTelescope();

    /**
     * @brief Framework managed pthread.
     */
    void main_thread() override;
};

#endif /* EXAMPLE_CONSUMER_H */
