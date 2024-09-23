#ifndef TIME_UTIL_DUMP_HPP
#define TIME_UTIL_DUMP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage

#include <time.h>
#include <string> // for string

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
class TimeUtilDump : public kotekan::Stage {
public:
    /**
     * @brief Constructor for the stage
     *   Note: you can use the macro STAGE_CONSTRUCTOR(ExampleConsumer)
     *   if your constructor does not need additional customisation
     *   and you wish to hide the complexity.
     */
    TimeUtilDump(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /**
     * @brief Deconstructor - what happens when Kotekan shuts down.
     */
    virtual ~TimeUtilDump();

    /**
     * @brief Framework managed pthread.
     */
    void main_thread() override;

private:
    // Input time
    timespec _gps_time;
};

#endif
