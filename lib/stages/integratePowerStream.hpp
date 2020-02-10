/**
 * @file
 * @brief A simple stage to take an intensity stream and sum down.
 *  - integratePowerStream : public kotekan::Stage
 */

#ifndef INTEGRATE_POWER_STREAM_H
#define INTEGRATE_POWER_STREAM_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class integratePowerStream
 * @brief A simple stage to take a @c radioPowerStream and sum down.
 *
 * This is a simple signal processing block which takes power data from an input buffer,
 * integrates over time, and stuffs the results into an output buffer.
 * Input and output buffers need not be the same length.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer containing power data to be integrated.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer, where integrated samples will be placed.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   integration   Int. Number of time samples to sum.
 *
 * @todo    Update once we have a formal radioPowerStream buffer format.
 * @todo    Add metadata to allow different data types for in/out.
 *
 * @author Keith Vanderlinde
 *
 */
class integratePowerStream : public kotekan::Stage {
public:
    /// Constructor.
    integratePowerStream(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);

    /// Destructor.
    virtual ~integratePowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

private:
    void tcpConnect();

    /// Kotekan buffer which this stage consumes from.
    /// Data should be packed with IntensityPacketHeader's.
    struct Buffer* in_buf;
    /// Kotekan buffer which this stage produces into.
    struct Buffer* out_buf;

    /// Number of frequencies in the buffer
    int freqs;
    /// Number of times in the buffer
    int times;
    /// Number of elems in the buffer
    int elems;

    // options
    /// Number of timesteps to sum for each output value.
    int integration_length;
};

#endif
