/**
 * @file
 * @brief A simple process to take an intensity stream and sum down.
 *  - integratePowerStream : public KotekanProcess
 */

#ifndef INTEGRATE_POWER_STREAM_H
#define INTEGRATE_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <atomic>

/**
 * @class simpleAutocorr
 * @brief A simple process to take a @c radioPowerStream and sum down.
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
class integratePowerStream : public KotekanProcess {
public:
    /// Constructor.
    integratePowerStream(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container);

    /// Destructor.
    virtual ~integratePowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

    /// Re-parse config, not yet implemented.
    virtual void apply_config(uint64_t fpga_seq) override;

private:
	void tcpConnect();

    /// Kotekan buffer which this process consumes from.
    /// Data should be packed with IntensityPacketHeader's.
    struct Buffer *in_buf;
    /// Kotekan buffer which this process produces into.
    struct Buffer *out_buf;

    ///Number of frequencies in the buffer
    int freqs;
    ///Number of times in the buffer
    int times;
    ///Number of elems in the buffer
    int elems;

    //options
    /// Number of timesteps to sum for each output value.
    int integration_length;
};

#endif