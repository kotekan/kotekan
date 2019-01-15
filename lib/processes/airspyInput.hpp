/**
 * @file
 * @brief Contains airspy producer for kotekan.
 *  - airspyInput : public kotekan::Stage
 */

#ifndef AIRSPY_INPUT_HPP
#define AIRSPY_INPUT_HPP

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <libairspy/airspy.h>
#include <signal.h>
#include <unistd.h>

#define BYTES_PER_SAMPLE 2

#include <string>
using std::string;

/**
 * @class airspyInput
 * @brief Producer ``kotekan::Stage`` which streams radio data from an AirSpy SDR device
 * into a
 * ``Buffer``.
 *
 * This is a simple producer which initializes an AirSpy dongle (https://airspy.com)
 * in streaming mode, filling samples into the provided kotekan buffer.
 * It is agnostic about buffer framesize, simply filling in its data until a frame is full,
 * marking it so, requesting a new one, and continuing.
 * An internal sub-frame pointer is used to position data in subsequent callbacks within frames.
 * A pthread mutex is used to ensure callbacks don't clobber one another.
 *
 * This producer depends on ``libairspy``.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 *
 * @conf   freq        Float (default 1420.0). LO tuning frequency, in MHz.
 * @conf   sample_bw   Float (default 2.5). Bandwidth to sample from the airspy, in MHz.
 * @conf   gain_lna    Int (default 5). Gain setting of the LNA, in the range 0-14.
 * @conf   gain_mix    Int (default 5). Gain setting of the mixer amplifier, from 0-15.
 * @conf   gain_if     Int (default 5). Gain setting of the IF amplifier, from 0-15.
 * @conf   biast_power Bool (default false). Whether or not to enable the 4.5V DC bias on the AirSpy
 * RF input.
 *
 * @warning Just realized that if things bog down and new 2 callbacks come while one is active,
 *          the order of the others will be undefined. This process may produce out-of-order
 * samples.
 * @remark  Only operates in I/Q mode currently, and only with packed int16_t samples.
 *          No support for raw samples, etc.
 * @todo    Only handles one device currently -- add checks and index handling.
 *          Undefined behaviour (segfault) if it doesn't exist or is already in use.
 *
 * @author Keith Vanderlinde
 *
 */
class airspyInput : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    airspyInput(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Destructor, cleans up local allocs.
    virtual ~airspyInput();

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

    /// Initializes the airspy device.
    /// Also configures runtime parameters: gains, LO tuning, samplerate, and sample type.
    struct airspy_device* init_device();

    /**
     * Static function to service the callback generated by the AirSpy after
     * data has been read and returned.
     * This function immediately passes processing off to the airspyInput object
     * controlling the AirSpy device which produced it,
     * no processing of any sort is done here.
     * @param transfer Contains the airspy transfer buffer, defined in liairspy/airspy.h
     *                 Useful portions of the transfer object include:
     *                 @li @c samples - pointer to the raw stream of samples),
     *                 @li @c ctx - pointer to the object which called it), and
     *                 @li @c sample_count - number of samples included in the buffer)
     * @warning Nobody should ever call this directly, it's only meant to service the
     *          data-ready AirSpy callback.
     */
    static int airspy_callback(airspy_transfer_t* transfer);

    /**
     * Function which actually processes the data returned from the airspy.
     * Waits for an available frame from @c buf, then copies data into it
     * until either the entire set of samples is transferred, or the frame is filled.
     * In the latter case, marks the frame full, waits for a new one, and continues.
     * @c frame_loc is updated after copying each chunk of data.
     * Locks the @c recv_busy mutex before beginning, and only releases it after all
     * samples are copied to into @c buf.
     * @param transfer Contains the airspy transfer buffer, defined in liairspy/airspy.h
     *                 Useful portions of the transfer object include:
     *                 @li @c samples - pointer to the raw stream of samples),
     *                 @li @c ctx - pointer to the object which called it), and
     *                 @li @c sample_count - number of samples included in the buffer)
     * @warning        Nobody should ever call this directly, it's only meant to service the
     *                 data-ready AirSpy callback.
     */
    void airspy_producer(airspy_transfer_t* transfer);


private:
    /// kotekan buffer object which will be fed
    struct Buffer* buf;
    /// handle to the airspy device itself
    struct airspy_device* a_device;

    /// pointer to the current frame's memory
    unsigned char* frame_ptr;
    /// memory offset from the start of the current frame
    /// where the next incoming data will go
    unsigned int frame_loc;
    /// index of the current frame
    int frame_id;
    /// mutex to keep multiple callbacks from clobbering one another:
    /// this forces subsequent callbacks to wait their turn before writing to memory,
    /// marking buffers full, and/or incrementing the relevant counters
    pthread_mutex_t recv_busy;


    // options
    /// Frequency of the LO, in Hz
    uint freq;
    /// Sampling bandwidth, in Hz. Should be 2500000 or 10000000
    uint sample_bw;
    /// Gain of the LNA, should be an integer in the range 0-14
    int gain_lna;
    /// Gain of the mixer amp, should be an integer in the range 0-15
    int gain_mix;
    /// Gain of the IF amp, should be an integer in the range 0-15
    int gain_if;
    /// Whether or not the AirSpy should apply 4.5V DC bias on the RF line.
    /// Binary flag, should be 0 or 1.
    int biast_power;
};


#endif
