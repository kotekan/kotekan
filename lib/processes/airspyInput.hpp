/**
 * @class airspyInput
 * @brief Kotekan Process to stream data from an AirSpy SDR device.
 *
 * This is a simple producer which initializes an AirSpy dongle (https://airspy.com)
 * in streaming mode, filling samples into the provided kotekan buffer.
 * It is agnostic about buffer framesize, simply filling in its data until a frame is full,
 * marking it so, requesting a new one, and continuing.
 * An internal sub-frame pointer is used to position data in subsequent callbacks within frames.
 * A pthread mutex is used to ensure callbacks don't clobber one another.
 * 
 * This producer depends on libairspy.
 *
 * @param   out_buf     Float. The kotekan buffer which will be fed, can be any size.
 * @param   freq        Float. LO tuning frequency, in MHz. Defaults to 1420.0.
 * @param   sample_bw   Bandwidth to sample from the airspy, in MHz. Defaults to 2.5.
 * @param   gain_lna    Gain setting of the LNA, in the range 0-14. Defaults to 5.
 * @param   gain_mix    Gain setting of the mixer amplifier, from 0-15. Defaults to 5.
 * @param   gain_if     Gain setting of the IF amplifier, from 0-15. Defaults to 5.
 * @param   biast_power Whether or not to enable the 4.5V DC bias on the AirSpy RF input.
 *
 * @warning Just realized that if things bog down and new 2 callbacks come while one is active,
 *          the order of the others will be undefined. This process may produce out-of-order samples.
 * @todo    Only handles one device currently -- add checks and index handling.
 *          Undefined behaviour (segfault) if it doesn't exist or is already in use.
 *
 * @author Keith Vanderlinde
 *
 */

#ifndef AIRSPY_INPUT_HPP
#define AIRSPY_INPUT_HPP

#include <unistd.h>
#include <libairspy/airspy.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#define BYTES_PER_SAMPLE 2

#include <string>
using std::string;

class airspyInput : public KotekanProcess {
public:
    /// Constructor, also initializes internal variables from config.
    airspyInput(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);

    /// Destructor, cleans up local allocs.
    virtual ~airspyInput();

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread();

    /// Not yet implemented, should update runtime parameters.
    virtual void apply_config(uint64_t fpga_seq);

    /// Initializes the airspy device.
    /// Also configues runtime parameters, such as gains, LO tuning, samplerate, etc.
    struct airspy_device *init_device();

    /// Static callback when the device returns a buffer of samples.
    static int airspy_callback(airspy_transfer_t* transfer);
    /// Function to process a new buffer of samples.
    void airspy_producer(airspy_transfer_t* transfer);

private:
    /// kotekan buffer object which will be fed
    struct Buffer *buf;
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


    //options
    /// Frequency of the LO, in Hz
    int freq;
    /// Sampling bandwidth, in Hz. Should be 2500000 or 10000000
    int sample_bw;
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
