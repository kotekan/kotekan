/**
 * @file
 * @brief Process to transmit VDIF data as a UDP stream.
 *  - streamSingleDishVDIF : public kotekan::Stage
 */

#ifndef STREAM_SINGLE_DISH_VDIF_H
#define STREAM_SINGLE_DISH_VDIF_H

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"

/**
 * @class streamSingleDishVDIF
 * @brief Process to transmit VDIF data as a UDP stream.
 *
 * This is a consumer process which takes VDIF-formatted data from a buffer and streams
 * it via UDP to a remote client.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer containing VDIF data to be transmitted.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   num_freq               Int. Number of time samples to sum.
 * @conf   dest_port              Int. Number of time samples to sum.
 * @conf   dest_server_ip         Int. Number of time samples to sum.
 *
 * @note    Hasn't been tested lately, should confirm this still works!
 *
 * @author Andre Renard
 *
 */
class streamSingleDishVDIF : public kotekan::Stage {
public:
    /// Constructor
    streamSingleDishVDIF(kotekan::Config& config, const string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~streamSingleDishVDIF();

    /// Main loop, just waits for frames and fires 'em off.
    void main_thread() override;

private:
    /// Kotekan buffer which this process consumes from.
    /// Data should be packed into VDIF frames, see e.g. @c vdif_function.h.
    struct Buffer* in_buf;

    /// Port of the listening receiver.
    uint32_t dest_port;
    /// IP of the listening receiver.
    string dest_ip;

    /// Number of frequencies in the buffer
    int num_freq;
};

#endif
