/**
 * @file
 * @brief Process to gather, order, and assemble CHIME node VDIF streams.
 *  - psrRecv : public KotekanProcess
 */

#ifndef NETWORK_INPUT_POWER_STREAM_H
#define NETWORK_INPUT_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <atomic>

/**
 * @class psrRecv
 * @brief Process to gather, order, and assemble CHIME node VDIF streams.
 *
 * This is a producer process which listens for CHIME VDIF UDP packets,
 * assembling them into waiting buffers. VDIF packets marked invalid are ignored,
 * others are copied into the correct position in the output buffer.
 *
 * This process holds @c recv_depth frames open at any given time,
 * closing the oldest and opening a new one as soon as it receives a
 * packet requiring that new frame. On opening a new frame, it is seeded
 * with the anticipated VDIF headers (ensuring all written data contains
 * valid headers and is compatible with the baseband python package),
 * and marked as invalid until replaced by a valid incoming packet.
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer where the incoming packets will be stuffed.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   timesamples_per_frame  Int. Number of time samples to pack into each kotekan frame.
 * @conf   num_freq               Int. Number of frequencies to be contained in each frame.
 * @conf   port                   Int. Port to listen on for the incoming UDP samples.
 * @conf   recv_depth             Int. Number of frames to hold open at a time..
 *
 *
 * @author Keith Vanderlinde
 *
 */
class psrRecv : public KotekanProcess {
public:
    ///Constructor.
    psrRecv(Config& config,const string& unique_name,
                           bufferContainer &buffer_container);
    ///Destructor.
    virtual ~psrRecv();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread();

    /// Re-parse config, not yet implemented.
    virtual void apply_config(uint64_t fpga_seq);


private:
    ///Output kotekanBuffer.
    struct Buffer *out_buf;

    ///Port of the listening receiver.
    uint32_t port;
    ///Number of time samples to pack into each outgoing frame.
    uint timesamples_per_frame;
    ///Number of frequencies to pack into each outgoing frame.
    uint num_freq;
    ///Port to listen for UDP packets on.
    uint recv_depth;
};

#endif