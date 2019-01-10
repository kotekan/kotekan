/**
 * @file
 * @brief Process to receive an intensity stream from a remote client.
 *  - networkInputPowerStream : public kotekan::KotekanProcess
 */

#ifndef NETWORK_INPUT_POWER_STREAM_H
#define NETWORK_INPUT_POWER_STREAM_H

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "powerStreamUtil.hpp"

#include <atomic>
#include <sys/socket.h>


/**
 * @class networkInputPowerStream
 * @brief Process to take an intensity stream and stream to a remote client.
 *
 * This is a consumer process which takes intensity data from a buffer and streams
 * it via TCP (and some day UDP) to a remote client, primarily for visualization purposes.
 *
 * In TCP mode, the process should continually attempt to establish a TCP connection,
 * then transmit data once successful.
 *
 * @par Buffers
 * @buffer out_buf Input kotekan buffer containing power data to be sent.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   samples_per_data_set   Int. Number of time samples to sum.
 * @conf   integration_length     Int. Number of time samples to sum.
 * @conf   num_freq               Int. Number of time samples to sum.
 * @conf   num_elements           Int. Number of time samples to sum.
 * @conf   port                   Int. Number of time samples to sum.
 * @conf   ip                     Int. Number of time samples to sum.
 * @conf   protocol               String. Should be @c "TCP" or @c "UDP"
 *
 * @warning UDP stream receiption doesn't work at the moment.
 * @note    Lots of updating required once buffers are typed...
 *
 * @author Keith Vanderlinde
 *
 */
class networkInputPowerStream : public kotekan::KotekanProcess {
public:
    /// Constructor.
    networkInputPowerStream(kotekan::Config& config, const string& unique_name,
                            kotekan::bufferContainer& buffer_container);
    /// Destructor.
    virtual ~networkInputPowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

private:
    /// Simple function to receive data of @c length bytes.
    void receive_packet(void* buffer, int length, int socket_fd);

    /// Output kotekanBuffer.
    struct Buffer* out_buf;

    /// Port of the listening receiver.
    uint32_t port;
    /// IP of the listening receiver.
    string server_ip;
    /// Protocol to use: TCP or UDP. (Only TCP works now)
    string protocol;


    /// Number of frequencies in the buffer
    int freqs;
    /// Number of times in the buffer
    int times;
    /// Number of elems in the buffer
    int elems;
};

#endif