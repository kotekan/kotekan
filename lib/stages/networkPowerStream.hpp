/**
 * @file
 * @brief Stage to take an intensity stream and stream to a remote client.
 *  - networkPowerStream : public kotekan::Stage
 */

#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"
#include "powerStreamUtil.hpp" // for IntensityHeader

#include <atomic>      // for atomic_flag
#include <stdint.h>    // for uint32_t, uint64_t
#include <string>      // for string
#include <sys/types.h> // for uint
#include <thread>      // for thread

/**
 * @class networkPowerStream
 * @brief Stage to take an intensity stream and stream to a remote client.
 *
 * This is a consumer stage which takes intensity data from a buffer and streams
 * it via TCP (and some day UDP) to a remote client, primarily for visualization purposes.
 *
 * In TCP mode, the stage should continually attempt to establish a TCP connection,
 * then transmit data once successful.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer containing power data to be sent.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   samples_per_data_set   Int. Number of time samples to sum.
 * @conf   power_integration_length     Int. Number of time samples to sum.
 * @conf   num_freq               Int. Number of time samples to sum.
 * @conf   num_elements           Int. Number of time samples to sum.
 * @conf   freq0                  Float (default 1420.0).
 * @conf   sample_bw              Float (default 10.0).
 * @conf   dest_port               Int. Number of time samples to sum.
 * @conf   dest_server_ip           Int. Number of time samples to sum.
 * @conf   dest_protocol          String. Should be @c "TCP" or @c "UDP"
 *
 * @warning UDP stream doesn't work at the moment.
 * @note    Lots of updating required once buffers are typed...
 *
 * @author Keith Vanderlinde
 *
 */
class networkPowerStream : public kotekan::Stage {
public:
    /// Constructor.
    networkPowerStream(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container);

    /// Destructor.
    virtual ~networkPowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

private:
    /// Function to attempt to establish a TCP link with the receiver.
    void tcpConnect();

    /// Input kotekanBuffer.
    struct Buffer* in_buf;

    /// Port of the listening receiver.
    uint32_t dest_port;
    /// IP of the listening receiver.
    std::string dest_server_ip;
    /// Protocol to use: TCP or UDP. (Only TCP works now)
    std::string dest_protocol;

    // Socket handle for link
    int socket_fd;
    // Flag showing whether the link is up
    bool tcp_connected = false;
    // Flag showing whether we're trying to make the link
    bool tcp_connecting = false;
    // Thread off to the side that establishes the connection for us
    std::thread connect_thread;
    // Lock to prevent race conditions with the connection thread.
    std::atomic_flag socket_lock;

    /// Number of frequencies in the buffer
    int freqs;
    /// Number of times in the buffer
    int times;
    /// Number of elems in the buffer
    int elems;

    /// Frequency of the center of the band. Temporary until we have better metadata.
    float freq0;
    /// Bandwidth of the data stream. Temporary until we have better metadata.
    float sample_bw;

    /// Index of active frame in input buffer.
    uint frame_idx = 0;

    /// Sequence number of the transmit handshake.
    uint64_t handshake_idx = -1;
    /// Timestamp of the transmit handshake.
    double handshake_utc = -1;

    /// Header used for establishing the communication link.
    IntensityHeader header;
};

#endif
