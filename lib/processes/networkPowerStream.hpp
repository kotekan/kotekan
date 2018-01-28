/**
 * @file
 * @brief Process to take an intensity stream and stream to a remote client.
 *  - networkPowerStream : public KotekanProcess
 */

#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <atomic>

/**
 * @class networkPowerStream
 * @brief Process to take an intensity stream and stream to a remote client.
 *
 * This is a consumer process which takes intensity data from a buffer and streams
 * it via TCP (and some day UDP) to a remote client, primarily for visualization purposes.
 *
 * In TCP mode, the process should continually attempt to establish a TCP connection,
 * then transmit data once successful.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer containing power data to be sent.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   samples_per_data_set   Int. Number of time samples to sum.
 * @conf   integration_length     Int. Number of time samples to sum.
 * @conf   num_freq               Int. Number of time samples to sum.
 * @conf   num_elements           Int. Number of time samples to sum.
 * @conf   freq0                  Float (default 1420.0).
 * @conf   sample_bw              Float (default 10.0).
 * @conf   dest_port               Int. Number of time samples to sum.
 * @conf   dest_server_ip           Int. Number of time samples to sum.
 * @conf   dest_protocol          String. Should be @c "TCP" or @c "UDP"
 *
 * @warning UDP stream doesn't work at the moment.
 * @bug     Kotekan exits when TCP receiver goes away. Figure out!
 * @note    Lots of updating required once buffers are typed...
 *
 * @author Keith Vanderlinde
 *
 */
class networkPowerStream : public KotekanProcess {
public:
    ///Constructor.
    networkPowerStream(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);

    ///Destructor.
    virtual ~networkPowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread();

    /// Re-parse config, not yet implemented.
    virtual void apply_config(uint64_t fpga_seq);

private:
    ///Function to attempt to establish a TCP link with the receiver.
	void tcpConnect();

    ///Input kotekanBuffer.
    struct Buffer *buf;

    ///Port of the listening receiver.
    uint32_t dest_port;
    ///IP of the listening receiver.
    string dest_server_ip;
    ///Protocol to use: TCP or UDP. (Only TCP works now)
    string dest_protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;
    std::atomic_flag socket_lock;

    ///Number of frequencies in the buffer
    int freqs;
    ///Number of times in the buffer
    int times;
    ///Number of elems in the buffer
    int elems;

    float freq0;
    float sample_bw;

    int id;

    uint frame_idx=0;

    uint64_t handshake_idx=-1;
    double   handshake_utc=-1;

	IntensityHeader header;
};

#endif