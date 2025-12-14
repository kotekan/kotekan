/**
 * @file
 * @brief Stage to receive an intensity stream from a remote client.
 *  - networkInputPowerStream : public kotekan::Stage
 */

#ifndef NETWORK_INPUT_POWER_STREAM_H
#define NETWORK_INPUT_POWER_STREAM_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string


/**
 * @class networkInputPowerStream
 * @brief Stage to take an intensity stream and stream to a remote client.
 *
 * This is a consumer stage which reads power/intensity frames and sends them to a remote client
 * over TCP (UDP is stubbed). It opens a socket to the configured `ip:port` and writes each frame
 * sequentially; if the connection drops it reconnects and continues. The stage does not alter
 * data or metadata; it simply forwards raw buffer contents.
 *
 * @par Buffers
 * @buffer out_buf Input kotekan buffer containing power data to be sent.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf out_buf             String. Input power buffer to stream.
 * @conf samples_per_data_set Int. Number of time samples in each frame.
 * @conf integration_length  Int. Integration length used upstream (metadata only).
 * @conf num_freq            Int. Number of frequencies per frame.
 * @conf num_elements        Int. Number of elements per frame.
 * @conf port                Int. Destination port.
 * @conf ip                  String. Destination IP address.
 * @conf protocol            String. "TCP" or "UDP" (only TCP supported).
 *
 * @par Example
 * @code
 * networkInputPowerStream:
 *   out_buf: power_out
 *   samples_per_data_set: 1024
 *   integration_length: 1
 *   num_freq: 1024
 *   num_elements: 2048
 *   ip: 127.0.0.1
 *   port: 14000
 *   protocol: TCP
 * @endcode
 *
 * @warning UDP stream receiption doesn't work at the moment.
 * @note    Lots of updating required once buffers are typed...
 *
 * @author Keith Vanderlinde
 *
 */
class networkInputPowerStream : public kotekan::Stage {
public:
    /// Constructor.
    networkInputPowerStream(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container);
    /// Destructor.
    virtual ~networkInputPowerStream();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

private:
    /// Simple function to receive data of @c length bytes.
    void receive_packet(void* buffer, int length, int socket_fd);

    /// Output kotekanBuffer.
    Buffer* out_buf;

    /// Port of the listening receiver.
    uint32_t port;
    /// IP of the listening receiver.
    std::string server_ip;
    /// Protocol to use: TCP or UDP. (Only TCP works now)
    std::string protocol;


    /// Number of frequencies in the buffer
    int freqs;
    /// Number of times in the buffer
    int times;
    /// Number of elems in the buffer
    int elems;
};

#endif
