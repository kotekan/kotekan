/**
 * @file
 * @brief Stage to transmit VDIF data as a UDP stream.
 *  - streamSingleDishVDIF : public kotekan::Stage
 */

#ifndef STREAM_SINGLE_DISH_VDIF_H
#define STREAM_SINGLE_DISH_VDIF_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string

/**
 * @class streamSingleDishVDIF
 * @brief Stage to transmit VDIF data as a UDP stream.
 *
 * This is a consumer stage which takes VDIF-formatted data from a buffer and streams
 * it via UDP to a remote client.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer containing VDIF data to be transmitted.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   in_buf           String. Input buffer.
 * @conf   num_freq         Int. Number of frequencies in each VDIF packet.
 * @conf   dest_port        Int. Destination UDP port.
 * @conf   dest_server_ip   String. Destination IP.
 *
 * @note    Hasn't been tested lately, should confirm this still works!
 *
 * @par Example
 * @code
 * streamSingleDishVDIF:
 *   in_buf: vdif_in
 *   num_freq: 1024
 *   dest_port: 12001
 *   dest_server_ip: 239.0.0.1
 * @endcode
 *
 * @author Andre Renard
 *
 */
class streamSingleDishVDIF : public kotekan::Stage {
public:
    /// Constructor
    streamSingleDishVDIF(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~streamSingleDishVDIF();

    /// Main loop, just waits for frames and fires 'em off.
    void main_thread() override;

private:
    /// Kotekan buffer which this stage consumes from.
    /// Data should be packed into VDIF frames, see e.g. @c vdif_function.h.
    Buffer* in_buf;

    /// Port of the listening receiver.
    uint32_t dest_port;
    /// IP of the listening receiver.
    std::string dest_ip;

    /// Number of frequencies in the buffer
    int num_freq;
};

#endif
