/**
 * @file
 * @brief Stage to receive VDIF data from a UDP stream.
 *  - recvSingleDishVDIF : public kotekan::Stage
 */

#ifndef RECV_SINGLE_DISH_VDIF_H
#define RECV_SINGLE_DISH_VDIF_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string

/**
 * @class recvSingleDishVDIF
 * @brief Stage to transmit VDIF data as a UDP stream.
 *
 * This is a producer stage which gathers VDIF-formatted data from a UDP stream and
 * packs it into a target buffer.
 *
 * @par Buffers
 * @buffer out_buf Output kotekan buffer containing VDIF data to be transmitted.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   num_freq         Int. Number of frequencies in each VDIF packet.
 * @conf   orig_port        Int. UDP port to listen on.
 * @conf   orig_server_ip   String. Source IP to bind to (optional).
 *
 * @par Example
 * @code
 * recv_single_dish_vdif:
 *   kotekan_stage: recvSingleDishVDIF
 *   out_buf: vdif_in
 *   num_freq: 1024
 *   orig_port: 12000
 *   orig_server_ip: 0.0.0.0
 * @endcode
 *
 * @bug    Currently broken, just returns empty frames!
 *
 * @author Andre Renard
 *
 */
class recvSingleDishVDIF : public kotekan::Stage {
public:
    /// Constructor
    recvSingleDishVDIF(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~recvSingleDishVDIF();

    /// Main loop, just waits for network data and stuffs info a frame.
    void main_thread() override;

private:
    Buffer* out_buf;

    /// Port of the listening receiver.
    uint32_t orig_port;
    /// IP of the listening receiver.
    std::string orig_ip;

    /// Number of frequencies in the buffer
    int num_freq;
};

#endif
