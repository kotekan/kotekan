#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string

class vdifStream : public kotekan::Stage {
public:
    vdifStream(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~vdifStream();
    void main_thread() override;

private:
    /**
     * @brief Streams VDIF packets from a buffer to a remote UDP receiver.
     *
     * Consumes `vdif_in_buf`, opens a UDP socket to `vdif_server_ip:vdif_port`, and sends each
     * 5032-byte packet from the frame in order with a small adaptive sleep to maintain ~1 s per
     * frame. No data transformation is performed.
     *
     * @par Buffers
     * @buffer vdif_in_buf Input VDIF packet buffer.
     *     @buffer_format Raw VDIF packets (5032-byte) concatenated
     *     @buffer_metadata none
     *
     * @conf vdif_in_buf    String. Input buffer.
     * @conf vdif_port      UInt. Destination UDP port.
     * @conf vdif_server_ip String. Destination IP address.
     *
     * @par Example
     * @code
     * vdifStream:
     *   vdif_in_buf: vdif_out
     *   vdif_port: 14002
     *   vdif_server_ip: 192.168.1.10
     * @endcode
     */
    Buffer* buf;

    uint32_t _vdif_port;
    std::string _vdif_server_ip;
};

#endif
