#ifndef ACQ_UPLINK_H
#define ACQ_UPLINK_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string, basic_string


/**
 * @brief TCP uplink for visibilities (and optional gating data) to CHIME/CHORD collection server.
 *
 * @par Buffers
 * @buffer chrx_in_buf Input visibilities (consumer), any format/metadata expected by receiver.
 * @buffer gate_in_buf Optional gating buffer (consumer), sent if enabled.
 *
 * @conf chrx_in_buf            String. Input visibility buffer.
 * @conf gate_in_buf            String. Input gating buffer.
 * @conf collection_server_ip   String. Destination IP.
 * @conf collection_server_port Int. Optional; derived from hostname suffix if omitted.
 * @conf enable_gating          Bool. Send gate buffer alongside visibilities.
 *
 * @par Example
 * @code
 * chrxUplink:
 *   chrx_in_buf: vis_tx
 *   gate_in_buf: gate_tx
 *   collection_server_ip: 10.10.0.12
 *   enable_gating: true
 * @endcode
 */
class chrxUplink : public kotekan::Stage {
public:
    chrxUplink(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~chrxUplink();
    void main_thread() override;

private:
    Buffer* vis_buf;
    Buffer* gate_buf;

    // Config variables
    std::string _collection_server_ip;
    int32_t _collection_server_port;
    bool _enable_gating;
};

#endif
