/*
 * @file rfiBroadcast.hpp
 * @brief Contains RFI data broadcaster for SK estimates in kotekan.
 *  - rfiBroadcast : public kotekan::Stage
 */
#ifndef RFI_BROADCAST_H
#define RFI_BROADCAST_H

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "visUtil.hpp"           // for movingAverage

#include "json.hpp" // for json

#include <mutex>    // for mutex
#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class rfiBroadcast
 * @brief Consumer ``kotekan::Stage`` which consumes a buffer filled with spectral kurtosis
 * estimates.
 *
 * This stage reads RFI data from a kotekan buffer before packaging it into UDP packets and
 * sending them to a user defined IP address. Each packet is fitted with a header which can be read
 * by the server to ensure that the config parameters of the packet match the server config. This
 * stage simply reads the spectral kurtosis estimates, averages them for a single frame, averages
 * frames_per_packet frames together, packages the results into a packet (header + data), and sends
 * the packets to a user defined IP address via UDP.
 *
 * @par Buffers
 * @buffer rfi_in	The kotekan buffer containing spectral kurtosis estimates to be read by the
 * stage.
 * 	@buffer_format	Array of @c floats
 * 	@buffer_metadata chimeMetadata
 *
 * @par REST Endpoints
 * @endpoint    /rfi_broadcast ``POST`` Updates frames per broadcast packet
 *              requires json values      "frames_per_packet"
 *              update config             "frames_per_packet"
 *
 * @conf   num_elements         Int . Number of elements.
 * @conf   num_local_freq       Int . Number of local freq.
 * @conf   num_total_freq       Int (default 1024). Number of total freq.
 * @conf   samples_per_data_set Int . Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   frames_per_packet    Int (default 1). The Number of frames to average over before sending
 * each UDP pack$
 * @conf   rfi_combined         Bool (default true). Whether or not the kurtosis measurements
 * include an input sum.
 * @conf   total_links          Int (default 1). Number of FPGA links per buffer
 * @conf   dest_port            Int, The port number for the stream destination (Example: 41214)
 * @conf   dest_server_ip       String, The IP address of the stream destination (Example:
 * 192.168.52.174)
 * @conf   dest_protocol        String, Currently only supports 'UDP'
 *
 * @author Jacob Taylor
 */
class rfiBroadcast : public kotekan::Stage {
public:
    // Constructor
    rfiBroadcast(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    // Deconstructor, cleans up / does nothing
    virtual ~rfiBroadcast();
    // Primary loop, reads buffer and sends out UDP stream
    void main_thread() override;
    // Callback function called by rest server
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    // Callback function called by rest server
    void rest_zero(kotekan::connectionInstance& conn);

private:
    /// Kotekan buffer containing kurtosis estimates
    Buffer* rfi_buf;
    /// Kotekan buffer containing kurtosis estimates
    Buffer* rfi_mask_buf;
    // General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_total_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    // RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t _sk_step;
    /// Flag for element summation in kurtosis estimation process
    bool _rfi_combined;
    /// Flag to tell stage whether or not to use FPGA seq nums
    bool replay;
    /// Number of frames to average per UDP packet
    uint32_t _frames_per_packet;
    // Stage-specific config parameters
    /// The total number of links processed by gpu
    uint32_t total_links;
    /// The port for UDP stream to be sent to
    uint32_t dest_port;
    /// The address for UDP stream to be sent to
    std::string dest_server_ip;
    /// The streaming protocol, only UDP is supported
    std::string dest_protocol;
    /// Internal socket error holder
    int socket_fd;
    /// Rest server callback mutex
    std::mutex rest_callback_mutex;
    /// Rest server callback mutex
    std::mutex rest_zero_callback_mutex;
    /// String to hold endpoint
    std::string endpoint;
    /// String to hold endpoint
    std::string endpoint_zero;
    /// Moving average of frame zeroing percentage to send to prometheus
    movingAverage perc_zeroed;
    /// Prometheus metrics to export
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& sample_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& flagged_sample_counter;
};

#endif
