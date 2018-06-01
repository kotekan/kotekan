/**
 * @file rfiBroadcast.hpp
 * @brief Contains RFI data broadcaster for SK estimates in kotekan.
 *  - rfiBroadcast : public KotekanProcess
 */
#ifndef RFI_BROADCAST_H
#define RFI_BROADCAST_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "chimeMetadata.h"

/*
 * @struct RFIHeader
 * @brief A structure that contains the header attached to each rfiBroadcast packet.
 */
struct __attribute__ ((packed)) RFIHeader {
    /// uint8_t indicating whether or not the SK value was summed over inputs.
    uint8_t rfi_combined;
    /// uint32_t indicating the time intergration length of the SK values.
    uint32_t sk_step;
    /// uint32_t indicating the number of inputs in the input data.
    uint32_t num_elements;
    /// uint32_t indicating the number of timesteps in each frame.
    uint32_t samples_per_data_set;
    /// uint32_t indicating the total number of frequencies under consideration (1024 by default).
    uint32_t num_total_freq;
    /// uint32_t indicating the number of frequencies in the packet.
    uint32_t num_local_freq;
    /// uint32_t indicating the number of frames which were averaged over.
    uint32_t frames_per_packet;
    /// int64_t containing the FPGA sequence number of the first packet in the average.
    int64_t seq_num;
    /// uint16_t holding the current stream ID value.
    uint16_t streamID;
};

/**
 * @class rfiBroadcast
 * @brief Consumer ``KotekanProcess`` which consumer a buffer filled with spectral kurtosis estimates.
 *
 * This process reads RFI data from a kotekan buffer before packaging it into UDP packets and sending them 
 * to a user defined IP address. Each packet is fitted with a header which can be read by the server to ensure 
 * that the config parameters of the packet match the server config. This process simply reads the spectral 
 * kurtosis estimates, averages them for a single frame, averages frames_per_packet frames toegther, packages 
 * the results into a packet (header + data), and sends the packets to a user defined IP address via UDP.
 *
 * @par Buffers
 * @buffer rfi_in	The kotekan buffer containing spectral kurtosis estimates to be read by the process.
 * 	@buffer_format	Array of @c floats
 * 	@buffer_metadata chimeMetadata
 *
 * @conf   num_elements         Int (default 2048). Number of elements.
 * @conf   num_local_freq       Int (default 1). Number of local freq.
 * @conf   num_local_freq       Int (default 1024). Number of total freq.
 * @conf   samples_per_data_set Int (default 32768). Number of time samples in a data set.
 * @conf   sk_step              Int Length of time integration in SK estimate.
 * @conf   frames_per_packet    Int The Number of frames to average over before sending each UDP packet.
 * @conf   rfi_combined         Bool Whether or not the kurtosis measurements include an input sum.
 * @conf   dest_port            Int, The port number for the stream destination (Example: 41214)
 * @conf   dest_server_ip       String, The IP address of the stream destination (Example: 192.168.52.174)
 * @conf   dest_protocol        String, Currently only supports 'UDP'
 *
 * @todo Merge this process with rfiBroadcastVdif
 *
 * @author Jacob Taylor
 */
class rfiBroadcast : public KotekanProcess {
public:
    //Constructor, intializes config variables via apply_config
    rfiBroadcast(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);

    //Deconstructor, cleans up / does nothing
    virtual ~rfiBroadcast();

    //Primary loop, reads buffer and sends out UDP stream
    void main_thread();

    //Intializes config variables
    virtual void apply_config(uint64_t fpga_seq);

private:
    /// Kotekan buffer containing kurtosis estimates
    struct Buffer *rfi_buf;

    //General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;

    //RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t  _sk_step;
    /// Flag for element summation in kurtosis estimation process
    bool COMBINED;
    /// Number of frames to average per UDP packet
    uint32_t frames_per_packet;

    //Process specific config parameters
    /// The total number of links processed by gpu
    uint32_t total_links;
    /// The port for UDP stream to be sent to
    uint32_t dest_port;
    /// The address for UDP stream to be sent to
    string dest_server_ip;
    /// The streaming protocol, only UDP is supported
    string dest_protocol;
    /// Internal socket error holder
    int socket_fd;
};

#endif
