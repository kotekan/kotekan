/**
 * @file pulsarSimProcess.hpp
 * @brief Packet simulator for pulsar transmission testing
 *  - pulsarSimProcess : public Stage
 */

#ifndef PULSAR_SIM_PROCESS
#define PULSAR_SIM_PROCESS

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for int32_t, uint16_t, uint64_t
#include <string>   // for string


/**
 * @class pulsarSimProcess
 * @brief pulsarSimProcess Simulates packets used for testing the transmission code
 *
 *
 * This is a Kotekan stage that simulated the VDIF headers for the pulsar transmission code.
 * Presently this stage simulates just the headers and this is enough for current level of testing
 * This stage is coded to simulate packets for all the 10 beams
 * In oreder to map the packets conviently to the L0 nodes the frequency ids are derived from the
 *hostname. The node IP address is derived by parsing the hostname.
 *
 * @par Buffers
 * @buffer out_buf The kotkean buffer to hold the packets to be transmitted to pulsar nodes
 *      @buffer_format Array of unsigned char.
 *      @buffer_metadata none
 *
 * @conf   udp_pulsar_packet_size  Int (default 6288). packet size including header
 * @conf   udp_pulsar_header_size  Int (default 32).
 * @conf   number_of_subnets    Int (default 2). Number of subnets or VLANS used for transmission of
 *PULSAR data
 *
 * @todo   simulated random data, this will need more CPU resorces not requied for present testing
 *
 * @author Arun Naidu
 *
 *
 **/

class pulsarSimProcess : public kotekan::Stage {
public:
    /// constructor
    pulsarSimProcess(kotekan::Config& config_, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);

    /// distructor
    virtual ~pulsarSimProcess();

    void main_thread() override;

    /// parses the hostname to get the unique node_id used for freqency_id
    void parse_host_name();

private:
    void fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header,
                      const uint64_t fpga_seq_num, struct timeval* time_now,
                      struct psrCoord* psr_coord, uint16_t* freq_ids);

    struct Buffer* pulsar_buf;

    /// number of GPUs same as number of frequencies per node.
    int32_t _num_gpus;

    /// same as _num_gpus different variable is used just for convience.
    int32_t _nfreq_coarse;

    /// number of pulsar beams. default: 10
    int32_t _num_pulsar;

    /// number of polarizations. default: 2
    int32_t _num_pol;

    /// packet size default: 6288
    int32_t _udp_packet_size;

    /// header size of the VDIF packet. default 32
    int32_t _udp_header_size;

    /// node id derived from the hostname
    int my_node_id;

    /// host name from the gethosename()
    char* my_host_name = new char[20];

    // number of subnets
    int number_of_subnets;

    /// node ip addresses
    std::string my_ip_address[2];
};

#endif
