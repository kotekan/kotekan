/**
 * @file rfiBroadcast.hpp
 * @brief Contains RFI data broadcaster for VDIF data in kotekan.
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


/**
 * @class rfiBroadcast
 * @brief Consumer ``KotekanProcess`` which consumer a buffer filled with output from the kurtosis estimator
 *
 * This process reads RFI data from a kotekan buffer before packaging it into UDP packets and sending them
 * to a user defined IP address. This process will most likely be absorbed into a more general rfiBroadcast 
 * process at a later date, so the documentation will be brief.
 *
 * @par Buffers
 * @buffer rfi_in	The kotekan buffer containing kurtosis estimates of a VDIF style input which 
 * 			will be read by the process.
 * 	@buffer_format	Array of floats
 * 	@buffer_metadata none
 *
 * @conf dest_port	Int, The port number for the stream destination (Example: 41214)
 * @conf dest_server_ip	String, The IP address of the stream destination (Example: 192.168.52.174)
 * @conf dest_protocol	String, Currently only supports 'UDP' 
 *
 * @todo Merge this process into a general rfiBroadcast process
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
    //Kotekan buffer containing kurtosis estimates from VDIF input
    struct Buffer *rfi_buf;
    
    //Base Config Parameters
    //Total number of frequencies of VDIF input
    uint32_t _num_freq;
    uint32_t _num_local_freq;
    //Total number f elements of VDIF Input
    uint32_t _num_elements;
    //Total number of tmesteps of VDIF input
    uint32_t _samples_per_data_set;

    //RFI config parameters
    //The kurtosis step , how many timesteps per kurtosis estimate
    uint32_t  _sk_step;
    //Flag for element summation in kurtosis estimation process
    bool COMBINED;
    //Number of frames to average per UDP packet
    uint32_t frames_per_packet;

    //Process config parameters
    //The port for UDP stream to be sent to
    uint32_t dest_port;
    //The address for UDP stream to be sent to
    string dest_server_ip;
    //The streaming protocol, only UDP is supported
    string dest_protocol;

    //Internal socket error holder
    int socket_fd;
};

#endif
