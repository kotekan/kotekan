/**
 * @file
 * @brief Packetizer for data destined for CHIME/Pulsar.
 *  - pulsarPostProcess625 : public KotekanProcess
 */

#ifndef PULSAR_POST_PROCESS_625
#define PULSAR_POST_PROCESS_625

#include "KotekanProcess.hpp"
#include <vector>

using std::vector;

/**
 * @class pulsarPostProcess625
 * @brief Post-processing engine for output of the CHIME/Pulsar kernel, 
 *        makes packets with 625 time samples.
 *
 * This engine gathers CHIME/Pulsar data from the 4 GPU streams in each CHIME node, 
 * packing it into CHIME/Pulsar VDIF format packets, which are stored in the output buffer.
 * Each packet consists of 4 freq, 625 time samples, 2 pol, and one beam (4f-625t-2p-1b), 
 * where freq is the fastest varying index. The header is of VDIF format and contains 32-B. * Prior to packing, the real and imag part of the (float) input values are scaled and 
 * offset to 4-bit unsigned ints (i.e., 0-15) independently. 
 * The scaling factor is an static input value provided by the scheduler, 
 * and can be different on a per beam basis. 
 *
 * @par Buffers
 * @buffer network_input_buffer_0 Kotekan buffer feeding data from GPU0.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer network_input_buffer_1 Kotekan buffer feeding data from GPU1.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer network_input_buffer_2 Kotekan buffer feeding data from GPU2.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer network_input_buffer_3 Kotekan buffer feeding data from GPU3.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer pulsar_out_buf Kotekan buffer that will be populated with packetized data.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata chimeMetadata
 *
 * @conf   num_gpus             Int. No. of GPUs.
 * @conf   samples_per_data_set Int. No. of baseband samples corresponding to each buffer.
 * @conf   nfreq_coarse         Int. No. of freq per GPU node, (should be 4).
 * @conf   num_pulsar           Int. No. of total pulsar beams (should be 10).
 * @conf   num_pol              Int. No. of polarization (should be 2).
 * @conf   timesamples_per_pulsar_packet    Int. Number of times that will go into each packet. (should be 625)
 * @conf   udp_packet_size      Int. Size of packet, incl. header (should be 5032).
 * @conf   udp_header_size      Int. Size of header (should be 32).
 *
 * @author Cherry Ng
 *
 */


class pulsarPostProcess625 : public KotekanProcess {
public:
    /// Constructor.
    pulsarPostProcess625(Config& config_,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    /// Destructor
    virtual ~pulsarPostProcess625();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread();

    /// Initializes internal variables from config
    virtual void apply_config(uint64_t fpga_seq);

private:
    void fill_headers(unsigned char * out_buf,
                struct VDIFHeader * vdif_header,
                const uint64_t fpga_seq_num,
                struct timespec * time_now,
                struct psrCoord psr_coord,
                uint8_t freq_id);

    struct Buffer **in_buf;
    struct Buffer *pulsar_buf;

    // Config variables
    uint32_t _num_gpus;
    uint32_t _samples_per_data_set;
    uint32_t _nfreq_coarse;
    uint32_t _num_pulsar;
    uint32_t _num_pol;
    uint32_t _timesamples_per_pulsar_packet;
    uint32_t _udp_packet_size;
    uint32_t _udp_header_size;
    struct timespec time_now;
};

#endif
