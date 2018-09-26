/**
 * @file
 * @brief Packetizer for data destined for CHIME/Pulsar.
 *  - pulsarPostProcess : public KotekanProcess
 */

#ifndef PULSAR_POST_PROCESS
#define PULSAR_POST_PROCESS

#include "KotekanProcess.hpp"
#include <vector>

using std::vector;

/**
 * @class pulsarPostProcess
 * @brief Post-processing engine for output of the CHIME/Pulsar kernel,
 *        makes packets with either 625 or 3125 time samples.
 *
 * This engine gathers CHIME/Pulsar data from the 4 GPU streams in each CHIME node,
 * which are stored in the output buffer.
 * There are two accepted configurations: 
 * -- packet with 625 time samples: each packet consists of 4 freq, 2 pol, 
 *    and one beam (4f-625t-2p-1b), where freq is the fastest varying index. 
 *    All 4 freq are assumed to be observing the same 10 pulsars.
 * -- packet with 3125 time samples: each packet consists of 1 freq, 2 pol, 
 *    and one beam (1f-3125t-2p-1b), where freq is the fastest varying index. 
 *    A padding of 6-B is required, making the 3125-format not strictly VDIF, hence undesirable. 
 * In both cases, the header contains 32-B.
 * Prior to packing, the real and imag part of the (float) input values are scaled and
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
 * @conf   num_pulsar           Int. No. of total pulsar beams (should be 10).
 * @conf   num_pol              Int. No. of polarization (should be 2).
 * @conf   timesamples_per_pulsar_packet    Int. Number of times that will go into each packet. (should be 3125 or 625)
 *
 * @author Cherry Ng
 *
 */

class pulsarPostProcess : public KotekanProcess {
public:
    /// Constructor.
    pulsarPostProcess(Config& config_,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    /// Destructor
    virtual ~pulsarPostProcess();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;
    /// Initializes internal variables from config
    virtual void apply_config(uint64_t fpga_seq) override;

private:
    void fill_headers(unsigned char * out_buf,
                struct VDIFHeader * vdif_header,
                const uint64_t fpga_seq_num,
                struct timespec * time_now,
                struct psrCoord * psr_coord,
                uint16_t * freq_ids);

    struct Buffer **in_buf;
    struct Buffer *pulsar_buf;

    /// Config variables
    uint32_t _num_gpus;
    uint32_t _samples_per_data_set;
    uint32_t _num_pulsar;
    uint32_t _num_pol;
    /// number of time samples per packet (3125 or 625)
    uint32_t _timesamples_per_pulsar_packet;
    /// UDP packet size (6288 for 3125; 5032 for 625)
    uint32_t _udp_pulsar_packet_size;
    /// number of packet per stream (16 for 3125; 80 for 625)
    uint32_t _num_packet_per_stream;
    /// number of stream (40 for 3125; 10 for 625)
    uint32_t _num_stream;

    /// Derived variables
    struct timespec time_now;
    uint32_t unix_offset;
};

#endif
