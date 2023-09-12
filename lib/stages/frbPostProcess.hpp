/**
 * @file
 * @brief Packetizer for data destined for CHIME/FRB L1.
 *  - frbPostProcess : public kotekan::Stage
 */

#ifndef FRB_POST_PROCESS
#define FRB_POST_PROCESS

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "frb_functions.h"       // for FRBHeader
#include "prometheusMetrics.hpp" // for Counter

#include <stdint.h> // for int32_t, uint16_t, int16_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class frbPostProcess
 * @brief Post-processing engine for data coming out of the CHIME/FRB kernel stack.
 *
 * This engine gathers CHIME/FRB data from the 4 GPU streams in each CHIME node,
 * packing it into CHIME/FRB L0-L1 packets, which are stored in the output buffer.
 * Prior to packing, the (float) input values are scaled and offset to 8-bit
 * unsigned ints (i.e., 0-255). The scaling is determined on a per-packet basis,
 * using AVX2 instructions to calculate and apply those parameters.
 *
 * This stage can also optionally produce a sum-of-all-beams "incoherent" beam,
 * which will be stored in the 0th beam position in output packets.
 *
 * Time samples with dropped packet are set to zero.
 *
 * This stage depends on ``AVX2`` intrinsics.
 *
 * @par Buffers
 * @buffer in_buf_0 Kotekan buffer feeding data from GPU0.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer in_buf_1 Kotekan buffer feeding data from GPU1.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer in_buf_2 Kotekan buffer feeding data from GPU2.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer in_buf_3 Kotekan buffer feeding data from GPU3.
 *     @buffer_format Array of @c floats
 *     @buffer_metadata chimeMetadata
 * @buffer out_buf Kotekan buffer that will be populated with packetized data.
 *     @buffer_format Array of @c uchars
 *     @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Koktekan buffer with drop packet info, where 1=dropped
 *     @buffer_metadata chimeMetadata
 *     @buffer_format Array of @c uint8
 *
 * @conf   num_gpus                   Int. Number of GPUs.
 * @conf   samples_per_data_set       Int. Number of baseband samples corresponding to each buffer.
 * @conf   downsample_time            Int. Number of time samples summed in the FRB kernels.
 * @conf   factor_upchan              Int. Total upchannelization in the FRB kernels.
 * @conf   factor_upchan_out          Int. Upchannelization that exits the FRB kernels,
 *                                        also number of freqs that will go into each output packet
 * @conf   num_beams_per_frb_packet   Int. Number of FRB beam in each output stream. (?) Should be
 * 4?
 * @conf   timesamples_per_frb_packet Int. Number of times that will go into each packet.
 * @conf   incoherent_beam            Bool (default=false). Form the incoherent beam or not.
 *                                        If true, will stuff incoherent beam in position 0.
 * @conf   incoherent_truncate        Float (default=1e10). To deal with inputs / times /freqs with
 *                                        anomalously high values, this limits values used prior to
 * summing into the incoherent beam.
 *
 * @par Metrics
 * @metric kotekan_frb_masked_packets_total
 *         Count of masked packets
 *
 * @author Keith Vanderlinde, Cherry Ng
 *
 */
class frbPostProcess : public kotekan::Stage {
public:
    /// Constructor.
    frbPostProcess(kotekan::Config& config_, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~frbPostProcess();

    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;

private:
    void write_header(unsigned char* dest);

    /// Lost sample - drop packet
    Buffer* lost_samples_buf;
    int32_t lost_samples_buf_id;

    struct Buffer** in_buf;
    struct Buffer* frb_buf;

    struct FRBHeader frb_header;
    float* ib;

    // Dynamic header
    uint16_t* frb_header_beam_ids;
    uint16_t* frb_header_coarse_freq_ids;
    float* frb_header_scale;
    float* frb_header_offset;

    // kotekan::Config variables
    int32_t _num_gpus;
    int32_t _samples_per_data_set;
    int32_t _downsample_time;
    int32_t _factor_upchan;
    int32_t _factor_upchan_out;
    int32_t _nbeams;
    int32_t _timesamples_per_frb_packet;
    std::vector<int32_t> _incoherent_beams;
    float _incoherent_truncation;

    // Derived useful things
    int32_t num_L1_streams;
    uint32_t num_samples;
    int32_t udp_packet_size;
    int32_t udp_header_size;
    int16_t fpga_counts_per_sample;

    uint8_t* droppacket;

    /// Count of masked packets
    kotekan::prometheus::Counter& masked_packets_counter;
};

#endif
