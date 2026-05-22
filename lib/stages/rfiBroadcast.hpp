/*
 * @file rfiBroadcast.hpp
 * @brief Send RFI metrics via UDP to an external broker.
 *  - rfiBroadcast : public kotekan::Stage
 */
#ifndef RFI_BROADCAST
#define RFI_BROADCAST

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <arpa/inet.h>
#include <string> // for string

/*
 * @class rfiBroadcast
 * @brief Computes and broadcasts UDP packets with several reduced RFI statistics.
 *
 * This stage consumes multiple outputs from `cudaRFISKBar` and `cudaRFISKTilde`, and
 * computes a separate metric from each input. All metrics are computed for each
 * local frequency.
 *
 * Fraction of flagged samples:
 *    - Fraction of the total number of flagged samples across
 *      all accumulated frames. Derived from `rfi_mask_buf`.
 * Average SK:
 *    - Average spectral kurtosis value across all accumulated frames,
 *      derived from the feed-averaged SK `sk_tilde_buf`.
 * Bad feed count:
 *    - Number of time samples exceeding `num_sigma_deviations` standard deviations
 *      from the median-subtracted per-feed SK `sk_bar_buf`.
 *
 * All of these inputs are accumulated for `frames_per_packet` frames and packed
 * into a UDP packet with format:
 *    - header: struct `RFIHeader`
 *    - freq_ids: uint32 [num_local_freq]
 *    - num_flagged: float [num_local_freq]
 *    - sktilde_avg: float [num_local_freq]
 *    - bad_feed_count: uint8 [num_local_freq * num_elements]
 *
 * For CHIME with `num_element=2048`, this works out to 2.06 kB per frequency,
 * plus a fixed 46 byte header. See `lib/utils/rfu_functions.h` for the header
 * definition.
 *
 * @par Buffers
 * @buffer sk_tilde_buf        Kotekan buffer containing feed-averaged spectral kurtosis.
 *     @buffer_format float32
 *     @buffer_shape [samples_per_data_set / rfi_downsampling_factor][num_local_freq][3]
 *     @buffer_metadata chordMetadata
 * @buffer sk_bar_buf          Kotekan buffer containing coarse-time per-feed spectral kurtosis.
 *     @buffer_format float32
 *     @buffer_shape [samples_per_data_set / rfi_downsampling_factor /
 *                     second_rfi_downsampling_factor][num_local_freq][3][num_elements]
 *     @buffer_metadata chordMetadata
 * @buffer rfi_mask_buf        Kotekan buffer containing the RFI mask in bit mask format.
 *     @buffer_format uint1x8
 *     @buffer_shape    As uint8: [samples_per_data_set / 1024][num_local_freq][128]
 *                      As uint1: [samples_per_data_set / 1024][num_local_freq][1024]
 *     @buffer_metadata chordMetadata
 *
 * @conf num_elements          Int. Number of elements/feeds.
 * @conf num_local_freq        Int. Number of frequencies per node.
 * @conf num_global_freq       Int. Total number of coarse system frequencies.
 * @conf samples_per_data_set  Int. Number of FPGA time samples per buffer.
 * @conf rfi_downsampling_factor    Int. Downsampling factor for high-cadence SK
 *                                   relative to `samples_per_data_set`.
 * @conf rfi_second_downsampling_factor    Int. Additional downsampling factor for low-cadence
 *                                         SK, where the overall cadence is `samples_per_data_set` /
 *                                         `rfi_downsampling_factor` /
 * `rfi_second_downsampling_factor`
 * @conf frames_per_packet                 Int. Number of frames to accumulate per UDP packet.
 * @conf dest_port                         Int. Destination port number.
 * @conf dest_ip                           String. Destination IP address.
 * @conf num_sigma_deviations              Int. Number of sigma deviations to use when
 * computing per-feed excursion count.
 *
 * @author Liam Gray
 */
class rfiBroadcast : public kotekan::Stage {
public:
    // Constructor
    rfiBroadcast(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    // Deconstructor
    virtual ~rfiBroadcast();
    // Primary loop, reads from various inputs and sends a UDP stream
    void main_thread() override;

private:
    // Buffer containing feed-averaged SK estimate
    Buffer* sk_tilde_buf;
    // Buffer containing per-feed SK estimate
    Buffer* sk_bar_buf;
    // Buffer containing derived RFI mask
    Buffer* rfi_mask_buf;

    // General config parameters
    size_t num_elements;                   // number of elements/feeds
    size_t num_local_freq;                 // number of frequencies per buffer
    size_t num_global_freq;                // total number of frequencies
    size_t samples_per_data_set;           // base number of FPGA time samples
    size_t rfi_downsampling_factor;        // Downsampling rate to high-cadence SK
    size_t rfi_second_downsampling_factor; // Additional downsampling factor for low-cadence SK

    // Derived buffer shapes
    size_t _downsampled_samples_per_data_set;
    size_t _second_downsampled_samples_per_data_set;

    // Packet/network
    size_t frames_per_packet; // Number of frames to gather into each packet
    size_t dest_port;         // broker port
    std::string dest_ip;      // broker IP

    // The number of standard deviations required
    // for a sample to be considered an outlier.
    uint16_t num_sigma_deviations;

    // Other
    struct sockaddr_in dest_addr; // Destination socket address
};

#endif
