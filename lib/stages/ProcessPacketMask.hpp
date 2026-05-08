#ifndef PROCESS_PACKET_MASK_HPP
#define PROCESS_PACKET_MASK_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @class ProcessPacketMask
 * @brief Processes voltage data using packet receipt bitmaps
 *
 * This stage takes voltage data and packet receipt bitmap data to process
 * and handle missing/received packets.
 *
 * @par Buffers
 * @buffer voltage_buf The input voltage buffer
 *     @buffer_format voltage_frame[time_long][frequency][element_long][time_short][element_short]
 *     @buffer_metadata chordMetadata
 * @buffer receipt_bitmap_bufs Array of packet receipt bitmap buffers
 *     @buffer_format packet_receipt_bitmap[time_long][source_id]
 *     @buffer_metadata chordMetadata
 * @buffer combined_receipt_bitmap_buf Output buffer for combined receipt bitmap
 *    @buffer_format combined_receipt_bitmap[time_long][source_id]
 *    @buffer_metadata chordMetadata
 * @buffer pl_mask_buf Output buffer for packet loss mask
 *    @buffer_format uint64_t pl_mask[T/64][F][E/8] where T=time_long*time_short, F=num_frequency,
 * E=element_long*element_short
 *    @buffer_metadata chordMetadata
 *
 * @conf time_long      Int. Number of long time samples per frame.
 * @conf num_frequency  Int. Number of frequency bins.
 * @conf element_long   Int. Number of long element samples.
 * @conf time_short     Int. Number of short time samples (default: 16).
 * @conf element_short  Int. Number of short element samples (default: 8).
 * @conf num_source_ids Int. Number of source IDs in the receipt bitmap.
 *
 * @author Andre Renard
 */
class ProcessPacketMask : public kotekan::Stage {
public:
    ProcessPacketMask(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~ProcessPacketMask();
    void main_thread() override;

private:
    /// Input voltage buffer
    Buffer* voltage_buf;

    /// Combined receipt bitmap output buffer
    Buffer* combined_receipt_bitmap_buf;

    /// Packet loss mask output buffer
    Buffer* pl_mask_buf;

    /// RFI mask output buffer
    Buffer* rfi_mask_buf;

    /// Array of packet receipt bitmap buffers
    std::vector<Buffer*> receipt_bitmap_bufs;

    /// Frame dimensions for voltage data
    uint32_t time_long;
    uint32_t num_frequency;
    uint32_t element_long;
    uint32_t time_short;
    uint32_t element_short;

    /// Number of source IDs in receipt bitmap
    uint32_t num_source_ids;

    /// Number of stream IDs (bits per source_id byte)
    static constexpr uint32_t num_stream_ids = 8;

    /// Number of frequencies per stream
    static constexpr uint32_t num_freq_per_stream = 48;

    /// List of missing source IDs
    std::vector<uint32_t> missing_source_ids;
    uint64_t missing_source_id_mask[2];

    /// Long-term packet loss counters
    uint64_t total_packets_received = 0;
    uint64_t total_packets_missing = 0;
    uint64_t total_frames_processed = 0;
};

#endif // PROCESS_PACKET_MASK_HPP
