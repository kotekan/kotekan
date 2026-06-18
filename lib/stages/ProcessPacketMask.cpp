#include "ProcessPacketMask.hpp"

#include <visUtil.hpp>          // for frameID, modulo
#include <assert.h>             // for assert
#include <cstring>              // for size_t, memset
#include <stdexcept>            // for runtime_error
#include <memory>               // for __shared_ptr_access, shared_ptr

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"   // for INFO, DEBUG2
#include "fmt.hpp"              // for compile_string_to_view, format, fmt
#include "DataType.hpp"         // for DataType

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(ProcessPacketMask);

STAGE_CONSTRUCTOR(ProcessPacketMask) {
    // Register as consumer on the voltage buffer
    voltage_buf = get_buffer("voltage_buf");
    voltage_buf->register_consumer(unique_name);

    combined_receipt_bitmap_buf = get_buffer("combined_receipt_bitmap_buf");
    combined_receipt_bitmap_buf->register_producer(unique_name);

    pl_mask_buf = get_buffer("pl_mask_buf");
    pl_mask_buf->register_producer(unique_name);

    // The RFI mask here is just a hack to avoid running the actual RFI kernels
    // It just replays a mask of whatever is in the frames at startup
    // (This should be set to 0xff as the "zeroing" value.)
    rfi_mask_buf = get_buffer("rfi_mask_buf");
    rfi_mask_buf->register_producer(unique_name);

    // Get the array of receipt bitmap buffer names and register as consumer on each
    std::vector<std::string> receipt_bitmap_buf_names =
        config.get<std::vector<std::string>>(unique_name, "receipt_bitmap_bufs");

    for (const auto& buf_name : receipt_bitmap_buf_names) {
        Buffer* buf = buffer_container.get_buffer(buf_name);
        buf->register_consumer(unique_name);
        receipt_bitmap_bufs.push_back(buf);
    }

    if (receipt_bitmap_bufs.empty()) {
        throw std::runtime_error(
            "ProcessPacketMask: At least one receipt_bitmap_buf must be provided");
    }

    // Get frame dimension configuration
    time_long = config.get<uint32_t>(unique_name, "time_long");
    num_frequency = config.get<uint32_t>(unique_name, "num_frequency");
    element_long = config.get<uint32_t>(unique_name, "element_long");
    time_short = config.get_default<uint32_t>(unique_name, "time_short", 16);
    element_short = config.get_default<uint32_t>(unique_name, "element_short", 8);
    num_source_ids = config.get<uint32_t>(unique_name, "num_source_ids");

    // Get a list of missing source_ids, e.g. which CRS boards are currently offline.
    missing_source_ids =
        config.get_default<std::vector<uint32_t>>(unique_name, "missing_source_ids", {});
    // Create mask of missing source IDs for OR'ing with receipt bitmaps
    missing_source_id_mask[0] = 0;
    missing_source_id_mask[1] = 0;
    for (const auto& src_id : missing_source_ids) {
        if (src_id >= num_source_ids) {
            throw std::runtime_error(fmt::format(fmt("ProcessPacketMask: missing source_id {:d} is "
                                                     "out of range (num_source_ids={:d})"),
                                                 src_id, num_source_ids));
        }
        if (src_id < 8) {
            missing_source_id_mask[0] |= (0xffull << src_id * 8);
        } else {
            missing_source_id_mask[1] |= (0xffull << (src_id - 8) * 8);
        }
    }

    // print the missing source ID mask
    INFO("ProcessPacketMask: Missing source ID mask: 0x{:016x}{:016x}", missing_source_id_mask[1],
         missing_source_id_mask[0]);

    // Validate voltage buffer size
    size_t expected_voltage_size =
        (size_t)time_long * num_frequency * element_long * time_short * element_short;
    if (voltage_buf->frame_size != expected_voltage_size) {
        throw std::runtime_error(fmt::format(
            fmt("ProcessPacketMask: voltage_buf frame size ({:d}) does not match expected "
                "size ({:d}) for shape [time_long={:d}][frequency={:d}][element_long={:d}]"
                "[time_short={:d}][element_short={:d}]"),
            voltage_buf->frame_size, expected_voltage_size, time_long, num_frequency, element_long,
            time_short, element_short));
    }

    // Validate receipt bitmap buffer sizes
    size_t expected_bitmap_size = (size_t)time_long * num_source_ids;
    for (size_t i = 0; i < receipt_bitmap_bufs.size(); i++) {
        if (receipt_bitmap_bufs[i]->frame_size != expected_bitmap_size) {
            throw std::runtime_error(fmt::format(
                fmt("ProcessPacketMask: receipt_bitmap_bufs[{:d}] frame size ({:d}) does not match "
                    "expected size ({:d}) for shape [time_long={:d}][source_id={:d}]"),
                i, receipt_bitmap_bufs[i]->frame_size, expected_bitmap_size, time_long,
                num_source_ids));
        }
    }

    // Validate combined receipt bitmap buffer size
    if (combined_receipt_bitmap_buf->frame_size != expected_bitmap_size) {
        throw std::runtime_error(fmt::format(
            fmt("ProcessPacketMask: combined_receipt_bitmap_buf frame size ({:d}) does not match "
                "expected size ({:d}) for shape [time_long={:d}][source_id={:d}]"),
            combined_receipt_bitmap_buf->frame_size, expected_bitmap_size, time_long,
            num_source_ids));
    }

    // Validate pl_mask buffer size
    // pl_mask shape: uint64_t [T/64][F][E/8] where T = time_long * time_short, E = element_long *
    // element_short
    size_t T = (size_t)time_long * time_short;
    size_t E = (size_t)element_long * element_short;
    size_t expected_pl_mask_size = (T / 64) * num_frequency * (E / 8) * sizeof(uint64_t);
    if (pl_mask_buf->frame_size != expected_pl_mask_size) {
        throw std::runtime_error(fmt::format(
            fmt("ProcessPacketMask: pl_mask_buf frame size ({:d}) does not match "
                "expected size ({:d}) for shape [T/64={:d}][F={:d}][E/8={:d}] * sizeof(uint64_t)"),
            pl_mask_buf->frame_size, expected_pl_mask_size, T / 64, num_frequency, E / 8));
    }

    pl_mask_buf->allocate_ndarray_frame_desc(
        kotekan::uint1x8, "pl_mask_exp",
        {time_long * time_short / 64, num_frequency, 2, element_long * element_short / 8 / 2, 8},
        {"Thi64", "F", "P", "D8", "Tlo64"});

    // Validate rfi_mask buffer size
    // rfi_mask shape: uint8_t [T/1024][F][128] where T = time_long * time_short
    size_t expected_rfi_mask_size = (T / 1024) * num_frequency * 128 * sizeof(uint8_t);
    if (rfi_mask_buf->frame_size != expected_rfi_mask_size) {
        throw std::runtime_error(
            fmt::format(fmt("ProcessPacketMask: rfi_mask_buf frame size ({:d}) does not match "
                            "expected size ({:d}) for shape [T/1024={:d}][F={:d}][128]"),
                        rfi_mask_buf->frame_size, expected_rfi_mask_size, T / 1024, num_frequency));
    }

    rfi_mask_buf->allocate_ndarray_frame_desc(kotekan::uint1x8, "RFImask",
                                              {time_long * time_short / 1024, num_frequency, 128},
                                              {"T8hi128", "F", "T8lo128"});

    INFO("ProcessPacketMask: Initialized with {:d} receipt bitmap buffers",
         receipt_bitmap_bufs.size());
    INFO("ProcessPacketMask: Voltage frame shape: [{:d}][{:d}][{:d}][{:d}][{:d}]", time_long,
         num_frequency, element_long, time_short, element_short);
    INFO("ProcessPacketMask: Receipt bitmap shape: [{:d}][{:d}]", time_long, num_source_ids);
    INFO("ProcessPacketMask: PL mask shape: [{:d}][{:d}][{:d}] (uint64_t)", T / 64, num_frequency,
         E / 8);
}

ProcessPacketMask::~ProcessPacketMask() {}

void ProcessPacketMask::main_thread() {
    frameID voltage_frame_id(voltage_buf);
    frameID combined_bitmap_frame_id(combined_receipt_bitmap_buf);
    frameID pl_mask_frame_id(pl_mask_buf);
    frameID rfi_mask_frame_id(rfi_mask_buf);
    std::vector<frameID> bitmap_frame_ids;
    for (auto* buf : receipt_bitmap_bufs) {
        bitmap_frame_ids.emplace_back(buf);
    }

    // Constants for pl_mask generation
    const size_t E_DIV_8 = (size_t)element_long * element_short / 8; // 16 for default config
    const size_t F = num_frequency;                                  // 384

    while (!stop_thread) {
        // Wait for voltage frame
        uint8_t* voltage_frame = voltage_buf->wait_for_full_frame(unique_name, voltage_frame_id);
        if (voltage_frame == nullptr)
            break;

        // Wait for combined receipt bitmap frame
        uint8_t* combined_bitmap_frame = combined_receipt_bitmap_buf->wait_for_empty_frame(
            unique_name, combined_bitmap_frame_id);
        if (combined_bitmap_frame == nullptr)
            break;

        // Wait for pl_mask frame
        uint8_t* pl_mask_frame = pl_mask_buf->wait_for_empty_frame(unique_name, pl_mask_frame_id);
        if (pl_mask_frame == nullptr)
            break;

        // Wait for rfi_mask frame
        uint8_t* rfi_mask_frame =
            rfi_mask_buf->wait_for_empty_frame(unique_name, rfi_mask_frame_id);
        if (rfi_mask_frame == nullptr)
            break;

        // Wait for all receipt bitmap frames
        std::vector<uint8_t*> bitmap_frames;
        bool got_all_frames = true;
        for (size_t i = 0; i < receipt_bitmap_bufs.size(); i++) {
            uint8_t* bitmap_frame =
                receipt_bitmap_bufs[i]->wait_for_full_frame(unique_name, bitmap_frame_ids[i]);
            if (bitmap_frame == nullptr) {
                got_all_frames = false;
                break;
            }
            bitmap_frames.push_back(bitmap_frame);
        }

        if (!got_all_frames)
            break;

        DEBUG2("ProcessPacketMask: Processing voltage frame {:d} with {:d} bitmap frames",
               (int)voltage_frame_id, bitmap_frames.size());

        // Combine receipt bitmaps casting to uint64_t for faster processing
        uint64_t* combined_bitmap_ptr = (uint64_t*)combined_bitmap_frame;
        size_t num_bitmap_words = combined_receipt_bitmap_buf->frame_size / sizeof(uint64_t);
        for (size_t w = 0; w < num_bitmap_words; w++) {
            uint64_t combined_word = missing_source_id_mask[w % 2];
            for (size_t b = 0; b < bitmap_frames.size(); b++) {
                uint64_t* bitmap_ptr = (uint64_t*)bitmap_frames[b];
                combined_word |= bitmap_ptr[w];
            }
            combined_bitmap_ptr[w] = combined_word;
        }

        // Count the number of missing packets in the combined bitmap
        size_t missing_packet_count = 0;
        for (size_t w = 0; w < num_bitmap_words; w++) {
            uint64_t word = combined_bitmap_ptr[w] ^ 0xffffffffffffffffull;
            if (word != 0) {
                // Count number of set bits in word
                missing_packet_count += __builtin_popcountll(word);
            }
        }
        DEBUG2("ProcessPacketMask: Voltage frame {:d} has {:d} missing packets in combined bitmap",
               (int)voltage_frame_id, missing_packet_count);

        // Calcuate a packet loss rate based on the missing packet count
        // and the number of working source IDs
        size_t working_source_ids = num_source_ids - missing_source_ids.size();
        size_t frame_total_packets = time_long * working_source_ids * 8; // num_stream_ids = 8
        double packet_loss_rate = 0.0;
        assert(frame_total_packets > 0);
        packet_loss_rate = (double)missing_packet_count / (double)frame_total_packets;
        double packet_loss_percentage = packet_loss_rate * 100.0;

        // Update long-term counters
        total_packets_received += frame_total_packets;
        total_packets_missing += missing_packet_count;
        total_frames_processed++;

        // Calculate cumulative packet loss rate
        double cumulative_loss_percentage = 0.0;
        if (total_packets_received > 0) {
            cumulative_loss_percentage =
                (double)total_packets_missing / (double)total_packets_received * 100.0;
        }

        INFO("ProcessPacketMask: Voltage frame {:d} packet loss = {:.4f}% ({:d} missing "
             "/ {:d} total) | Cumulative: {:.4f}%",
             (int)voltage_frame_id, packet_loss_percentage, (int)missing_packet_count,
             (int)frame_total_packets, cumulative_loss_percentage);

        // Generate the packet loss mask (pl_mask)
        // pl_mask shape: uint64_t [T/64][F][E/8]
        // Each bit represents whether data is good (1) or missing (0) at T resolution
        // Initialize to all 1s (all data good)
        uint64_t* pl_mask_ptr = (uint64_t*)pl_mask_frame;
        std::memset(pl_mask_frame, 0xff, pl_mask_buf->frame_size);

        // Process combined_receipt_bitmap to set zeros in pl_mask for missing packets
        // combined_receipt_bitmap layout: [time_long][source_id] with 8 stream_id bits per byte
        // Each zero bit means: set 16 time samples to 0 at all frequencies for that stream_id
        //                      for all E values corresponding to that source_id
        for (size_t w = 0; w < num_bitmap_words; w++) {
            uint64_t word = combined_bitmap_ptr[w];

            // Quick check: if all bits are 1, no packet loss in this word
            if (word == 0xffffffffffffffffull)
                continue;

            // Calculate which time_long indices and source_ids this word covers
            // Each word is 8 bytes, layout is [time_long][source_id] with source_id = 16 bytes
            size_t byte_offset = w * 8;
            size_t time_long_base = byte_offset / num_source_ids;
            size_t base_source_id = byte_offset % num_source_ids;

            // Process each byte (source_id) in the word
            for (size_t byte_idx = 0; byte_idx < 8; byte_idx++) {
                uint8_t byte_val = (word >> (byte_idx * 8)) & 0xff;

                // Quick check: if all bits are 1, no packet loss for this source_id
                if (byte_val == 0xff)
                    continue;

                size_t source_id = base_source_id + byte_idx;
                size_t time_long_idx = time_long_base + source_id / num_source_ids;

                // Process each stream_id bit
                for (uint32_t stream_bit = 0; stream_bit < num_stream_ids; stream_bit++) {
                    if ((byte_val & (1 << stream_bit)) == 0) {
                        // Packet loss detected!
                        // Set 16 time bits to 0 in pl_mask for all frequencies of this stream_id

                        // Compute T/64 index and bit position within the uint64_t
                        // T = time_long * time_short, time sample = time_long_idx * time_short
                        // T/64 index = (time_long_idx * time_short) / 64 = time_long_idx / 4 (when
                        // time_short=16) Bit position = (time_long_idx % 4) * 16
                        size_t t64_idx = (time_long_idx * time_short) / 64;
                        size_t bit_start = (time_long_idx * time_short) % 64;

                        // Create a mask to clear 16 bits at the bit_start position
                        uint64_t clear_mask = ~(0xffffull << bit_start);

                        // Set zeros for all frequencies corresponding to this stream_id
                        // and all elements at each (time, frequency) location
                        for (uint32_t freq_idx = 0; freq_idx < num_freq_per_stream; freq_idx++) {
                            // Local frequency ID = stream_bit + freq_idx * num_stream_ids
                            uint32_t freq_id = stream_bit + freq_idx * num_stream_ids;

                            // Apply clear_mask to all E/8 elements at this (time, frequency)
                            size_t base_pl_idx = t64_idx * F * E_DIV_8 + freq_id * E_DIV_8;
                            for (size_t e = 0; e < E_DIV_8; e++) {
                                assert(pl_mask_buf->frame_size
                                       > (base_pl_idx + e) * sizeof(uint64_t));
                                pl_mask_ptr[base_pl_idx + e] &= clear_mask;
                            }
                        }
                    }
                }
            }
        }


        // Zero out the receipt bitmaps that were just processed
        for (size_t i = 0; i < bitmap_frames.size(); i++) {
            memset(bitmap_frames[i], 0, receipt_bitmap_bufs[i]->frame_size);
        }

        // Set metadata for pl_mask_buf
        pl_mask_buf->allocate_new_metadata_object(pl_mask_frame_id);
        auto pl_mask_meta = get_chord_metadata(pl_mask_buf, pl_mask_frame_id);

        pl_mask_meta->set_fpga_seq_num(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_fpga_seq_num());
        pl_mask_meta->type = kotekan::uint1x8;
        pl_mask_meta->dims = 5;
        pl_mask_meta->set_array_dimension(0, time_long * time_short / 64, "Thi64");
        pl_mask_meta->set_array_dimension(1, num_frequency, "F");
        pl_mask_meta->set_array_dimension(2, 2, "P");
        pl_mask_meta->set_array_dimension(3, E_DIV_8 / 2, "D8");
        pl_mask_meta->set_array_dimension(4, 8, "Tlo64");
        pl_mask_meta->set_name("pl_mask_exp");

        pl_mask_meta->set_strides_simple();

        pl_mask_meta->set_coarse_freq(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_coarse_freq());
        pl_mask_meta->set_freq_upchan_factor(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_freq_upchan_factor());
        pl_mask_meta->set_time_downsampling_fpga(64);

        // Set metadata for rfi_mask_buf
        rfi_mask_buf->allocate_new_metadata_object(rfi_mask_frame_id);
        auto rfi_mask_meta = get_chord_metadata(rfi_mask_buf, rfi_mask_frame_id);

        rfi_mask_meta->set_fpga_seq_num(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_fpga_seq_num());
        rfi_mask_meta->type = kotekan::uint1x8;
        rfi_mask_meta->dims = 3;
        rfi_mask_meta->set_array_dimension(0, time_long * time_short / 1024, "T8hi128");
        rfi_mask_meta->set_array_dimension(1, num_frequency, "F");
        rfi_mask_meta->set_array_dimension(2, 128, "T8lo128");
        rfi_mask_meta->set_name("RFImask");

        rfi_mask_meta->set_strides_simple();

        rfi_mask_meta->set_coarse_freq(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_coarse_freq());
        rfi_mask_meta->set_freq_upchan_factor(
            get_chord_metadata(voltage_buf, voltage_frame_id)->get_freq_upchan_factor());
        rfi_mask_meta->set_time_downsampling_fpga(1024);

        // Mark frames as empty
        voltage_buf->mark_frame_empty(unique_name, voltage_frame_id);
        voltage_frame_id++;

        combined_receipt_bitmap_buf->mark_frame_full(unique_name, combined_bitmap_frame_id);
        combined_bitmap_frame_id++;

        pl_mask_buf->mark_frame_full(unique_name, pl_mask_frame_id);
        pl_mask_frame_id++;

        rfi_mask_buf->mark_frame_full(unique_name, rfi_mask_frame_id);
        rfi_mask_frame_id++;

        for (size_t i = 0; i < receipt_bitmap_bufs.size(); i++) {
            receipt_bitmap_bufs[i]->mark_frame_empty(unique_name, bitmap_frame_ids[i]);
            bitmap_frame_ids[i]++;
        }
    }
}
