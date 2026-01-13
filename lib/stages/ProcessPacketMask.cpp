#include "ProcessPacketMask.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"

#include "fmt.hpp"

#include <stdexcept>
#include <visUtil.hpp>

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

    INFO("ProcessPacketMask: Initialized with {:d} receipt bitmap buffers",
         receipt_bitmap_bufs.size());
    INFO("ProcessPacketMask: Voltage frame shape: [{:d}][{:d}][{:d}][{:d}][{:d}]", time_long,
         num_frequency, element_long, time_short, element_short);
    INFO("ProcessPacketMask: Receipt bitmap shape: [{:d}][{:d}]", time_long, num_source_ids);
}

ProcessPacketMask::~ProcessPacketMask() {}

void ProcessPacketMask::main_thread() {
    frameID voltage_frame_id(voltage_buf);
    frameID combined_bitmap_frame_id(combined_receipt_bitmap_buf);
    std::vector<frameID> bitmap_frame_ids;
    for (auto* buf : receipt_bitmap_bufs) {
        bitmap_frame_ids.emplace_back(buf);
    }

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

        DEBUG("ProcessPacketMask: Processing voltage frame {:d} with {:d} bitmap frames",
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
        INFO("ProcessPacketMask: Voltage frame {:d} has {:d} missing packets in combined bitmap",
             (int)voltage_frame_id, missing_packet_count);

        // Calcuate a packet loss rate based on the missing packet count
        // and the number of working source IDs
        size_t working_source_ids = num_source_ids - missing_source_ids.size();
        size_t total_packets = time_long * working_source_ids * 8; // num_stream_ids = 8
        double packet_loss_rate = 0.0;
        assert(total_packets > 0);
        packet_loss_rate = (double)missing_packet_count / (double)total_packets;
        double packet_loss_percentage = packet_loss_rate * 100.0;
        INFO("ProcessPacketMask: Voltage frame {:d} packet loss precentage = {:.4f}% ({:d} missing "
             "/ {:d} total)",
             (int)voltage_frame_id, packet_loss_percentage, (int)missing_packet_count,
             (int)total_packets);


        // Zero out the receipt bitmaps that were just processed
        for (size_t i = 0; i < bitmap_frames.size(); i++) {
            memset(bitmap_frames[i], 0, receipt_bitmap_bufs[i]->frame_size);
        }

        // Mark frames as empty
        voltage_buf->mark_frame_empty(unique_name, voltage_frame_id);
        voltage_frame_id++;

        combined_receipt_bitmap_buf->mark_frame_full(unique_name, combined_bitmap_frame_id);
        combined_bitmap_frame_id++;

        for (size_t i = 0; i < receipt_bitmap_bufs.size(); i++) {
            receipt_bitmap_bufs[i]->mark_frame_empty(unique_name, bitmap_frame_ids[i]);
            bitmap_frame_ids[i]++;
        }
    }
}
