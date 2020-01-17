/**
 * @file
 * @brief Capture of a single stream of IceBoard packets feeding them into one buffer
 * - iceBoardStandard : public iceBoardHandler
 */

#ifndef ICE_BOARD_STANDARD_HPP
#define ICE_BOARD_STANDARD_HPP

#include "buffer.h"
#include "chimeMetadata.h"
#include "gpsTime.h"
#include "iceBoardHandler.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "util.h"

/**
 * @brief DPDK Packet handler for capturing one port's data and placing it in one buffer
 *        with deterministic memory positioning (based on seq number) and missing frame zeroing.
 *
 * This handler strips header and footers and only places the data part of the
 * packet in the output frame in a location determined by the seq number in the header.
 * This means the location in the output frame corresponds to an exact seq number and time.
 *
 * @note It is important that this handler is paired with a zeroSample stage to zero out
 *       memory which this handler did not fill because the packet was lost or invalide.
 *
 * @par REST Endpoints
 * @endpoint /\<unique_name\>/port_data ``[GET]`` Returns stats about the PORT and the packets
 * received on it.
 *
 * @par Buffers
 * @buffer out_buf  Kotekan buffer to place the packets in.
 *       @buffer_format unit8_t array of FPGA packet contents
 *       @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Kotekan buffer of flags (one per time sample)
 *       @buffer_format unit8_t array of flags
 *       @buffer_metadata none
 *
 * @author Andre Renard
 */
class iceBoardStandard : public iceBoardHandler {

public:
    iceBoardStandard(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container, int port);

    virtual int handle_packet(rte_mbuf* mbuf);

protected:
    bool advance_frame(uint64_t new_seq, bool first_time = false);

    void handle_lost_samples(int64_t lost_samples);

    void copy_packet(struct rte_mbuf* mbuf);

    /// The output buffer
    struct Buffer* out_buf;

    /// The current frame
    uint8_t* out_frame;

    /// The ID of the current frame
    int32_t out_frame_id = 0;

    /// The flag buffer tracking lost samples
    struct Buffer* lost_samples_buf;

    /// The active lost sample frame
    uint8_t* lost_samples_frame;

    /// Frame IDs
    int lost_samples_frame_id = 0;
};

iceBoardStandard::iceBoardStandard(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& buffer_container, int port) :
    iceBoardHandler(config, unique_name, buffer_container, port) {

    DEBUG("iceBoardStandard: {:s}", unique_name);

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    register_producer(out_buf, unique_name.c_str());

    lost_samples_buf =
        buffer_container.get_buffer(config.get<std::string>(unique_name, "lost_samples_buf"));
    register_producer(lost_samples_buf, unique_name.c_str());
    // We want to make sure the flag buffers are zeroed between uses.
    zero_frames(lost_samples_buf);

    // TODO Some parts of this function are common to the various ICEboard
    // handlers, and could likely be factored out.
    std::string endpoint_name = unique_name + "/port_data";
    kotekan::restServer::instance().register_get_callback(endpoint_name,
                                                          [&](kotekan::connectionInstance& conn) {
                                                              json info = get_json_port_info();
                                                              conn.send_json_reply(info);
                                                          });
}

inline int iceBoardStandard::handle_packet(struct rte_mbuf* mbuf) {

    // Check if the packet is valid
    if (!iceBoardHandler::check_packet(mbuf))
        return 0; // Disgards the packet.

    if (unlikely(!got_first_packet)) {
        if (likely(!iceBoardHandler::align_first_packet(mbuf)))
            return 0; // Not the first packet.

        // Setup the first buffer frame for copying data into
        if (!iceBoardStandard::advance_frame(last_seq, true))
            return 0; // This catches the exit condition.
    } else {
        cur_seq = iceBoardHandler::get_mbuf_seq_num(mbuf);
    }

    int64_t diff = iceBoardHandler::get_packet_diff();

    // This checks if the FPGAs have reset the seq number count
    if (unlikely(!iceBoardHandler::check_for_reset(diff)))
        return -1;

    // Check if we have an out-of-order or duplicate packet
    if (unlikely(!iceBoardHandler::check_order(diff)))
        return 0; // For not we just disgard any dublicate/out-of-order packets.

    // Handle lost packets
    if (unlikely(diff > samples_per_packet))
        iceBoardStandard::handle_lost_samples(diff - samples_per_packet);

    // copy packet
    iceBoardStandard::copy_packet(mbuf);

    last_seq = cur_seq;

    return 0;
}

inline bool iceBoardStandard::advance_frame(uint64_t new_seq, bool first_time) {
    struct timeval now;
    gettimeofday(&now, NULL);

    // Advance the frame
    if (!first_time) {
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);

        // Advance frame ID
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
    }

    out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
    if (out_frame == NULL)
        return false;

    // Set metadata values.
    allocate_new_metadata_object(out_buf, out_frame_id);

    set_first_packet_recv_time(out_buf, out_frame_id, now);

    if (is_gps_global_time_set() == 1) {
        struct timespec gps_time = compute_gps_time(new_seq);
        set_gps_time(out_buf, out_frame_id, gps_time);
    }

    set_stream_id_t(out_buf, out_frame_id, port_stream_id);
    set_fpga_seq_num(out_buf, out_frame_id, new_seq);

    // Adcance the lost samples frame
    if (!first_time) {
        mark_frame_full(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        lost_samples_frame_id = (lost_samples_frame_id + 1) % lost_samples_buf->num_frames;
    }
    lost_samples_frame =
        wait_for_empty_frame(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
    if (lost_samples_frame == NULL)
        return false;

    return true;
}

// Note this function is almost identical to the handle_lost_samples function in the
// iceboardshuffle except for the call to advance_frame().  It might be possible to
// refactor some of this code.
inline void iceBoardStandard::handle_lost_samples(int64_t lost_samples) {

    int64_t lost_sample_location =
        last_seq + samples_per_packet - get_fpga_seq_num(out_buf, out_frame_id);
    uint64_t temp_seq = last_seq + samples_per_packet;

    // TODO this could be made more efficent by breaking it down into blocks of memsets.
    while (lost_samples > 0) {
        if (unlikely(lost_sample_location * sample_size == out_buf->frame_size)) {
            advance_frame(temp_seq);
            lost_sample_location = 0;
        }

        // Set the lost samples flags in the lost samples frame.
        lost_samples_frame[lost_sample_location] = 1;
        lost_sample_location += 1;
        lost_samples -= 1;
        rx_lost_samples_total += 1;
        temp_seq += 1;
    }
}

inline void iceBoardStandard::copy_packet(struct rte_mbuf* mbuf) {

    // Note this assumes that frame_size is divisable by samples_per_packet,
    // or the assert below will fail.
    int64_t sample_location = cur_seq - get_fpga_seq_num(out_buf, out_frame_id);
    assert(sample_location * sample_size <= out_buf->frame_size);

    // Check if we are at the end of the current frame
    if (unlikely(sample_location * sample_size == out_buf->frame_size)) {
        // If there are no new frames to fill, we are just dropping the packet
        if (!advance_frame(cur_seq))
            return;
        sample_location = 0;
    }

    // Initial packet offset, advances with each call to copy_block.
    int pkt_offset = header_offset;

    copy_block(&mbuf, (uint8_t*)&out_frame[sample_location * sample_size],
               sample_size * samples_per_packet, &pkt_offset);
}

#endif
