#ifndef ICE_BOARD_SHUFFLE_HPP
#define ICE_BOARD_SHUFFLE_HPP

#include "iceBoardHandler.hpp"
#include "prometheusMetrics.hpp"
#include "packet_copy.h"
#include "util.h"
#include "chimeMetadata.h"
#include "gpsTime.h"
#include "buffer.h"

class iceBoardShuffle : public iceBoardHandler {
public:

    iceBoardShuffle(Config &config, const std::string &unique_name,
                    bufferContainer &buffer_container, int port);

    virtual int handle_packet(struct rte_mbuf *mbuf);

    virtual void update_stats();

protected:
    void advance_frames(uint64_t new_seq, bool first_time = false);

    void copy_packet_shuffle(struct rte_mbuf *mbuf);

    void handle_lost_samples(int64_t lost_samples);

    bool check_fpga_shuffle_flags(struct rte_mbuf *mbuf);

    static const uint32_t shuffle_size = 4;
    struct Buffer * out_bufs[shuffle_size];
    uint8_t * out_buf_frame[shuffle_size];
    struct Buffer * lost_samples_buf;
    uint8_t * lost_samples_frame;
    int lost_samples_frame_id = 0;
    int out_buf_frame_ids[shuffle_size] = {0};

    // This is fixed by the FPGA design, so not exposed to the config
    const uint32_t sub_sample_size = 512;

    // Error counter for each of the 16 lanes of the 2nd stage (within-crate) data shuffle.
    uint64_t fpga_second_stage_shuffle_errors[16];

    // Error counter for each of the 8 lanes of the 3rd stage (between-crate) data shuffle.
    uint64_t fpga_third_stage_shuffle_errors[8];

};

iceBoardShuffle::iceBoardShuffle(Config &config, const std::string &unique_name,
                    bufferContainer &buffer_container, int port) :
    iceBoardHandler(config, unique_name, buffer_container, port) {

    WARN("iceBoardHandler: %s", unique_name.c_str());

    std::vector<std::string> buffer_names = config.get_string_array(unique_name, "out_bufs");
    if (shuffle_size != buffer_names.size()) {
        throw std::runtime_error("Expecting 4 buffers, got " + std::to_string(port));
    }
    for (uint32_t i = 0; i < shuffle_size; ++i) {
        out_bufs[i] = buffer_container.get_buffer(buffer_names[i]);
        register_producer(out_bufs[i], unique_name.c_str());
    }

    lost_samples_buf = buffer_container.get_buffer(config.get_string(unique_name, "lost_samples_buf"));
    register_producer(lost_samples_buf, unique_name.c_str());

}

inline void iceBoardShuffle::handle_lost_samples(int64_t lost_samples) {

    // By design all the seq numbers for all frames should be the same here.
    int64_t lost_sample_location = last_seq + samples_per_packet
                                    - get_fpga_seq_num(out_bufs[0], out_buf_frame_ids[0]);
    uint64_t temp_seq = last_seq + samples_per_packet;

    // TODO this could be made more efficent by breaking it down into blocks of memsets.
    while (lost_samples > 0) {
        // TODO this assumes the frame size of all the output buffers are the
        // same, which should be true in all cases, but should still be tested
        // elsewhere.
        if (unlikely(lost_sample_location * sample_size == out_bufs[0]->frame_size)) {
            advance_frames(temp_seq);
        }

        // This sets the flag to zero this sample with the zeroSamples process.
        // NOTE: I thought about using a bit field for this array, but doing so
        // opens up a huge number of problems getting the bit set atomically in
        // a way that's also efficent.  By using a byte array with values of either
        // 0 or 1 then just setting the value to 1 avoids any syncronization issues.
        // The linux function set_bit() might have worked (since DPDK is linux/x86 only),
        // but then there are endianness issues if reading the array as a uint8_t *
        // This might be less memory size efficent, but it's much easier to work with.
        // NOTE: This also introduces cache line contension since we are using one array
        // to for all 4 links, ideally we might use 4 arrays and a reduce operation to bring
        // it down to one on another core.
        lost_samples_frame[lost_sample_location] = 1;
        lost_sample_location += 1;
        lost_samples -= 1;
        rx_lost_samples_total += 1;
        temp_seq += 1;
    }
}

inline void iceBoardShuffle::advance_frames(uint64_t new_seq, bool first_time) {
    WARN("port %d, called advance_frames!, first_time %d", port, (int)first_time);
    struct timeval now;
    gettimeofday(&now, NULL);

    for (uint32_t i = 0; i < shuffle_size; ++i) {
        if (!first_time) {
            mark_frame_full(out_bufs[i], unique_name.c_str(), out_buf_frame_ids[i]);

            // Advance frame ID
            out_buf_frame_ids[i] = (out_buf_frame_ids[i] + 1) % out_bufs[i]->num_frames;
            WARN("port %d, out_buf_frame_ids[%d] = %d", port, i, out_buf_frame_ids[i]);
        }

        out_buf_frame[i] = wait_for_empty_frame(out_bufs[i], unique_name.c_str(), out_buf_frame_ids[i]);

        allocate_new_metadata_object(out_bufs[i], out_buf_frame_ids[i]);

        set_first_packet_recv_time(out_bufs[i], out_buf_frame_ids[i], now);

        if (is_gps_global_time_set() == 1) {
            struct timespec gps_time = compute_gps_time(new_seq);
            set_gps_time(out_bufs[i], out_buf_frame_ids[i], gps_time);
        }

        // We take the stream ID only from the first pair of crates,
        // to avoid overwriting it on different ports.
        // This makes the stream ID unique for down stream processes.
        if (port_stream_id.crate_id / 2 == 0)
            set_stream_id_t(out_bufs[i], out_buf_frame_ids[i], port_stream_id);

        set_fpga_seq_num(out_bufs[i], out_buf_frame_ids[i], new_seq);
    }

    if (!first_time) {
        mark_frame_full(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        lost_samples_frame_id = (lost_samples_frame_id + 1) % lost_samples_buf->num_frames;
    }
    lost_samples_frame = wait_for_empty_frame(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
}

inline void iceBoardShuffle::copy_packet_shuffle(struct rte_mbuf *mbuf) {

    const int sample_size = 2048;
    const int sub_sample_size = sample_size / shuffle_size;

    // Where in the buf frame we should write sample.
    // TODO by construction this value should be the same for all
    // frames, but that should be proven carefully...
    int64_t sample_location;

    // Check if we need to advance frames
    /*for (uint32_t i = 0; i < shuffle_size; ++i) {
        // This could be optimized to avoid this look up every time.
        // i.e. save the base FPGA seq number for this frame.
        // Note all 4 of these should be the same, maybe check this?
        //fprintf(stderr, "port: %d, i: %d, sample_location: %p, out_bufs: %p, out_buf_frame_ids: %p, out_bufs[i]: %p, out_bufs_frame: %d\n",
        //    port, i, sample_location, out_bufs, out_buf_frame_ids, out_bufs[i], out_buf_frame_ids[i]);
        sample_location[i] = 0;
        out_bufs[i];
        out_buf_frame_ids[i];
        //fprintf(stderr, "port: %d, metadata[%d]: %p\n", port, i, out_bufs[i]->metadata);
        get_fpga_seq_num(out_bufs[i], out_buf_frame_ids[i]);
        sample_location[i] = cur_seq - get_fpga_seq_num(out_bufs[i], out_buf_frame_ids[i]);

        // We reached the end of the output frame
        if (unlikely(sample_location[i] * sample_size == out_bufs[i]->frame_size)) {
            advance_frames(cur_seq);
            sample_location[i] = 0;
        }
    }*/

    sample_location = cur_seq - get_fpga_seq_num(out_bufs[0], out_buf_frame_ids[0]);
    assert(sample_location * sample_size <= out_bufs[0]->frame_size);
    if (unlikely(sample_location * sample_size == out_bufs[0]->frame_size)) {
        advance_frames(cur_seq);
        sample_location = 0;
    }

    // Where to place each of the 512 element blocks based on which crate they
    // are coming from.
    int sub_sample_pos = port_stream_id.crate_id / 2;

    // Initial packet offset, advances with each call to copy_block.
    int pkt_offset = header_offset;

    // Copy the packet in packet memory order.
    for (uint32_t sample_id = 0; sample_id < samples_per_packet; ++sample_id) {

        for (uint32_t sub_sample_id = 0; sub_sample_id < shuffle_size; ++sub_sample_id) {
            uint64_t copy_location = (sample_location + sample_id) * sample_size
                                     + sub_sample_pos * sub_sample_size;

            /*copy_block(&mbuf,
                       (uint8_t *) &out_buf_frame[copy_location],
                       sub_sample_size,
                       &pkt_offset);*/
        }
    }
}

inline int iceBoardShuffle::handle_packet(struct rte_mbuf *mbuf) {

    if (!iceBoardHandler::check_packet(mbuf))
        return 0;

    if (unlikely(!got_first_packet)) {
        if (unlikely(!iceBoardHandler::align_first_packet(mbuf))) {
            return 0;
        } else {
            // Get the first set of buffer frames to write into.
            // We use last seq in case there are missing frames,
            // we want to start at the alignment point.
            // See align_first_packet for details.
            advance_frames(last_seq, true);
        }
    } else {
        cur_seq = iceBoardHandler::get_mbuf_seq_num(mbuf);
    }

    // Check footers
    if (unlikely(!iceBoardShuffle::check_fpga_shuffle_flags(mbuf)))
        return 0;

    int64_t diff = iceBoardHandler::get_packet_diff();

    if (unlikely(!iceBoardHandler::check_order(diff)))
        return 0;

    if (unlikely(!iceBoardHandler::check_for_reset(diff)))
        return -1;

    // Handle lost packets
    // Note this handles packets for all loss reasons,
    // because we don't update the last_seq number value if the
    // packet isn't accepted for any reason.
    if (unlikely(diff > samples_per_packet))
        iceBoardShuffle::handle_lost_samples(diff);

    // copy packet
    iceBoardShuffle::copy_packet_shuffle(mbuf);

    return 0;
}

inline bool iceBoardShuffle::check_fpga_shuffle_flags(struct rte_mbuf *mbuf) {

    const int flag_len = 4; // 32-bits = 4 bytes

    // Go to the last part of the packet
    // Note this assumes that the footer doesn't cross two mbuf
    // segment, but based on the packet design this should never happen.
    while(mbuf->next != NULL) {
        mbuf = mbuf->next;
    }

    int cur_mbuf_len = mbuf->data_len;
    assert(cur_mbuf_len >= flag_len);
    const uint8_t * mbuf_data = rte_pktmbuf_mtod_offset(mbuf, uint8_t *, cur_mbuf_len - flag_len);

    uint32_t flag_value = *(uint32_t *)mbuf_data;

    // If no flags are set then we hvae no errors to check,
    // so we accept the packet
    if (flag_value == 0) {
        return true;
    }

    // The 32 bits of the flag contain:
    // - The most significant 16 bits indicate an error for each of the 16 lanes
    //   of the 2nd stage (within-crate) data shuffle.
	// - The middle 8 bits are always 0.
	// - The least significant 8 bits indicate an error for each of the 8 lanes
    //   of the 3rd stage (between-crate) data shuffle.
    // The FPGA sends data in Little-endian byte order, so the operation below works
    // only on systems which are little-endian.  Therefore this code is not portiable.

    int i;
    for (i = 0; i < 8; ++i) {
        fpga_third_stage_shuffle_errors[i] += (flag_value >> i) & 1;
    }

    for (i = 0; i < 16; ++i) {
        fpga_second_stage_shuffle_errors[i] += (flag_value >> (16 + i)) & 1;
    }

    // One of the flags was set, so let's not process this packet.
    return false;
}

void iceBoardShuffle::update_stats() {
    iceBoardHandler::update_stats();

    prometheusMetrics &metrics = prometheusMetrics::instance();

    std::string tags = "port=\"" + std::to_string(port) + "\"";

    for (int i = 0; i < 8; ++i) {
        metrics.add_process_metric("kotekan_dpdk_shuffle_fpga_third_stage_shuffle_errors_total",
                                    unique_name,
                                    fpga_third_stage_shuffle_errors[i],
                                    tags + ",fpga_lane=\"" + std::to_string(i) + "\"");
    }
    for (int i = 0; i < 16; ++i) {
        metrics.add_process_metric("kotekan_dpdk_shuffle_fpga_second_stage_shuffle_errors_total",
                                    unique_name,
                                    fpga_second_stage_shuffle_errors[i],
                                    tags + ",fpga_lane=\"" + std::to_string(i) + "\"");
    }
}

#endif