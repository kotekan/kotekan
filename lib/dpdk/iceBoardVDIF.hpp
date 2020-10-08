/**
 * @file
 * @brief Contains the handler for generating VDIF frames from an ICEBoard in shuffle16 mode
 * - iceBoardVDIF : public iceBoardHandler
 */

#ifndef ICE_BOARD_VDIF
#define ICE_BOARD_VDIF

#include "Config.hpp"
#include "ICETelescope.hpp"
#include "Telescope.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "iceBoardHandler.hpp"
#include "packet_copy.h"
#include "restServer.hpp"
#include "util.h"
#include "vdif_functions.h"

/**
 * @brief Handler for extacting two elements from 8 links from an
 *        ICEboard running in PFB 16-element mode
 *
 * This mode only works when attached to an ICEboard running in shuffle16 mode with the PFB enabled.
 * The receiving system must be connected to all 8 FPGA links.  And this hander must be used in a
 * group of 8, with all 8 handers attached to the same output buffer and lost samples buffer. The
 * order of the links attached to the node however does not matter.
 *
 * The output data stream is standard VDIF see: https://vlbi.org/vdif/
 *
 * Note that when one more packets is lost the entire VDIF frame is marked as invalid. This requires
 * the use of the invalidateVDIFframes stage to mark the VDIF frames as invalid.  Note once a
 * frame is marked as invalid there is no guarantee any data will be good, including the time stamp.
 * Only the frame lenght can be considered be correct.
 *
 * @par REST Endpoints
 * @endpoint /\<unique_name\>/port_data ``[GET]`` Returns stats about the PORT and the packets
 * received on it.
 *
 * @par Buffers
 * @buffer out_buf  Kotekan buffer to place the VDIF frames in.
 *       @buffer_format unit8_t array of VDIF frame
 *       @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Kotekan buffer of flags (one per time sample)
 *       @buffer_format unit8_t array of flags
 *       @buffer_metadata none
 *
 * @conf station_id   Int   Default 0x4151 ('AQ') Interger stored ascii denoting the standard VDIF
 *                            Station ID.  AQ == ARO
 * @conf offset       Int   Defailt 0.  The offset from the first element.  i.e. a value of 2
 * would select the 3nd and 4rd element (one based), a value of 0 gives 1st and 2nd element
 *
 * @author Andre Renard
 */
class iceBoardVDIF : public iceBoardHandler {

public:
    iceBoardVDIF(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container, int port);

    virtual int handle_packet(struct rte_mbuf* mbuf) override;

protected:
    bool advance_vdif_frame(uint64_t new_seq, bool first_time = false);

    void copy_packet_vdif(struct rte_mbuf* mbuf);

    void handle_lost_samples(int64_t lost_samples);

    void set_vdif_header_options(int vdif_frame_location, uint64_t seq);

    struct Buffer* out_buf;
    int32_t out_buf_frame_id = 0;
    uint8_t* out_buf_frame;

    /// The flag buffer tracking lost samples
    struct Buffer* lost_samples_buf;

    /// The active lost sample frame
    uint8_t* lost_samples_frame;

    /// Current lost samples frame id
    int lost_samples_frame_id = 0;

    /// We use the two ADC inputs starting at this offset.
    /// So an offset of 2 reads the 3th and 4th ADC input.
    uint32_t offset;

    /// VDIF station ID
    uint32_t station_id;

    /// Time in nano seconds of the 0 ICEBoard seq number, from the 2000 epoch
    uint64_t vdif_base_time;

    // Note: It might be possible to make some of these more dynamic

    /// The standard VDIF header lenght
    const int64_t vdif_header_len = 32;

    /// The packet size, which in this case is always 1024
    const int64_t vdif_packet_len = vdif_header_len + 1024;

    /// The total number of elements in each packet.
    const int64_t total_num_elements = 16;

    /// The number of elements to extract.
    const int64_t num_elements = 2; // This is also the number of threads.
};

iceBoardVDIF::iceBoardVDIF(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container, int port) :
    iceBoardHandler(config, unique_name, buffer_container, port) {

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    register_producer(out_buf, unique_name.c_str());

    lost_samples_buf =
        buffer_container.get_buffer(config.get<std::string>(unique_name, "lost_samples_buf"));
    register_producer(lost_samples_buf, unique_name.c_str());
    // We want to make sure the flag buffers are zeroed between uses.
    zero_frames(lost_samples_buf);

    station_id = config.get_default<uint32_t>(unique_name, "station_id", 0x4151); // AQ
    offset = config.get_default<uint32_t>(unique_name, "offset", 0);

    if (offset > 14) {
        throw std::runtime_error(fmt::format(fmt("The offset value is too large: {:d}"), offset));
    }

    std::string endpoint_name = unique_name + "/port_data";
    kotekan::restServer::instance().register_get_callback(
        endpoint_name, [&](kotekan::connectionInstance& conn) {
            nlohmann::json info = get_json_port_info();
            conn.send_json_reply(info);
        });
}

int iceBoardVDIF::handle_packet(struct rte_mbuf* mbuf) {

    // Check if the packet is valid
    if (!iceBoardHandler::check_packet(mbuf))
        return 0; // Disgards the packet.

    if (unlikely(!got_first_packet)) {
        if (likely(!iceBoardHandler::align_first_packet(mbuf))) {
            return 0; // Not the first packet.
        }

        // Setup the first buffer frame for copying data into
        if (!iceBoardVDIF::advance_vdif_frame(last_seq, true))
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
        iceBoardVDIF::handle_lost_samples(diff - samples_per_packet);

    // copy packet
    iceBoardVDIF::copy_packet_vdif(mbuf);

    last_seq = cur_seq;

    return 0;
}

bool iceBoardVDIF::advance_vdif_frame(uint64_t new_seq, bool first_time) {

    auto& tel = Telescope::instance();

    struct timeval now;
    gettimeofday(&now, nullptr);

    // Advance the frame
    if (!first_time) {
        mark_frame_full(out_buf, unique_name.c_str(), out_buf_frame_id);

        // Advance frame ID
        out_buf_frame_id = (out_buf_frame_id + 1) % out_buf->num_frames;
    }

    out_buf_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_frame_id);
    if (out_buf_frame == nullptr)
        return false;

    // Setup the VDIF time offsets from the seq number
    if (first_time && port == 0) {

        // TODO This reference epoch offset doesn't account for leap seconds after the epoch.
        // So this will go wrong once the next leap second happens.  See: issue #498
        // Unix time of the 2018 VDIF epoch, corresponds to 2018.01.01.0:0:0 in UTC
        const uint64_t ref_epoch_offset = 1514764800;
        if (tel.gps_time_enabled()) {
            timespec gps_time = tel.to_time(new_seq);
            // Compute the time at fpga_seq_num == 0 in nano seconds
            // relative to the 2018 epoch (see above)
            vdif_base_time = gps_time.tv_sec * 1000000000 + gps_time.tv_nsec
                             - new_seq * tel.seq_length_nsec() - ref_epoch_offset * 1000000000;
            DEBUG("Using GPS based vdif_base_time: {:d}", vdif_base_time);
        } else {
            vdif_base_time = now.tv_sec * 1000000000 + now.tv_usec * 1000
                             - new_seq * tel.seq_length_nsec() - ref_epoch_offset * 1000000000;
            DEBUG("Using system clock based vdif_base_time: {:d}", vdif_base_time);
        }
    }

    allocate_new_metadata_object(out_buf, out_buf_frame_id);

    if (port == 0)
        set_first_packet_recv_time(out_buf, out_buf_frame_id, now);

    if (tel.gps_time_enabled()) {
        struct timespec gps_time = tel.to_time(new_seq);
        set_gps_time(out_buf, out_buf_frame_id, gps_time);
    }

    set_fpga_seq_num(out_buf, out_buf_frame_id, new_seq);

    // Advance the lost samples frame
    if (!first_time) {
        mark_frame_full(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        lost_samples_frame_id = (lost_samples_frame_id + 1) % lost_samples_buf->num_frames;
    }
    lost_samples_frame =
        wait_for_empty_frame(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
    if (lost_samples_frame == nullptr)
        return false;

    return true;
}

inline void iceBoardVDIF::handle_lost_samples(int64_t lost_samples) {

    const int64_t frame_size = vdif_packet_len * num_elements;

    int64_t lost_sample_location =
        last_seq + samples_per_packet - get_fpga_seq_num(out_buf, out_buf_frame_id);
    uint64_t temp_seq = last_seq + samples_per_packet;

    // TODO this could be made more efficient by breaking it down into blocks of memsets.
    while (lost_samples > 0) {
        if (unlikely(lost_sample_location * frame_size == out_buf->frame_size)) {
            advance_vdif_frame(temp_seq);
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

void iceBoardVDIF::copy_packet_vdif(struct rte_mbuf* mbuf) {

    const int64_t frame_size = vdif_packet_len * num_elements;

    int64_t vdif_frame_location = cur_seq - get_fpga_seq_num(out_buf, out_buf_frame_id);

    if (unlikely(vdif_frame_location * frame_size == out_buf->frame_size)) {
        advance_vdif_frame(cur_seq);
        vdif_frame_location = 0;
    }

    assert(vdif_frame_location * frame_size <= out_buf->frame_size);

    // Set the VDIF headers only on the first port
    // since all ports would generate the same header
    if (port == 0) {
        set_vdif_header_options(vdif_frame_location * frame_size, cur_seq);
    }

    auto& tel = Telescope::instance();
    stream_t encoded_id = ice_encode_stream_id(port_stream_id);

    // Create the parts of the VDIF frame that are in this packet.
    int from_idx = header_offset + offset;
    int mbuf_len = mbuf->data_len;
    for (uint32_t time_step = 0; time_step < samples_per_packet; ++time_step) {
        for (int freq = 0; freq < 128; ++freq) {
            for (int elem = 0; elem < num_elements; ++elem) {

                // Advance to the next mbuf in the chain.
                if (unlikely(from_idx >= mbuf_len)) {
                    mbuf = mbuf->next;
                    assert(mbuf);
                    from_idx -= mbuf_len; // Subtract the last mbuf_len from the current idx.
                    mbuf_len = mbuf->data_len;
                }

                int output_idx =
                    vdif_frame_location * frame_size +           // Frame location in output buffer.
                    vdif_packet_len * num_elements * time_step + // Time step in output frame.
                    vdif_packet_len * elem + // VDIF pack for the correct element (ThreadID).
                    vdif_header_len +        // Offset for the vdif header.
                    tel.to_freq_id(encoded_id,
                                   freq); // Location in the VDIF packet is just frequency.

                // After all that indexing copy one byte :)
                out_buf_frame[output_idx] = *(rte_pktmbuf_mtod(mbuf, char*) + from_idx);

                from_idx += 1;
            }
            // If we only take 2 elements, then we have to skip 14
            from_idx += total_num_elements - num_elements;
        }
    }
}

inline void iceBoardVDIF::set_vdif_header_options(int vdif_frame_location, uint64_t seq) {

    uint64_t fpga_ns = Telescope::instance().seq_length_nsec();

    for (uint32_t time_step = 0; time_step < samples_per_packet; ++time_step) {
        for (int elem = 0; elem < num_elements; ++elem) {
            int header_idx = vdif_frame_location + vdif_packet_len * num_elements * time_step
                             + vdif_packet_len * elem;

            assert(header_idx < out_buf->frame_size);

            struct VDIFHeader* vdif_header = (struct VDIFHeader*)&out_buf_frame[header_idx];

            vdif_header->invalid = 0;
            vdif_header->legacy = 0;
            vdif_header->vdif_version = 1;
            vdif_header->data_type = 1;
            vdif_header->unused = 0;
            // First half of 2018, corresponds to 2018.01.01.0:0:0 in UTC
            vdif_header->ref_epoch = 36;
            vdif_header->frame_len = 132;
            vdif_header->log_num_chan = 10;
            vdif_header->bits_depth = 3;
            vdif_header->edv = 0;
            vdif_header->eud1 = 0;
            vdif_header->eud2 = 0;
            vdif_header->eud3 = 0;
            vdif_header->eud4 = 0;
            vdif_header->station_id = station_id;
            vdif_header->thread_id = elem;

            // Current time in nano seconds relative to epoch
            uint64_t cur_time = (seq + time_step) * fpga_ns + vdif_base_time;
            vdif_header->seconds = (cur_time) / 1000000000;
            vdif_header->data_frame = (cur_time % 1000000000) / fpga_ns;
        }
    }
}

#endif
