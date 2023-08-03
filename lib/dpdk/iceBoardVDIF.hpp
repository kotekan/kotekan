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
 * @brief Handler for extacting elements and frequency ranges from 8 links from an
 *        ICEboard running in PFB 16-element mode
 *
 * This mode only works when attached to an ICEboard running in shuffle16 mode with the PFB enabled.
 * The receiving system must be connected to all 8 FPGA links.  And this handler must be used in a
 * group of 8, with all 8 handlers attached to the same output buffer and lost samples buffer. The
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
 * @conf station_id   Int  Default 0x4151 ('AQ') Interger stored ascii denoting the standard VDIF
 *                                        Station ID.  AQ == ARO
 * @conf num_threads  Int  Default 0.     The number of VDIF threads to spread the data over.
 *                                        If 0, taken from num_elements for backwards compatibility
 * @conf num_elements Int  Default 2.     Number of input elements to store in a thread
 * @conf num_freq     Int  Default 1024.  Number of frequency channels to store in a thread
 * @conf offsets      Array               Starting input offset for each thread (of num_elements)
 * @conf frequencies  Array               Starting frequency channel for each thread (of num_freq)
 * @conf offset       Int  Defailt 0.     The offset from the first element.  i.e. a value of 2
 *                                        would select the 3nd and 4rd element (one based), a value
 *                                        of 0 gives 1st and 2nd element.
 *                                        Present for backwards compatibility with old config giles.
 *
 * @author Andre Renard
 * @author Marten van Kerkwijk
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

    /// The total number of frequency channels.
    const uint32_t total_num_freq = 1024;

    /// The total number of input elements in each packet.
    const uint32_t total_num_elements = 16;

    /// The number of threads to write.
    uint32_t num_threads;
    /// The number of elements per thread.  Also used for backwards compatibility
    /// if num_threads is not given.  In that case, it is the number of threads,
    /// with each thread holding one element.
    uint32_t num_elements;
    /// The number of frequency channels per thread.
    uint32_t num_freq;
    /// Start input element and frequency for each thread in the input buffer.
    std::vector<uint32_t> offsets = {};
    std::vector<uint32_t> frequencies = {};
    /// Corresponding offset in the input buffer: offset + (frequency / 8) * 16
    std::vector<uint32_t> buffer_offsets = {};

    /// VDIF station ID
    uint32_t station_id;

    /// Time in nano seconds of the 0 ICEBoard seq number, from the 2000 epoch
    uint64_t vdif_base_time;

    /// The number of VDIF channels, num_freq * num_elements
    uint32_t vdif_num_chan;
    uint32_t vdif_log2_num_chan;
    /// The standard VDIF header length in bytes
    const uint32_t vdif_header_len = 32;
    /// vdif frame size in bytes = vdif_num_chan + vdif_header_len
    uint32_t vdif_frame_size;
    /// vdif frame set size in bytes = vdif_frame_size * num_threads
    uint32_t vdif_frameset_size;

    /// frequency offset for thread (here set to sentinel indicating it has not yet
    /// been set; will be set to real value (0 - 7) in copy_packet_vdif.
    uint32_t freq_offset = total_num_freq;
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

    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 0);
    num_elements = config.get_default<uint32_t>(unique_name, "num_elements", 2);
    if (num_threads > 0) {
        // Default new configuration style: get number of frequencies
        num_freq = config.get_default<int64_t>(unique_name, "num_freq", total_num_freq);
        // and the starting input offsets and frequency channels for each thread.
        offsets = config.get<std::vector<uint32_t>>(unique_name, "offsets");
        frequencies = config.get<std::vector<uint32_t>>(unique_name, "frequencies");
    } else {
        // Backward compatibility: information is in num_elements and offset.
        num_threads = num_elements;
        num_elements = 1;
        num_freq = total_num_freq;
        uint32_t offset = config.get_default<uint32_t>(unique_name, "offset", 0);
        offsets.resize(num_threads);
        std::iota(offsets.begin(), offsets.end(), offset); // offset, offset+1, ...
        frequencies.assign(num_threads, 0);
    }
    // Sanity checks of configuration parameters.
    if (!(num_elements == 1 || num_elements == 2 || num_elements == 4 || num_elements == 8)) {
        throw std::runtime_error(
            fmt::format(fmt("num_elements must be 1, 2, 4, or 8, not {:d}"), num_elements));
    }
    if (total_num_freq % num_freq != 0) {
        throw std::runtime_error(fmt::format(
            fmt("Number of frequences {:d} per thread must divide into 1024"), num_freq));
    }
    if (offsets.size() != num_threads) {
        throw std::runtime_error(fmt::format(
            fmt("Number of input offsets should be {:d}, not {:d}"), num_threads, offsets.size()));
    }
    if (frequencies.size() != num_threads) {
        throw std::runtime_error(
            fmt::format(fmt("Number of start frequencies should be {:d}, not {:d}"), num_threads,
                        frequencies.size()));
    }
    for (uint32_t i_thread = 0; i_thread < num_threads; i_thread++) {
        if (offsets[i_thread] > total_num_elements - num_elements) {
            throw std::runtime_error(fmt::format(fmt("The offset range extends too far: {:d}-{:d}"),
                                                 offsets[i_thread],
                                                 offsets[i_thread] + num_elements - 1));
        }
        if (frequencies[i_thread] > total_num_freq - num_freq) {
            throw std::runtime_error(
                fmt::format(fmt("The frequency range extends too far: {:d}-{:d}"),
                            frequencies[i_thread], frequencies[i_thread] + num_freq - 1));
        }
        // With frequencies split 8 ways, insisting on a multiple of 8 for the start frequency
        // ensures we can copy the same frequency chunk in each input buffer.
        if (frequencies[i_thread] % 8 != 0) {
            throw std::runtime_error(fmt::format(
                fmt("The start frequency must be a multiple of 8: {:d}"), frequencies[i_thread]));
        }
    }
    buffer_offsets.resize(num_threads);
    for (uint32_t i_thread = 0; i_thread < num_threads; i_thread++) {
        // Calculate true offset in input buffer. This has every eighth frequency,
        // so for buffer 0, 16 elements for freq 0, then 16 for freq 8, etc.
        buffer_offsets[i_thread] =
            offsets[i_thread] + (frequencies[i_thread] / 8) * total_num_elements;
    }
    // Calculate fixed VDIF properties.
    vdif_num_chan = num_freq * num_elements;
    vdif_log2_num_chan = int(std::log2(vdif_num_chan));
    vdif_frame_size = vdif_num_chan + vdif_header_len; // Each channel is 4+4 bits = 1 byte
    vdif_frameset_size = vdif_frame_size * num_threads;
    if (port == 0) {
        DEBUG("VDIF: {:d} threads, each containing {:d} input(s) and {:d} frequencies.",
              num_threads, num_elements, num_freq);
        DEBUG("VDIF: number of channels = {:d} = 2**{:d}; frame size={:d}, frameset size={:d}.",
              vdif_num_chan, vdif_log2_num_chan, vdif_frame_size, vdif_frameset_size);
        for (uint32_t i_thread = 0; i_thread < num_threads; i_thread++) {
            DEBUG("VDIF: thread {:d} will start at input {:2d} and frequency {:4d}; "
                  "buffer offset {:4d}",
                  i_thread, offsets[i_thread], frequencies[i_thread], buffer_offsets[i_thread]);
        }
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
        return 0; // Discards the packet.

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
        return 0; // For now we just discard any dublicate/out-of-order packets.

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

    int64_t lost_sample_location =
        last_seq + samples_per_packet - get_fpga_seq_num(out_buf, out_buf_frame_id);
    uint64_t temp_seq = last_seq + samples_per_packet;

    // TODO this could be made more efficient by breaking it down into blocks of memsets.
    while (lost_samples > 0) {
        if (unlikely((size_t)(lost_sample_location * vdif_frameset_size) == out_buf->frame_size)) {
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

    int64_t vdif_frame_location = cur_seq - get_fpga_seq_num(out_buf, out_buf_frame_id);

    if (unlikely((size_t)(vdif_frame_location * vdif_frameset_size) == out_buf->frame_size)) {
        advance_vdif_frame(cur_seq);
        vdif_frame_location = 0;
    }

    assert((size_t)(vdif_frame_location * vdif_frameset_size) <= out_buf->frame_size);

    // Set the VDIF headers only on the first port
    // since all ports would generate the same header
    if (port == 0) {
        set_vdif_header_options(vdif_frame_location * vdif_frameset_size, cur_seq);
    }

    // Offset in output frame for first frequency of this receiver buffer thread (0-7).
    if (freq_offset == total_num_freq) {
        auto& tel = Telescope::instance();
        stream_t encoded_id = ice_encode_stream_id(port_stream_id);
        freq_offset = tel.to_freq_id(encoded_id, 0);
    }
    // Pointer to where we should start storing our data.
    uint8_t* out_t0_f0 =
        (out_buf_frame                              // Start of output buffer.
         + vdif_frame_location * vdif_frameset_size // Frame location in output buffer.
         + vdif_header_len                          // Offset for the vdif header.
         + freq_offset * num_elements);             // Offset to first frequency.

    // Buffer and offset to first sample in the input.
    auto mbuf0 = mbuf;
    // First mbuf in packet includes header, but later ones do not.
    uint32_t mbuf_start_offset = header_offset;

    // Times are in separate frames, so time stride equals the size of the VDIF framesets.
    for (uint8_t* out_t_f0 = out_t0_f0;
         out_t_f0 < out_t0_f0 + vdif_frameset_size * samples_per_packet;
         out_t_f0 += vdif_frameset_size) {
        // Copy data by threads, to help cache performance.
        for (uint32_t i_thread = 0; i_thread < num_threads; i_thread++) {
            // Set up start point for this thread.
            mbuf = mbuf0;
            // Pointer to start of buffer
            char* in = rte_pktmbuf_mtod(mbuf, char*);
            // Pointer to last place where a full element can be read.
            char* mbuf_last_sample = in + mbuf->data_len - num_elements;
            // Adjust in pointer to starting element and frequency for this VDIF thread.
            in += mbuf_start_offset + buffer_offsets[i_thread];
            // Output start location.
            uint8_t* out_t_fi = out_t_f0 + i_thread * vdif_frame_size;
            for (uint8_t* out = out_t_fi; out < out_t_fi + num_freq * num_elements;
                 out += 8 * num_elements) {
                while (unlikely(in > mbuf_last_sample)) {
                    // Input is beyond start of last sample.
                    uint32_t beyond_last_sample = in - mbuf_last_sample;
                    // Go to the next buffer, keeping the pointer in case of a partial sample.
                    auto in_old = in;
                    mbuf = mbuf->next;
                    assert(mbuf);
                    in = rte_pktmbuf_mtod(mbuf, char*);
                    mbuf_last_sample = in + mbuf->data_len - num_elements;
                    // Go to new sample location (subsequent mbuf do not have header).
                    in += beyond_last_sample - num_elements;
                    if (unlikely(beyond_last_sample < num_elements)) {
                        // Partial sample.
                        uint32_t n_before = num_elements - beyond_last_sample;
                        // Copy piece from previous buffer (leaves tmp at out+n_before).
                        auto tmp = std::copy_n(in_old, n_before, out);
                        // Copy rest from new buffer (in + n_before is start of buffer data).
                        std::copy_n(in + n_before, beyond_last_sample, tmp);
                        // 2023-07-23, MHvK: Tested that this is hit for the first frame
                        // with num_threads=1, num_elements=8, num_freq=512, frequencies=[512]
                        // (the debug statement leads to large packet losses, unsurprisingly).
                        // DEBUG("Hit partial sample {:x}", *(uint64_t*)out);
                        goto next_element;
                    }
                } // end of while; input pointer is at sample.
                std::copy_n(in, num_elements, out);
            next_element:
                in += total_num_elements;
            }
        } // end of loop over threads.
        // Go to next time sample (of 128 frequencies and 16 inputs)
        mbuf_start_offset += 128 * total_num_elements;
        // If needed, advance to the next mbuf in the chain.
        while (mbuf_start_offset > mbuf0->data_len) {
            mbuf_start_offset -= mbuf0->data_len;
            mbuf0 = mbuf0->next;
            assert(mbuf0);
        }
    } // end of loop over times.
}

inline void iceBoardVDIF::set_vdif_header_options(int vdif_frame_location, uint64_t seq) {

    uint64_t fpga_ns = Telescope::instance().seq_length_nsec();

    for (uint32_t time_step = 0; time_step < samples_per_packet; ++time_step) {
        for (uint32_t i_thread = 0; i_thread < num_threads; ++i_thread) {
            size_t header_idx =
                vdif_frame_location + vdif_frameset_size * time_step + vdif_frame_size * i_thread;

            assert(header_idx < out_buf->frame_size);

            struct VDIFHeader* vdif_header = (struct VDIFHeader*)&out_buf_frame[header_idx];

            vdif_header->invalid = 0;
            vdif_header->legacy = 0;
            vdif_header->vdif_version = 1;
            vdif_header->data_type = 1;
            vdif_header->unused = 0;
            // First half of 2018, corresponds to 2018-01-01T00:00:00 in UTC
            vdif_header->ref_epoch = 36;
            vdif_header->frame_len = vdif_frame_size / 8;
            vdif_header->log_num_chan = vdif_log2_num_chan;
            vdif_header->bits_depth = 3;
            vdif_header->edv = 0;
            vdif_header->eud1 = 0;
            vdif_header->eud2 = 0;
            vdif_header->eud3 = 0;
            vdif_header->eud4 = 0;
            vdif_header->station_id = station_id;
            vdif_header->thread_id = i_thread;

            // Current time in nano seconds relative to epoch
            uint64_t cur_time = (seq + time_step) * fpga_ns + vdif_base_time;
            vdif_header->seconds = (cur_time) / 1000000000;
            vdif_header->data_frame = (cur_time % 1000000000) / fpga_ns;
        }
    }
}

#endif
