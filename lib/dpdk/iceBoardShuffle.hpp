/**
 * @file
 * @brief Contains the handler for doing the final stage shuffle in a larger than 512 element
 * system.
 * - iceBoardShuffle : public iceBoardHandler
 */

#ifndef ICE_BOARD_SHUFFLE_HPP
#define ICE_BOARD_SHUFFLE_HPP

#include "Config.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chimeMetadata.hpp"
#include "iceBoardHandler.hpp"
#include "kotekanLogging.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"
#include "util.h"


/**
 * @brief DPDK Packet handler which adds a final stage shuffle for systems larger than 512 elements
 *
 * @par REST Endpoints
 * @endpoint /\<unique_name\>/port_data ``[GET]`` Returns a large amount of stats about the port and
 * FPGA flags
 *
 * @par Buffers
 * @buffer out_bufs  Array of kotekan buffers of lenght shuffle_size
 *       @buffer_format unit8_t array of FPGA packet contents
 *       @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Kotekan buffer of flags (one per time sample)
 *       @buffer_format unit8_t array of flags
 *       @buffer_metadata none
 *
 * @par Metrics
 * @metric kotekan_dpdk_shuffle_fpga_third_stage_shuffle_errors_total
 *         The total number of FPGA thrid stage shuffle errors seen
 * @metric kotekan_dpdk_shuffle_fpga_second_stage_shuffle_errors_total
 *         The total number of FPGA second stage shuffle errors seen
 *
 * @conf  fpga_dataset          String. The dataset ID for the data being received from
 *                              the F-engine.
 *
 * @todo Some parts of the port_data endpoint could be refactored into the base classes
 *
 * @author Andre Renard
 */
class iceBoardShuffle : public iceBoardHandler {
public:
    /// Default constructor
    iceBoardShuffle(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

    /**
     * @brief The packet processor, called each time there is a new packet
     *
     * @param mbuf The DPDK rte_mbuf containing the packet.
     * @return -1 if there is a serious error requiring shutdown, 0 otherwise.
     */
    virtual int handle_packet(struct rte_mbuf* mbuf) override;

    /// Updates the prometheus metrics
    virtual void update_stats() override;

protected:
    /**
     * @brief Advances the @c shuffle_size output frames, and the lost sample frame
     *
     * This function is used to move the system to the next set of output frames.
     * It updates the active frame pointers, and also fills the metadata for the
     * new frame; including GPS/System time, and FPGA seq number/streamID.
     *
     * @param new_seq The seq of the start of this new frame.
     * @param first_time Default false.  Set to true if we are setting up the first frame for start
     * up.
     * @return true if the frame was advanced.  false if the system is exiting, and there are no new
     * frames.
     */
    bool advance_frames(uint64_t new_seq, bool first_time = false);

    /**
     * @brief Checks that the rules for streamIDs are met.  i.e. correct cabling.
     *
     * @return True if cable/streamID rules are met, False otherwise.
     */
    bool check_stream_id();

    /**
     * @brief Copies the given packet accounting for the last stage suffle.
     *
     * This means it copies the packet into 4 buffer frames, and can advance
     * all 4 buffers.
     *
     * @param mbuf The rte_mbuf containing the packet
     */
    void copy_packet_shuffle(struct rte_mbuf* mbuf);

    /**
     * @brief Processes lost samples
     *
     * @param lost_samples The number of lost samples to record
     * @todo This could be make slightly more efficent, see notes in code
     * @return Returns false if the function encountered an exit condition,
     *         returns true otherwise.
     */
    bool handle_lost_samples(int64_t lost_samples);

    /**
     * @brief Checks the FPGA shuffle flags in the footer.
     *
     * Also adds to the FPGA flag counters.
     *
     * @param mbuf The rte_mbuf containing the packet
     * @return true if there are no flags set, and false if any flag is set.
     */
    bool check_fpga_shuffle_flags(struct rte_mbuf* mbuf);

    /// The size of the final full shuffle
    /// This might be possible to change someday.
    static const uint32_t shuffle_size = 4;

    /// The buffers which are filled by this port
    Buffer* out_bufs[shuffle_size];

    /// The active frame for the buffers to fill
    uint8_t* out_buf_frame[shuffle_size];

    /// The flag buffer tracking lost samples
    Buffer* lost_samples_buf;

    // Parameters saved from the config files
    dset_id_t fpga_dataset;

    /// The active lost sample frame
    uint8_t* lost_samples_frame;

    /// Frame IDs
    int lost_samples_frame_id = 0;

    /// Frame IDs
    int out_buf_frame_ids[shuffle_size] = {0};

    /// The stream_ids for all iceBoardShuffle objects.
    /// This might be an issue in the case of multiple indepdent
    /// shuffle operations. In that case this will need to be factored out.
    static ice_stream_id_t all_stream_ids[shuffle_size];

    // ** FPGA Second stage error counters **

    /// Error counter for each of the 16 lanes of the 2nd stage (within-crate) data shuffle.
    uint64_t fpga_second_stage_shuffle_errors[16] = {0};

    /// Counter for flag if there is a CRC error in ANY of the second stage input lanes
    uint64_t fpga_second_stage_crc_errors = 0;

    /// Counter for flag if the packet was missing or was too short on ANY second stage input lane
    uint64_t fpga_second_stage_missing_short_errors = 0;

    /// Counter for flag if the packet was too long on ANY second stage input lane
    uint64_t fpga_second_stage_long_errors = 0;

    /// Counter for flag if the data or frame fifo has overflowed on ANY second stage input lane
    /// (sticky)
    uint64_t fpga_second_stage_fifo_overflow_errors = 0;

    // ** FPGA Third stage error counters **

    /// Error counter for each of the 8 lanes of the 3rd stage (between-crate) data shuffle.
    uint64_t fpga_third_stage_shuffle_errors[8] = {0};

    /// Counter for flag if there is a CRC error in ANY of the third stage input lanes
    uint64_t fpga_third_stage_crc_errors = 0;

    /// Counter for flag if the packet was missing or was too short on ANY third stage input lane
    uint64_t fpga_third_stage_missing_short_errors = 0;

    /// Counter for flag if the packet was too long on ANY third stage input lane
    uint64_t fpga_third_stage_long_errors = 0;

    /// Counter for flag if the data or frame fifo has overflowed on ANY third stage input lane
    /// (sticky)
    uint64_t fpga_third_stage_fifo_overflow_errors = 0;

    /// Tracks the number of times at least one of the flags in the second or
    /// thrid stage shuffle were set.  Not including the sticky flags.
    uint64_t rx_shuffle_flags_set = 0;

    /// Prometheus metrics
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& third_shuffle_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& third_crc_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        third_missing_short_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& third_long_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        third_fifo_overflow_errors_counter;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& second_shuffle_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& second_crc_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        second_missing_short_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& second_long_errors_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        second_fifo_overflow_errors_counter;
};

iceBoardShuffle::iceBoardShuffle(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& buffer_container, int port) :
    iceBoardHandler(config, unique_name, buffer_container, port),
    third_shuffle_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_third_stage_shuffle_errors_total", unique_name,
        {"port", "fpga_lane"})),
    third_crc_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_third_stage_crc_errors_total", unique_name, {"port"})),
    third_missing_short_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_third_stage_missing_short_errors_total", unique_name, {"port"})),
    third_long_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_third_stage_long_errors_total", unique_name, {"port"})),
    third_fifo_overflow_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_third_stage_fifo_overflow_errors_total", unique_name, {"port"})),
    second_shuffle_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_second_stage_shuffle_errors_total", unique_name,
        {"port", "fpga_lane"})),
    second_crc_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_second_stage_crc_errors_total", unique_name, {"port"})),
    second_missing_short_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_second_stage_missing_short_errors_total", unique_name,
        {"port"})),
    second_long_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_second_stage_long_errors_total", unique_name, {"port"})),
    second_fifo_overflow_errors_counter(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_shuffle_fpga_second_stage_fifo_overflow_errors_total", unique_name,
        {"port"})) {

    DEBUG("iceBoardHandler: {:s}", unique_name);

    all_stream_ids[port] = {255, 255, 255, 255};

    // Read config
    fpga_dataset = config.get_default<dset_id_t>("/fpga_dataset", "id", dset_id_t::null);

    std::vector<std::string> buffer_names =
        config.get<std::vector<std::string>>(unique_name, "out_bufs");
    if (shuffle_size != buffer_names.size()) {
        throw std::runtime_error("Expecting 4 buffers, got " + std::to_string(port));
    }
    for (uint32_t i = 0; i < shuffle_size; ++i) {
        out_bufs[i] = buffer_container.get_buffer(buffer_names[i]);
        out_bufs[i]->register_producer(unique_name);
    }

    lost_samples_buf =
        buffer_container.get_buffer(config.get<std::string>(unique_name, "lost_samples_buf"));
    lost_samples_buf->register_producer(unique_name);
    // We want to make sure the flag buffers are zeroed between uses.
    zero_frames(lost_samples_buf);

    std::string endpoint_name = unique_name + "/port_data";
    kotekan::restServer::instance().register_get_callback(
        endpoint_name, [&](kotekan::connectionInstance& conn) {
            nlohmann::json info = get_json_port_info();

            std::vector<uint64_t> second_stage_errors;
            second_stage_errors.assign(fpga_second_stage_shuffle_errors,
                                       fpga_second_stage_shuffle_errors + 16);
            info["fpga_second_stage_shuffle_errors"] = second_stage_errors;
            info["fpga_second_stage_crc_errors"] = fpga_second_stage_crc_errors;
            info["fpga_second_stage_missing_short_errors"] = fpga_second_stage_missing_short_errors;
            info["fpga_second_stage_long_errors"] = fpga_second_stage_long_errors;
            info["fpga_second_stage_fifo_overflow_errors"] = fpga_second_stage_fifo_overflow_errors;

            std::vector<uint64_t> third_stage_errors;
            third_stage_errors.assign(fpga_third_stage_shuffle_errors,
                                      fpga_third_stage_shuffle_errors + 8);
            info["fpga_thrid_stage_shuffle_errors"] = third_stage_errors;
            info["fpga_third_stage_crc_errors"] = fpga_third_stage_crc_errors;
            info["fpga_third_stage_missing_short_errors"] = fpga_third_stage_missing_short_errors;
            info["fpga_third_stage_long_errors"] = fpga_third_stage_long_errors;
            info["fpga_third_stage_fifo_overflow_errors"] = fpga_third_stage_fifo_overflow_errors;

            info["shuffle_flags_set"] = rx_shuffle_flags_set;

            conn.send_json_reply(info);
        });
}

inline int iceBoardShuffle::handle_packet(struct rte_mbuf* mbuf) {

    if (!iceBoardHandler::check_packet(mbuf))
        return 0;

    if (unlikely(!got_first_packet)) {
        if (unlikely(!iceBoardHandler::align_first_packet(mbuf))) {
            return 0;
        } else {
            // Check that the set of streamIDs matches the shuffle rules.
            if (!check_stream_id())
                return -1; // Exit if check_stream_id is false.

            // Get the first set of buffer frames to write into.
            // We use last seq in case there are missing frames,
            // we want to start at the alignment point.
            // See align_first_packet for details.
            if (!advance_frames(last_seq, true))
                return -1; // Exit condition reached
        }
    } else {
        cur_seq = iceBoardHandler::get_mbuf_seq_num(mbuf);
    }

    // Check footers
    // iceBoardShuffle::check_fpga_shuffle_flags(mbuf);
    if (unlikely(!iceBoardShuffle::check_fpga_shuffle_flags(mbuf)))
        return 0;

    int64_t diff = iceBoardHandler::get_packet_diff();

    if (unlikely(!iceBoardHandler::check_for_reset(diff)))
        return -1;

    if (unlikely(!iceBoardHandler::check_order(diff)))
        return 0;

    // Handle lost packets
    // Note this handles packets for all loss reasons,
    // because we don't update the last_seq number value if the
    // packet isn't accepted for any reason.
    if (unlikely(diff > samples_per_packet))
        if (unlikely(!iceBoardShuffle::handle_lost_samples(diff - samples_per_packet)))
            return -1; // Exit condition hit, don't copy packet below.

    // copy packet
    iceBoardShuffle::copy_packet_shuffle(mbuf);

    last_seq = cur_seq;

    return 0;
}

inline bool iceBoardShuffle::check_stream_id() {

    // Lock this to only one thread at a time.
    static std::mutex alignment_mutex;
    std::lock_guard<std::mutex> alignment_lock(alignment_mutex);

    all_stream_ids[port] = port_stream_id;

    uint8_t crate_id = port_stream_id.crate_id;
    uint8_t slot_id = port_stream_id.slot_id;
    uint8_t link_id = port_stream_id.link_id;
    bool even = crate_id % 2 == 0;

    for (uint32_t i = 0; i < shuffle_size; ++i) {
        // No need to check the current port, or if the link hasn't been initialized
        if (i == port || all_stream_ids[i].crate_id == 255)
            continue;

        // Check that all the slots and links are the same.
        if (all_stream_ids[i].slot_id != slot_id || all_stream_ids[i].link_id != link_id) {
            FATAL_ERROR("One of the link_ids or slot_ids don't match! There is a cabling problem.");
            return false;
        }

        // Check that we don't have the same crate ID as another link
        // This should be impossible unless there is an FPGA problem
        if (all_stream_ids[i].crate_id == crate_id) {
            FATAL_ERROR("Two of the crate_ids are the same! There is a cabling problem.");
            return false;
        }

        // Check that all the crates are from the same group (all even/odd)
        if (even != ((all_stream_ids[i].crate_id % 2) == 0)) {
            FATAL_ERROR("The crate IDs are not all even or all odd. There is a cabling problem.");
            return false;
        }
    }
    return true;
}

inline bool iceBoardShuffle::advance_frames(uint64_t new_seq, bool first_time) {

    auto& tel = Telescope::instance();

    struct timeval now;
    gettimeofday(&now, nullptr);

    struct timespec gps_time;
    gps_time.tv_sec = 0;
    gps_time.tv_nsec = 0;
    if (tel.gps_time_enabled()) {
        gps_time = tel.to_time(new_seq);
    }

    for (uint32_t i = 0; i < shuffle_size; ++i) {
        if (!first_time) {
            mark_frame_full(out_bufs[i], unique_name.c_str(), out_buf_frame_ids[i]);

            // Advance frame ID
            out_buf_frame_ids[i] = (out_buf_frame_ids[i] + 1) % out_bufs[i]->num_frames;
        }

        out_buf_frame[i] =
            wait_for_empty_frame(out_bufs[i], unique_name.c_str(), out_buf_frame_ids[i]);
        if (out_buf_frame[i] == nullptr)
            return false;

        allocate_new_metadata_object(out_bufs[i], out_buf_frame_ids[i]);

        set_first_packet_recv_time(out_bufs[i], out_buf_frame_ids[i], now);
        set_gps_time(out_bufs[i], out_buf_frame_ids[i], gps_time);

        // We take the stream ID only from the first pair of crates,
        // to avoid overwriting it on different ports.
        // This makes the stream ID unique for down stream stages.
        if (port_stream_id.crate_id / 2 == 0) {
            ice_stream_id_t tmp_stream_id = port_stream_id;
            // Set the unused flag to store the post shuffle freq bin number.
            tmp_stream_id.unused = i;
            ice_set_stream_id_t(out_bufs[i], out_buf_frame_ids[i], tmp_stream_id);
        }

        set_fpga_seq_num(out_bufs[i], out_buf_frame_ids[i], new_seq);

        set_dataset_id(out_bufs[i], out_buf_frame_ids[i], fpga_dataset);
    }

    if (!first_time) {
        mark_frame_full(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        lost_samples_frame_id = (lost_samples_frame_id + 1) % lost_samples_buf->num_frames;
    }
    lost_samples_frame =
        wait_for_empty_frame(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
    if (lost_samples_frame == nullptr)
        return false;

    allocate_new_metadata_object(lost_samples_buf, lost_samples_frame_id);
    set_fpga_seq_num(lost_samples_buf, lost_samples_frame_id, new_seq);
    set_first_packet_recv_time(lost_samples_buf, lost_samples_frame_id, now);
    set_gps_time(lost_samples_buf, lost_samples_frame_id, gps_time);
    set_dataset_id(lost_samples_buf, lost_samples_frame_id, fpga_dataset);

    // The lost samples buffer is the same for all 4 frequencies,
    // so the stream ID actually covers all 4 possible `unused` freq values.
    if (port_stream_id.crate_id / 2 == 0) {
        ice_stream_id_t tmp_stream_id = port_stream_id;
        tmp_stream_id.unused = 0;
        ice_set_stream_id_t(lost_samples_buf, lost_samples_frame_id, tmp_stream_id);
    }

    return true;
}

inline bool iceBoardShuffle::handle_lost_samples(int64_t lost_samples) {

    // By design all the seq numbers for all frames should be the same here.
    int64_t lost_sample_location =
        last_seq + samples_per_packet - get_fpga_seq_num(out_bufs[0], out_buf_frame_ids[0]);
    uint64_t temp_seq = last_seq + samples_per_packet;

    // TODO this could be made more efficient by breaking it down into blocks of memsets.
    while (lost_samples > 0) {
        // TODO this assumes the frame size of all the output buffers are the
        // same, which should be true in all cases, but should still be tested
        // elsewhere.
        if (unlikely((size_t)(lost_sample_location * sample_size) == out_bufs[0]->frame_size)) {
            // If advance_frames() returns false then we are in shutdown mode.
            if (!advance_frames(temp_seq))
                return false;
            lost_sample_location = 0;
        }

        // This sets the flag to zero this sample with the zeroSamples stage.
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
        // WARN("port {:d}, adding lost packets at: {:d}", port, lost_sample_location);
        lost_samples_frame[lost_sample_location] = 1;
        lost_sample_location += 1;
        lost_samples -= 1;
        rx_lost_samples_total += 1;
        temp_seq += 1;
    }
    return true;
}

inline void iceBoardShuffle::copy_packet_shuffle(struct rte_mbuf* mbuf) {

    const int sample_size = 2048;
    const int sub_sample_size = sample_size / shuffle_size;

    // Where in the buf frame we should write sample.
    // TODO by construction this value should be the same for all
    // frames, but that should be proven carefully...
    int64_t sample_location;

    sample_location = cur_seq - get_fpga_seq_num(out_bufs[0], out_buf_frame_ids[0]);
    assert((size_t)(sample_location * sample_size) <= out_bufs[0]->frame_size);
    assert(sample_location >= 0);
    assert(get_mbuf_seq_num(mbuf) == cur_seq);
    if (unlikely((size_t)(sample_location * sample_size) == out_bufs[0]->frame_size)) {
        // If there are no new frames to fill, we are just dropping the packet
        if (!advance_frames(cur_seq))
            return;
        sample_location = 0;
    }

    // Where to place each of the 512 element blocks based on which crate they
    // are coming from.
    int sub_sample_pos = port_stream_id.crate_id / 2;

    // Initial packet offset, advances with each call to copy_block.
    int pkt_offset = header_offset;

    // Copy the packet in packet memory order.
    for (uint32_t sample_id = 0; sample_id < samples_per_packet; ++sample_id) {

        for (uint32_t sub_sample_freq = 0; sub_sample_freq < shuffle_size; ++sub_sample_freq) {
            uint64_t copy_location =
                (sample_location + sample_id) * sample_size + sub_sample_pos * sub_sample_size;

            copy_block(&mbuf, (uint8_t*)&out_buf_frame[sub_sample_freq][copy_location],
                       sub_sample_size, &pkt_offset);
        }
    }
}

inline bool iceBoardShuffle::check_fpga_shuffle_flags(struct rte_mbuf* mbuf) {

    const int flag_len = 4; // 32-bits = 4 bytes
    const int rounding_factor = 2;

    // Go to the last part of the packet
    // Note this assumes that the footer doesn't cross two mbuf
    // segment, but based on the packet design this should never happen.
    while (mbuf->next != nullptr) {
        mbuf = mbuf->next;
    }

    int cur_mbuf_len = mbuf->data_len;
    assert(cur_mbuf_len >= flag_len);
    assert(2048 * 2 + cur_mbuf_len - flag_len - rounding_factor
           == 4922); // Make sure the flag address is correct.
    const uint8_t* mbuf_data =
        rte_pktmbuf_mtod_offset(mbuf, uint8_t*, cur_mbuf_len - flag_len - rounding_factor);

    uint32_t flag_value = *(uint32_t*)mbuf_data;

    // If no flags (excluding the FIFO overflow flags) are set then
    // we hvae no errors to check, so we accept the packet.
    // The FIFO overflow errors are sticky bits, so we exclude them
    // in testing if a packet is valid.  However even if the packet is showing as
    // valid after excluding the sticky flags, then we should still count that the
    // sticky flag is being set.
    if ((flag_value & 0x70000700) == 0) {

        fpga_third_stage_fifo_overflow_errors += (flag_value >> 11) & 1;
        fpga_second_stage_fifo_overflow_errors += (flag_value >> 31) & 1;

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

    fpga_third_stage_missing_short_errors += (flag_value >> 8) & 1;
    fpga_third_stage_long_errors += (flag_value >> 9) & 1;
    fpga_third_stage_crc_errors += (flag_value >> 10) & 1;
    fpga_third_stage_fifo_overflow_errors += (flag_value >> 11) & 1;

    for (i = 0; i < 16; ++i) {
        fpga_second_stage_shuffle_errors[i] += (flag_value >> (12 + i)) & 1;
    }

    fpga_second_stage_missing_short_errors += (flag_value >> 28) & 1;
    fpga_second_stage_long_errors += (flag_value >> 29) & 1;
    fpga_second_stage_crc_errors += (flag_value >> 30) & 1;
    fpga_second_stage_fifo_overflow_errors += (flag_value >> 31) & 1;

    // One of the flags was set, so let's not process this packet.
    rx_shuffle_flags_set += 1;
    rx_errors_total += 1;

    return false;
}

void iceBoardShuffle::update_stats() {
    iceBoardHandler::update_stats();

    std::string port_str = std::to_string(port);

    for (int i = 0; i < 8; ++i) {
        third_shuffle_errors_counter.labels({port_str, std::to_string(i)})
            .set(fpga_third_stage_shuffle_errors[i]);
    }

    third_crc_errors_counter.labels({port_str}).set(fpga_third_stage_crc_errors);
    third_missing_short_errors_counter.labels({port_str})
        .set(fpga_third_stage_missing_short_errors);
    third_long_errors_counter.labels({port_str}).set(fpga_third_stage_long_errors);
    third_fifo_overflow_errors_counter.labels({port_str})
        .set(fpga_third_stage_fifo_overflow_errors);

    for (int i = 0; i < 16; ++i) {
        second_shuffle_errors_counter.labels({port_str, std::to_string(i)})
            .set(fpga_second_stage_shuffle_errors[i]);
    }

    second_crc_errors_counter.labels({port_str}).set(fpga_second_stage_crc_errors);
    second_missing_short_errors_counter.labels({port_str})
        .set(fpga_second_stage_missing_short_errors);
    second_long_errors_counter.labels({port_str}).set(fpga_second_stage_long_errors);
    second_fifo_overflow_errors_counter.labels({port_str})
        .set(fpga_second_stage_fifo_overflow_errors);
}

#endif
