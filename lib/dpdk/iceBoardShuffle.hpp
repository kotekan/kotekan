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
#include "chordMetadata.hpp"
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
 *       @buffer_metadata chordMetadata
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
 * @conf  link_group            Unsigned int.  The link group this handler is a member of.
 * @conf  link_group_subid      Int.  The position of this handler within the link group.
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

    /// The maximum number of link groups.
    static const uint32_t link_group_max = 8;

    /// The stream_ids for all iceBoardShuffle objects.
    /// There are 4 stream_ids per link group.
    inline static ice_stream_id_t all_stream_ids[link_group_max][shuffle_size];

    /// The link group this handler is a member of, must be less than link_group_max
    uint32_t link_group;

    /// The position of this handler within the link group,
    /// must be less than shuffle_size.
    uint32_t link_group_subid;

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

    link_group = config.get<uint32_t>(unique_name, "link_group");
    link_group_subid = config.get<uint32_t>(unique_name, "link_group_subid");

    if (link_group >= link_group_max) {
        throw std::runtime_error(fmt::format("link_group {} is too large.  Maximum is {}",
                                             link_group, link_group_max - 1));
    }

    if (link_group_subid >= shuffle_size) {
        throw std::runtime_error(fmt::format("link_group_subid {} is too large.  Maximum is {}",
                                             link_group_subid, shuffle_size - 1));
    }

    all_stream_ids[link_group][link_group_subid] = {255, 255, 255, 255};

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
        /* new style array description */
        out_bufs[i]
            ->allocate_ndarray_frame_desc<
                kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type, 3>(
                "E", {1, ptrdiff_t(out_bufs[i]->frame_size) / sample_size, sample_size},
                {"F", "T", "E"});
    }

    lost_samples_buf =
        buffer_container.get_buffer(config.get<std::string>(unique_name, "lost_samples_buf"));
    lost_samples_buf->register_producer(unique_name);
    // We want to make sure the flag buffers are zeroed between uses.
    lost_samples_buf->zero_frames();

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

    // Lock this to only one thread per groups at a time.
    static std::mutex alignment_mutex[link_group_max];
    std::lock_guard<std::mutex> alignment_lock(alignment_mutex[link_group]);

    all_stream_ids[link_group][link_group_subid] = port_stream_id;

    uint8_t crate_id = port_stream_id.crate_id;
    uint8_t slot_id = port_stream_id.slot_id;
    uint8_t link_id = port_stream_id.link_id;
    bool even = crate_id % 2 == 0;

    for (uint32_t i = 0; i < shuffle_size; ++i) {
        // No need to check the current port, or if the link hasn't been initialized
        if (i == link_group_subid || all_stream_ids[link_group][i].crate_id == 255)
            continue;

        // Check that all the slots and links are the same.
        if (all_stream_ids[link_group][i].slot_id != slot_id
            || all_stream_ids[link_group][i].link_id != link_id) {
            /// Print out all the stream IDs for this group.
            for (uint32_t j = 0; j < shuffle_size; ++j) {
                ERROR("link_group {:d} subid {:d} stream_id: crate {:d} slot {:d} link {:d} "
                      "dpdk_port {:d}",
                      link_group, j, all_stream_ids[link_group][j].crate_id,
                      all_stream_ids[link_group][j].slot_id, all_stream_ids[link_group][j].link_id,
                      port);
            }
            FATAL_ERROR("At least one of the link_ids or slot_ids don't match! There is a cabling "
                        "problem.");
            return false;
        }

        // Check that we don't have the same crate ID as another link
        // This should be impossible unless there is an FPGA problem
        if (all_stream_ids[link_group][i].crate_id == crate_id) {
            FATAL_ERROR("Two of the crate_ids are the same! There is a cabling problem.");
            return false;
        }

        // Check that all the crates are from the same group (all even/odd)
        if (even != ((all_stream_ids[link_group][i].crate_id % 2) == 0)) {
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
            out_bufs[i]->mark_frame_full(unique_name, out_buf_frame_ids[i]);

            // Advance frame ID
            out_buf_frame_ids[i] = (out_buf_frame_ids[i] + 1) % out_bufs[i]->num_frames;
        }

        out_buf_frame[i] = out_bufs[i]->wait_for_empty_frame(unique_name, out_buf_frame_ids[i]);
        if (out_buf_frame[i] == nullptr)
            return false;

        out_bufs[i]->allocate_new_metadata_object(out_buf_frame_ids[i]);

        // Add metadata to the output buffer

        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_first_packet_recv_time(now);
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_fpga_seq_num(new_seq);
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_time_downsampling_fpga(1);
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_gps_time(gps_time);
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_dataset_id(fpga_dataset);

        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])
            ->set_freq_upchan_factor(std::vector<int>(1 /* nfreq */, 1));
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])
            ->set_freq_upchan_index(std::vector<int>(1 /* nfreq */, 0));

        ice_stream_id_t tmp_stream_id = port_stream_id;
        // Set the unused flag to store the post shuffle freq bin number.
        tmp_stream_id.unused = i;
        tmp_stream_id.crate_id = tmp_stream_id.crate_id % 2;
        std::vector<int> coarse_freq(1);
        coarse_freq[0] = tel.to_freq_id(ice_encode_stream_id(tmp_stream_id));
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_coarse_freq(coarse_freq);

        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_lost_timesamples(0);
        get_chord_metadata(out_bufs[i], out_buf_frame_ids[i])->set_rfi_flagged_samples(0);

        auto meta = get_chord_metadata(out_bufs[i], out_buf_frame_ids[i]);

        // HACK for now. DONOTPUSH
        meta->ndishes = 1024;
        meta->n_dish_locations_ew = 4;
        meta->n_dish_locations_ns = 256;
        const static dish_index_t const_dish_indices[1024] = {
            0, 256, 512, 768,
            1, 257, 513, 769,
            2, 258, 514, 770,
            3, 259, 515, 771,
            4, 260, 516, 772,
            5, 261, 517, 773,
            6, 262, 518, 774,
            7, 263, 519, 775,
            8, 264, 520, 776,
            9, 265, 521, 777,
            10, 266, 522, 778,
            11, 267, 523, 779,
            12, 268, 524, 780,
            13, 269, 525, 781,
            14, 270, 526, 782,
            15, 271, 527, 783,
            16, 272, 528, 784,
            17, 273, 529, 785,
            18, 274, 530, 786,
            19, 275, 531, 787,
            20, 276, 532, 788,
            21, 277, 533, 789,
            22, 278, 534, 790,
            23, 279, 535, 791,
            24, 280, 536, 792,
            25, 281, 537, 793,
            26, 282, 538, 794,
            27, 283, 539, 795,
            28, 284, 540, 796,
            29, 285, 541, 797,
            30, 286, 542, 798,
            31, 287, 543, 799,
            32, 288, 544, 800,
            33, 289, 545, 801,
            34, 290, 546, 802,
            35, 291, 547, 803,
            36, 292, 548, 804,
            37, 293, 549, 805,
            38, 294, 550, 806,
            39, 295, 551, 807,
            40, 296, 552, 808,
            41, 297, 553, 809,
            42, 298, 554, 810,
            43, 299, 555, 811,
            44, 300, 556, 812,
            45, 301, 557, 813,
            46, 302, 558, 814,
            47, 303, 559, 815,
            48, 304, 560, 816,
            49, 305, 561, 817,
            50, 306, 562, 818,
            51, 307, 563, 819,
            52, 308, 564, 820,
            53, 309, 565, 821,
            54, 310, 566, 822,
            55, 311, 567, 823,
            56, 312, 568, 824,
            57, 313, 569, 825,
            58, 314, 570, 826,
            59, 315, 571, 827,
            60, 316, 572, 828,
            61, 317, 573, 829,
            62, 318, 574, 830,
            63, 319, 575, 831,
            64, 320, 576, 832,
            65, 321, 577, 833,
            66, 322, 578, 834,
            67, 323, 579, 835,
            68, 324, 580, 836,
            69, 325, 581, 837,
            70, 326, 582, 838,
            71, 327, 583, 839,
            72, 328, 584, 840,
            73, 329, 585, 841,
            74, 330, 586, 842,
            75, 331, 587, 843,
            76, 332, 588, 844,
            77, 333, 589, 845,
            78, 334, 590, 846,
            79, 335, 591, 847,
            80, 336, 592, 848,
            81, 337, 593, 849,
            82, 338, 594, 850,
            83, 339, 595, 851,
            84, 340, 596, 852,
            85, 341, 597, 853,
            86, 342, 598, 854,
            87, 343, 599, 855,
            88, 344, 600, 856,
            89, 345, 601, 857,
            90, 346, 602, 858,
            91, 347, 603, 859,
            92, 348, 604, 860,
            93, 349, 605, 861,
            94, 350, 606, 862,
            95, 351, 607, 863,
            96, 352, 608, 864,
            97, 353, 609, 865,
            98, 354, 610, 866,
            99, 355, 611, 867,
            100, 356, 612, 868,
            101, 357, 613, 869,
            102, 358, 614, 870,
            103, 359, 615, 871,
            104, 360, 616, 872,
            105, 361, 617, 873,
            106, 362, 618, 874,
            107, 363, 619, 875,
            108, 364, 620, 876,
            109, 365, 621, 877,
            110, 366, 622, 878,
            111, 367, 623, 879,
            112, 368, 624, 880,
            113, 369, 625, 881,
            114, 370, 626, 882,
            115, 371, 627, 883,
            116, 372, 628, 884,
            117, 373, 629, 885,
            118, 374, 630, 886,
            119, 375, 631, 887,
            120, 376, 632, 888,
            121, 377, 633, 889,
            122, 378, 634, 890,
            123, 379, 635, 891,
            124, 380, 636, 892,
            125, 381, 637, 893,
            126, 382, 638, 894,
            127, 383, 639, 895,
            128, 384, 640, 896,
            129, 385, 641, 897,
            130, 386, 642, 898,
            131, 387, 643, 899,
            132, 388, 644, 900,
            133, 389, 645, 901,
            134, 390, 646, 902,
            135, 391, 647, 903,
            136, 392, 648, 904,
            137, 393, 649, 905,
            138, 394, 650, 906,
            139, 395, 651, 907,
            140, 396, 652, 908,
            141, 397, 653, 909,
            142, 398, 654, 910,
            143, 399, 655, 911,
            144, 400, 656, 912,
            145, 401, 657, 913,
            146, 402, 658, 914,
            147, 403, 659, 915,
            148, 404, 660, 916,
            149, 405, 661, 917,
            150, 406, 662, 918,
            151, 407, 663, 919,
            152, 408, 664, 920,
            153, 409, 665, 921,
            154, 410, 666, 922,
            155, 411, 667, 923,
            156, 412, 668, 924,
            157, 413, 669, 925,
            158, 414, 670, 926,
            159, 415, 671, 927,
            160, 416, 672, 928,
            161, 417, 673, 929,
            162, 418, 674, 930,
            163, 419, 675, 931,
            164, 420, 676, 932,
            165, 421, 677, 933,
            166, 422, 678, 934,
            167, 423, 679, 935,
            168, 424, 680, 936,
            169, 425, 681, 937,
            170, 426, 682, 938,
            171, 427, 683, 939,
            172, 428, 684, 940,
            173, 429, 685, 941,
            174, 430, 686, 942,
            175, 431, 687, 943,
            176, 432, 688, 944,
            177, 433, 689, 945,
            178, 434, 690, 946,
            179, 435, 691, 947,
            180, 436, 692, 948,
            181, 437, 693, 949,
            182, 438, 694, 950,
            183, 439, 695, 951,
            184, 440, 696, 952,
            185, 441, 697, 953,
            186, 442, 698, 954,
            187, 443, 699, 955,
            188, 444, 700, 956,
            189, 445, 701, 957,
            190, 446, 702, 958,
            191, 447, 703, 959,
            192, 448, 704, 960,
            193, 449, 705, 961,
            194, 450, 706, 962,
            195, 451, 707, 963,
            196, 452, 708, 964,
            197, 453, 709, 965,
            198, 454, 710, 966,
            199, 455, 711, 967,
            200, 456, 712, 968,
            201, 457, 713, 969,
            202, 458, 714, 970,
            203, 459, 715, 971,
            204, 460, 716, 972,
            205, 461, 717, 973,
            206, 462, 718, 974,
            207, 463, 719, 975,
            208, 464, 720, 976,
            209, 465, 721, 977,
            210, 466, 722, 978,
            211, 467, 723, 979,
            212, 468, 724, 980,
            213, 469, 725, 981,
            214, 470, 726, 982,
            215, 471, 727, 983,
            216, 472, 728, 984,
            217, 473, 729, 985,
            218, 474, 730, 986,
            219, 475, 731, 987,
            220, 476, 732, 988,
            221, 477, 733, 989,
            222, 478, 734, 990,
            223, 479, 735, 991,
            224, 480, 736, 992,
            225, 481, 737, 993,
            226, 482, 738, 994,
            227, 483, 739, 995,
            228, 484, 740, 996,
            229, 485, 741, 997,
            230, 486, 742, 998,
            231, 487, 743, 999,
            232, 488, 744, 1000,
            233, 489, 745, 1001,
            234, 490, 746, 1002,
            235, 491, 747, 1003,
            236, 492, 748, 1004,
            237, 493, 749, 1005,
            238, 494, 750, 1006,
            239, 495, 751, 1007,
            240, 496, 752, 1008,
            241, 497, 753, 1009,
            242, 498, 754, 1010,
            243, 499, 755, 1011,
            244, 500, 756, 1012,
            245, 501, 757, 1013,
            246, 502, 758, 1014,
            247, 503, 759, 1015,
            248, 504, 760, 1016,
            249, 505, 761, 1017,
            250, 506, 762, 1018,
            251, 507, 763, 1019,
            252, 508, 764, 1020,
            253, 509, 765, 1021,
            254, 510, 766, 1022,
            255, 511, 767, 1023,
        };
        meta->dish_index = const_cast<dish_index_t*>(const_dish_indices);

        // The dimensions are time (T) and "element" (E) which is the "correlator ordered"
        // feed and polarization.  Note that off the F-engine polarization is _not_ a defined
        // axis.
        /* old style array description */
        std::strncpy(meta->dim_name[0], "F", sizeof meta->dim_name[0]);
        std::strncpy(meta->dim_name[1], "T", sizeof meta->dim_name[1]);
        std::strncpy(meta->dim_name[2], "E", sizeof meta->dim_name[2]);
        meta->dims = 3;
        meta->type = kotekan::int4x2_swapped_withoffset;
        // Somewhat confusingly E in this context is the electric field...
        std::strncpy(meta->name, "E", sizeof meta->name);
        meta->dim[0] = 1;
        meta->dim[1] = out_bufs[i]->frame_size / sample_size;
        meta->dim[2] = sample_size;
        meta->set_strides_simple();
        // frame_desc set in constructor
        /* test that things are consistent */
        meta->check_frame_desc(out_bufs[i]->get_ndarray_frame_desc());

        // Print out the chordMetadata
        DEBUG("chordMetadata: seq: {:d} freq_id: {:d} dim[0]: {:d} dim[1]: {:d}",
              meta->get_fpga_seq_num(), meta->get_coarse_freq()[0], meta->dim[0], meta->dim[1]);
    }

    if (!first_time) {
        lost_samples_buf->mark_frame_full(unique_name, lost_samples_frame_id);
        lost_samples_frame_id = (lost_samples_frame_id + 1) % lost_samples_buf->num_frames;
    }
    lost_samples_frame = lost_samples_buf->wait_for_empty_frame(unique_name, lost_samples_frame_id);
    if (lost_samples_frame == nullptr)
        return false;

    // Add metadata to the lost samples buffer
    lost_samples_buf->allocate_new_metadata_object(lost_samples_frame_id);
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_fpga_seq_num(new_seq);
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_time_downsampling_fpga(1);
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)
        ->set_freq_upchan_factor(std::vector<int>(1 /* nfreq */, 1));
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)
        ->set_freq_upchan_index(std::vector<int>(1 /* nfreq */, 0));
    // TODO: are these required for the lost_samples buffer? or is having them
    // in the corresponding data buffer sufficient?
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_first_packet_recv_time(now);
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_gps_time(gps_time);
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_dataset_id(fpga_dataset);

    // Set the frequency bins
    // Note that there are 4 frequencies assoicated with this packet loss buffer
    // because of the final stage shuffle, but that dimension is collapsed.
    ice_stream_id_t tmp_stream_id = port_stream_id;

    std::vector<int> coarse_freq(4);
    for (int i = 0; i < 4; i++) {
        tmp_stream_id.unused = i;
        tmp_stream_id.crate_id = tmp_stream_id.crate_id % 2;
        coarse_freq[i] = tel.to_freq_id(ice_encode_stream_id(tmp_stream_id));
    }
    get_chord_metadata(lost_samples_buf, lost_samples_frame_id)->set_coarse_freq(coarse_freq);

    auto meta = get_chord_metadata(lost_samples_buf, lost_samples_frame_id);

    meta->type = kotekan::uint8;

    std::strncpy(meta->dim_name[0], "T", sizeof meta->dim_name[0]);
    meta->dim[0] = lost_samples_buf->frame_size; // One byte per time sample
    meta->dims = 1;
    std::strncpy(meta->name, "lost_samples", sizeof meta->name);
    meta->set_strides_simple();
    /* new style array description */
    lost_samples_buf->allocate_ndarray_frame_desc<kotekan::GetType<kotekan::uint8>::type, 1>(
        "lost_samples", {ptrdiff_t(lost_samples_buf->frame_size)}, {"T"});
    /* test that things are consistent */
    meta->check_frame_desc(lost_samples_buf->get_ndarray_frame_desc());


    return true;
}

inline bool iceBoardShuffle::handle_lost_samples(int64_t lost_samples) {

    // By design all the seq numbers for all frames should be the same here.
    int64_t lost_sample_location;

    if (out_bufs[0]->metadata_pool->type_name == "chordMetadata") {
        lost_sample_location =
            last_seq + samples_per_packet
            - get_chord_metadata(out_bufs[0], out_buf_frame_ids[0])->get_fpga_seq_num();
    } else {
        FATAL_ERROR("Unsupported metadata type: {:s}", out_bufs[0]->metadata_pool->type_name);
        return false;
    }

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

    if (out_bufs[0]->metadata_pool->type_name == "chordMetadata") {
        sample_location =
            cur_seq - get_chord_metadata(out_bufs[0], out_buf_frame_ids[0])->get_fpga_seq_num();
    } else {
        FATAL_ERROR("Unsupported metadata type: {:s}", out_bufs[0]->metadata_pool->type_name);
        return;
    }

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
